import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from models.layers import MultiHeadSpatialNet, CrossFFN


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim=2048, num_heads=8):
        """ Attetion module with vectorized version

        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation
            nongt_dim: number of objects consider relations per image
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
            m: dimension of memory matrix
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(self.feat_dim, self.feat_dim)
        self.key = nn.Linear(self.feat_dim, self.feat_dim)
        self.value = nn.Linear(self.feat_dim, self.feat_dim)
        self.linear_out_ = nn.Conv2d(in_channels=self.num_heads * self.feat_dim,
                                     out_channels=self.feat_dim,
                                     kernel_size=(1, 1),
                                     groups=self.num_heads)

    def forward(self, obj_feats, cross_feats, adj_matrix, label_biases_att):
        """
        Args:
            obj_feats: [B, N, feat_dim]
            adj_matrix: [B, N, N]
            position_embedding: [N, N, pos_emb_dim]
            text_feats: [B, len, feat_dim]
        Returns:
            output: [B, N, output_dim]
        """
        B, N = obj_feats.shape[:2]

        # Q
        q_data = self.query(obj_feats)
        q_data_batch = q_data.view(B, N, self.num_heads, self.head_dim)
        q_data_batch = torch.transpose(q_data_batch, 1, 2)  # [B,num_heads, N, head_dim]
        #
        # K
        k_data = self.key(cross_feats)
        k_data_batch = k_data.view(B, N, self.num_heads, self.head_dim)
        k_data_batch = torch.transpose(k_data_batch, 1, 2)  # [B, num_heads,N, head_dim]

        # V
        v_data = self.value(cross_feats)

        att = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))  # [B, num_heads, N, N]
        att = (1.0 / math.sqrt(float(self.head_dim))) * att
        weighted_att = att.transpose(1, 2)  # (B,N,num_heads,N)

        if adj_matrix is not None:
            weighted_att = weighted_att.transpose(2, 3)  # [B,N, N, num_heads]
            zero_vec = -9e15 * torch.ones_like(weighted_att)

            adj_matrix = adj_matrix.view(adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2], 1)
            adj_matrix_expand = adj_matrix.expand((-1, -1, -1, weighted_att.shape[-1]))
            weighted_att_masked = torch.where(adj_matrix_expand > 0, weighted_att, zero_vec)

            weighted_att_masked = weighted_att_masked + label_biases_att.unsqueeze(-1)

            weighted_att = weighted_att_masked
            weighted_att = weighted_att.transpose(2, 3)

        # aff_softmax, [B, N, num_heads, N]
        att_softmax = nn.functional.softmax(weighted_att, 3)
        aff_softmax_reshape = att_softmax.view((B, N, self.num_heads, -1))

        output_t = torch.matmul(aff_softmax_reshape.reshape(B, N * self.num_heads, -1),
                                v_data)  # (B,N*num_heads,N)*(B,N,768)
        output_t = output_t.view((-1, self.num_heads * self.feat_dim, 1, 1))
        linear_out = self.linear_out_(output_t)
        output = linear_out.view((B, N, self.feat_dim))

        return output


class Re_Encoder(nn.Module):
    def __init__(self, lay_num, gat_num, in_feat_dim, out_feat_dim, binary_num, n_ary_num, hidden_dim=2048,
                 dropout=0.15,
                 num_heads=16, ):
        super(Re_Encoder, self).__init__()
        self.lay_num = lay_num
        self.gat_num = gat_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.dropout = nn.Dropout(dropout)
        self.feat_fc = nn.Linear(in_feat_dim, out_feat_dim, bias=True)
        self.bias = nn.Linear(1, 1, bias=True)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.binary_k = binary_num
        self.n_ary_k = n_ary_num

        self.re_crossattn = CrossFFN(n_head=self.num_heads, d_model=self.out_feat_dim, d_hidden=hidden_dim,
                                     dropout=dropout)
        self.multire_crossattn = CrossFFN(n_head=self.num_heads, d_model=self.out_feat_dim, d_hidden=hidden_dim,
                                          dropout=dropout)
        self.re_clf1 = nn.Sequential(nn.Linear(self.out_feat_dim, self.out_feat_dim // 2),
                                     nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(self.out_feat_dim // 2, 1))
        self.re_clf2 = nn.Sequential(nn.Linear(self.out_feat_dim, self.out_feat_dim // 2),
                                     nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(self.out_feat_dim // 2, 1))

        self.sp_agg = nn.ModuleList([MultiHeadSpatialNet(n_head=self.num_heads, d_model=self.out_feat_dim,
                                                         d_hidden=self.hidden_dim, dropout=dropout)
                                     for _ in range(lay_num)])

        self.obj_crossattns = nn.ModuleList([CrossFFN(n_head=num_heads, d_model=out_feat_dim, d_hidden=self.hidden_dim,
                                                      dropout=dropout)
                                             for _ in range(lay_num)])

        self.g_atts = nn.ModuleList([GraphSelfAttentionLayer(hidden_dim=self.hidden_dim,
                                                             num_heads=num_heads,
                                                             feat_dim=out_feat_dim)
                                     for _ in range(gat_num)])

    def pair_aware_perbatch(self, obj_feats, text_feats, geo_feats, mask):
        R, N = obj_feats.shape[:2]
        obj_inds = torch.arange(0, N).to(obj_feats.device)
        v_text_feats = text_feats[None].repeat(R, 1, 1)
        obj_feats = self.re_crossattn(obj_feats.reshape(R, N, -1), v_text_feats)  # R, N, C
        obj2obj_feats = (obj_feats[:, None] * obj_feats[:, :, None] + geo_feats).reshape(R, N * N, -1)
        # obj2obj_feats = (obj_feats[:, None] * obj_feats[:, :, None]).reshape(R, N * N, -1)
        o2o_logits = self.re_clf1(obj2obj_feats).squeeze(-1)  # R, N * N
        o2o_logits = o2o_logits.reshape(R, -1).sum(dim=0) / R
        pair_probs = o2o_logits.sigmoid() * mask.ravel()

        obj2obj_inds = torch.stack([obj_inds[None].repeat(N, 1), obj_inds[:, None].repeat(1, N)],
                                   dim=-1).reshape(N * N, 2)
        k = min(self.binary_k, N * N)
        _, topk_inds = torch.topk(pair_probs, k, dim=0)  # K
        a_obj2obj_feats = obj2obj_feats.sum(dim=0) / R
        topk_obj2obj_feats = a_obj2obj_feats[topk_inds, :]  # ( K, C)
        topk_obj2obj_inds = obj2obj_inds[topk_inds, :]

        if self.n_ary_k > 0:
            pair_feats = self.multire_crossattn(topk_obj2obj_feats[None], text_feats[None]).squeeze(0)  # K, C
            pair2pair_feats = (pair_feats[None] * pair_feats[:, None]).reshape(k * k, -1)  # B, K * K, C
            p2p_logits = self.re_clf2(pair2pair_feats[None]).squeeze(-1).squeeze(0).reshape(k, k)
            p2p_inds = torch.cat(
                [topk_obj2obj_inds[None].repeat(k, 1, 1), topk_obj2obj_inds[:, None].repeat(1, k, 1)],
                dim=-1).reshape(k, k, 4)
        else:
            p2p_logits, p2p_inds = None, None

        return obj_feats.reshape(R, N, -1), obj2obj_feats.reshape(R, N, N, -1), \
               o2o_logits.reshape(N, N), p2p_logits, p2p_inds, topk_obj2obj_inds

    def forward(self, obj_feats, geo_feats, text_feats, box_feats, real_nums):
        B, R, N = box_feats.shape[:3]
        mask = torch.zeros(B, N, N).to(obj_feats.device)
        for bi in range(B):
            mask[bi, :real_nums[bi], :real_nums[bi]] = 1.0
            mask[bi, torch.arange(N), torch.arange(N)] = 0

        v_text_feats = text_feats[:, None].repeat(1, R, 1, 1).reshape(B * R, -1, self.in_feat_dim)  # (B*R,len,768)
        obj_feats = obj_feats[:, None] + box_feats

        K = self.binary_k
        geo_feats = geo_feats.reshape(B, R, N, N, -1)
        res_obj_feats = torch.zeros_like(obj_feats).to(obj_feats.device)
        res_obj2obj_feats = torch.zeros_like(geo_feats).to(obj_feats.device)
        o2o_logits = torch.zeros(B, N, N).to(obj_feats.device) - 1e+5
        p2p_logits = torch.zeros(B, K, K).to(obj_feats.device) - 1e+5
        p2p_inds = torch.zeros(B, K, K, 4).to(obj_feats.device) - 1
        top_o2o_inds = torch.zeros(B, K, 2).to(obj_feats.device) - 1
        for bi in range(B):
            bi_obj_feats, bi_obj2obj_feats, bi_o2o_logits, bi_p2p_logits, bi_p2p_inds, bi_o2o_inds = \
                self.pair_aware_perbatch(obj_feats[bi, :, :real_nums[bi]], text_feats[bi],
                                         geo_feats[bi, :, :real_nums[bi], :real_nums[bi]],
                                         mask[bi, :real_nums[bi], :real_nums[bi]])
            res_obj_feats[bi, :, : real_nums[bi]] = bi_obj_feats
            res_obj2obj_feats[bi, :, :real_nums[bi], :real_nums[bi]] = bi_obj2obj_feats
            o2o_logits[bi, :real_nums[bi], :real_nums[bi]] = bi_o2o_logits
            top_o2o_inds[bi, :bi_o2o_inds.shape[0]] = bi_o2o_inds
            if self.n_ary_k > 0:
                k = bi_p2p_logits.shape[0]
                p2p_logits[bi, :k, :k] = bi_p2p_logits
                p2p_inds[bi, :k, :k] = bi_p2p_inds

        if self.n_ary_k > 0:
            p2p_inds = p2p_inds.reshape(B, K * K, -1)
            p2p_logits = p2p_logits.reshape(B, -1)
            _, topk_inds = torch.topk(p2p_logits, self.n_ary_k, dim=1)
            topk_p2p_inds = p2p_inds[torch.arange(B)[:, None], topk_inds, :]
            adj_matrix = torch.zeros(B, N, N, 1).to(obj_feats.device)
            for bi in range(B):
                indx = topk_p2p_inds[bi].long()
                adj_matrix[bi, indx[:, 0], indx[:, 1]] = 1
                adj_matrix[bi, indx[:, 1], indx[:, 0]] = 1
                adj_matrix[bi, indx[:, 2], indx[:, 3]] = 1
                adj_matrix[bi, indx[:, 3], indx[:, 2]] = 1
            adj_matrix = adj_matrix * mask[..., None]
        elif self.binary_k > 0:
            adj_matrix = torch.zeros(B, N, N, 1).to(obj_feats.device)
            for bi in range(B):
                indx = top_o2o_inds[bi].long()
                adj_matrix[bi, indx[:, 0], indx[:, 1]] = 1
                adj_matrix[bi, indx[:, 1], indx[:, 0]] = 1
            adj_matrix = adj_matrix * mask[..., None]
        else:
            adj_matrix = mask[..., None]

        input_adj_matrix = adj_matrix[:, None].repeat(1, R, 1, 1, 1).reshape(B * R, N, N, 1)
        v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)  # (B,N,N)
        obj_feats = res_obj_feats.reshape(B * R, N, -1)
        for i in range(self.lay_num):
            fuse_feats = obj_feats[:, None].repeat(1, N, 1, 1) + res_obj2obj_feats.reshape(B * R, N, N, -1)
            dec_feats = self.sp_agg[i](fuse_feats.reshape(B * R * N, N, -1)).reshape(B * R, N, -1)
            obj_feats = self.obj_crossattns[i](obj_feats + dec_feats, v_text_feats)  # obj-text
            if i < self.gat_num:
                re_feats = self.g_atts[i](obj_feats, obj_feats, input_adj_matrix, v_biases_neighbors)
                obj_feats = (obj_feats + re_feats.reshape(B * R, N, -1))

        return obj_feats.reshape(B, R, N, -1).sum(dim=1) / R, o2o_logits, p2p_logits, p2p_inds
