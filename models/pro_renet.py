import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.backbones.point_net_pp import PointNetNew, PointNetPP
from transformers import BertModel, BertConfig
import time
from models.utils import *
from models.gat.matrix_emb import *
from models.gat.re_att2 import Re_Encoder


class MultiReNet(nn.Module):

    def __init__(self,
                 cfg,
                 n_obj_classes,
                 ignore_index,
                 class_name_tokens=None,
                 clip_text_feats=None):

        super().__init__()
        self.view_number = cfg.view_number
        self.rotate_number = cfg.rotate_number
        self.points_per_object = cfg.points_per_object
        self.class2idx = cfg.class_to_idx
        self.ignore_index = ignore_index

        self.n_obj_classes = n_obj_classes
        self.class_name_tokens = class_name_tokens
        self.clip_text_feats = clip_text_feats
        self.clip_dim = cfg.clip_dim

        self.object_dim = cfg.object_latent_dim
        self.inner_dim = cfg.inner_dim
        self.head_num = cfg.head_num
        self.hidden_dim = cfg.hidden_dim

        self.dropout_rate = cfg.dropout_rate
        self.lang_cls_alpha = cfg.lang_cls_alpha
        self.pp_cls_alpha = cfg.pp_cls_alpha

        self.clip_point = cfg.clip_pp
        self.clip_pp_dim = cfg.clip_pp_dim
        self.geo_dim = cfg.geo_dim

        self.binary_k = cfg.binary_num
        self.n_ary_k = cfg.n_ary_num

        if self.clip_point:
            self.object_encoder = PointNetNew(sa_n_points=[32, 16, None],
                                              sa_n_samples=[[32], [32], [None]],
                                              sa_radii=[[0.2], [0.4], [None]],
                                              sa_mlps=[[[3, 64, 64, 128]],
                                                       [[128, 128, 128, 256]],
                                                       [[256, 256, self.object_dim, self.object_dim]]])
        else:
            self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                             sa_n_samples=[[32], [32], [None]],
                                             sa_radii=[[0.2], [0.4], [None]],
                                             sa_mlps=[[[3, 64, 64, 128]],
                                                      [[128, 128, 128, 256]],
                                                      [[256, 256, self.object_dim, self.object_dim]]])

        self.relation_encoder = Re_Encoder(lay_num=cfg.lay_number,
                                           gat_num=cfg.gat_number,
                                           in_feat_dim=self.inner_dim,
                                           out_feat_dim=self.inner_dim,
                                           num_heads=self.head_num,
                                           hidden_dim=self.hidden_dim,
                                           binary_num=self.binary_k,
                                           n_ary_num=self.n_ary_k)

        self.language_encoder = BertModel.from_pretrained(cfg.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:3]

        # Classifier heads
        self.language_target_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                                 nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim // 2, n_obj_classes))

        self.pp_object_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                           nn.ReLU(), nn.Dropout(self.dropout_rate),
                                           nn.Linear(self.inner_dim // 2, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim // 2),
                                                 nn.ReLU(), nn.Dropout(self.dropout_rate),
                                                 nn.Linear(self.inner_dim // 2, 1))

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(6, self.inner_dim),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.inner_dim),
        )

        self.geo_feature_mapping = nn.Sequential(
            nn.Linear(self.geo_dim, self.inner_dim),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.inner_dim),
        )

        if self.clip_point:
            self.clip_feature_mapping = nn.Sequential(
                nn.Linear(cfg.clip_dim, self.clip_pp_dim),
                nn.Dropout(self.dropout_rate),
                nn.LayerNorm(self.clip_pp_dim),
            )

        self.logit_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.n_ary_loss = nn.BCEWithLogitsLoss()
        self.binary_loss = nn.BCEWithLogitsLoss()

    def get_bsize(self, r, bsize):
        B = r.shape[0]
        new_size = bsize.clone()
        for bi in range(B):
            if r[bi] == 1 or r[bi] == 3:
                new_size[bi, :, 0] = bsize[bi, :, 1]
                new_size[bi, :, 1] = bsize[bi, :, 0]
        return new_size

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)  # (B,52,1024,7)
        box_infos = box_infos.float().to(self.device)  # (B,52,6) cx,cy,cz,lx,ly,lz
        xyz = input_points[:, :, :, :3]  # (B,52,1024,3)
        bxyz = box_infos[:, :, :3]  # B,52,3
        bsize = box_infos[:, :, -3:]
        B, N, P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)]).to(
            self.device)
        view_theta_arr = torch.Tensor([i * 2.0 * np.pi / self.view_number for i in range(self.view_number)]).to(
            self.device)

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            r = torch.randint(0, self.rotate_number, (B,))
            theta = rotate_theta_arr[r]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(self.device)[
                None].repeat(B, 1, 1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B, N * P, 3), rotate_matrix).reshape(B, N, P, 3)
            bxyz = torch.matmul(bxyz.reshape(B, N, 3), rotate_matrix).reshape(B, N, 3)
            bsize = self.get_bsize(r, bsize)

        # multi-view
        boxs = []
        for r, theta in enumerate(view_theta_arr):
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                          [math.sin(theta), math.cos(theta), 0.0],
                                          [0.0, 0.0, 1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            new_size = self.get_bsize(torch.zeros((B,)) + r, bsize)
            boxs.append(torch.cat([rxyz, new_size], dim=-1))

        boxs = torch.stack(boxs, dim=1)  # (B,view_num,N,4)
        if self.view_number == 1:
            boxs = torch.squeeze(boxs, 1)
        return input_points, boxs

    def n_ary_relation_loss(self, p2p_logits, p2p_inds, target_pos):
        p2p_labels = p2p_inds == target_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, p2p_inds.shape[1], 4)
        no_target = p2p_labels.float().sum(dim=-1) == 0
        neg_samples = p2p_logits[no_target]
        neg_loss = self.n_ary_loss(neg_samples, torch.zeros_like(neg_samples).to(self.device))

        max_probs = []
        for bi in range(p2p_logits.shape[0]):
            bi_ll = p2p_logits[bi][~no_target[bi]]
            if len(bi_ll) == 0:
                continue
            max_probs.append(max(bi_ll))
        if len(max_probs) > 0:
            max_probs = torch.stack(max_probs).sigmoid()
            pos_loss = -torch.log(max_probs + 1e-8).mean()
        else:
            pos_loss = 0.0

        return (neg_loss + pos_loss) / 2

    def forward(self, batch: dict):
        self.device = self.obj_feature_mapping[0].weight.device

        # rotation augmentation and multi_view generation
        obj_points, boxes = self.aug_input(batch['objects'], batch['box_info'])
        B, R, N = boxes.shape[:3]

        real_nums = batch['context_size']
        box_embedding = self.box_feature_mapping(boxes)

        if self.clip_point:
            clip_feats = self.clip_feature_mapping(batch['clip_feats'].float())
            obj_feats, aux_feats = get_hybrid_features(self.object_encoder, obj_points, clip_feats,
                                                       aggregator=torch.stack)  # (B,52,768)
        else:
            obj_feats = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)  # (B,52,768)

        obj_feats = self.obj_feature_mapping(obj_feats)  # (B,N,768)

        if self.clip_point:
            obj_feats = obj_feats + aux_feats
        PP_LOGITS = self.pp_object_clf(obj_feats)

        # language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]  # (B,len,768)
        # <LOSS>: lang_cls
        LANG_LOGITS = self.language_target_clf(lang_infos[:, 0])
        _, lang_label = torch.max(F.softmax(LANG_LOGITS, -1), dim=-1)

        geo_feats = get_pairwise_distance(boxes.reshape(B * R, N, -1))  # B*R,N,N,48
        geo_feats = self.geo_feature_mapping(geo_feats)

        final_feats, o2o_logits, p2p_logits, p2p_inds = self.relation_encoder(obj_feats, geo_feats,
                                                                              lang_infos,
                                                                              box_embedding, real_nums)

        LOGITS = self.object_language_clf(final_feats).squeeze(-1)
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])

        binary_loss = torch.tensor(0.0).to(self.device)
        n_ary_loss = torch.tensor(0.0).to(self.device)
        if self.training:
            if self.binary_k > 0:
                binary_loss = self.binary_loss(o2o_logits.reshape(B, -1), batch['re_labels'].reshape(B, -1))
            if self.n_ary_k > 0:
                n_ary_loss = self.n_ary_relation_loss(p2p_logits, p2p_inds, batch['target_pos'])

        if self.n_ary_k > 0:
            flat_indices = p2p_inds.view(B, -1).long()  # shape: (B, 256*4)
            expanded_scores = p2p_logits.unsqueeze(-1).expand(B, -1, 4).reshape(B, -1)  # shape: (B, 256*4)

            RObjectLogits = []
            for bi in range(B):
                RL = torch.full((N,),float('-inf'), device=self.device)
                igmask = flat_indices[bi] != -1
                RL = RL.scatter_reduce_(dim=0, index=flat_indices[bi][igmask], src=expanded_scores[bi][igmask],
                                        reduce='amax',
                                        include_self=False)
                RObjectLogits.append(RL)
            RObjectLogits = torch.stack(RObjectLogits).sigmoid()
            LOGITS = LOGITS.softmax(dim=-1) + RObjectLogits

        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])

        if PP_LOGITS is not None:
            pp_clf_loss = self.class_logits_loss(PP_LOGITS.transpose(2, 1), batch['class_labels'])
        else:
            pp_clf_loss = torch.tensor(0.0).to(self.device)

        LOSS = referential_loss + 2.0 * (binary_loss + n_ary_loss)
        LOSS = LOSS + self.lang_cls_alpha * lang_clf_loss + self.pp_cls_alpha * pp_clf_loss

        if not self.training and self.n_ary_k > 0:
            p2p_logits = p2p_logits.softmax(dim=-1)
            topk_values, topk_inds = torch.topk(p2p_logits, self.n_ary_k, dim=1)  # b,8
            topk_p2p_inds = p2p_inds[torch.arange(B)[:, None], topk_inds, :]  # b,8,4

            return LOSS, PP_LOGITS, LANG_LOGITS, LOGITS, (topk_values, topk_p2p_inds)
        else:
            return LOSS, PP_LOGITS, LANG_LOGITS, LOGITS, None
