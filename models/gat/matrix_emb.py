import numpy as np
import math
import torch
from torch.autograd import Variable


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    feat_range = torch.arange(0, feat_dim / 6)  # [0,1,2,3,4,5,6,7]
    dim_mat = torch.pow(torch.ones((1,)) * wave_length,
                        (6 / feat_dim) * feat_range)  # (8,)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)  # (1,1,1,8)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)  # (B,N,n,3,1)
    div_mat = torch.div(position_mat.to(device), dim_mat)  # (B,N,n,3,8)
    sin_mat = torch.sin(div_mat)  # (B,N,n,3,8)
    cos_mat = torch.cos(div_mat)  # (B,N,n,3,8)
    embedding = torch.cat([sin_mat, cos_mat], -1)  # (B,N,n,3,16)
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], -1)
    return embedding


def torch_extract_position_matrix(bbox):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 6]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    cx, cy, cz, lx, ly, lz = torch.split(bbox, 1, dim=-1)

    # [batch_size,num_boxes, num_boxes]
    delta_x = cx - torch.transpose(cx, 1, 2)
    delta_x = torch.div(delta_x, lx + 1e-10)
    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)

    delta_y = cy - torch.transpose(cy, 1, 2)
    delta_y = torch.div(delta_y, ly + 1e-10)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)

    delta_z = cz - torch.transpose(cz, 1, 2)
    delta_z = torch.div(delta_z, lz + 1e-10)
    delta_z = torch.abs(delta_z)
    delta_z[delta_z < threshold] = threshold
    delta_z = torch.log(delta_z)

    concat_list = [delta_x[..., None], delta_y[..., None], delta_z[..., None]]
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


def prepare_graph_variables(bbox, pos_emb_dim, device):
    bbox = bbox.to(device)
    pos_mat = torch_extract_position_matrix(bbox)  # (B,N,N,3)
    pos_emb = torch_extract_position_embedding(pos_mat, feat_dim=pos_emb_dim, device=device)
    pos_emb_var = Variable(pos_emb).to(device)
    return pos_emb_var


def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)


def get_pairwise_distance(boxes):
    x = boxes[..., :3]
    B, N, _ = x.shape
    relative_positions = x[:, None] - x[:, :, None]
    # Obtain the xy distances
    xy_distances = relative_positions[..., :2].norm(dim=-1, keepdim=True) + 1e-9
    r = xy_distances.squeeze(-1)
    phi = torch.atan2(relative_positions[..., 1], relative_positions[..., 0])  # Azimuth angle
    theta = torch.atan2(r, relative_positions[..., 2])  # Elevation angle
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    relative_positions = torch.cat([relative_positions, xy_distances, sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-1),
                                    sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)
    # Normalize x-y plane to unit vectors
    relative_positions[..., :2] = relative_positions[..., :2] / xy_distances
    # Scale z values so that max(z) - min(z) = 1
    relative_positions[..., 2] = scale_to_unit_range(relative_positions[..., 2])
    # Scale d values between 0 and 1 for each set of relative positions independently.
    relative_positions[..., 3] = scale_to_unit_range(relative_positions[..., 3])

    # cx, cy, cz, lx, ly, lz = torch.split(boxes, 1, dim=-1)
    # delta_x, delta_y, delta_z = cx - cx.transpose(1, 2), cy - cy.transpose(1, 2), cz - cz.transpose(1, 2)
    # x_distance = (abs(delta_x) - (lx + lx.transpose(1, 2)) / 2) / (delta_x + 1e-9)
    # y_distance = (abs(delta_y) - (ly + ly.transpose(1, 2)) / 2) / (delta_y + 1e-9)
    # z_distance = (abs(delta_z) - (lz + lz.transpose(1, 2)) / 2) / (delta_z + 1e-9)
    # xyz_distances = torch.cat([delta_x[..., None], delta_y[..., None], delta_z[..., None]], dim=-1).norm(dim=-1, keepdim=True) + 1e-9
    #
    # relative_positions = torch.cat(
    #     [relative_positions, x_distance[..., None], y_distance[..., None], z_distance[..., None], xyz_distances],
    #     dim=-1)

    return relative_positions
