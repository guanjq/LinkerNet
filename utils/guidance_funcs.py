import torch

from utils import so3
from utils.geometry import local_to_global


def compute_frag_distance_loss(p1, p2, min_d, max_d, mode='frag_center_distance'):
    """
    :param p1: (B, 3). center of fragment 1.
    :param p2: (B, 3). center of fragment 2.
    :param min_d: min distance
    :param max_d: max distance
    :param mode: constrained distance mode
    :return
    """
    if mode == 'frag_center_distance':
        dist = torch.norm(p1 - p2, p=2, dim=-1)  # (B, )
        loss = torch.mean(torch.clamp(min_d - dist, min=0) ** 2 + torch.clamp(dist - max_d, min=0) ** 2)
    else:
        raise ValueError(mode)
    return loss


def compute_anchor_prox_loss(x_linker, v1, v2, p1, p2,
                             frags_local_pos_filled, frags_local_pos_mask,
                             fragment_mask, cand_anchors_mask, batch, min_d=1.2, max_d=1.9):
    """
    :param x_linker: (num_linker_atoms, 3)
    :param v1: (B, 3)
    :param v2: (B, 3)
    :param p1: (B, 3)
    :param p2: (B, 3)
    :param frags_local_pos_filled: (B, max_num_atoms, 3)
    :param frags_local_pos_mask:  (B, max_num_atoms) of BoolTensor.
    :param fragment_mask: (N, )
    :param cand_anchors_mask: (N, ) of BoolTensor.
    :param batch: (N, )
    :param min_d
    :param max_d
    :return
    """

    # transform rotation and translation to positions
    R1 = so3.so3vec_to_rotation(v1)
    R2 = so3.so3vec_to_rotation(v2)
    f1_local_pos = frags_local_pos_filled[::2]
    f2_local_pos = frags_local_pos_filled[1::2]
    f1_local_pos_mask = frags_local_pos_mask[::2]
    f2_local_pos_mask = frags_local_pos_mask[1::2]
    x_f1 = local_to_global(R1, p1, f1_local_pos)[f1_local_pos_mask]
    x_f2 = local_to_global(R2, p2, f2_local_pos)[f2_local_pos_mask]

    f1_anchor_mask = cand_anchors_mask[fragment_mask == 1]
    f2_anchor_mask = cand_anchors_mask[fragment_mask == 2]
    f1_batch, f2_batch = batch[fragment_mask == 1], batch[fragment_mask == 2]

    # approach 1: distance constraints on fragments only (unreasonable)
    # c_a1 = scatter_mean(x_f1[f1_anchor_mask], f1_batch[f1_anchor_mask], dim=0)  # (B, 3)
    # c_a2 = scatter_mean(x_f2[f2_anchor_mask], f2_batch[f2_anchor_mask], dim=0)
    # c_na1 = scatter_mean(x_f1[~f1_anchor_mask], f1_batch[~f1_anchor_mask], dim=0)
    # c_na2 = scatter_mean(x_f2[~f2_anchor_mask], f2_batch[~f2_anchor_mask], dim=0)
    # loss = 0.
    # d_a1_a2 = torch.norm(c_a1 - c_a2, p=2, dim=-1)
    # if c_na1.size(0) > 0:
    #     d_na1_a2 = torch.norm(c_na1 - c_a2, p=2, dim=-1)
    #     loss += torch.mean(torch.clamp(d_a1_a2 - d_na1_a2, min=0))
    # if c_na2.size(0) > 0:
    #     d_a1_na2 = torch.norm(c_a1 - c_na2, p=2, dim=-1)
    #     loss += torch.mean(torch.clamp(d_a1_a2 - d_a1_na2, min=0))

    # approach 2: min dist of (linker, anchor) can form bond, (linker, non-anchor) cannot form bond
    linker_batch = batch[fragment_mask == 0]

    num_graphs = batch.max().item() + 1
    batch_losses = 0.
    for idx in range(num_graphs):
        linker_pos = x_linker[linker_batch == idx]
        loss_f1 = compute_prox_loss(x_f1[f1_batch == idx], linker_pos, f1_anchor_mask[f1_batch == idx], min_d, max_d)
        loss_f2 = compute_prox_loss(x_f2[f2_batch == idx], linker_pos, f2_anchor_mask[f2_batch == idx], min_d, max_d)
        batch_losses += loss_f1 + loss_f2

    return batch_losses / num_graphs


def compute_prox_loss(frags_pos, linker_pos, anchor_mask, min_d=1.2, max_d=1.9):
    pairwise_dist = torch.norm(frags_pos[anchor_mask].unsqueeze(1) - linker_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = pairwise_dist.min()
    # 1.2 < min dist < 1.9
    loss_anchor = torch.mean(torch.clamp(min_d - min_dist, min=0) ** 2 + torch.clamp(min_dist - max_d, min=0) ** 2)

    # non anchor min dist > 1.9
    loss_non_anchor = 0.
    non_anchor_pairwise_dist = torch.norm(frags_pos[~anchor_mask].unsqueeze(1) - linker_pos.unsqueeze(0), p=2, dim=-1)
    if non_anchor_pairwise_dist.size(0) > 0:
        non_anchor_min_dist = non_anchor_pairwise_dist.min()
        loss_non_anchor = torch.mean(torch.clamp(max_d - non_anchor_min_dist, min=0) ** 2)

    loss = loss_anchor + loss_non_anchor
    # print(f'loss anchor: {loss_anchor}  loss non anchor: {loss_non_anchor}')
    return loss
