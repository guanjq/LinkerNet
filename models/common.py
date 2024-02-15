import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import radius_graph, knn_graph
import math


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, expansion_mode='linear'):
        super().__init__()
        self.start, self.stop, self.num_gaussians = start, stop, num_gaussians
        self.expansion_mode = expansion_mode
        if expansion_mode == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start+1), end=np.log(stop+1), steps=num_gaussians)) - 1
            diff = torch.diff(offset)
            diff = torch.cat([diff[:1], diff])
            coeff = -0.5 / (diff ** 2)
            self.register_buffer('coeff', coeff)
        elif expansion_mode == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
            self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        else:
            raise NotImplementedError('type_ must be either exp or linear')

        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, ' \
               f'num_gaussians={self.num_gaussians}, expansion_mode={self.expansion_mode})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    if isinstance(h_protein, list):
        h_ctx_sca = torch.cat([h_protein[0], h_ligand[0]], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
        h_ctx_vec = torch.cat([h_protein[1], h_ligand[1]], dim=0)[sort_idx]  # (N_protein+N_ligand, H, 3)
        h_ctx = [h_ctx_sca, h_ctx_vec]
    else:
        h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            # print(f'batch: {i}, ligand_index: {ligand_index} {len(ligand_index)}')
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                # print('len protein index: ', len(protein_index))
                # print('max index: ', max(p_src), max(p_dst))
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)
            # edge_index.append(torch.cat([ll_edge_index, pl_edge_index], -1))

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index


# Time Embedding
def basic_time_embedding(beta):
    emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (N, 3)
    return emb


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'basic':
        emb_func = (lambda x: basic_time_embedding(x))
    elif embedding_type == 'sin':
        emb_func = (lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func
