import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from tqdm.auto import tqdm

import utils.guidance_funcs as guidance
from models.common import get_timestep_embedding
from models.eps_net import EpsilonNet
from models.transition import get_transitions
from utils import so3
from utils.geometry import local_to_global


def center_pos(pos, fragment_mask, batch_node, mode='frags'):
    if mode == 'frags':
        mu1 = scatter_mean(pos[fragment_mask == 1], batch_node[fragment_mask == 1], dim=0)
        mu2 = scatter_mean(pos[fragment_mask == 2], batch_node[fragment_mask == 2], dim=0)
        offset = (mu1 + mu2) / 2
        new_pos = pos - offset[batch_node]
    else:
        raise NotImplementedError
    return new_pos, offset


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)  # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)  # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)  # (*, )
    return loss


class AtomEncoder(nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim):
        # first element of feature_dims tuple is a list with the length of each categorical feature
        # and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.num_categorical_features = feature_dims[0]
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        infeat_dim = self.num_categorical_features + self.num_scalar_features
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, 2 * emb_dim), nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

    def forward(self, atom_types, scalar_feats, t_embed):
        cat_feats = F.one_hot(atom_types, num_classes=self.num_categorical_features).float()
        x_embedding = self.mlp(torch.cat([cat_feats, scalar_feats, t_embed], dim=-1))
        return x_embedding


class BondEncoder(nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim):
        # first element of feature_dims tuple is a list with the length of each categorical feature
        # and the second is the number of scalar features
        super(BondEncoder, self).__init__()
        self.num_categorical_features = feature_dims[0]
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        infeat_dim = self.num_categorical_features + self.num_scalar_features
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, 2 * emb_dim), nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

    def forward(self, bond_types, scalar_feats, t_embed):
        cat_feats = F.one_hot(bond_types, num_classes=self.num_categorical_features).float()
        x_embedding = self.mlp(torch.cat([cat_feats, scalar_feats, t_embed], dim=-1))
        return x_embedding


class DiffPROTACModel(nn.Module):

    def __init__(self, cfg, num_classes, num_bond_classes, atom_feature_dim, edge_feature_dim):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_bond_classes = num_bond_classes
        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim

        self.node_emb_dim = cfg.node_emb_dim
        self.edge_emb_dim = cfg.edge_emb_dim
        self.time_emb_dim = cfg.time_emb_dim
        self.num_steps = cfg.num_steps

        self.train_frag_rot = cfg.get('train_frag_rot', True)
        self.train_frag_pos = cfg.get('train_frag_pos', True)
        self.train_link = cfg.get('train_link', True)
        self.train_bond = cfg.get('train_bond', True)

        self.frag_pos_prior = cfg.frag_pos_prior
        self.use_rel_geometry = (cfg.get('rel_geometry', 'distance_and_two_rot') == 'relative_pos_and_rot')
        print(f'Use relative geometry: {self.use_rel_geometry}')
        self.trans_rot, self.trans_dis, self.trans_pos, self.trans_link_pos, \
            self.trans_link_cls, self.trans_link_bond = get_transitions(cfg, num_classes, num_bond_classes)
        for trans_name in ['rot', 'dis', 'pos', 'link_pos', 'link_cls', 'link_bond']:
            trans = getattr(self, 'trans_' + trans_name)
            if trans is not None:
                if hasattr(trans, 'var_sched'):
                    alpha_bars = trans.var_sched.alpha_bars.tolist()
                else:
                    alpha_bars = trans.alpha_bars.tolist()
                print(f'Transition {trans_name:10s} alpha bar: ',
                      ['%.4f' % e for e in alpha_bars[:5]] + ['...'] + ['%.4f' % e for e in alpha_bars[-5:]])

        # embedding
        self.atom_embed = AtomEncoder(self.node_emb_dim, [num_classes, atom_feature_dim], self.time_emb_dim)
        self.bond_embed = BondEncoder(self.edge_emb_dim, [num_bond_classes, edge_feature_dim], self.time_emb_dim)
        self.time_emb_type = cfg.time_emb_type
        if self.time_emb_type != 'plain':
            self.timestep_emb_func = get_timestep_embedding(
                embedding_type=self.time_emb_type,
                embedding_dim=self.time_emb_dim,
                embedding_scale=cfg.time_emb_scale)

        self.eps_net = EpsilonNet(
            cfg.eps_net, self.node_emb_dim, self.edge_emb_dim,
            self.time_emb_dim, num_classes, num_bond_classes,
            train_frag_rot=self.train_frag_rot, train_frag_pos=self.train_frag_pos,
            train_link=self.train_link, train_bond=self.train_bond,
            pred_frag_dist=True if self.trans_dis else False,
            use_rel_geometry=self.use_rel_geometry,
            softmax_last=True
        )
        self.register_buffer('_dummy', torch.empty([0, ]))

    def forward(self, x_noisy, c_noisy, edge_noisy, edge_index, atom_feat, edge_feat,
                R1_noisy, R2_noisy, t, fragment_mask, linker_bond_mask, inner_edge_mask,
                rel_fragment_mask, rel_R_noisy, batch_node, batch_edge):
        t_batch = t[batch_node]
        if self.time_emb_type == 'plain':
            node_t_emb = t[batch_node].unsqueeze(-1) / self.num_steps
            edge_t_emb = t[batch_edge].unsqueeze(-1) / self.num_steps
            graph_t_emb = t.unsqueeze(-1) / self.num_steps
        elif self.time_emb_type == 'basic':
            betas = self.trans_pos.var_sched.betas
            node_t_emb = self.timestep_emb_func(betas[t_batch])
            edge_t_emb = self.timestep_emb_func(betas[t[batch_edge]])
            graph_t_emb = self.timestep_emb_func(betas[t])
        elif self.time_emb_type in ['sin', 'fourier']:
            node_t_emb = self.timestep_emb_func(t_batch / self.num_steps)
            edge_t_emb = self.timestep_emb_func(t[batch_edge] / self.num_steps)
            graph_t_emb = self.timestep_emb_func(t / self.num_steps)
        else:
            raise ValueError

        c_emb = self.atom_embed(c_noisy, atom_feat, node_t_emb)
        b_emb = self.bond_embed(edge_noisy, edge_feat, edge_t_emb)
        outputs = self.eps_net(x_noisy, c_emb, R1_noisy, R2_noisy, edge_index,
                               b_emb, fragment_mask, edge_mask=linker_bond_mask, inner_edge_mask=inner_edge_mask,
                               batch_node=batch_node, node_t_emb=node_t_emb, edge_t_emb=edge_t_emb, graph_t_emb=graph_t_emb,
                               rel_fragment_mask=rel_fragment_mask, rel_R_noisy=rel_R_noisy)
        return outputs

    @staticmethod
    def compose_noisy_graphs(batch, R1_noisy, R2_noisy, p1_noisy, p2_noisy,
                             x_linker_noisy, c_linker_noisy, bond_linker_noisy):

        # transform rotation and translation to positions
        f1_local_pos = batch.frags_local_pos_filled[::2]
        f2_local_pos = batch.frags_local_pos_filled[1::2]
        f1_local_pos_mask = batch.frags_local_pos_mask[::2]
        f2_local_pos_mask = batch.frags_local_pos_mask[1::2]
        x_f1 = local_to_global(R1_noisy, p1_noisy, f1_local_pos)[f1_local_pos_mask]
        x_f2 = local_to_global(R2_noisy, p2_noisy, f2_local_pos)[f2_local_pos_mask]

        # convert noisy fragment geometry into x_noisy
        x_noisy = torch.zeros_like(batch.pos)
        x_noisy[batch.fragment_mask == 1] = x_f1
        x_noisy[batch.fragment_mask == 2] = x_f2
        x_noisy[batch.linker_mask] = x_linker_noisy
        c_noisy = batch.atom_type.clone()
        c_noisy[batch.linker_mask] = c_linker_noisy
        b_noisy = batch.edge_type.clone()
        b_noisy[batch.linker_bond_mask] = bond_linker_noisy
        return x_noisy, c_noisy, b_noisy

    @staticmethod
    def get_prior_frags_distance(batch):
        num_linker_atoms = scatter_sum(batch.linker_mask.long(), batch.batch)
        frag_idx_mask = batch.fragment_mask[batch.fragment_mask > 0]
        frag_batch = batch.batch[batch.fragment_mask > 0]
        f1_batch = frag_batch[frag_idx_mask == 1]
        r1 = batch.frags_local_pos[frag_idx_mask == 1].norm(p=2, dim=-1)
        max_r1, _ = scatter_max(r1, f1_batch)
        f2_batch = frag_batch[frag_idx_mask == 2]
        r2 = batch.frags_local_pos[frag_idx_mask == 2].norm(p=2, dim=-1)
        max_r2, _ = scatter_max(r2, f2_batch)
        prior_frag_d = 0.4 * num_linker_atoms - 0.9 + (max_r1 + max_r2)
        return prior_frag_d

    def add_noise_to_batch(
            self, batch, t,add_rot_noise=True, add_pos_noise=True, add_linker_noise=True, add_bond_noise=True):
        # Add noise to frags rotations
        if add_rot_noise:
            frags_v0 = so3.rotation_to_so3vec(batch.frags_R)
            v_frags_noisy, _ = self.trans_rot.add_noise(frags_v0, t.repeat_interleave(2))
            R_frags_noisy = so3.so3vec_to_rotation(v_frags_noisy)
            noisy_R = torch.where(batch.frags_rel_mask.view(-1, 1, 1), R_frags_noisy, batch.frags_R)
            batch.R1_noisy = noisy_R[::2]
            batch.R2_noisy = noisy_R[1::2]
            if self.use_rel_geometry:
                batch.rel_R_noisy = R_frags_noisy[batch.frags_rel_mask]
                batch.rel_R = batch.frags_R[batch.frags_rel_mask]
        else:
            batch.R1_noisy = batch.frags_R[::2]
            batch.R2_noisy = batch.frags_R[1::2]
        batch.R1 = batch.frags_R[::2]
        batch.R2 = batch.frags_R[1::2]

        # Add noise to frags positions
        if add_pos_noise:
            if self.trans_dis is None:
                p_frags_noisy, eps_p_frags = self.trans_pos.add_noise(batch.frags_t, t.repeat_interleave(2),
                                                                      shift_center=True)
                noisy_p = torch.where(batch.frags_rel_mask.view(-1, 1), p_frags_noisy, batch.frags_t)
                batch.p1_noisy = noisy_p[::2]
                batch.p2_noisy = noisy_p[1::2]
                if self.use_rel_geometry:
                    batch.rel_eps_p = eps_p_frags[batch.frags_rel_mask]
                else:
                    batch.eps_p1 = eps_p_frags[::2]
                    batch.eps_p2 = eps_p_frags[1::2]
            else:
                if self.frag_pos_prior == 'stat_distance':
                    prior_frag_d = self.get_prior_frags_distance(batch).unsqueeze(-1)
                    d_noisy, eps_d = self.trans_dis.add_noise(batch.frags_d.unsqueeze(-1) - prior_frag_d, t)
                    d_noisy = d_noisy + prior_frag_d
                else:
                    raise ValueError(self.frag_pos_prior)
                batch.p1_noisy = F.pad(-d_noisy / 2, pad=(0, 2))
                batch.p2_noisy = F.pad(d_noisy / 2, pad=(0, 2))
                batch.eps_d = eps_d
                batch.d_noisy = d_noisy

        else:
            batch.p1_noisy = batch.frags_t[::2]
            batch.p2_noisy = batch.frags_t[1::2]
        batch.p1 = batch.frags_t[::2]
        batch.p2 = batch.frags_t[1::2]

        # Add noise to linker
        if add_linker_noise:
            pos_linker = batch.pos[batch.linker_mask]
            cls_linker = batch.atom_type[batch.linker_mask]
            t_linker = t[batch.batch][batch.linker_mask]
            x_linker_noisy, _ = self.trans_link_pos.add_noise(pos_linker, t_linker)
            _, c_linker_noisy = self.trans_link_cls.add_noise(cls_linker, t_linker)
        else:
            x_linker_noisy = batch.pos[batch.linker_mask]
            c_linker_noisy = batch.atom_type[batch.linker_mask]

        # Add noise to bond
        if add_bond_noise:
            bond_linker = batch.edge_type[batch.linker_bond_mask]
            t_bond = t[batch.edge_type_batch][batch.linker_bond_mask]
            _, bond_linker_noisy = self.trans_link_bond.add_noise(bond_linker, t_bond)
        else:
            # bond_linker_noisy = batch.edge_type[batch.linker_bond_mask]
            bond_linker_noisy = torch.zeros_like(batch.edge_type[batch.linker_bond_mask])

        batch.x_noisy, batch.c_noisy, batch.edge_noisy = self.compose_noisy_graphs(
            batch, batch.R1_noisy, batch.R2_noisy, batch.p1_noisy, batch.p2_noisy,
            x_linker_noisy, c_linker_noisy, bond_linker_noisy)

        return batch

    def get_loss(self, batch, t=None, return_outputs=False, add_noise=True, pos_noise_std=0.):
        linker_pos = batch.pos[batch.linker_mask]
        batch.pos[batch.linker_mask] += torch.randn_like(linker_pos) * pos_noise_std
        # add noise
        if add_noise:
            if t is None:
                t = torch.randint(0, self.num_steps, size=(batch.num_graphs,)).to(self._dummy.device)
            batch = self.add_noise_to_batch(
                batch, t, self.train_frag_rot, self.train_frag_pos, self.train_link, self.train_bond)
        batch.x_noisy, offset = center_pos(batch.x_noisy, batch.fragment_mask, batch.batch)
        batch.p1_noisy -= offset
        batch.p2_noisy -= offset

        out = self.forward(
            x_noisy=batch.x_noisy,
            c_noisy=batch.c_noisy,
            edge_noisy=batch.edge_noisy,
            edge_index=batch.edge_index,
            atom_feat=batch.atom_feat, edge_feat=batch.edge_feat,
            R1_noisy=batch.R1_noisy, R2_noisy=batch.R2_noisy,
            t=t, fragment_mask=batch.fragment_mask, linker_bond_mask=batch.linker_bond_mask,
            inner_edge_mask=batch.inner_edge_mask,
            batch_node=batch.batch, batch_edge=batch.edge_type_batch,
            rel_fragment_mask=batch.get('frags_atom_rel_mask', None),
            rel_R_noisy=batch.get('rel_R_noisy', None)
        )

        if self.train_frag_rot:
            # Fragment rotation loss
            if self.use_rel_geometry:
                loss_frag_rot = rotation_matrix_cosine_loss(out['frag_R_next'], batch.rel_R)
            else:
                loss_frag_rot = rotation_matrix_cosine_loss(out['frag_R_next'][0], batch.R1) + \
                                rotation_matrix_cosine_loss(out['frag_R_next'][1], batch.R2)  # (G, )
            loss_frag_rot = torch.mean(loss_frag_rot)
        else:
            loss_frag_rot = torch.tensor(0.)

        if self.train_frag_pos:
            if self.trans_dis is None:
                # Fragment position loss
                if self.use_rel_geometry:
                    loss_frag_pos = F.mse_loss(out['frag_eps_pos'], batch.rel_eps_p, reduction='none').sum(-1)
                else:
                    if out['frag_eps_pos'][0] is not None:
                        loss_frag_pos = F.mse_loss(out['frag_eps_pos'][0], batch.eps_p1, reduction='none').sum(-1) + \
                                        F.mse_loss(out['frag_eps_pos'][1], batch.eps_p2, reduction='none').sum(-1)  # (G, )
                    else:
                        loss_frag_pos = F.mse_loss(out['frag_recon_x'][0], batch.p1, reduction='none').sum(-1) + \
                                        F.mse_loss(out['frag_recon_x'][1], batch.p2, reduction='none').sum(-1)  # (G, )
                loss_frag_pos = torch.mean(loss_frag_pos)
            else:
                # Fragment distance loss
                loss_frag_pos = F.mse_loss(out['frag_eps_d'], batch.eps_d, reduction='none').sum(-1)
                loss_frag_pos = torch.mean(loss_frag_pos)
        else:
            loss_frag_pos = torch.tensor(0.)

        if self.train_link:
            # Linker position loss
            gt_pos = batch.pos - offset[batch.batch]
            linker_x0 = gt_pos[batch.fragment_mask == 0]
            batch_linker = batch.batch[batch.fragment_mask == 0]
            loss_link_pos = scatter_mean(((out['linker_x'] - linker_x0) ** 2).sum(-1), batch_linker, dim=0)
            loss_link_pos = torch.mean(loss_link_pos)

            # Linker atom type loss
            linker_c0 = batch.atom_type[batch.fragment_mask == 0]
            linker_ct = batch.c_noisy[batch.fragment_mask == 0]
            linker_t = t[batch_linker]
            post_true = self.trans_link_cls.posterior(linker_ct, linker_c0, linker_t)
            log_post_pred = torch.log(self.trans_link_cls.posterior(linker_ct, out['linker_c'], linker_t) + 1e-8)
            kl_div = F.kl_div(
                input=log_post_pred,
                target=post_true,
                reduction='none',
                log_target=False
            ).sum(dim=-1)  # (F, )
            loss_link_cls = scatter_mean(kl_div, batch_linker, dim=0)
            loss_link_cls = torch.mean(loss_link_cls)
        else:
            loss_link_pos, loss_link_cls = torch.tensor(0.), torch.tensor(0.)

        if self.train_bond:
            linker_b0 = batch.edge_type[batch.linker_bond_mask]
            linker_bt = batch.edge_noisy[batch.linker_bond_mask]
            batch_linker_bond = batch.edge_type_batch[batch.linker_bond_mask]
            linker_bond_t = t[batch_linker_bond]
            post_true = self.trans_link_bond.posterior(linker_bt, linker_b0, linker_bond_t)
            log_post_pred = torch.log(self.trans_link_bond.posterior(linker_bt, out['linker_bond'], linker_bond_t) + 1e-8)
            kl_div = F.kl_div(
                input=log_post_pred,
                target=post_true,
                reduction='none',
                log_target=False
            ).sum(dim=-1)  # (F, )
            loss_link_bond = scatter_mean(kl_div, batch_linker_bond, dim=0)
            loss_link_bond = torch.mean(loss_link_bond)
        else:
            loss_link_bond = torch.tensor(0.)

        loss_dict = {
            'frag_rot': loss_frag_rot,
            'frag_pos': loss_frag_pos,
            'link_pos': loss_link_pos,
            'link_cls': loss_link_cls,
            'link_bond': loss_link_bond
        }
        if return_outputs:
            return loss_dict, out
        else:
            return loss_dict

    def sample_init(self, batch, sample_frag_rot, sample_frag_pos, sample_link, sample_bond,
                    p_init_mode='stat_distance'):
        batch_size = batch.num_graphs
        v1_init, v2_init = None, None
        if sample_frag_rot:
            # init fragment rotation
            if self.use_rel_geometry:
                v1_init = torch.zeros([batch_size, 3]).to(self._dummy.device)
                v2_init = so3.random_uniform_so3([batch_size]).to(self._dummy.device)
            else:
                v_frags_init = so3.random_uniform_so3([2, batch_size]).to(self._dummy.device)
                v1_init, v2_init = v_frags_init[0], v_frags_init[1]

        p1_init, p2_init, d_init = None, None, None
        if sample_frag_pos:
            # init fragment translation
            if self.trans_dis is None:
                if self.use_rel_geometry:
                    p1_init = torch.zeros([batch_size, 3]).to(self._dummy.device)
                    p2_init = torch.randn([batch_size, 3]).to(self._dummy.device)
                else:
                    p_frags_init = torch.randn([2, batch_size, 3]).to(self._dummy.device)
                    p_frags_init -= p_frags_init.mean(0, keepdim=True)
                    p1_init, p2_init = p_frags_init[0], p_frags_init[1]
            else:
                if p_init_mode == 'stat_distance':
                    prior_frag_d = self.get_prior_frags_distance(batch)
                    # # to test
                    # prior_frag_d = torch.ones_like(prior_frag_d) * 13.
                    d_init = torch.clamp(prior_frag_d + torch.randn(batch_size).to(self._dummy.device), min=2.)[:, None]
                else:
                    raise ValueError(p_init_mode)

        num_linker_atoms = scatter_sum(batch.linker_mask.long(), batch.batch)
        num_linker_bonds = scatter_sum(batch.linker_bond_mask.long(), batch.edge_type_batch)
        x_linker_init, c_linker_init, bond_linker_init = None, None, None
        if sample_link:
            # init linker position
            x_linker_init = torch.randn([num_linker_atoms.sum(), 3]).to(self._dummy.device)
            # init linker atom type
            c_linker_init = torch.randint(size=(num_linker_atoms.sum(),),
                                          low=0, high=self.num_classes).to(self._dummy.device)
        if sample_bond:
            # init linker bond
            bond_linker_init = torch.randint(size=(num_linker_bonds.sum().item() // 2,),
                                             low=0, high=self.num_bond_classes)
            bond_linker_init = bond_linker_init.repeat_interleave(2).to(self._dummy.device)

        init_batch = self.get_next_step(
            batch, sample_frag_rot, sample_frag_pos, sample_link, sample_bond,
            v1_next=v1_init, v2_next=v2_init,
            p1_next=p1_init, p2_next=p2_init, d_next=d_init,
            x_linker_next=x_linker_init, c_linker_next=c_linker_init, bond_linker_next=bond_linker_init)

        return init_batch

    def get_next_step(self, batch, sample_frag_rot, sample_frag_pos, sample_link, sample_bond,
                      v1_next, v2_next, p1_next, p2_next, d_next, x_linker_next, c_linker_next, bond_linker_next):
        if sample_frag_rot:
            batch.R1_noisy = so3.so3vec_to_rotation(v1_next)
            batch.R2_noisy = so3.so3vec_to_rotation(v2_next)
        else:
            batch.R1_noisy = batch.frags_R[::2]
            batch.R2_noisy = batch.frags_R[1::2]

        if sample_frag_pos:
            if self.trans_dis is None:
                batch.p1_noisy = p1_next
                batch.p2_noisy = p2_next
            else:
                batch.d_noisy = d_next
                batch.p1_noisy = F.pad(-batch.d_noisy / 2, pad=(0, 2))
                batch.p2_noisy = F.pad(batch.d_noisy / 2, pad=(0, 2))
        else:
            batch.p1_noisy = batch.frags_t[::2]
            batch.p2_noisy = batch.frags_t[1::2]

        if sample_link:
            x_linker = x_linker_next
            c_linker = c_linker_next
        else:
            x_linker = batch.pos[batch.linker_mask]
            c_linker = batch.atom_type[batch.linker_mask]

        if sample_bond:
            bond_noisy = bond_linker_next
        else:
            bond_noisy = batch.edge_type[batch.linker_bond_mask]

        batch.x_noisy, batch.c_noisy, batch.edge_noisy = self.compose_noisy_graphs(
            batch, batch.R1_noisy, batch.R2_noisy, batch.p1_noisy, batch.p2_noisy, x_linker, c_linker, bond_noisy)
        return batch

    @torch.no_grad()
    def sample(self, batch,
               sample_frag_rot=None, sample_frag_pos=None, sample_link=None, sample_bond=None,
               p_init_mode='stat_distance', guidance_opt=None, pbar=True):
        if sample_frag_rot is None:
            sample_frag_rot = self.train_frag_rot
        if sample_frag_pos is None:
            sample_frag_pos = self.train_frag_pos
        if sample_link is None:
            sample_link = self.train_link
        if sample_bond is None:
            sample_bond = self.train_bond
        # initialize
        batch = self.sample_init(batch, sample_frag_rot, sample_frag_pos, sample_link, sample_bond,
                                 p_init_mode=p_init_mode)
        batch.x_noisy, offset = center_pos(batch.x_noisy, batch.fragment_mask, batch.batch)
        batch.p1_noisy -= offset
        batch.p2_noisy -= offset
        # reverse diffusion sampling
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        num_graphs = batch.num_graphs
        traj = [(batch.x_noisy.cpu(), batch.c_noisy.cpu(), batch.edge_noisy.cpu(),
                 batch.R1_noisy.cpu(), batch.R2_noisy.cpu(), batch.p1_noisy.cpu(), batch.p2_noisy.cpu())]
        for i in pbar(range(self.num_steps, 0, -1)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=self._dummy.device)
            node_t = t[batch.batch]
            edge_t = t[batch.edge_type_batch]

            outputs = self.forward(
                x_noisy=batch.x_noisy,
                c_noisy=batch.c_noisy,
                edge_noisy=batch.edge_noisy,

                edge_index=batch.edge_index,
                atom_feat=batch.atom_feat,
                edge_feat=batch.edge_feat,

                R1_noisy=batch.R1_noisy, R2_noisy=batch.R2_noisy,
                t=t, fragment_mask=batch.fragment_mask, linker_bond_mask=batch.linker_bond_mask,
                inner_edge_mask=batch.inner_edge_mask,
                batch_node=batch.batch, batch_edge=batch.edge_type_batch,
                rel_fragment_mask=(batch.fragment_mask == 2) if self.use_rel_geometry else None,  # in sampling, we assume the first fragment is reference
                rel_R_noisy=batch.R2_noisy if self.use_rel_geometry else None
            )
            v1_next, v2_next = None, None
            if sample_frag_rot:
                # fragment denoising
                if self.use_rel_geometry:
                    v1_next = so3.rotation_to_so3vec(batch.R1_noisy)  # should always be zeros
                    v2_next = self.trans_rot.denoise(v_t=so3.rotation_to_so3vec(batch.R2_noisy),
                                                     v_0_pred=outputs['frag_v_next'], t=t)
                else:
                    v1_next = self.trans_rot.denoise(v_t=so3.rotation_to_so3vec(batch.R1_noisy),
                                                     v_0_pred=outputs['frag_v_next'][0], t=t)
                    v2_next = self.trans_rot.denoise(v_t=so3.rotation_to_so3vec(batch.R2_noisy),
                                                     v_0_pred=outputs['frag_v_next'][1], t=t)

            p1_next, p2_next, d_next = None, None, None
            if sample_frag_pos:
                if self.trans_dis is None:
                    if self.use_rel_geometry:
                        rel_p_noisy = batch.p2_noisy - batch.p1_noisy
                        rel_p_next = self.trans_pos.denoise(rel_p_noisy, outputs['frag_eps_pos'], t)
                        p1_next = - rel_p_next / 2
                        p2_next = rel_p_next / 2
                    else:
                        p_noisy = torch.stack([batch.p1_noisy, batch.p2_noisy], dim=1).view(-1, 3)
                        if outputs['frag_eps_pos'][0] is not None:
                            eps_pos = torch.stack(outputs['frag_eps_pos'], dim=1).view(-1, 3)
                            p_next = self.trans_pos.denoise(p_noisy, eps_pos, t.repeat_interleave(2), shift_center=True)
                        else:
                            recon_p = torch.stack(outputs['frag_recon_x'], dim=1).view(-1, 3)
                            p_next = self.trans_pos.posterior_sample(
                                recon_p, p_noisy, t.repeat_interleave(2), shift_center=True)
                        p1_next, p2_next = p_next[::2], p_next[1::2]
                else:
                    d_next = self.trans_dis.denoise(batch.d_noisy, outputs['frag_eps_d'], t)

            x_linker_next, c_linker_next, bond_linker_next = None, None, None
            if sample_link:
                # linker pos denoising
                x_linker_next = self.trans_link_pos.posterior_sample(
                    outputs['linker_x'], batch.x_noisy[batch.linker_mask], node_t[batch.linker_mask])
                # linker atom type denoising
                # try:
                _, c_linker_next = self.trans_link_cls.denoise(
                    batch.c_noisy[batch.linker_mask], outputs['linker_c'], node_t[batch.linker_mask])
                # except:
                #     print(outputs['linker_c'], torch.isnan(outputs['linker_c']))
            if sample_bond:
                _, bond_linker_next = self.trans_link_bond.denoise(
                    batch.edge_noisy[batch.linker_bond_mask], outputs['linker_bond'], edge_t[batch.linker_bond_mask])

            if guidance_opt is not None:
                energy_all, input_names_all, energy_grads_all = self.guidance(
                    x_linker=batch.x_noisy[batch.linker_mask],
                    v1=so3.rotation_to_so3vec(batch.R1_noisy),
                    v2=so3.rotation_to_so3vec(batch.R2_noisy),
                    p1=batch.p1_noisy,
                    p2=batch.p2_noisy,
                    d=batch.get('d_noisy', None),
                    batch=batch, guidance_opt=guidance_opt,
                    sample_frag_rot=sample_frag_rot, sample_frag_pos=sample_frag_pos, sample_link=sample_link,
                    t=i
                )
                inputs_next = {
                    'x_linker': x_linker_next,
                    'v1': v1_next, 'v2': v2_next,
                    'p1': p1_next, 'p2': p2_next,
                    'd': d_next
                }
                for input_names, losses, grads in zip(input_names_all, energy_all, energy_grads_all):
                    for var_name, grad in zip(input_names, grads):
                        var = inputs_next[var_name]
                        var -= grad

            batch = self.get_next_step(
                batch, sample_frag_rot, sample_frag_pos, sample_link, sample_bond,
                v1_next, v2_next, p1_next, p2_next, d_next, x_linker_next, c_linker_next, bond_linker_next)
            batch.x_noisy, offset = center_pos(batch.x_noisy, batch.fragment_mask, batch.batch)
            batch.p1_noisy -= offset
            batch.p2_noisy -= offset

            traj.append((
                batch.x_noisy.cpu(), batch.c_noisy.cpu(), batch.edge_noisy.cpu(),
                batch.R1_noisy.cpu(), batch.R2_noisy.cpu(), batch.p1_noisy.cpu(), batch.p2_noisy.cpu()
            ))
        batch = batch.to('cpu')
        final_x = [batch.x_noisy[batch.ptr[idx]:batch.ptr[idx + 1]] for idx in range(num_graphs)]
        final_c = [batch.c_noisy[batch.ptr[idx]:batch.ptr[idx + 1]] for idx in range(num_graphs)]
        edge_ptr = [0] + scatter_sum(torch.ones_like(batch.edge_type_batch), batch.edge_type_batch).cumsum(0).tolist()
        final_bond = [batch.edge_noisy[edge_ptr[idx]:edge_ptr[idx + 1]] for idx in range(num_graphs)]
        return traj, final_x, final_c, final_bond

    def guidance(self, x_linker, v1, v2, p1, p2, d, batch, guidance_opt,
                 sample_frag_rot, sample_frag_pos, sample_link, t):
        if sample_frag_rot:
            v1.requires_grad = True
            v2.requires_grad = True
        if sample_frag_pos:
            if self.trans_dis is None:
                p1.requires_grad = True
                p2.requires_grad = True
            else:
                d.requires_grad = True
        if sample_link:
            x_linker.requires_grad = True

        with torch.enable_grad():
            energy_grads_all = []
            input_names_all = []
            energy_all = []
            for drift in guidance_opt:
                if drift['type'] == 'anchor_prox':
                    energy = guidance.compute_anchor_prox_loss(
                        x_linker, v1, v2, p1, p2,
                        batch.frags_local_pos_filled, batch.frags_local_pos_mask,
                        batch.fragment_mask, batch.cand_anchors_mask, batch.batch,
                        min_d=drift['min_d'], max_d=drift['max_d'])

                    if drift['update'] == 'frag_rot':
                        inputs = [v1, v2]
                        input_names = ['v1', 'v2']
                    elif drift['update'] == 'frag':
                        inputs = [v1, v2, p1, p2]
                        input_names = ['v1', 'v2', 'p1', 'p2']
                    elif drift['update'] == 'all':
                        inputs = [v1, v2, p1, p2, x_linker]
                        input_names = ['v1', 'v2', 'p1', 'p2', 'x_linker']
                    else:
                        raise ValueError(drift['update'])
                    if drift['decay']:
                        weight = t / self.num_steps * 9. + 1.
                    else:
                        weight = 1.
                    energy_grads = torch.autograd.grad(energy * weight, inputs)

                elif drift['type'] == 'frag_distance':
                    dist = batch.frags_d  # (B, )
                    if drift['constraint_mode'] == 'dynamic':
                        min_d = dist - drift['sigma'] * dist
                        max_d = dist + drift['sigma'] * dist
                    elif drift['constraint_mode'] == 'const':
                        min_d, max_d = drift['min_d'], drift['max_d']
                    else:
                        raise ValueError(drift['constraint_mode'])

                    energy = guidance.compute_frag_distance_loss(p1, p2, min_d, max_d, mode=drift['mode'])
                    if self.trans_dis is None:
                        inputs = [p1, p2]
                        input_names = ['p1', 'p2']
                    else:
                        inputs = [d]
                        input_names = ['d']
                    energy_grads = torch.autograd.grad(energy, inputs)
                else:
                    raise ValueError(drift['type'])

                # inputs_all.append(inputs)
                energy_all.append(energy)
                input_names_all.append(input_names)
                energy_grads_all.append(energy_grads)

        return energy_all, input_names_all, energy_grads_all
