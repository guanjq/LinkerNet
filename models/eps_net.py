import torch
import torch.nn as nn
from models.encoders import get_refine_net
from utils.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from utils.so3 import so3vec_to_rotation, rotation_to_so3vec
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from models.common import MLP
import numpy as np


class ForceLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, time_dim,
                 separate_att=False, act_fn='relu', norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.act_fn = act_fn
        self.separate_att = separate_att

        kv_input_dim = input_dim * 2 + edge_feat_dim
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim + time_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, rel_x, edge_feat, edge_index, inner_edge_mask, t, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        kv_input = torch.cat([edge_feat, hi, hj], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(torch.cat([kv_input, t[dst]], -1))
        e_w = e_w.view(-1, 1) if e_w is not None else 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)   # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        if self.separate_att:
            k_in, v_in = k[inner_edge_mask], v[inner_edge_mask]
            alpha_in = scatter_softmax((q[dst[inner_edge_mask]] * k_in / np.sqrt(k_in.shape[-1])).sum(-1),
                                       dst[inner_edge_mask], dim=0)  # (E, heads)
            m_in = alpha_in.unsqueeze(-1) * v_in  # (E, heads, 3)
            inner_forces = scatter_sum(m_in, dst[inner_edge_mask], dim=0, dim_size=N).mean(1)

            k_out, v_out = k[~inner_edge_mask], v[~inner_edge_mask]
            alpha_out = scatter_softmax((q[dst[~inner_edge_mask]] * k_out / np.sqrt(k_out.shape[-1])).sum(-1),
                                        dst[~inner_edge_mask], dim=0)  # (E, heads)
            m_out = alpha_out.unsqueeze(-1) * v_out  # (E, heads, 3)
            outer_forces = scatter_sum(m_out, dst[~inner_edge_mask], dim=0, dim_size=N).mean(1)

        else:
            # Compute attention weights
            alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # (E, heads)

            # Perform attention-weighted message-passing
            m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
            inner_forces = scatter_sum(m[inner_edge_mask], dst[inner_edge_mask], dim=0, dim_size=N).mean(1)
            outer_forces = scatter_sum(m[~inner_edge_mask], dst[~inner_edge_mask], dim=0, dim_size=N).mean(1)
        # output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        return inner_forces, outer_forces  # [num_nodes, 3]


class SymForceLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_feat_dim, time_dim,
                 act_fn='relu', norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.act_fn = act_fn
        self.pred_layer = MLP(input_dim + edge_feat_dim + time_dim, 1, hidden_dim,
                              num_layer=3, norm=norm, act_fn=act_fn)

    def forward(self, h, rel_x, edge_feat, edge_index, inner_edge_mask, t):
        N = h.size(0)
        src, dst = edge_index[:, ~inner_edge_mask]
        rel_x = rel_x[~inner_edge_mask]
        distance = torch.norm(rel_x, p=2, dim=-1)
        hi, hj = h[dst], h[src]

        feat = torch.cat([edge_feat[~inner_edge_mask], (hi + hj) / 2, t[dst]], -1)
        forces = self.pred_layer(feat) * rel_x / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        outer_forces = scatter_sum(forces, dst, dim=0, dim_size=N)
        return None, outer_forces  # [num_nodes, 3]


class EpsilonNet(nn.Module):

    def __init__(self, cfg, node_emb_dim, edge_emb_dim, time_emb_dim, num_classes, num_bond_classes,
                 train_frag_rot, train_frag_pos, train_link, train_bond, pred_frag_dist=False,
                 use_rel_geometry=False, softmax_last=True):
        super().__init__()
        self.encoder_type = cfg.net_type
        self.encoder = get_refine_net(cfg.encoder, cfg.net_type, node_emb_dim, edge_emb_dim, train_link)
        self.num_frags = 2
        self.pred_frag_dist = pred_frag_dist
        self.use_rel_geometry = use_rel_geometry
        self.train_frag_rot = train_frag_rot
        self.train_frag_pos = train_frag_pos
        self.train_link = train_link
        self.train_bond = train_bond
        self.tr_output_type = cfg.get('tr_output_type', 'invariant_eps')
        self.rot_output_type = cfg.rot_output_type
        print('EpsNet Softmax Last: ', softmax_last)

        if 'newton_equation' in self.tr_output_type or 'euler_equation' in self.rot_output_type:
            if cfg.get('sym_force', False):
                self.force_layer = SymForceLayer(
                    input_dim=self.encoder.node_hidden_dim,
                    hidden_dim=self.encoder.node_hidden_dim,
                    edge_feat_dim=self.encoder.edge_hidden_dim,
                    time_dim=time_emb_dim
                )
            else:
                self.force_layer = ForceLayer(
                    input_dim=self.encoder.node_hidden_dim,
                    hidden_dim=self.encoder.node_hidden_dim,
                    output_dim=self.encoder.node_hidden_dim,
                    n_heads=cfg.output_n_heads,
                    edge_feat_dim=self.encoder.edge_hidden_dim,
                    time_dim=time_emb_dim,
                    separate_att=cfg.get('separate_att', False),
                )
            print(self.force_layer)
        if self.tr_output_type == 'invariant_eps' or self.rot_output_type == 'invariant_eps':
            self.frag_aggr = nn.Sequential(
                nn.Linear(node_emb_dim, node_emb_dim * 2), nn.ReLU(),
                nn.Linear(node_emb_dim * 2, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, node_emb_dim)
            )
            if self.tr_output_type == 'invariant_eps':
                self.eps_crd_net = nn.Sequential(
                    nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                    nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                    nn.Linear(node_emb_dim, 1 if pred_frag_dist else 3)
                )
            if self.rot_output_type == 'invariant_eps':
                self.eps_rot_net = nn.Sequential(
                    nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                    nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                    nn.Linear(node_emb_dim, 3)
                )

        if softmax_last:
            self.eps_cls_net = nn.Sequential(
                nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, num_classes), nn.Softmax(dim=-1)
            )
        else:
            self.eps_cls_net = nn.Sequential(
                nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(),
                nn.Linear(node_emb_dim, num_classes)
            )
        if self.train_bond:
            if softmax_last:
                self.bond_pred_net = nn.Sequential(
                    nn.Linear(edge_emb_dim, edge_emb_dim), nn.ReLU(),
                    nn.Linear(edge_emb_dim, num_bond_classes), nn.Softmax(dim=-1)
                )
            else:
                self.bond_pred_net = nn.Sequential(
                    nn.Linear(edge_emb_dim, edge_emb_dim), nn.ReLU(),
                    nn.Linear(edge_emb_dim, num_bond_classes)
                )

    def update_frag_geometry(self, final_x, final_h, final_h_bond,
                             edge_index, inner_edge_mask, batch_node, mask, node_t_emb, R):

        eps_d, eps_pos, recon_x = None, None, None
        v_next, R_next = None, None
        if self.tr_output_type == 'invariant_eps' or self.rot_output_type == 'invariant_eps':
            # Aggregate features for fragment update
            frag_h = final_h[mask]
            frag_batch = batch_node[mask]
            frag_h_mean = scatter_mean(frag_h, frag_batch, dim=0)
            frag_h_mean = self.frag_aggr(frag_h_mean)  # (G, H)

            # Position / Distance changes
            if self.tr_output_type == 'invariant_eps':
                if self.pred_frag_dist:
                    eps_d = self.eps_crd_net(frag_h_mean)  # (G, 1)
                else:
                    eps_crd = self.eps_crd_net(frag_h_mean)  # local coordinates (G, 3)
                    eps_pos = apply_rotation_to_vector(R, eps_crd)  # (G, 3)

            if self.rot_output_type == 'invariant_eps':
                # New orientation
                eps_rot = self.eps_rot_net(frag_h_mean)  # (G, 3)
                U = quaternion_1ijk_to_rotation_matrix(eps_rot)  # (G, 3, 3)
                R_next = R @ U
                v_next = rotation_to_so3vec(R_next)  # (G, 3)

        if 'newton_equation' in self.tr_output_type or 'euler_equation' in self.rot_output_type:
            src, dst = edge_index
            rel_x = final_x[dst] - final_x[src]
            inner_forces, outer_forces = self.force_layer(
                final_h, rel_x, final_h_bond, edge_index, inner_edge_mask, t=node_t_emb)  # equivariant
            if 'outer' in self.tr_output_type:
                assert 'outer' in self.rot_output_type
                forces = outer_forces
            else:
                forces = inner_forces + outer_forces

            if 'newton_equation' in self.tr_output_type:
                frag_batch = batch_node[mask]
                frag_center = scatter_mean(final_x[mask], frag_batch, dim=0)  # (G, 3)
                recon_x = frag_center + scatter_mean(forces[mask], frag_batch, dim=0)

            if 'euler_equation' in self.rot_output_type:
                x_f1, force_1, batch_node_f1 = final_x[mask], forces[mask], batch_node[mask]
                mu_1 = scatter_mean(x_f1, batch_node_f1, dim=0)[batch_node_f1]
                tau = scatter_sum(torch.cross(x_f1 - mu_1, force_1), batch_node_f1, dim=0)  # (num_graphs, 3)
                inertia_mat = scatter_sum(
                    torch.sum((x_f1 - mu_1) ** 2, dim=-1)[:, None, None] * torch.eye(3)[None].to(x_f1) -
                    (x_f1 - mu_1).unsqueeze(-1) @ (x_f1 - mu_1).unsqueeze(-2), batch_node_f1,
                    dim=0)  # (num_graphs, 3, 3)
                omega = torch.linalg.solve(inertia_mat, tau.unsqueeze(-1)).squeeze(-1)  # (num_graphs, 3)
                R_next = so3vec_to_rotation(-omega) @ R
                v_next = rotation_to_so3vec(R_next)

        assert (eps_pos is not None) or (eps_d is not None) or (recon_x is not None)
        return eps_pos, eps_d, recon_x, v_next, R_next

    def forward(self,
                x_noisy, node_attr, R1_noisy, R2_noisy, edge_index, edge_attr,
                fragment_mask, edge_mask, inner_edge_mask, batch_node,
                node_t_emb, edge_t_emb, graph_t_emb, rel_fragment_mask, rel_R_noisy
                ):
        """
        Args:
            x_noisy:    (N, 3)
            node_attr:      (N, H)
            R1_noisy:   (num_graphs, 3, 3)
            f_mask:     (N, )

            v_t:    (F, 3).
            p_t:    (F, 3).
            node_feat:   (N, Hn).
            edge_feat:  (E, He).
            edge_index: (2, E)
            mask_ligand:    (N, ).
            mask_ll_edge:   (E, ).
            beta:   (F, ).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orientations, (F, 3).
            R_next: (F, 3, 3).
            eps_pos: (F, 3).
            c_denoised: (F, C).
        """
        linker_mask = (fragment_mask == 0)

        final_x, final_h, final_h_bond = self.encoder(
            pos_node=x_noisy, h_node=node_attr, h_edge=edge_attr, edge_index=edge_index,
            linker_mask=linker_mask, node_time=node_t_emb, edge_time=edge_t_emb)

        # Update linker
        outputs = {}
        if self.train_link:
            x_denoised = final_x[linker_mask]  # (L, 3)
            c_denoised = self.eps_cls_net(final_h[linker_mask])  # may have softmax-ed, (L, K)
            outputs.update({
                'linker_x': x_denoised,
                'linker_c': c_denoised
            })

        if self.train_bond:
            in_bond_feat = final_h_bond[edge_mask]
            in_bond_feat = (in_bond_feat[::2] + in_bond_feat[1::2]).repeat_interleave(2, dim=0)
            bond_denoised = self.bond_pred_net(in_bond_feat)
            outputs['linker_bond'] = bond_denoised

        # Update fragment geometry
        if self.train_frag_rot or self.train_frag_pos:
            if self.use_rel_geometry:
                eps_pos, _, recon_x, v_next, R_next = self.update_frag_geometry(
                    final_x, final_h, final_h_bond,
                    edge_index, inner_edge_mask, batch_node, rel_fragment_mask, node_t_emb, rel_R_noisy)
                outputs.update({
                    'frag_eps_pos': eps_pos,
                    'frag_v_next': v_next,
                    'frag_R_next': R_next
                })
            else:
                eps_pos1, eps_d1, recon_x1, v_next1, R_next1 = self.update_frag_geometry(
                    final_x, final_h, final_h_bond,
                    edge_index, inner_edge_mask, batch_node, (fragment_mask == 1), node_t_emb, R1_noisy)
                eps_pos2, eps_d2, recon_x2, v_next2, R_next2 = self.update_frag_geometry(
                    final_x, final_h, final_h_bond,
                    edge_index, inner_edge_mask, batch_node, (fragment_mask == 2), node_t_emb, R2_noisy)

                # zero center
                if not self.pred_frag_dist:
                    if self.tr_output_type == 'invariant_eps':
                        center = (eps_pos1 + eps_pos2) / 2
                        eps_pos1, eps_pos2 = eps_pos1 - center, eps_pos2 - center
                    elif 'newton_equation' in self.tr_output_type:
                        center = (recon_x1 + recon_x2) / 2
                        recon_x1, recon_x2 = recon_x1 - center, recon_x2 - center
                    else:
                        raise ValueError(self.tr_output_type)

                outputs.update({
                    'frag_eps_pos': (eps_pos1, eps_pos2),
                    'frag_eps_d': eps_d1 + eps_d2 if self.pred_frag_dist else None,
                    'frag_recon_x': (recon_x1, recon_x2),
                    'frag_v_next': (v_next1, v_next2),
                    'frag_R_next': (R_next1, R_next2)
                })

        return outputs
