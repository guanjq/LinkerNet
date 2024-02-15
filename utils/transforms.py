import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.transforms import BaseTransform

from utils import so3
from utils.geometry import local_to_global, global_to_local, rotation_matrix_from_vectors


# required by model:
# x_noisy, c_noisy, atom_feat, edge_index, edge_feat,
# R1_noisy, R2_noisy, R1, R2, eps_p1, eps_p2, x_0, c_0, t, fragment_mask, batch


def modify_frags_conformer(frags_local_pos, frag_idx_mask, v_frags, p_frags):
    R_frags = so3.so3vec_to_rotation(v_frags)
    x_frags = torch.zeros_like(frags_local_pos)
    for i in range(2):
        noisy_pos = local_to_global(R_frags[i], p_frags[i],
                                    frags_local_pos[frag_idx_mask == i + 1])
        x_frags[frag_idx_mask == i + 1] = noisy_pos
    return x_frags


def dataset_info(dataset):  # qm9, zinc, cep
    if dataset == 'qm9':
        return {'atom_types': ["H", "C", "N", "O", "F"],
                'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                }
    elif dataset == 'zinc' or dataset == 'protac':
        return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                               'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
                'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3},
                'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'},
                'bucket_sizes': np.array(
                    [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
                }

    elif dataset == "cep":
        return {'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                'bucket_sizes': np.array([25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 43, 46])
                }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)


class FeaturizeAtom(BaseTransform):

    def __init__(self, dataset_name, known_anchor=False,
                 add_atom_type=True, add_atom_feat=True):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info(dataset_name)
        self.known_anchor = known_anchor
        self.add_atom_type = add_atom_type
        self.add_atom_feat = add_atom_feat

    @property
    def num_classes(self):
        return len(self.dataset_info['atom_types'])

    @property
    def feature_dim(self):
        n_feat_dim = 2
        if self.known_anchor:
            n_feat_dim += 2
        return n_feat_dim

    def get_index(self, atom_num, valence, charge):
        if self.dataset_name in ['zinc', 'protac']:
            pt = Chem.GetPeriodicTable()
            atom_str = "%s%i(%i)" % (pt.GetElementSymbol(atom_num), valence, charge)
            return self.dataset_info['atom_types'].index(atom_str)
        else:
            raise ValueError

    def get_element_from_index(self, index):
        pt = Chem.GetPeriodicTable()
        symb = self.dataset_info['number_to_atom'][index]
        return pt.GetAtomicNumber(symb)

    def __call__(self, data):
        if self.add_atom_type:
            x = [self.get_index(int(e), int(v), int(c)) for e, v, c in zip(data.element, data.valence, data.charge)]
            data.atom_type = torch.tensor(x)
        if self.add_atom_feat:
            # fragment / linker indicator, independent with atom types
            linker_flag = F.one_hot((data.fragment_mask == 0).long(), 2)
            all_feats = [linker_flag]
            # fragment anchor flag
            if self.known_anchor:
                anchor_flag = F.one_hot((data.anchor_mask == 1).long(), 2)
                all_feats.append(anchor_flag)
            data.atom_feat = torch.cat(all_feats, -1)
        return data


class BuildCompleteGraph(BaseTransform):

    def __init__(self, known_linker_bond=False, known_cand_anchors=False):
        super().__init__()
        self.known_linker_bond = known_linker_bond
        self.known_cand_anchors = known_cand_anchors

    @property
    def num_bond_classes(self):
        return 5

    @property
    def bond_feature_dim(self):
        return 4

    @staticmethod
    def _get_interleave_edge_index(edge_index):
        edge_index_sym = torch.stack([edge_index[1], edge_index[0]])
        e = torch.zeros_like(torch.cat([edge_index, edge_index_sym], dim=-1))
        e[:, ::2] = edge_index
        e[:, 1::2] = edge_index_sym
        return e

    def _build_interleave_fc(self, n1_atoms, n2_atoms):
        eij = torch.triu_indices(n1_atoms, n2_atoms, offset=1)
        e = self._get_interleave_edge_index(eij)
        return e

    def __call__(self, data):
        # fully connected graph
        num_nodes = len(data.pos)
        fc_edge_index = self._build_interleave_fc(num_nodes, num_nodes)
        data.edge_index = fc_edge_index

        # (ll, lf, fl, ff) indicator
        src, dst = data.edge_index
        num_edges = len(fc_edge_index[0])
        edge_type = torch.zeros(num_edges).long()
        l_ind_src = data.fragment_mask[src] == 0
        l_ind_dst = data.fragment_mask[dst] == 0
        edge_type[l_ind_src & l_ind_dst] = 0
        edge_type[l_ind_src & ~l_ind_dst] = 1
        edge_type[~l_ind_src & l_ind_dst] = 2
        edge_type[~l_ind_src & ~l_ind_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        data.edge_feat = edge_type

        # bond type 0: none 1: singe 2: double 3: triple 4: aromatic
        bond_type = torch.zeros(num_edges).long()

        id_fc_edge = fc_edge_index[0] * num_nodes + fc_edge_index[1]
        id_frag_bond = data.bond_index[0] * num_nodes + data.bond_index[1]
        idx_edge = torch.tensor([torch.nonzero(id_fc_edge == id_).squeeze() for id_ in id_frag_bond])
        bond_type[idx_edge] = data.bond_type
        # data.edge_type = F.one_hot(bond_type, num_classes=5)
        data.edge_type = bond_type
        if self.known_linker_bond:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) ^ (data.fragment_mask[dst] == 0)
        elif self.known_cand_anchors:
            ll_bond = (data.fragment_mask[src] == 0) & (data.fragment_mask[dst] == 0)
            fl_bond = (data.cand_anchors_mask[src] == 1) & (data.fragment_mask[dst] == 0)
            lf_bond = (data.cand_anchors_mask[dst] == 1) & (data.fragment_mask[src] == 0)
            data.linker_bond_mask = ll_bond | fl_bond | lf_bond
        else:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) | (data.fragment_mask[dst] == 0)

        data.inner_edge_mask = (data.fragment_mask[src] == data.fragment_mask[dst])
        return data


class SelectCandAnchors(BaseTransform):

    def __init__(self, mode='exact', k=2):
        super().__init__()
        self.mode = mode
        assert mode in ['exact', 'k-hop']
        self.k = k

    @staticmethod
    def bfs(nbh_list, node, k=2, valid_list=[]):
        visited = [node]
        queue = [node]
        level = [0]
        bfs_perm = []

        while len(queue) > 0:
            m = queue.pop(0)
            l = level.pop(0)
            if l > k:
                break
            bfs_perm.append(m)

            for neighbour in nbh_list[m]:
                if neighbour not in visited and neighbour in valid_list:
                    visited.append(neighbour)
                    queue.append(neighbour)
                    level.append(l + 1)
        return bfs_perm

    def __call__(self, data):
        # link_indices = (data.linker_mask == 1).nonzero()[:, 0].tolist()
        # frag_indices = (data.linker_mask == 0).nonzero()[:, 0].tolist()
        # anchor_indices = [j for i, j in zip(*data.bond_index.tolist()) if i in link_indices and j in frag_indices]
        # data.anchor_indices = anchor_indices
        cand_anchors_mask = torch.zeros_like(data.fragment_mask).bool()
        if self.mode == 'exact':
            cand_anchors_mask[data.anchor_indices] = True
            data.cand_anchors_mask = cand_anchors_mask

        elif self.mode == 'k-hop':
            # data.nbh_list = {i.item(): [j.item() for k, j in enumerate(data.bond_index[1])
            #                             if data.bond_index[0, k].item() == i] for i in data.bond_index[0]}
            # all_cand = []
            for anchor in data.anchor_indices:
                a_frag_id = data.fragment_mask[anchor]
                a_valid_list = (data.fragment_mask == a_frag_id).nonzero(as_tuple=True)[0].tolist()
                a_cand = self.bfs(data.nbh_list, anchor, k=self.k, valid_list=a_valid_list)
                a_cand = [a for a in a_cand if data.frag_mol.GetAtomWithIdx(a).GetTotalNumHs() > 0]
                cand_anchors_mask[a_cand] = True
                # all_cand.append(a_cand)
            data.cand_anchors_mask = cand_anchors_mask
        else:
            raise ValueError(self.mode)
        return data


class StackFragLocalPos(BaseTransform):
    def __init__(self, max_num_atoms=30):
        super().__init__()
        self.max_num_atoms = max_num_atoms

    def __call__(self, data):
        frag_idx_mask = data.fragment_mask[data.fragment_mask > 0]
        f1_pos = data.frags_local_pos[frag_idx_mask == 1]
        f2_pos = data.frags_local_pos[frag_idx_mask == 2]
        assert len(f1_pos) <= self.max_num_atoms
        assert len(f2_pos) <= self.max_num_atoms
        # todo: use F.pad
        f1_fill_pos = torch.cat([f1_pos, torch.zeros(self.max_num_atoms - len(f1_pos), 3)], dim=0)
        f1_mask = torch.cat([torch.ones(len(f1_pos)), torch.zeros(self.max_num_atoms - len(f1_pos))], dim=0)
        f2_fill_pos = torch.cat([f2_pos, torch.zeros(self.max_num_atoms - len(f2_pos), 3)], dim=0)
        f2_mask = torch.cat([torch.ones(len(f2_pos)), torch.zeros(self.max_num_atoms - len(f2_pos))], dim=0)
        data.frags_local_pos_filled = torch.stack([f1_fill_pos, f2_fill_pos], dim=0)
        data.frags_local_pos_mask = torch.stack([f1_mask, f2_mask], dim=0).bool()
        return data


class RelativeGeometry(BaseTransform):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def __call__(self, data):
        if self.mode == 'relative_pos_and_rot':
            # randomly take first / second fragment as the reference
            idx = torch.randint(0, 2, [1])[0]
            pos = (data.pos - data.frags_t[idx]) @ data.frags_R[idx]
            frags_R = data.frags_R[idx].T @ data.frags_R
            frags_t = (data.frags_t - data.frags_t[idx]) @ data.frags_R[idx]
            # frags_d doesn't change
            data.frags_rel_mask = torch.tensor([True, True])
            data.frags_rel_mask[idx] = False  # the reference fragment will not be added noise later
            data.frags_atom_rel_mask = data.fragment_mask == (2 - idx)

        elif self.mode == 'two_pos_and_rot':
            # still guarantee the center of two fragments' centers is the origin
            rand_rot = get_random_rot()
            pos = data.pos @ rand_rot
            frags_R = rand_rot.T @ data.frags_R
            frags_t = data.frags_t @ rand_rot
            data.frags_rel_mask = torch.tensor([True, True])

        elif self.mode == 'distance_and_two_rot_aug':
            # only the first row of frags_R unchanged
            rand_rot = get_random_rot()
            tmp_pos = data.pos @ rand_rot
            tmp_frags_R = rand_rot.T @ data.frags_R
            tmp_frags_t = data.frags_t @ rand_rot

            rot = rotation_matrix_from_vectors(tmp_frags_t[1] - tmp_frags_t[0], torch.tensor([1., 0., 0.]))
            tr = -rot @ ((tmp_frags_t[0] + tmp_frags_t[1]) / 2)
            pos = tmp_pos @ rot.T + tr
            frags_R = rot @ tmp_frags_R
            frags_t = tmp_frags_t @ rot.T + tr
            data.frags_rel_mask = torch.tensor([True, True])

        elif self.mode == 'distance_and_two_rot':
            # unchanged
            frags_R = data.frags_R
            frags_t = data.frags_t
            pos = data.pos
            data.frags_rel_mask = torch.tensor([True, True])

        else:
            raise ValueError(self.mode)

        data.frags_R = frags_R
        data.frags_t = frags_t
        data.pos = pos
        # print('frags_R: ', data.frags_R,  'frags_t: ', frags_t)
        return data


def get_random_rot():
    M = np.random.randn(3, 3)
    Q, __ = np.linalg.qr(M)
    rand_rot = torch.from_numpy(Q.astype(np.float32))
    return rand_rot


class ReplaceLocalFrame(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        frags_R1 = get_random_rot()
        frags_R2 = get_random_rot()
        f1_local_pos = global_to_local(frags_R1, data.frags_t[0], data.pos[data.fragment_mask == 1])
        f2_local_pos = global_to_local(frags_R2, data.frags_t[1], data.pos[data.fragment_mask == 2])
        data.frags_local_pos = torch.cat([f1_local_pos, f2_local_pos], dim=0)
        return data
