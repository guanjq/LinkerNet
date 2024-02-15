import torch
import numpy as np
from scipy.spatial.distance import cdist

from rdkit import Chem, Geometry

from utils import const
from copy import deepcopy
import re
import itertools


class MolReconsError(Exception):
    pass


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def build_linker_mol(linker_x, linker_c, add_bonds=False):
    mol = Chem.RWMol()
    for atom in linker_c:
        a = Chem.Atom(int(atom))
        mol.AddAtom(a)

    if add_bonds:
        # predict bond order
        n = len(linker_x)
        ptable = Chem.GetPeriodicTable()
        dists = cdist(linker_x, linker_x, 'euclidean')
        E = torch.zeros((n, n), dtype=torch.int)
        A = torch.zeros((n, n), dtype=torch.bool)
        for i in range(n):
            for j in range(i):
                atom_i = ptable.GetElementSymbol(int(linker_c[i]))
                atom_j = ptable.GetElementSymbol(int(linker_c[j]))
                order = get_bond_order(atom_i, atom_j, dists[i, j], margins=const.MARGINS_EDM)
                if order > 0:
                    A[i, j] = 1
                    E[i, j] = order
        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])
    return mol


def reconstruct_mol(frag_mol, frag_x, linker_x, linker_c):
    # construct linker mol
    linker_mol = build_linker_mol(linker_x, linker_c, add_bonds=True)

    # combine mol and assign conformer
    mol = Chem.CombineMols(frag_mol, linker_mol)
    mol = Chem.RWMol(mol)
    # frag_x = frag_mol.GetConformer().GetPositions()
    all_x = np.concatenate([frag_x, linker_x], axis=0)
    mol.RemoveAllConformers()
    mol.AddConformer(create_conformer(all_x))

    # add frag-linker bond
    n_frag_atoms = frag_mol.GetNumAtoms()
    n = mol.GetNumAtoms()

    dists = cdist(all_x, all_x, 'euclidean')
    E = torch.zeros((n, n), dtype=torch.int)
    A = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n_frag_atoms):
        for j in range(n_frag_atoms, n):
            atom_i = mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = mol.GetAtomWithIdx(j).GetSymbol()
            order = get_bond_order(atom_i, atom_j, dists[i, j], margins=const.MARGINS_EDM)
            if order > 0:
                A[i, j] = 1
                E[i, j] = order
    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])

    # frag_c = [atom.GetAtomicNum() for atom in frag_mol.GetAtoms()]
    # all_x = np.concatenate([frag_x, linker_x], axis=0)
    # all_c = np.concatenate([frag_c, linker_c], axis=0)
    # print('all c: ', all_c)
    # mol = build_linker_mol(all_x, all_c)
    #
    # try:
    #     Chem.SanitizeMol(mol)
    #     fixed = True
    # except Exception as e:
    #     fixed = False
    #
    # if not fixed:
    #     mol, fixed = fix_valence(mol)
    return mol


def reconstruct_mol_with_bond(frag_mol, frag_x, linker_x, linker_c,
                              linker_bond_index, linker_bond_type, known_linker_bonds=True, check_validity=True):
    # construct linker mol
    linker_mol = build_linker_mol(linker_x, linker_c, add_bonds=known_linker_bonds)

    # combine mol and assign conformer
    mol = Chem.CombineMols(frag_mol, linker_mol)
    linker_atom_idx = list(range(mol.GetNumAtoms() - linker_mol.GetNumAtoms(), mol.GetNumAtoms()))
    mol = Chem.RWMol(mol)
    # frag_x = frag_mol.GetConformer().GetPositions()
    all_x = np.concatenate([frag_x, linker_x], axis=0)
    mol.RemoveAllConformers()
    mol.AddConformer(create_conformer(all_x))

    linker_bond_index, linker_bond_type = linker_bond_index.tolist(), linker_bond_type.tolist()
    anchor_indices = set()
    # add bonds
    for i, type_this in enumerate(linker_bond_type):
        node_i, node_j = linker_bond_index[0][i], linker_bond_index[1][i]
        if node_i < node_j:
            if type_this == 0:
                continue
            else:
                if node_i in linker_atom_idx and node_j not in linker_atom_idx:
                    anchor_indices.add(int(node_j))
                elif node_j in linker_atom_idx and node_i not in linker_atom_idx:
                    anchor_indices.add(int(node_i))

                if type_this == 1:
                    mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
                elif type_this == 2:
                    mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
                elif type_this == 3:
                    mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
                elif type_this == 4:
                    mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
                else:
                    raise Exception('unknown bond order {}'.format(type_this))

    mol = mol.GetMol()
    for anchor_idx in anchor_indices:
        atom = mol.GetAtomWithIdx(anchor_idx)
        atom.SetNumExplicitHs(0)

    if check_validity:
        mol = fix_validity(mol, linker_atom_idx)

    # check valid
    # rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    # if (rd_mol_check is None) and check_validity:
    #     raise MolReconsError()
    return mol


def fix_validity(mol, linker_atom_idx):
    try:
        Chem.SanitizeMol(mol)
        fixed = True
    except Exception as e:
        fixed = False

    if not fixed:
        try:
            Chem.Kekulize(deepcopy(mol))
        except Chem.rdchem.KekulizeException as e:
            err = e
            if 'Unkekulized' in err.args[0]:
                mol, fixed = fix_aromatic(mol)

    # valence error for N
    if not fixed:
        mol, fixed = fix_valence(mol, linker_atom_idx)

    # print('s2')
    if not fixed:
        mol, fixed = fix_aromatic(mol, True, linker_atom_idx)

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        # raise MolReconsError()
        return None
    return mol


def fix_valence(mol, linker_atom_idx=None):
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            if linker_atom_idx is None or int(index[0]) in linker_atom_idx:
                mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    return mol, fixed


def get_ring_sys(mol):
    all_rings = Chem.GetSymmSSSR(mol)
    if len(all_rings) == 0:
        ring_sys_list = []
    else:
        ring_sys_list = [all_rings[0]]
        for ring in all_rings[1:]:
            form_prev = False
            for prev_ring in ring_sys_list:
                if set(ring).intersection(set(prev_ring)):
                    prev_ring.extend(ring)
                    form_prev = True
                    break
            if not form_prev:
                ring_sys_list.append(ring)
    ring_sys_list = [list(set(x)) for x in ring_sys_list]
    return ring_sys_list


def get_all_subsets(ring_list):
    all_sub_list = []
    for n_sub in range(len(ring_list)+1):
        all_sub_list.extend(itertools.combinations(ring_list, n_sub))
    return all_sub_list


def fix_aromatic(mol, strict=False, linker_atom_idx=None):
    mol_orig = mol
    atomatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
    N_ring_list = []
    S_ring_list = []
    for ring_sys in get_ring_sys(mol):
        if set(ring_sys).intersection(set(atomatic_list)):
            idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
            if len(idx_N) > 0:
                idx_N.append(-1) # -1 for not add to this loop
                N_ring_list.append(idx_N)
            idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
            if len(idx_S) > 0:
                idx_S.append(-1) # -1 for not add to this loop
                S_ring_list.append(idx_S)
    # enumerate S
    fixed = False
    if strict:
        S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
        permutation = get_all_subsets(S_ring_list)
    else:
        permutation = list(itertools.product(*S_ring_list))
    for perm in permutation:
        mol = deepcopy(mol_orig)
        perm = [x for x in perm if x != -1]
        for idx in perm:
            if linker_atom_idx is None or idx in linker_atom_idx:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        try:
            if strict:
                mol, fixed = fix_valence(mol, linker_atom_idx)
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except:
            continue
    # enumerate N
    if not fixed:
        if strict:
            N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
            permutation = get_all_subsets(N_ring_list)
        else:
            permutation = list(itertools.product(*N_ring_list))
        for perm in permutation:  # each ring select one atom
            perm = [x for x in perm if x != -1]
            # print(perm)
            actions = itertools.product([0, 1], repeat=len(perm))
            for action in actions: # add H or charge
                mol = deepcopy(mol_orig)
                for idx, act_atom in zip(perm, action):
                    if linker_atom_idx is None or idx in linker_atom_idx:
                        if act_atom == 0:
                            mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                        else:
                            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                try:
                    if strict:
                        mol, fixed = fix_valence(mol, linker_atom_idx)
                    Chem.SanitizeMol(mol)
                    fixed = True
                    break
                except:
                    continue
            if fixed:
                break
    return mol, fixed


def parse_sampling_result(data_list, final_x, final_c, atom_featurizer):
    all_mols = []
    for data, x_gen, c_gen in zip(data_list, final_x, final_c):
        frag_pos = x_gen[data.fragment_mask > 0].cpu().numpy().astype(np.float64)
        linker_pos = x_gen[data.fragment_mask == 0].cpu().numpy().astype(np.float64)
        linker_ele = [atom_featurizer.get_element_from_index(int(c)) for c in c_gen[data.linker_mask]]
        full_mol = reconstruct_mol(data.frag_mol, frag_pos, linker_pos, linker_ele)
        all_mols.append(full_mol)
    return all_mols


def parse_sampling_result_with_bond(data_list, final_x, final_c, final_bond, atom_featurizer,
                                    known_linker_bonds=True, check_validity=False):
    all_mols = []
    if not isinstance(data_list, list):
        data_list = [data_list] * len(final_x)
    for data, x_gen, c_gen, b_gen in zip(data_list, final_x, final_c, final_bond):
        frag_pos = x_gen[data.fragment_mask > 0].cpu().numpy().astype(np.float64)
        linker_pos = x_gen[data.fragment_mask == 0].cpu().numpy().astype(np.float64)
        linker_ele = [atom_featurizer.get_element_from_index(int(c)) for c in c_gen[data.linker_mask]]
        linker_bond_type = b_gen[data.linker_bond_mask]
        linker_bond_index = data.edge_index[:, data.linker_bond_mask]
        full_mol = reconstruct_mol_with_bond(
            data.frag_mol, frag_pos, linker_pos, linker_ele, linker_bond_index, linker_bond_type,
            known_linker_bonds, check_validity)
        all_mols.append(full_mol)
    return all_mols
