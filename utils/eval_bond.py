"""Utils for evaluating bond length."""

import collections
from typing import Tuple, Sequence, Dict
from rdkit.Chem.rdMolTransforms import GetAngleDeg
from rdkit import Chem
import numpy as np
from utils.data import BOND_TYPES

BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution
DISTANCE_BINS = np.arange(1.1, 1.7, 0.005)[:-1]
ANGLE_BINS = np.arange(0., 180., 1.)[:-1]
DIHEDRAL_BINS = np.arange(-180., 180., 1.)[:-1]

# bond distance
BOND_DISTS = (
    (6, 6, 1),
    (6, 6, 2),
    (6, 7, 1),
    (6, 7, 2),
    (6, 8, 1),
    (6, 8, 2),
    (6, 6, 4),
    (6, 7, 4),
    (6, 8, 4),
)
BOND_ANGLES = [
    'CCC',
    'CCO',
    'CNC',
    'OPO',
    'NCC',
    'CC=O',
    'COC'
]
DIHEDRAL_ANGLES = [
    # 'C1C-C1C-C1C',
    # 'C12C-C12C-C12C',
    # 'C1C-C1C-C1O',
    # 'O1C-C1C-C1O',
    # 'C1c-c12c-c12c',
    # 'C1C-C2C-C1C'
]


def get_bond_angle(mol, fragment_mask, bond_smi='CCC'):
    """
    Find bond pairs (defined by bond_smi) in mol and return the angle of the bond pair
    bond_smi: bond pair smiles, e.g. 'CCC'
    """
    deg_list = []
    substructure = Chem.MolFromSmiles(bond_smi)
    bond_pairs = mol.GetSubstructMatches(substructure)
    for pair in bond_pairs:
        if (fragment_mask[pair[0]] == 0) | (fragment_mask[pair[1]] == 0) | (fragment_mask[pair[2]] == 0):
            deg_list += [GetAngleDeg(mol.GetConformer(), *pair)]
            assert mol.GetBondBetweenAtoms(pair[0], pair[1]) is not None
            assert mol.GetBondBetweenAtoms(pair[2], pair[1]) is not None
    return deg_list


def get_bond_symbol(bond):
    """
    Return the symbol representation of a bond
    """
    a0 = bond.GetBeginAtom().GetSymbol()
    a1 = bond.GetEndAtom().GetSymbol()
    b = str(int(bond.GetBondType()))  # single: 1, double: 2, triple: 3, aromatic: 12
    return ''.join([a0, b, a1])


def get_triple_bonds(mol):
    """
    Get all the bond triplets in a molecule
    """
    valid_triple_bonds = []
    for idx_bond, bond in enumerate(mol.GetBonds()):
        idx_begin_atom = bond.GetBeginAtomIdx()
        idx_end_atom = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(idx_begin_atom)
        end_atom = mol.GetAtomWithIdx(idx_end_atom)
        begin_bonds = begin_atom.GetBonds()
        valid_left_bonds = []
        for begin_bond in begin_bonds:
            if begin_bond.GetIdx() == idx_bond:
                continue
            else:
                valid_left_bonds.append(begin_bond)
        if len(valid_left_bonds) == 0:
            continue

        end_bonds = end_atom.GetBonds()
        for end_bond in end_bonds:
            if end_bond.GetIdx() == idx_bond:
                continue
            else:
                for left_bond in valid_left_bonds:
                    valid_triple_bonds.append([left_bond, bond, end_bond])
    return valid_triple_bonds


def get_dihedral_angle(mol, bonds_ref_sym):
    """
    find bond triplets (defined by bonds_ref_sym) in mol and return the dihedral angle of the bond triplet
    bonds_ref_sym: a symbol string of bond triplet, e.g. 'C1C-C1C-C1C'
    """
    # bonds_ref_sym = '-'.join(get_bond_symbol(bonds_ref))
    bonds_list = get_triple_bonds(mol)
    angles_list = []
    for bonds in bonds_list:
        sym = '-'.join([get_bond_symbol(b) for b in bonds])
        sym1 = '-'.join([get_bond_symbol(b) for b in bonds][::-1])
        atoms = []
        if (sym == bonds_ref_sym) or (sym1 == bonds_ref_sym):
            if sym1 == bonds_ref_sym:
                bonds = bonds[::-1]
            bond0 = bonds[0]
            atom0 = bond0.GetBeginAtomIdx()
            atom1 = bond0.GetEndAtomIdx()

            bond1 = bonds[1]
            atom1_0 = bond1.GetBeginAtomIdx()
            atom1_1 = bond1.GetEndAtomIdx()
            if atom0 == atom1_0:
                i, j, k = atom1, atom0, atom1_1
            elif atom0 == atom1_1:
                i, j, k = atom1, atom0, atom1_0
            elif atom1 == atom1_0:
                i, j, k = atom0, atom1, atom1_1
            elif atom1 == atom1_1:
                i, j, k = atom0, atom1, atom1_0

            bond2 = bonds[2]
            atom2_0 = bond2.GetBeginAtomIdx()
            atom2_1 = bond2.GetEndAtomIdx()
            if atom2_0 == k:
                l = atom2_1
            elif atom2_1 == k:
                l = atom2_0
            # print(i,j,k,l)
            angle = Chem.rdMolTransforms.GetDihedralDeg(mol.GetConformer(), i, j, k, l)
            angles_list.append(angle)
    return angles_list


def bond_distance_from_mol(mol, fragment_mask):
    # only consider linker-related distance
    pos = mol.GetConformer().GetPositions()
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    all_distances = []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (fragment_mask[s_idx] == 0) | (fragment_mask[e_idx] == 0):
            bond_type = BOND_TYPES[bond.GetBondType()]
            distance = pdist[s_idx, e_idx]
            all_distances.append(((s_sym, e_sym, bond_type), distance))
    return all_distances


def _format_bond_type(bond_type: BondType) -> BondType:
    atom1, atom2, bond_category = bond_type
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1
    return atom1, atom2, bond_category


def get_distribution(distances: Sequence[float], bins=DISTANCE_BINS) -> np.ndarray:
    """Get the distribution of distances.

    Args:
        distances (list): List of distances.
        bins (list): bins of distances
    Returns:
        np.array: empirical distribution of distances with length equals to DISTANCE_BINS.
    """
    bin_counts = collections.Counter(np.searchsorted(bins, distances))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts


def get_bond_length_profile(mol_list, fragment_mask_list) -> BondLengthProfile:
    bond_lengths = []
    for mol, mask in zip(mol_list, fragment_mask_list):
        mol = Chem.RemoveAllHs(mol)
        bond_lengths += bond_distance_from_mol(mol, mask)

    bond_length_profile = collections.defaultdict(list)
    for bond_type, bond_length in bond_lengths:
        bond_type = _format_bond_type(bond_type)
        bond_length_profile[bond_type].append(bond_length)
    bond_length_profile = {k: get_distribution(v) for k, v in bond_length_profile.items()}
    return bond_length_profile


def get_bond_angles_dict(mol_list, fragment_mask_list):
    bond_angles = collections.defaultdict(list)
    dihedral_angles = collections.defaultdict(list)
    for mol, mask in zip(mol_list, fragment_mask_list):
        mol = Chem.RemoveAllHs(mol)
        for angle_type in BOND_ANGLES:
            bond_angles[angle_type] += get_bond_angle(mol, mask, bond_smi=angle_type)
        for angle_type in DIHEDRAL_ANGLES:
            dihedral_angles[angle_type] += get_dihedral_angle(mol, angle_type)
    return bond_angles, dihedral_angles


def get_bond_angles_profile(mol_list, fragment_mask_list):
    angles, dihedrals = get_bond_angles_dict(mol_list, fragment_mask_list)
    angles_profile = {}
    for k, v in angles.items():
        angles_profile[k] = get_distribution(v, ANGLE_BINS)
    for k, v in dihedrals.items():
        angles_profile[k] = get_distribution(v, DIHEDRAL_BINS)
    return angles_profile
