import os
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from openbabel import openbabel as ob
import torch


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


def process_from_mol(rdmol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element, valence, charge = [], [], []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        valence.append(atom.GetTotalValence())
        charge.append(atom.GetFormalCharge())
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int)
    valence = np.array(valence, dtype=np.int)
    charge = np.array(charge, dtype=np.int)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'rdmol': rdmol,
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'valence': valence,
        'charge': charge
    }
    return data


# rdmol conformer
def compute_3d_coors(mol, random_seed=0):
    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if success == -1:
        return 0, 0
    mol = Chem.RemoveHs(mol)
    c = mol.GetConformer(0)
    pos = c.GetPositions()
    return pos, 1


def compute_3d_coors_multiple(mol, numConfs=20, maxIters=400, randomSeed=1):
    # mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=0, randomSeed=randomSeed)
    if mol.GetConformers() == ():
        return None, 0
    try:
        result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=maxIters, numThreads=0)
    except Exception as e:
        print(str(e))
        return None, 0
    mol = Chem.RemoveHs(mol)
    result = [tuple((result[i][0], result[i][1], i)) for i in range(len(result)) if result[i][0] == 0]
    if result == []:  # no local minimum on energy surface is found
        return None, 0
    result.sort()
    return mol.GetConformers()[result[0][-1]].GetPositions(), 1


def compute_3d_coors_frags(mol, numConfs=20, maxIters=400, randomSeed=1):
    du = Chem.MolFromSmiles('*')
    clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)),du,Chem.MolFromSmiles('[H]'),True)[0])
    frag = Chem.CombineMols(clean_frag, Chem.MolFromSmiles("*.*"))
    mol_to_link_carbon = AllChem.ReplaceSubstructs(mol, du, Chem.MolFromSmiles('C'), True)[0]
    pos, _ = compute_3d_coors_multiple(mol_to_link_carbon, numConfs, maxIters, randomSeed)
    return pos


# -----
def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    assert mol.GetNumAtoms() == pos.shape[0]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.AddConformer(conf, assignId=True)

    # for i in range(pos.shape[0]):
    #     mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def set_rdmol_positions(rdkit_mol, pos, reset=True):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = copy.deepcopy(rdkit_mol)
    if reset:
        mol.RemoveAllConformers()
    set_rdmol_positions_(mol, pos)
    return mol
