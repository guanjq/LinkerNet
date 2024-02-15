import argparse
import pickle
from tqdm.auto import tqdm
from rdkit import Chem
import torch
import os
import numpy as np
from utils.data import compute_3d_coors_multiple, set_rdmol_positions


def read_protac_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for i, line in enumerate(lines):
        toks = line.strip().split(' ')
        if len(toks) == 4:
            smi_protac, smi_linker, smi_warhead, smi_ligase = toks
        else:
            raise ValueError("Incorrect input format.")
        data.append({'smi_protac': smi_protac,
                     'smi_linker': smi_linker,
                     'smi_warhead': smi_warhead,
                     'smi_ligase': smi_ligase})
    return data


def validity_check(num_atoms, m1, m2, m3):
    if len(set(m1).intersection(set(m2))) != 0 or len(set(m2).intersection(set(m3))) != 0 or \
            len(set(m1).intersection(set(m3))) != 0 or len(m1) + len(m2) + len(m3) != num_atoms:
        return False
    else:
        return True


def preprocess_protac(raw_file_path, save_path):
    if raw_file_path.endswith('.txt'):
        raw_data = read_protac_file(raw_file_path)
    else:
        with open(raw_file_path, 'rb') as f:
            raw_data = pickle.load(f)
    processed_data = []
    total = len(raw_data)
    for i, d in enumerate(tqdm(raw_data, desc='Generate 3D conformer')):
        smi_protac, smi_linker, smi_warhead, smi_ligase = d['smi_protac'], d['smi_linker'], \
                                                          d['smi_warhead'], d['smi_ligase']

        # frag_0 (3d pos), frag_1 (3d pos), linker (3d pos)
        mol = Chem.MolFromSmiles(smi_protac)
        # generate 3d coordinates of mols
        pos, _ = compute_3d_coors_multiple(mol, maxIters=1000)
        if pos is None:
            print('Generate conformer fail!')
            continue
        mol = set_rdmol_positions(mol, pos)

        if raw_file_path.endswith('.txt'):
            warhead_m = mol.GetSubstructMatch(Chem.MolFromSmiles(smi_warhead))
            ligase_m = mol.GetSubstructMatch(Chem.MolFromSmiles(smi_ligase))
            linker_m = mol.GetSubstructMatch(Chem.MolFromSmiles(smi_linker))
            d.update({
                'atom_indices_warhead': warhead_m,
                'atom_indices_ligase': ligase_m,
                'atom_indices_linker': linker_m
            })
        else:
            warhead_m, ligase_m, linker_m = d['atom_indices_warhead'], d['atom_indices_ligase'], d['atom_indices_linker']

        valid = validity_check(mol.GetNumAtoms(), warhead_m, ligase_m, linker_m)
        if not valid:
            print('Validity check fail!')
            continue

        processed_data.append({
            'mol': mol,
            **d
        })

    print('Saving data')
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print('Length raw data: \t%d' % total)
    print('Length processed data: \t%d' % len(processed_data))


def preprocess_zinc_from_difflinker(raw_file_dir, save_path, mode):
    datasets = {}
    if mode == 'full':
        all_subsets = ['train', 'val', 'test']
    else:
        all_subsets = ['val', 'test']
    for subset in all_subsets:
        data_list = []
        n_fail = 0
        all_data = torch.load(os.path.join(raw_file_dir, f'zinc_final_{subset}.pt'), map_location='cpu')
        full_rdmols = Chem.SDMolSupplier(os.path.join(raw_file_dir, f'zinc_final_{subset}_mol.sdf'), sanitize=False)
        frag_rdmols = Chem.SDMolSupplier(os.path.join(raw_file_dir, f'zinc_final_{subset}_frag.sdf'), sanitize=False)
        link_rdmols = Chem.SDMolSupplier(os.path.join(raw_file_dir, f'zinc_final_{subset}_link.sdf'), sanitize=False)
        assert len(all_data) == len(frag_rdmols) == len(link_rdmols)
        for i in tqdm(range(len(all_data)), desc=subset):
            data = all_data[i]
            mol, frag_mol, link_mol = full_rdmols[i], frag_rdmols[i], link_rdmols[i]
            # if mol is None or frag_mol is None or link_mol is None:
            #     print('Fail i: ', i)
            #     n_fail += 1
            #     continue
            pos = data['positions']
            fragment_mask, linker_mask = data['fragment_mask'].bool(), data['linker_mask'].bool()
            # align full mol with positions, etc.
            mol_pos = mol.GetConformer().GetPositions()
            mapping = np.linalg.norm(mol_pos[None] - data['positions'].numpy()[:, None], axis=-1).argmin(axis=1)
            assert len(np.unique(mapping)) == len(mapping)
            new_mol = Chem.RenumberAtoms(mol, mapping.tolist())
            Chem.SanitizeMol(new_mol)
            # check frag mol and link mol are aligned
            assert np.allclose(frag_mol.GetConformer().GetPositions(), pos[fragment_mask].numpy())
            assert np.allclose(link_mol.GetConformer().GetPositions(), pos[linker_mask].numpy())
            # print(mapping)

            # print(data['anchors'].nonzero(as_tuple=True)[0].tolist())
            # Note: anchor atom index may be wrong!
            frag_mols = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
            assert len(frag_mols) == 2

            all_frag_atom_idx = set((fragment_mask == 1).nonzero()[:, 0].tolist())
            frag_atom_idx_list = []
            for m1 in new_mol.GetSubstructMatches(frag_mols[0]):
                for m2 in new_mol.GetSubstructMatches(frag_mols[1]):
                    if len(set(m1).intersection(set(m2))) == 0 and set(m1).union(set(m2)) == all_frag_atom_idx:
                        frag_atom_idx_list = [m1, m2]
                        break

            try:
                assert len(frag_atom_idx_list) == 2 and all([x is not None and len(x) > 0 for x in frag_atom_idx_list])
            except:
                print('Fail i: ', i)
                n_fail += 1
                continue
            new_fragment_mask = torch.zeros_like(fragment_mask).long()
            new_fragment_mask[list(frag_atom_idx_list[0])] = 1
            new_fragment_mask[list(frag_atom_idx_list[1])] = 2

            # extract frag mol directly from new_mol, in case the Kekulize error
            bond_ids = []
            for bond_idx, bond in enumerate(new_mol.GetBonds()):
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if (new_fragment_mask[start] > 0) == (new_fragment_mask[end] == 0):
                    bond_ids.append(bond_idx)
            assert len(bond_ids) == 2
            break_mol = Chem.FragmentOnBonds(new_mol, bond_ids, addDummies=False)
            frags = [f for f in Chem.GetMolFrags(break_mol, asMols=True)
                     if f.GetNumAtoms() != link_mol.GetNumAtoms()
                     or not np.allclose(f.GetConformer().GetPositions(), link_mol.GetConformer().GetPositions())]
            assert len(frags) == 2
            new_frag_mol = Chem.CombineMols(*frags)
            assert np.allclose(new_frag_mol.GetConformer().GetPositions(), frag_mol.GetConformer().GetPositions())

            data_list.append({
                'id': data['uuid'],
                'smiles': data['name'],
                'mol': new_mol,
                'frag_mol': new_frag_mol,
                'link_mol': link_mol,
                'fragment_mask': new_fragment_mask,
                'atom_indices_f1': list(frag_atom_idx_list[0]),
                'atom_indices_f2': list(frag_atom_idx_list[1]),
                'linker_mask': linker_mask,
                # 'anchors': data['anchors'].bool()
            })
        print('n fail: ', n_fail)
        datasets[subset] = data_list

    print('Saving data')
    with open(save_path, 'wb') as f:
        pickle.dump(datasets, f)
    print('Length processed data: ', [f'{x}: {len(datasets[x])}' for x in datasets])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='protac', choices=['protac', 'zinc_difflinker'])
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--mode', type=str, default='full', choices=['tiny', 'full'])
    args = parser.parse_args()

    if args.dataset == 'protac':
        preprocess_protac(args.raw_path, args.dest)
    elif args.dataset == 'zinc_difflinker':
        preprocess_zinc_from_difflinker(args.raw_path, args.dest, args.mode)
    else:
        raise NotImplementedError
