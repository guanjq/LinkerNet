import collections
import copy
import os.path
import pickle
import re
from collections import defaultdict
from typing import Sequence
import csv
import pandas as pd

import numpy as np
import torch
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize, QED
from rdkit.Chem.rdMolAlign import GetBestRMS, GetO3A
from scipy import spatial as sci_spatial
from tqdm.auto import tqdm

import utils.transforms as trans
from models.diff_protac_bond import rotation_matrix_cosine_loss
from utils import eval_bond
from utils import frag_utils
from utils.geometry import find_rigid_transform
from utils.reconstruct_linker import parse_sampling_result_with_bond
from utils.calc_SC_RDKit import calc_SC_RDKit_score
from utils import sascorer
import joblib

# BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
# BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
# BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution


DISTANCE_BINS = np.arange(1.1, 1.7, 0.005)[:-1]


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


def get_distance_jsd(ref_dists, gen_dists, bins=DISTANCE_BINS):
    ref = get_distribution(ref_dists, bins)
    gen = get_distribution(gen_dists, bins)
    jsd = sci_spatial.distance.jensenshannon(ref, gen)
    return jsd


def eval_success_rate(gen_mols_list):
    n_recon, n_complete = 0, 0
    for i, mol in enumerate(gen_mols_list):
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        n_recon += 1
        smiles = Chem.MolToSmiles(mol)
        if '.' not in smiles:
            n_complete += 1
        else:
            continue
    return n_recon / len(gen_mols_list), n_complete / len(gen_mols_list)


def fix_explicit_hs(data):
    # find num explicit hs of linker mol
    num_hs = defaultdict(int)
    bond_order = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3
    }
    linker_indices = (data.linker_mask).nonzero(as_tuple=True)[0].tolist()
    for bond in data.rdmol.GetBonds():
        sidx, eidx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if sidx in data.anchor_indices and eidx in linker_indices:
            num_hs[eidx] += bond_order[bond_type]
        elif (eidx in data.anchor_indices and sidx in linker_indices):
            num_hs[sidx] += bond_order[bond_type]

    for atom_idx in linker_indices:
        if atom_idx not in num_hs:
            atom = data.rdmol.GetAtomWithIdx(atom_idx)
            num_hs[atom_idx] = atom.GetNumExplicitHs()

    num_hs_link_mol = {linker_indices.index(k): v for k, v in num_hs.items()}
    linker_mol = data.link_mol
    for k, v in num_hs_link_mol.items():
        atom = linker_mol.GetAtomWithIdx(k)
        if atom.GetSymbol() in ['N']:
            atom.SetNumExplicitHs(v)
    # print(num_hs_link_mol)
    # for bond in linker_mol.GetBonds():
    #     print(bond.GetBondType())
    # for atom in linker_mol.GetAtoms():
    #     print(atom.GetNumImplicitHs(), atom.GetNumExplicitHs())
    return linker_mol


def standardise_linker(linker):
    # Standardise linkers
    linker_canon = Chem.MolFromSmiles(re.sub('[0-9]+\*', '*', linker))
    Chem.rdmolops.RemoveStereochemistry(linker_canon)
    standard_linker = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker_canon))
    return standard_linker


def process_generated_mols(gen_mols_list, gen_all_data_list, ori_data_list):
    # validity
    valid_all_results = []
    for i, (gen_mols, gen_data_list, ori_data) in enumerate(tqdm(zip(gen_mols_list, gen_all_data_list, ori_data_list),
                                                            total=len(gen_mols_list), desc='process gen mols')):
        valid_results = []
        ref_smi = ori_data.smiles
        frag_smi = Chem.MolToSmiles(ori_data.frag_mol)
        if len(gen_mols) == 0:
            print(f'No gen mols for data {i}')

        for gen_mol, gen_data in zip(gen_mols, gen_data_list):
            try:
                smi = Chem.MolToSmiles(gen_mol)
                gen_smi = Chem.CanonSmiles(smi)
                # gen_mols is chemically valid
                Chem.SanitizeMol(Chem.MolFromSmiles(gen_smi), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except:
                continue
                
            # gen_mols should contain both fragments
            if len(Chem.MolFromSmiles(gen_smi).GetSubstructMatch(Chem.MolFromSmiles(frag_smi))) != Chem.MolFromSmiles(
                    frag_smi).GetNumAtoms():
                # print('No frags')
                continue

            # Determine linkers of generated molecules
            try:
                # todo: may need to update to consider PROTAC special cases (more than one anchor atom)
                linker = frag_utils.get_linker(Chem.MolFromSmiles(gen_smi), Chem.MolFromSmiles(frag_smi), frag_smi)
                linker_smi = standardise_linker(linker)
            except:
                # print('No Linker')
                continue

            gen_frag = frag_utils.get_frags(gen_mol, Chem.MolFromSmiles(frag_smi))
            Chem.SanitizeMol(gen_frag)
            valid_results.append({
                'ref_smi': ref_smi,
                'frag_smi': frag_smi,
                'gen_smi': gen_smi,
                'linker_smi': linker_smi,
                'gen_mol': gen_mol,
                'gen_frag_mol': gen_frag,
                'gen_data': gen_data,
                'metrics': {}
            })
            # valid_gen_mols.append(gen_mol)
            # valid_indices.append(idx)
            # valid_frag_mols.append(frags)
        valid_all_results.append(valid_results)
        # results_all.append(valid_results)
        # valid_gen_mols_list.append(valid_gen_mols)
        # valid_frag_mols_list.append(valid_frag_mols)
        # valid_indices_list.append(valid_indices)
        
    # validity = [len(valid_r) / len(gen_mols) if len(gen_mols) > 0 else 0
    #             for valid_r, gen_mols in zip(results_all, gen_mols_list)]
    # print("Valid molecules: Mean %.2f%% Median %.2f%%" % (np.mean(validity) * 100, np.median(validity) * 100))
    return valid_all_results


def process_training_linkers(train_set):
    # Determine linkers of training set
    train_linker_smiles = []
    for idx, data in enumerate(tqdm(train_set, desc='Extracting linkers in the training set')):
        # linker = frag_utils.get_linker(data.rdmol, data.frag_mol, Chem.MolToSmiles(data.frag_mol))
        # print(idx, linker)
        linker_indices = (data.linker_mask).nonzero(as_tuple=True)[0].tolist()
        bond_ids = []
        for bond in data.rdmol.GetBonds():
            sidx, eidx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (sidx in data.anchor_indices and eidx in linker_indices) or (
                    eidx in data.anchor_indices and sidx in linker_indices):
                bond_ids.append(bond.GetIdx())
        assert len(bond_ids) == 2
        frags = Chem.FragmentOnBonds(data.rdmol, bond_ids)
        frags = Chem.GetMolFrags(frags, asMols=True)
        assert np.allclose(frags[-1].GetConformer().GetPositions()[:-2], data.link_mol.GetConformer().GetPositions())
        linker = Chem.MolToSmiles(frags[-1])
        standard_linker = standardise_linker(linker)
        train_linker_smiles.append(standard_linker)

    # Remove duplicates
    linkers_train_canon_unique = list(set(train_linker_smiles))
    print("Number of unique linkers: %d" % len(linkers_train_canon_unique))
    return linkers_train_canon_unique


def print_dict(d):
    for k, v in d.items():
        if v is not None:
            print(f'{k}:\t{float(v):.4f}')
        else:
            print(f'{k}:\tNone')
            
            
class LinkerEvaluator:
    def __init__(self, raw_gen_results, train_set=None,
                 train_linker_path='utils/train_linker_smiles.pkl', reconstruct=False
                 ):
        """
        raw_gen_results: list of dict
        """
        super().__init__()
        self.train_linker_path = train_linker_path
        if os.path.exists(self.train_linker_path):
            with open(self.train_linker_path, 'rb') as f:
                self.train_linker_smiles = pickle.load(f)
        else:
            print('No training linker smiles found! Need to extract them from the training set')
            assert train_set is not None
            self.train_linker_smiles = process_training_linkers(train_set)
            with open(self.train_linker_path, 'wb') as f:
                pickle.dump(self.train_linker_smiles, f)

        gen_mols_list = []
        gen_all_data_list = []
        recon_rates, complete_rates = [], []
        for r in raw_gen_results:
            if reconstruct:
                atom_featurizer = trans.FeaturizeAtom('zinc', known_anchor=False)
                gen_mols = parse_sampling_result_with_bond(
                    r['data_list'], r['final_x'], r['final_c'], r['final_bond'], atom_featurizer,
                    known_linker_bonds=False, check_validity=True)
            else:
                gen_mols = r['gen_mols']
            mols = []
            gen_data_list = r['data_list']
            valid_gen_data = []
            n_recon, n_complete = 0, 0
            for gen_data, mol in zip(gen_data_list, gen_mols):
                try:
                    Chem.SanitizeMol(mol)
                except:
                    continue
                n_recon += 1
                smiles = Chem.MolToSmiles(mol)
                if '.' not in smiles:
                    n_complete += 1
                else:
                    continue
                mols.append(mol)
                valid_gen_data.append(gen_data)

            recon_rates.append(n_recon / len(r['gen_mols']))
            complete_rates.append(n_complete / len(r['gen_mols']))
            gen_mols_list.append(mols)
            gen_all_data_list.append(valid_gen_data)
        print(f'recon rate:    mean: {np.mean(recon_rates):.4f} median: {np.median(recon_rates):.4f}')
        print(f'complete rate: mean: {np.mean(complete_rates):.4f} median: {np.median(complete_rates):.4f}')
        
        # gen_all_data_list = [r['data_list'] for r in raw_gen_results]
        self.ref_data_list = [r['ref_data'] for r in raw_gen_results]
        
        # ref_smiles, frag_smiles, gen_smiles, gen_linker_smiles
        # self.gen_all_smiles, self.gen_mols_list, self.frag_mols_list, self.valid_indices_list = \
        #     process_generated_mols(gen_mols_list, self.ref_data_list)
        self.gen_all_results = process_generated_mols(gen_mols_list, gen_all_data_list, self.ref_data_list)

        # self.gen_valid_data_list = [[gen_all_data_list[i][idx] for idx in valid_indices]
        #                             for i, valid_indices in enumerate(self.valid_indices_list)]
        self.all_rec_idx = None
        self.bond_length_profile, self.ref_bond_length_profile = None, None
        self.bond_angle_profile, self.ref_bond_angle_profile = None, None
        self.summary = {}
        self.validity(num_samples=len(raw_gen_results[0]['gen_mols']))

    def validity(self, num_samples):
        self.num_valid_list = [len(r) for r in self.gen_all_results]
        print(f'Final valid rate: mean: {np.mean(self.num_valid_list) / num_samples:.4f} '
              f'median: {np.median(self.num_valid_list) / num_samples:.4f}')
        self.summary['valid'] = np.mean(self.num_valid_list) / num_samples

    def evaluate(self, eval_3d=True):
        self.uniqueness()
        self.novelty()
        self.recovery()
        self.chem_2d()
        # # self.pass_2d_filters()
        if eval_3d:
            self.bond_geometry()
            self.conf_energy()
            self.constraints_validity()

    def save_metrics(self, save_path):
        summary_table = pd.DataFrame([self.summary])
        summary_table.to_csv(save_path, index=False)

        eval_save_path = os.path.join(os.path.dirname(save_path), 'eval_results.pt')
        new_gen_all_results = []
        for results in self.gen_all_results:
            new_r = []
            for d in results:
                d.pop('gen_data')
                new_r.append(d)
            new_gen_all_results.append(new_r)
        torch.save(new_gen_all_results, eval_save_path)

    def overall_prop(self, prop_list, num_list):
        assert len(prop_list) == len(num_list)
        m = [p * n for p, n in zip(prop_list, num_list)]
        return sum(m) / sum(num_list)

    def uniqueness(self):
        # Check number of unique molecules
        all_uniqueness = []
        num_valid = []
        for results in self.gen_all_results:
            if len(results) == 0:
                continue
            # Create dictionary of results
            results_dict = {}
            for d in results:
                res = [d['ref_smi'], d['frag_smi'], d['gen_smi'], d['linker_smi']]
                if res[0] + '.' + res[1] in results_dict:  # Unique identifier - starting fragments and original molecule
                    results_dict[res[0] + '.' + res[1]].append(tuple(res))
                else:
                    results_dict[res[0] + '.' + res[1]] = [tuple(res)]
            uniqueness = frag_utils.unique(results_dict.values())
            all_uniqueness.append(uniqueness)
            num_valid.append(len(results))
        overall_unique = self.overall_prop(all_uniqueness, num_valid) * 100
        print("Unique molecules: Mean %.2f%% Median %.2f%% Overall %.2f%%" % (
            np.mean(all_uniqueness) * 100, np.median(all_uniqueness) * 100, overall_unique))
        self.summary['unique'] = overall_unique
        return overall_unique

    def novelty(self):
        # Check novelty of generated molecules
        all_novelty = []
        num_valid = []
        for results in self.gen_all_results:
            if len(results) == 0:
                continue
            count_novel = 0
            for d in results:
                if d['linker_smi'] in self.train_linker_smiles:
                    d['metrics']['novel'] = False
                    continue
                else:
                    d['metrics']['novel'] = True
                    count_novel += 1
            novelty = count_novel / len(results)
            all_novelty.append(novelty)
            num_valid.append(len(results))
        overall_novelty = self.overall_prop(all_novelty, num_valid) * 100
        print("Novel linkers: Mean %.2f%% Median %.2f%% Overall %.2f%%" % (
            np.mean(all_novelty) * 100, np.median(all_novelty) * 100, overall_novelty))
        self.summary['novel'] = overall_novelty
        return overall_novelty

    def recovery(self):
        # Check proportion recovered
        # Create dictionary of results
        all_recovered, all_rec_idx = [], []
        for results in self.gen_all_results:
            if len(results) == 0:
                all_rec_idx.append([])
                continue
            results_dict_with_idx = {}
            for i, d in enumerate(results):
                res = [d['ref_smi'], d['frag_smi'], d['gen_smi'], d['linker_smi']]
                if res[0] + '.' + res[1] in results_dict_with_idx:  # Unique identifier - starting fragments and original molecule
                    results_dict_with_idx[res[0] + '.' + res[1]].append([res, i])
                else:
                    results_dict_with_idx[res[0] + '.' + res[1]] = [[res, i]]
            recovered, rec_idx = frag_utils.check_recovered_original_mol_with_idx(list(results_dict_with_idx.values()))
            for i, d in enumerate(results):
                if i in rec_idx:
                    d['metrics']['recover'] = True
                else:
                    d['metrics']['recover'] = False
            all_recovered.append(recovered[0])
            all_rec_idx.append(rec_idx)
        recovery = sum(all_recovered) / len(self.gen_all_results)
        print("Recovered: %.2f%%" % (recovery * 100))
        self.all_rec_idx = all_rec_idx
        self.summary['recovered'] = recovery
        return recovery, all_rec_idx

    def chem_2d(self):
        sa_values, qed_values = [], []
        for results in self.gen_all_results:
            for r in results:
                # mol = copy.deepcopy(r['gen_mol'])
                mol = Chem.MolFromSmiles(r['gen_smi'])
                sa = sascorer.calculateScore(mol)
                qed = QED.qed(mol)
                sa_values.append(sa)
                qed_values.append(qed)
                r['metrics']['qed'] = qed
                r['metrics']['sa'] = sa
        print(f'Mean SA: {np.mean(sa_values):.3f}')
        print(f'Mean QED: {np.mean(qed_values):.3f}')
        self.summary['sa'] = np.mean(sa_values)
        self.summary['qed'] = np.mean(qed_values)

    def pass_2d_filters(self, pains_smarts_path='utils/wehi_pains.csv'):
        # Check if molecules pass 2D filters
        flat_gen_all_smiles = []
        for results in self.gen_all_results:
            flat_gen_all_smiles += [[d['ref_smi'], d['frag_smi'], d['gen_smi'], d['linker_smi']] for d in results]

        filters_2d = frag_utils.calc_filters_2d_dataset(
            flat_gen_all_smiles, pains_smarts_loc=pains_smarts_path, n_cores=4)
        results_filt = []
        for res, filt in zip(flat_gen_all_smiles, filters_2d):
            if filt[0] and filt[1] and filt[2]:
                results_filt.append(res)

        print("Pass all 2D filters: \t\t%.2f%%" % (len(results_filt) / len(flat_gen_all_smiles) * 100))
        print("Pass synthetic accessibility (SA) filter: \t%.2f%%" % (
                    len([f for f in filters_2d if f[0]]) / len(filters_2d) * 100))
        print("Pass ring aromaticity filter: \t\t\t%.2f%%" % (
                    len([f for f in filters_2d if f[1]]) / len(filters_2d) * 100))
        print("Pass PAINS filters: \t\t\t\t%.2f%%" % (len([f for f in filters_2d if f[2]]) / len(filters_2d) * 100))

    def linker_rmsd(self):
        if self.all_rec_idx is None:
            _, all_rec_idx = self.recovery()
            self.all_rec_idx = all_rec_idx
        
        all_rmsd, all_linker_rmsd_dummy, all_linker_rmsd = [], [], []
        for i, rec_idx in enumerate(self.all_rec_idx):
            if len(rec_idx) == 0:
                continue
            ref_data = self.ref_data_list[i]
            gen_results = self.gen_all_results[i]
            ref_mol = ref_data.rdmol
            rmsd, linker_rmsd_dummy, linker_rmsd = [], [], []
            for idx in rec_idx:
                gen_mol = gen_results[idx]['gen_mol']
                gen_data = gen_results[idx]['gen_data']
                # check if they are 3d recovered
                G1 = frag_utils.topology_from_rdkit(gen_mol)
                G2 = frag_utils.topology_from_rdkit(ref_mol)
                GM = isomorphism.GraphMatcher(G1, G2)
                flag = GM.is_isomorphic()
                if flag:  # check if isomorphic
                    error = GetBestRMS(gen_mol, ref_mol)
                    gen_linker = extract_linker(gen_mol, gen_data.linker_mask)
                    ref_linker = extract_linker(ref_mol, ref_data.linker_mask)
                    linker_dummy_error = GetBestRMS(gen_linker, ref_linker)

                    gen_linker = remove_dummys_mol(gen_linker)
                    ref_linker = remove_dummys_mol(ref_linker)
                    linker_error = GetBestRMS(gen_linker, ref_linker)

                    rmsd.append(error)
                    linker_rmsd_dummy.append(linker_dummy_error)
                    linker_rmsd.append(linker_error)

                    # # num_linker = mol2.GetNumAtoms() - Chem.MolFromSmiles(frag_mols[rec_idx[i]]).GetNumAtoms() + 2
                    # num_linker =
                    # num_atoms = mol1.GetNumAtoms()
                    # error *= np.sqrt(num_atoms / num_linker)  # only count rmsd on linker
                    # rmsd.append(error)
                else:
                    print('Not 3D recovered: ', rec_idx)

            if len(rmsd) > 0:
                all_rmsd.append(np.mean(rmsd))
                all_linker_rmsd_dummy.append(np.mean(linker_rmsd_dummy))
                all_linker_rmsd.append(np.mean(linker_rmsd))
        
        num_rec = [len(r) for r in self.all_rec_idx]
        if sum(num_rec) == 0:
            print('No recovered linkers!')
            return

        print('Average num recovered Mean %.4f Median %.4f' % (np.mean(num_rec), np.median(num_rec)))
        print('Average RMSD is Mean %.4f Median %.4f Overall %.4f' % (np.mean(all_rmsd), np.median(all_rmsd), self.overall_prop(all_rmsd, num_rec)))
        print('Average linker(w/  dummy) RMSD is Mean %.4f Median %.4f Overall %.4f' % (
            np.mean(all_linker_rmsd_dummy), np.median(all_linker_rmsd_dummy), self.overall_prop(all_linker_rmsd_dummy, num_rec)))
        print('Average linker(w/o dummy) RMSD is Mean %.4f Median %.4f Overall %.4f' % (
            np.mean(all_linker_rmsd), np.median(all_linker_rmsd), self.overall_prop(all_linker_rmsd, num_rec)))

    def fragment_geometry(self):
        losses = []
        for i, rec_idx in enumerate(self.all_rec_idx):
            if len(rec_idx) == 0:
                continue
            ref_data = self.ref_data_list[i]
            gen_results = self.gen_all_results[i]
            for idx in rec_idx:
                gen_mol = gen_results[idx]['gen_mol']
                error = fape_error(ref_data, gen_mol)
                losses.append(error)

        rot_losses, tr_losses = [], []
        for ref_data, gen_results in zip(self.ref_data_list, self.gen_all_results):
            for idx, r in enumerate(gen_results):
                gen_data = r['gen_data']
                gen_mol = r['gen_mol']
                gen_pos = gen_mol.GetConformer().GetPositions()
                rot_loss, tr_loss = rot_tr_loss(ref_data, gen_data, gen_pos)
                rot_losses.append(rot_loss)
                tr_losses.append(tr_loss)
        print('Average FAPE loss is %f (std: %f)' % (np.mean(losses), np.std(losses)))
        print('Average Rot  loss is %f (std: %f)' % (np.mean(rot_losses), np.std(rot_losses)))
        print('Average Tr   loss is %f (std: %f)' % (np.mean(tr_losses), np.std(tr_losses)))

    def bond_geometry(self):
        # # Anchor bond distance
        def _bond_type_str(bond_type) -> str:
            atom1, atom2, bond_category = bond_type
            return f'{atom1}-{atom2}|{bond_category}'
        
        flat_ref_mols, flat_gen_mols = [], []
        flat_ref_masks, flat_gen_masks = [], []
        for ref_data, gen_results in zip(self.ref_data_list, self.gen_all_results):
            gen_mols = [r['gen_mol'] for r in gen_results]
            flat_gen_mols += gen_mols
            flat_ref_mols += [ref_data.rdmol] * len(gen_mols)
            flat_gen_masks += [r['gen_data'].fragment_mask for r in gen_results]
            flat_ref_masks += [ref_data.fragment_mask] * len(gen_mols)
        
        if self.bond_length_profile is None:
            self.bond_length_profile = eval_bond.get_bond_length_profile(flat_gen_mols, flat_gen_masks)

        if self.ref_bond_length_profile is None:
            self.ref_bond_length_profile = eval_bond.get_bond_length_profile(flat_ref_mols, flat_ref_masks)

        if self.bond_angle_profile is None:
            self.bond_angle_profile = eval_bond.get_bond_angles_profile(flat_gen_mols, flat_gen_masks)

        if self.ref_bond_angle_profile is None:
            self.ref_bond_angle_profile = eval_bond.get_bond_angles_profile(flat_ref_mols, flat_ref_masks)

        metrics = {}
        # for bond_type in eval_bond.BOND_DISTS:
        for bond_type in self.ref_bond_length_profile.keys():
            if bond_type not in self.bond_length_profile:
                metrics[f'JSD_{_bond_type_str(bond_type)}'] = 'nan'
                continue
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(
                self.ref_bond_length_profile[bond_type], self.bond_length_profile[bond_type])

        angle_metrics = {}
        # for angle_type in eval_bond.BOND_ANGLES:
        for angle_type in self.ref_bond_angle_profile.keys():
            if angle_type not in self.bond_angle_profile:
                angle_metrics[f'JSD_{angle_type}'] = 'nan'
                continue
            angle_metrics[f'JSD_{angle_type}'] = sci_spatial.distance.jensenshannon(
                self.ref_bond_angle_profile[angle_type], self.bond_angle_profile[angle_type])
        print('Bond distribuition JSD: ')
        print_dict(metrics)
        print('Angle distribuition JSD: ')
        print_dict(angle_metrics)
        return metrics, angle_metrics

    def conf_energy(self):
        if self.all_rec_idx is None:
            _, all_rec_idx = self.recovery()
            self.all_rec_idx = all_rec_idx

        ref_energy_list = []
        gen_energies_list = []
        ff_opt_energies_list = []
        ff_opt_rmsd_list = []
        ff_opt_linker_delta_energies_list = []
        ff_opt_linker_energies_list = []
        delta_energies_list = []
        for ref_data, gen_results in tqdm(zip(self.ref_data_list, self.gen_all_results),
                                          desc='Computing Energy', total=len(self.ref_data_list)):
            ref_mol = ref_data.rdmol
            ref_mol = Chem.RemoveAllHs(ref_mol)
            r = ff_optimize(ref_mol, addHs=False, mode='score_only')
            ref_energy_list.append(r['energy_before_ff'])

            gen_energies = []
            ff_opt_rmsd = []
            ff_opt_energies = []
            ff_opt_linker_energies = []
            ff_opt_linker_delta_energies = []
            delta_energies = []
            for results in gen_results:
                mol = results['gen_mol']
                data = results['gen_data']
                mol = Chem.RemoveAllHs(mol)
                r = ff_optimize(mol, addHs=False, mode='minimize')
                if r is not None:
                    gen_energies.append(r['energy_before_ff'])
                    ff_opt_rmsd.append(r['rmsd'])
                    ff_opt_energies.append(r['energy_after_ff'])
                    delta_energies.append(r['energy_before_ff'] - r['energy_after_ff'])
                    results['metrics']['energy_before_ff'] = r['energy_before_ff']
                    results['metrics']['energy_after_ff'] = r['energy_after_ff']
                    results['metrics']['rmsd'] = r['rmsd']
                else:
                    gen_energies.append(np.nan)
                    ff_opt_rmsd.append(np.nan)
                    ff_opt_energies.append(np.nan)
                    results['metrics']['energy_before_ff'] = None
                    results['metrics']['energy_after_ff'] = None
                    results['metrics']['rmsd'] = None

                fixed_points = (data.fragment_mask > 0).nonzero()[:, 0]
                r = ff_optimize(mol, addHs=False, mode='minimize', fixed_points=fixed_points)
                if r is not None:
                    ff_opt_linker_energies.append(r['energy_after_ff'])
                    ff_opt_linker_delta_energies.append(r['energy_before_ff'] - r['energy_after_ff'])
                    results['metrics']['fix_linker_energy_before_ff'] = r['energy_before_ff']
                    results['metrics']['fix_linker_energy_after_ff'] = r['energy_after_ff']
                else:
                    ff_opt_linker_energies.append(np.nan)
                    ff_opt_linker_delta_energies.append(np.nan)
                    results['metrics']['fix_linker_energy_before_ff'] = None
                    results['metrics']['fix_linker_energy_after_ff'] = None

            gen_energies_list.append(gen_energies)
            ff_opt_energies_list.append(ff_opt_energies)
            ff_opt_rmsd_list.append(ff_opt_rmsd)
            ff_opt_linker_energies_list.append(ff_opt_linker_energies)
            delta_energies_list.append(delta_energies)
            ff_opt_linker_delta_energies_list.append(ff_opt_linker_delta_energies)

        min_energies = [np.nanmin(x) for x in gen_energies_list if len(x) > 0]
        median_energies = [np.nanmedian(x) for x in gen_energies_list if len(x) > 0]
        ff_min_energies = [np.nanmin(x) for x in ff_opt_energies_list if len(x) > 0]
        ff_median_energies = [np.nanmedian(x) for x in ff_opt_energies_list if len(x) > 0]

        delta_min_energies = [np.nanmin(x) for x in delta_energies_list if len(x) > 0]
        delta_median_energies = [np.nanmedian(x) for x in delta_energies_list if len(x) > 0]
        delta_mean_energies = [np.nanmean(x) for x in delta_energies_list if len(x) > 0]

        ff_min_rmsd = [np.nanmin(x) for x in ff_opt_rmsd_list if len(x) > 0]
        ff_median_rmsd = [np.nanmedian(x) for x in ff_opt_rmsd_list if len(x) > 0]
        ff_mean_rmsd = [np.nanmean(x) for x in ff_opt_rmsd_list if len(x) > 0]

        ff_linker_min_energies = [np.nanmin(x) for x in ff_opt_linker_energies_list if len(x) > 0]
        ff_linker_median_energies = [np.nanmedian(x) for x in ff_opt_linker_energies_list if len(x) > 0]
        ff_linker_mean_energies = [np.nanmean(x) for x in ff_opt_linker_energies_list if len(x) > 0]

        linker_delta_min_energies = [np.nanmin(x) for x in ff_opt_linker_delta_energies_list if len(x) > 0]
        linker_delta_median_energies = [np.nanmedian(x) for x in ff_opt_linker_delta_energies_list if len(x) > 0]
        linker_delta_mean_energies = [np.nanmean(x) for x in ff_opt_linker_delta_energies_list if len(x) > 0]

        re_min_energies = [np.nanmin([x[idx] for idx in self.all_rec_idx[i]])
                           for i, x in enumerate(gen_energies_list) if len(self.all_rec_idx[i]) > 0]
        re_median_energies = [np.nanmedian([x[idx] for idx in self.all_rec_idx[i]])
                              for i, x in enumerate(gen_energies_list) if len(self.all_rec_idx[i]) > 0]

        print('Ref    energy Mean %.4f Median %.4f' % (np.mean(ref_energy_list), np.median(ref_energy_list)))
        print('Min    energy Mean %.4f Median %.4f' % (np.mean(min_energies), np.median(min_energies)))
        print('Median energy Mean %.4f Median %.4f' % (np.mean(median_energies), np.median(median_energies)))
        # print('FF Opt Min    energy Mean %.4f Median %.4f' % (np.mean(ff_min_energies), np.median(ff_min_energies)))
        # print('FF Opt Median energy Mean %.4f Median %.4f' % (np.mean(ff_median_energies), np.median(ff_median_energies)))
        print('Delta Min energy     Mean %.4f Median %.4f' % (np.mean(delta_min_energies), np.median(delta_min_energies)))
        print('Delta Median energy  Mean %.4f Median %.4f' % (np.mean(delta_median_energies), np.median(delta_median_energies)))
        print('Delta Mean energy    Mean %.4f Median %.4f' % (np.mean(delta_mean_energies), np.median(delta_mean_energies)))
        print('Min    RMSD Mean %.4f Median %.4f' % (np.mean(ff_min_rmsd), np.median(ff_min_rmsd)))
        print('Median RMSD Mean %.4f Median %.4f' % (np.mean(ff_median_rmsd), np.median(ff_median_rmsd)))
        print('Mean   RMSD Mean %.4f Median %.4f' % (np.mean(ff_mean_rmsd), np.median(ff_mean_rmsd)))

        print('Linker Min    energy Mean %.4f Median %.4f' % (np.mean(ff_linker_min_energies), np.median(ff_linker_min_energies)))
        print('Linker Median energy Mean %.4f Median %.4f' % (np.mean(ff_linker_median_energies), np.median(ff_linker_median_energies)))
        print('Linker Mean   energy Mean %.4f Median %.4f' % (np.mean(ff_linker_mean_energies), np.median(ff_linker_mean_energies)))
        print('Linker Delta Min energy     Mean %.4f Median %.4f' % (np.mean(linker_delta_min_energies), np.median(linker_delta_min_energies)))
        print('Linker Delta Median energy  Mean %.4f Median %.4f' % (np.mean(linker_delta_median_energies), np.median(linker_delta_median_energies)))
        print('Linker Delta Mean energy    Mean %.4f Median %.4f' % (np.mean(linker_delta_mean_energies), np.median(linker_delta_mean_energies)))
        # print('Recovered Min    energy Mean %.4f Median %.4f' % (np.mean(re_min_energies), np.median(re_min_energies)))
        # print('Recovered Median energy Mean %.4f Median %.4f' % (np.mean(re_median_energies), np.median(re_median_energies)))
        self.summary.update({
            'min_energy': np.mean(min_energies),
            'median_energy': np.mean(median_energies),
            # 'ff_min_energy': np.mean(ff_min_energies),
            # 'ff_median_energy': np.mean(ff_median_energies),
            'delta_min_energy': np.mean(delta_min_energies),
            'delta_median_energy': np.mean(delta_median_energies),
            'delta_mean_energy': np.mean(delta_mean_energies),

            'ff_min_rmsd': np.mean(ff_min_rmsd),
            'ff_median_rmsd': np.mean(ff_median_rmsd),
            'ff_mean_rmsd': np.mean(ff_mean_rmsd),

            'ff_linker_min_energy': np.mean(ff_linker_min_energies),
            'ff_linker_median_energy': np.mean(ff_linker_median_energies),
            'recover_min_energy': np.mean(re_min_energies),
            'recover_median_energy': np.mean(re_median_energies)
        })
        # return gen_energies_list, ff_opt_energies_list

    def constraints_validity(self, frag_dist_std=0.2):
        # hit cand anchors
        anchor_success_rate = []
        frag_dist_success_rate = []
        overall_success_rate = []
        num_valid = []
        for gen_results in self.gen_all_results:
            if len(gen_results) == 0:
                continue

            anchor_success = 0
            frag_dist_success = 0
            overall_success = 0
            for r in gen_results:
                gen_data, gen_mol = r['gen_data'], r['gen_mol']
                if not hasattr(gen_data, 'cand_anchors_mask'):
                    cand_anchors_mask = torch.zeros_like(gen_data.fragment_mask).bool()
                    cand_anchors_mask[gen_data.anchor_indices] = True
                else:
                    cand_anchors_mask = gen_data.cand_anchors_mask
                cand_anchor_indices = set(cand_anchors_mask.nonzero(as_tuple=True)[0].tolist())

                anchor_indices = find_anchor_indices(gen_mol, gen_data.linker_mask)
                if set(anchor_indices).issubset(cand_anchor_indices):
                    anchor_success += 1
                    r['metrics']['anchor_succ'] = True
                else:
                    r['metrics']['anchor_succ'] = False

                # compute frag distance
                gen_pos = gen_mol.GetConformer().GetPositions()
                f1_center = gen_pos[gen_data.fragment_mask == 1].mean(0)
                f2_center = gen_pos[gen_data.fragment_mask == 2].mean(0)
                frag_d = np.linalg.norm(f1_center - f2_center, ord=2, axis=-1)
                true_frag_d = gen_data.frags_d
                if frag_d < true_frag_d * (1 + frag_dist_std) and frag_d > true_frag_d * (1 - frag_dist_std):
                    frag_dist_success += 1
                    r['metrics']['frag_dist_succ'] = True
                else:
                    r['metrics']['frag_dist_succ'] = False

                if r['metrics']['anchor_succ'] and r['metrics']['frag_dist_succ']:
                    overall_success += 1
                    r['metrics']['overall_succ'] = True
                else:
                    r['metrics']['overall_succ'] = False

            anchor_success_rate.append(anchor_success / len(gen_results))
            frag_dist_success_rate.append(frag_dist_success / len(gen_results))
            overall_success_rate.append(overall_success / len(gen_results))
            num_valid.append(len(gen_results))

        overall_anchor_success = self.overall_prop(anchor_success_rate, num_valid)
        overall_frag_dist_success = self.overall_prop(frag_dist_success_rate, num_valid)
        overall_success = self.overall_prop(overall_success_rate, num_valid)
        print('Hit Cand Anchors Mean %.4f Median %.4f Overall %.4f' % (
            np.mean(anchor_success_rate), np.median(anchor_success_rate), overall_anchor_success))
        print('Meet Fragment Distance Constraint Mean %.4f Median %.4f Overall %.4f' % (
            np.mean(frag_dist_success_rate), np.median(frag_dist_success_rate), overall_frag_dist_success))
        print('Meet Overall Constraint Mean %.4f Median %.4f Overall %.4f' % (
            np.mean(overall_success_rate), np.median(overall_success_rate), overall_success))
        self.summary['anchor_success'] = overall_anchor_success
        self.summary['frag_dist_success'] = overall_frag_dist_success
        self.summary['overall_success'] = overall_success

    @staticmethod
    def _cal_sc_rdkit(mols):
        gen_frag_mol, ref_frag_mol, gen_mol, ref_mol = mols
        frag_score = calc_sc_rdkit_mol(gen_frag_mol, ref_frag_mol)
        full_score = 0.
        # full_score = calc_sc_rdkit_mol(gen_mol, ref_mol)
        return frag_score, full_score

    def SC_RDKit(self):
        sc_rdkit_full, sc_rdkit_frag = [], []
        frag_rmsd_list = []

        tasks = []
        for ref_data, gen_results in zip(self.ref_data_list, self.gen_all_results):
            for r in gen_results:
                tasks.append([r['gen_frag_mol'], ref_data.frag_mol, r['gen_mol'], ref_data.rdmol])

        score_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(self._cal_sc_rdkit)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Computing SC RDKit')
        )

        for score in score_list:
            sc_rdkit_frag.append(score[0])
            sc_rdkit_full.append(score[1])

        sc_rdkit_full, sc_rdkit_frag = np.array(sc_rdkit_full), np.array(sc_rdkit_frag)
        frag_rmsd_list = np.array(frag_rmsd_list)
        for name, score_list in zip(['Full', 'Frag'], [sc_rdkit_full, sc_rdkit_frag]):
            sc_rdkit_7 = (score_list > 0.7).sum() / len(score_list) * 100
            sc_rdkit_8 = (score_list > 0.8).sum() / len(score_list) * 100
            sc_rdkit_9 = (score_list > 0.9).sum() / len(score_list) * 100
            sc_rdkit_mean = np.mean(score_list)
            print(f'SC_RDKit {name} > 0.7: {sc_rdkit_7:3f}%')
            print(f'SC_RDKit {name} > 0.8: {sc_rdkit_8:3f}%')
            print(f'SC_RDKit {name} > 0.9: {sc_rdkit_9:3f}%')
            print(f'Mean SC_RDKit {name}: {sc_rdkit_mean}')


def calc_sc_rdkit_mol(gen_mol, ref_mol):
    # try:
    gen_mol, ref_mol = copy.deepcopy(gen_mol), copy.deepcopy(ref_mol)
    Chem.SanitizeMol(gen_mol)
    Chem.SanitizeMol(ref_mol)
    _ = GetO3A(gen_mol, ref_mol).Align()
    sc_score = calc_SC_RDKit_score(gen_mol, ref_mol)
    return sc_score
    # except:
    #     return -0.5


def find_anchor_indices(mol, linker_mask):
    linker_indices = linker_mask.nonzero(as_tuple=True)[0].tolist()
    anchor_indices = []
    # add bonds
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if start in linker_indices and end not in linker_indices:
            anchor_indices.append(end)
        elif start not in linker_indices and end in linker_indices:
            anchor_indices.append(start)
    return anchor_indices


def extract_linker(rdmol, linker_mask):
    linker_indices = linker_mask.nonzero(as_tuple=True)[0].tolist()
    bond_ids = []
    for bond in rdmol.GetBonds():
        sidx, eidx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (sidx in linker_indices and eidx not in linker_indices) or (
                eidx in linker_indices and sidx not in linker_indices):
            bond_ids.append(bond.GetIdx())
    assert len(bond_ids) == 2
    frags = Chem.FragmentOnBonds(rdmol, bond_ids)
    frags = Chem.GetMolFrags(frags, asMols=True)
    # assert np.allclose(frags[-1].GetConformer().GetPositions()[:-2], data.link_mol.GetConformer().GetPositions())
    return frags[-1]


def remove_dummys_mol(rdmol):
    return Chem.RemoveHs(AllChem.ReplaceSubstructs(rdmol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), True)[0])


def fape_error(data, gen_mol):
    total_loss = 0.
    for i in [1, 2]:
        align_f_mask = (data.fragment_mask == i)
        # align ref / gen mol by matching i-th fragment to its local pos
        ref_pos = data.pos
        ref_pos_true = data.frags_local_pos_filled[i - 1][data.frags_local_pos_mask[i - 1]].numpy().astype(np.float64)
        R_ref, t_ref = find_rigid_transform(
            ref_pos_true, ref_pos[align_f_mask].numpy().astype(np.float64))

        gen_pos = gen_mol.GetConformer().GetPositions()
        R_gen, t_gen = find_rigid_transform(ref_pos_true, gen_pos[align_f_mask])

        # FAPE loss
        ref_new_pos = ref_pos @ R_ref.T + t_ref
        gen_new_pos = gen_pos @ R_gen.T + t_gen

        other_f_mask = (data.fragment_mask > 0) & (data.fragment_mask != i)
        ref_other = ref_new_pos[other_f_mask]
        gen_other = gen_new_pos[other_f_mask]
        loss = np.linalg.norm(ref_other - gen_other, ord=2, axis=-1).mean()
        total_loss += loss
    return total_loss / 2


def rot_tr_loss(data, gen_data, gen_pos):
    ref_pos = data.pos.numpy().astype(np.float64)

    # align first fragment
    align_f1_mask = data.fragment_mask == 1
    gen_f1_mask = gen_data.fragment_mask == 1
    rel_R, rel_t = find_rigid_transform(ref_pos[align_f1_mask], gen_pos[gen_f1_mask])
    gen_new_pos = gen_pos @ rel_R.T + rel_t

    align_f2_mask = data.fragment_mask == 2
    gen_f2_mask = gen_data.fragment_mask == 2
    rel_R2, rel_t2 = find_rigid_transform(ref_pos[align_f2_mask], gen_new_pos[gen_f2_mask])
    rot_loss = rotation_matrix_cosine_loss(torch.from_numpy(rel_R2), torch.eye(3))
    tr_loss = np.linalg.norm(rel_t2)
    return rot_loss, tr_loss


def ff_optimize(ori_mol, addHs=False, mode='score_only', fixed_points=[],
                # enable_torsion=False,
                ):
    mol = copy.deepcopy(ori_mol)
    Chem.GetSymmSSSR(mol)
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if mp is None:
        return None

    # turn off angle-related terms
    #     mp.SetMMFFOopTerm(enable_torsion)
    #     mp.SetMMFFAngleTerm(True)
    #     mp.SetMMFFTorsionTerm(enable_torsion)

    #     # optimize unrelated to angles
    #     mp.SetMMFFStretchBendTerm(True)
    #     mp.SetMMFFBondTerm(True)
    #     mp.SetMMFFVdWTerm(True)
    #     mp.SetMMFFEleTerm(True)

    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        for i in fixed_points:
            ff.AddFixedPoint(int(i))

        energy_before_ff = ff.CalcEnergy()
        result = {'energy_before_ff': energy_before_ff}
        if mode == 'minimize':
            # ff.Minimize(maxIts=1000)
            ff.Minimize()
            energy_after_ff = ff.CalcEnergy()
            # print(f'Energy: {energy_before_ff} --> {energy_after_ff}')
            energy_change = energy_before_ff - energy_after_ff
            Chem.SanitizeMol(ori_mol)
            Chem.SanitizeMol(mol)
            rmsd = GetBestRMS(ori_mol, mol)
            result.update({'energy_after_ff': energy_after_ff, 'rmsd': rmsd, 'ff_mol': mol})
    except:
        return None
    return result
