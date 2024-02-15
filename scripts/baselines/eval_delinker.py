import argparse

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets.linker_dataset import get_linker_dataset
from utils.evaluation import LinkerEvaluator, standardise_linker, remove_dummys_mol
from utils.visualize import *
from utils import frag_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--num_eval', type=int, default=400)
    parser.add_argument('--num_samples', type=int, default=250)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config_path', type=str, default='configs/sampling/zinc.yml')

    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    mols = []
    with open(args.sample_path, 'r') as f:
        for line in tqdm(f.readlines()):
            parts = line.strip().split(' ')
            data = {
                'fragments': parts[0],
                'true_molecule': parts[1],
                'pred_molecule': parts[2],
                'pred_linker': parts[3] if len(parts) > 3 else '',
            }
            mols.append(Chem.MolFromSmiles(data['pred_molecule']))
    print('Num data: ', len(mols))

    config = misc.load_config(args.config_path)
    # Transforms
    atom_featurizer = trans.FeaturizeAtom(
        config.dataset.name, known_anchor=False, add_atom_type=True, add_atom_feat=True)
    graph_builder = trans.BuildCompleteGraph(known_linker_bond=False)
    test_transform = Compose([
        atom_featurizer,
        trans.SelectCandAnchors(mode='k-hop', k=1),
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=config.dataset.get('max_num_atoms', 30)),
        # trans.RelativeGeometry(mode=cfg_model.get('rel_geometry', 'distance_and_two_rot'))
    ])
    dataset, subsets = get_linker_dataset(
        cfg=config.dataset,
        transform_map={'train': None, 'val': None, 'test': test_transform}
    )
    test_set = subsets['test']

    all_results = []
    for i in range(args.num_eval):
        ref_data = test_set[i]
        gen_mols = [mols[idx] for idx in range(args.num_samples * i, args.num_samples * (i + 1))]
            
        num_frag_atoms = sum(ref_data.fragment_mask > 0)
        gen_data_list = []
        for mol in gen_mols:
            if mol is None:
                gen_data_list.append(None)
                continue
            gen_data = ref_data.clone()
            gen_num_atoms = mol.GetNumAtoms()
            # gen_data['pos'] = torch.from_numpy(mol.GetConformer().GetPositions().astype(np.float32))
            num_linker_atoms = gen_num_atoms - num_frag_atoms
            if num_linker_atoms < 0:
                gen_data_list.append(None)
                continue
            # gen_data['fragment_mask'] = torch.cat(
            #     [ref_data.fragment_mask[:num_frag_atoms], torch.zeros(num_linker_atoms).long()])
            # gen_data['linker_mask'] = (gen_data['fragment_mask'] == 0)
            gen_data_list.append(gen_data)

        results = {
            'gen_mols': gen_mols,
            'ref_data': ref_data,
            'data_list': gen_data_list
        }
        all_results.append(results)

    # fix the mismatch between gen mols and ref mol
    data_list = []
    with open(args.sample_path, 'r') as f:
        for line in tqdm(f.readlines()):
            parts = line.strip().split(' ')
            data = {
                'fragments': parts[0],
                'true_molecule': parts[1],
                'pred_molecule': parts[2],
                'pred_linker': parts[3] if len(parts) > 3 else '',
            }
            data_list.append(data)

    valid_all_results = []
    for i in range(args.num_eval):
        valid_results = []
        gen_data_list = [data_list[idx] for idx in range(args.num_samples * i, args.num_samples * (i + 1))]
        for gen_data in gen_data_list:
            gen_smi = gen_data['pred_molecule']
            ref_smi = gen_data['true_molecule']
            raw_frag_smi = gen_data['fragments']
            frag_smi = Chem.MolToSmiles(remove_dummys_mol(Chem.MolFromSmiles(raw_frag_smi)))
            try:
                # gen_mols is chemically valid
                Chem.SanitizeMol(Chem.MolFromSmiles(gen_smi), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except:
                continue

            # gen_mols should contain both fragments
            if len(Chem.MolFromSmiles(gen_smi).GetSubstructMatch(Chem.MolFromSmiles(frag_smi))) != Chem.MolFromSmiles(
                    frag_smi).GetNumAtoms():
                continue

            # Determine linkers of generated molecules
            try:
                linker = frag_utils.get_linker(Chem.MolFromSmiles(gen_smi), Chem.MolFromSmiles(frag_smi), frag_smi)
                linker_smi = standardise_linker(linker)
            except:
                continue

            valid_results.append({
                'ref_smi': ref_smi,
                'frag_smi': frag_smi,
                'gen_smi': gen_smi,
                'linker_smi': linker_smi,
                'metrics': {}
            })
        valid_all_results.append(valid_results)

    evaluator = LinkerEvaluator(all_results, reconstruct=False)
    evaluator.gen_all_results = valid_all_results
    evaluator.validity(args.num_samples)

    evaluator.evaluate(eval_3d=False)
    if args.save_path is not None:
        evaluator.save_metrics(args.save_path)


