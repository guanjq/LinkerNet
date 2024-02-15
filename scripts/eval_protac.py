import argparse
from rdkit import RDLogger
import torch
from utils.evaluation import LinkerEvaluator
from glob import glob
import os
from tqdm.auto import tqdm
from torch_geometric.transforms import Compose
import utils.misc as misc
import utils.transforms as trans
from datasets.linker_dataset import get_linker_dataset
from utils.evaluation import LinkerEvaluator
from utils.visualize import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--num_eval', type=int, default=None)
    parser.add_argument('--recon', type=eval, default=False)
    parser.add_argument('--save', type=eval, default=True)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    if args.sample_path.endswith('.pt'):
        print(f'There is only one sampling file to evaluate')
        results = [torch.load(args.sample_path)]
    else:
        results_file_list = sorted(glob(os.path.join(args.sample_path, 'sampling_*.pt')))
        if args.num_eval:
            results_file_list = results_file_list[:args.num_eval]
        print(f'There are {len(results_file_list)} files to evaluate')
        results = []
        for f in tqdm(results_file_list, desc='Load sampling files'):
            results.append(torch.load(f))

    if 'data_list' not in results[0].keys():
        print('Can not find data_list in result keys -- add data_list based on the test set')
        config = misc.load_config('configs/sampling/zinc.yml')
        # Transforms
        atom_featurizer = trans.FeaturizeAtom(
            config.dataset.name, known_anchor=False, add_atom_type=True, add_atom_feat=True)
        graph_builder = trans.BuildCompleteGraph(known_linker_bond=False)
        test_transform = Compose([
            atom_featurizer,
            trans.SelectCandAnchors(mode='k-hop', k=2),
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
        for i in range(len(results)):
            ref_data = test_set[i]
            gen_mols = results[i]['gen_mols']
            num_frag_atoms = sum(ref_data.fragment_mask > 0)
            gen_data_list = [ref_data.clone() for _ in range(100)]

            new_results = {
                'gen_mols': gen_mols,
                'ref_data': ref_data,
                'data_list': gen_data_list,
                'final_x': results[i]['final_x'],
                'final_c': results[i]['final_c'],
                'final_bond': results[i]['final_bond']
            }
            all_results.append(new_results)
        results = all_results

    evaluator = LinkerEvaluator(results, reconstruct=args.recon)
    evaluator.evaluate()
    save_path = os.path.join(args.sample_path, 'summary.csv')
    if args.save:
        evaluator.save_metrics(save_path)
