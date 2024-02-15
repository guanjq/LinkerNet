import argparse

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets.linker_dataset import get_linker_dataset
from utils.evaluation import LinkerEvaluator
from utils.visualize import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--num_eval', type=int, default=400)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config_path', type=str, default='configs/sampling/zinc.yml')

    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    mols = []
    num_valid_mols = 0
    for fidx in range(args.num_eval):
        sdf_path = os.path.join(args.sample_path, f'data_{fidx}.sdf')
        if os.path.exists(sdf_path):
            m = Chem.SDMolSupplier(sdf_path, sanitize=False)
            mols.append(m)
            num_valid_mols += 1
        else:
            mols.append([])
    print('Num datapoints: ', args.num_eval)
    print('Num valid mols: ', num_valid_mols)

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
        gen_mols = mols[i]
        if len(gen_mols) == 0:
            continue

        num_frag_atoms = sum(ref_data.fragment_mask > 0)
        gen_data_list = []
        for mol in gen_mols:
            if mol is None:
                gen_data_list.append(None)
                continue
            gen_data = ref_data.clone()
            gen_data['pos'] = torch.from_numpy(mol.GetConformer().GetPositions().astype(np.float32))
            num_linker_atoms = len(gen_data['pos']) - num_frag_atoms
            if num_linker_atoms < 0:
                gen_data_list.append(None)
                continue
            gen_data['fragment_mask'] = torch.cat(
                [ref_data.fragment_mask[:num_frag_atoms], torch.zeros(num_linker_atoms).long()])
            gen_data['linker_mask'] = (gen_data['fragment_mask'] == 0)
            gen_data_list.append(gen_data)

        results = {
            'gen_mols': gen_mols,
            'ref_data': ref_data,
            'data_list': gen_data_list
        }
        all_results.append(results)

    evaluator = LinkerEvaluator(all_results, reconstruct=False)
    evaluator.evaluate()
    if args.save_path is not None:
        evaluator.save_metrics(args.save_path)
