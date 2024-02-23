import argparse
import os

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets.linker_dataset import get_linker_dataset
from models.diff_protac_bond import DiffPROTACModel
from utils.reconstruct_linker import parse_sampling_result, parse_sampling_result_with_bond
from torch_geometric.data import Batch
from utils.evaluation import eval_success_rate
from utils.prior_num_atoms import setup_configs, sample_atom_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--subset', choices=['val', 'test'], default='val')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--outdir', type=str, default='./outputs_test')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--save_traj', type=eval, default=False)
    parser.add_argument('--cand_anchors_mode', type=str, default='k-hop', choices=['k-hop', 'exact'])
    parser.add_argument('--cand_anchors_k', type=int, default=2)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.sample.seed)
    logger, writer, log_dir, ckpt_dir, vis_dir = misc.setup_logdir(
        args.config, args.outdir, mode='eval', tag=args.tag, create_dir=True)
    logger.info(args)

    # Load checkpoint
    ckpt_path = config.model.checkpoint if args.ckpt_path is None else args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location=args.device)
    logger.info(f'Successfully load the model! {ckpt_path}')
    cfg_model = ckpt['configs'].model

    # Transforms
    test_transform = Compose([
        trans.FeaturizeAtom(config.dataset.name, add_atom_feat=False)
    ])
    atom_featurizer = trans.FeaturizeAtom(
        config.dataset.name, known_anchor=cfg_model.known_anchor, add_atom_type=False, add_atom_feat=True)
    graph_builder = trans.BuildCompleteGraph(known_linker_bond=cfg_model.known_linker_bond,
                                             known_cand_anchors=config.sample.get('cand_bond_mask', False))
    init_transform = Compose([
        atom_featurizer,
        trans.SelectCandAnchors(mode=args.cand_anchors_mode, k=args.cand_anchors_k),
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=config.dataset.get('max_num_atoms', 30)),
        trans.RelativeGeometry(mode=cfg_model.get('rel_geometry', 'distance_and_two_rot'))
    ])
    logger.info(f'Init transform: {init_transform}')

    # Datasets and loaders
    logger.info('Loading dataset...')
    if config.dataset.split_mode == 'full':
        dataset = get_linker_dataset(
            cfg=config.dataset,
            transform_map=test_transform
        )
        test_set = dataset
    else:
        dataset, subsets = get_linker_dataset(
            cfg=config.dataset,
            transform_map={'train': None, 'val': test_transform, 'test': test_transform}
        )
        test_set = subsets[args.subset]
    logger.info(f'Test: {len(test_set)}')

    FOLLOW_BATCH = ['edge_type']
    COLLATE_EXCLUDE_KEYS = ['nbh_list']

    # Model
    logger.info('Building model...')
    model = DiffPROTACModel(
        cfg_model,
        num_classes=atom_featurizer.num_classes,
        num_bond_classes=graph_builder.num_bond_classes,
        atom_feature_dim=atom_featurizer.feature_dim,
        edge_feature_dim=graph_builder.bond_feature_dim
    ).to(args.device)
    logger.info('Num of parameters is %.2f M' % (np.sum([p.numel() for p in model.parameters()]) / 1e6))
    model.load_state_dict(ckpt['model'])
    logger.info(f'Load model weights done!')
    
    # Sampling
    logger.info(f'Begin sampling [{args.start_id}, {args.end_id})...')
    assert config.sample.num_atoms in ['ref', 'prior']
    if config.sample.num_atoms == 'prior':
        num_atoms_config = setup_configs(mode='frag_center_distance')
    else:
        num_atoms_config = None

    if args.end_id == -1:
        args.end_id = len(test_set)
    for idx in tqdm(range(args.start_id, args.end_id)):
        raw_data = test_set[idx]
        data_list = [raw_data.clone() for _ in range(args.num_samples)]
        # modify data list
        if num_atoms_config is None:
            new_data_list = [init_transform(data) for data in data_list]
        else:
            new_data_list = []
            for data in data_list:
                # sample num atoms
                dist = torch.floor(data.frags_d)  # (B, )
                num_linker_atoms = sample_atom_num(dist, num_atoms_config).astype(int)
                num_f1_atoms = len(data.atom_indices_f1)
                num_f2_atoms = len(data.atom_indices_f2)
                num_f_atoms = num_f1_atoms + num_f2_atoms
                frag_pos = data.pos[data.fragment_mask > 0]
                frag_atom_type = data.atom_type[data.fragment_mask > 0]
                frag_bond_idx = (data.bond_index[0] < num_f_atoms) & (data.bond_index[1] < num_f_atoms)

                data.fragment_mask = torch.LongTensor([1] * num_f1_atoms + [2] * num_f2_atoms + [0] * num_linker_atoms)
                data.linker_mask = (data.fragment_mask == 0)
                data.anchor_mask = torch.cat([data.anchor_mask[:num_f_atoms], torch.zeros([num_linker_atoms]).long()])
                data.bond_index = data.bond_index[:, frag_bond_idx]
                data.bond_type = data.bond_type[frag_bond_idx]
                data.pos = torch.cat([frag_pos, torch.zeros([num_linker_atoms, 3])], dim=0)
                data.atom_type = torch.cat([frag_atom_type, torch.zeros([num_linker_atoms]).long()])
                new_data = init_transform(data)
                new_data_list.append(new_data)

        batch = Batch.from_data_list(
            new_data_list, follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS).to(args.device)
        traj_batch, final_x, final_c, final_bond = model.sample(
            batch,
            p_init_mode=cfg_model.frag_pos_prior,
            guidance_opt=config.sample.guidance_opt
        )
        if model.train_bond:
            gen_mols = parse_sampling_result_with_bond(
                new_data_list, final_x, final_c, final_bond, atom_featurizer,
                known_linker_bonds=cfg_model.known_linker_bond, check_validity=True)
        else:
            gen_mols = parse_sampling_result(new_data_list, final_x, final_c, atom_featurizer)
        save_path = os.path.join(log_dir, f'sampling_{idx:06d}.pt')
        save_dict = {
            'ref_data': init_transform(raw_data),
            'data_list': new_data_list,  # don't save it to reduce the size of outputs
            'final_x': final_x, 'final_c': final_c, 'final_bond': final_bond,
            'gen_mols': gen_mols
        }
        if args.save_traj:
            save_dict['traj'] = traj_batch
            
        torch.save(save_dict, save_path)
    logger.info('Sample done!')

    # Quick Eval
    recon_rate, complete_rate = [], []
    anchor_dists = []
    for idx in range(args.start_id, args.end_id):
        load_path = os.path.join(log_dir, f'sampling_{idx:06d}.pt')
        results = torch.load(load_path)
        rr, cr = eval_success_rate(results['gen_mols'])
        logger.info(f'idx: {idx} recon rate: {rr} complete rate: {cr}')
        recon_rate.append(rr)
        complete_rate.append(cr)
    logger.info(f'recon rate:    mean: {np.mean(recon_rate):.4f} median: {np.median(recon_rate):.4f}')
    logger.info(f'complete rate: mean: {np.mean(complete_rate):.4f} median: {np.median(complete_rate):.4f}')
