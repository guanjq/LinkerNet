import argparse
import os
import time

import numpy as np
import torch.utils.tensorboard
from rdkit import RDLogger
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets.linker_dataset import get_linker_dataset
from models.diff_protac_bond import DiffPROTACModel
from utils.evaluation import eval_success_rate
from utils.reconstruct_linker import parse_sampling_result_with_bond, parse_sampling_result
from utils.train import *

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/training/zinc.yml')
    parser.add_argument('--overfit_one', type=eval, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--sampling_iter', type=int, default=10000)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    # Load configs
    config = misc.load_config(args.config)
    misc.seed_all(config.train.seed)
    logger, writer, log_dir, ckpt_dir, vis_dir = misc.setup_logdir(args.config, args.logdir)
    logger.info(args)

    # Transforms
    cfg_model = config.model
    atom_featurizer = trans.FeaturizeAtom(config.dataset.name, known_anchor=cfg_model.known_anchor)
    graph_builder = trans.BuildCompleteGraph(known_linker_bond=cfg_model.known_linker_bond)
    max_num_atoms = config.dataset.get('max_num_atoms', 30)
    train_transform = Compose([
        atom_featurizer,
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=max_num_atoms),
        trans.RelativeGeometry(mode=cfg_model.rel_geometry)
    ])
    test_transform = Compose([
        atom_featurizer,
        graph_builder,
        trans.StackFragLocalPos(max_num_atoms=max_num_atoms),
        trans.RelativeGeometry(mode=cfg_model.rel_geometry)
    ])
    logger.info(f'Relative Geometry: {cfg_model.rel_geometry}')
    logger.info(f'Train transform: {train_transform}')
    logger.info(f'Test transform: {test_transform}')
    if cfg_model.rel_geometry in ['two_pos_and_rot', 'relative_pos_and_rot']:
        assert cfg_model.frag_pos_prior is None

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_linker_dataset(
        cfg=config.dataset,
        transform_map={'train': train_transform, 'val': test_transform, 'test': test_transform}
    )

    if config.dataset.version == 'tiny':
        if args.overfit_one:
            train_set, val_set, test_set = [subsets['val'][0].clone() for _ in range(config.train.batch_size)], \
                                           [subsets['val'][0]], [subsets['val'][0]]
        else:
            train_set, val_set, test_set = subsets['val'], subsets['val'], subsets['val']
    else:
        train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    # train_set, val_set, test_set = [subsets['train'][697].clone() for _ in range(config.train.batch_size)], \
    #                                [subsets['val'][0]], [subsets['val'][0]]
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    FOLLOW_BATCH = ['edge_type']
    COLLATE_EXCLUDE_KEYS = ['nbh_list']
    train_iterator = inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=COLLATE_EXCLUDE_KEYS
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS)

    # Model
    logger.info('Building model...')
    model = DiffPROTACModel(
        config.model,
        num_classes=atom_featurizer.num_classes,
        num_bond_classes=graph_builder.num_bond_classes,
        atom_feature_dim=atom_featurizer.feature_dim,
        edge_feature_dim=graph_builder.bond_feature_dim
    ).to(args.device)
    logger.info('Num of parameters is %.2f M' % (np.sum([p.numel() for p in model.parameters()]) / 1e6))
    if config.train.get('ckpt_path', None):
        ckpt = torch.load(config.train.ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        logger.info(f'load checkpoint from {config.train.ckpt_path}!')

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    # if config.train.get('ckpt_path', None):
    #     logger.info('Resuming optimizer states...')
    #     optimizer.load_state_dict(ckpt['optimizer'])
    #     logger.info('Resuming scheduler states...')
    #     scheduler.load_state_dict(ckpt['scheduler'])

    def train(it, batch):
        optimizer.zero_grad()
        loss_dict = model.get_loss(batch, pos_noise_std=config.train.get('pos_noise_std', 0.))
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm,
                                         error_if_nonfinite=True)  # 5% running time
        optimizer.step()

        # Logging
        log_losses(loss_dict, it, 'train', args.train_report_iter, logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr']
        })

    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                for t in np.linspace(0, model.num_steps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch.num_graphs).to(args.device)
                    batch = batch.to(args.device)
                    loss_dict = model.get_loss(batch, t=time_step)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    def test(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(test_loader, desc='Test', dynamic_ncols=True)):
                for t in np.linspace(0, model.num_steps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch.num_graphs).to(args.device)
                    batch = batch.to(args.device)
                    loss_dict = model.get_loss(batch, t=time_step)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_tape.update(loss_dict, 1)
        avg_loss = loss_tape.log(it, logger, writer, 'test')
        return avg_loss


    try:
        model.train()
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # try:
            t1 = time.time()
            batch = next(train_iterator).to(args.device)
            t2 = time.time()
            # print('data processing time: ', t2 - t1)
            train(it, batch)
            if it % args.sampling_iter == 0:
                data = test_set[0]
                data_list = [data.clone() for _ in range(100)]
                batch = Batch.from_data_list(
                    data_list, follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS).to(args.device)
                traj_batch, final_x, final_c, final_bond = model.sample(batch, p_init_mode=cfg_model.frag_pos_prior)
                if model.train_bond:
                    gen_mols = parse_sampling_result_with_bond(
                        data_list, final_x, final_c, final_bond, atom_featurizer,
                        known_linker_bonds=cfg_model.known_linker_bond, check_validity=True)
                else:
                    gen_mols = parse_sampling_result(data_list, final_x, final_c, atom_featurizer)
                save_path = os.path.join(vis_dir, f'sampling_results_{it}.pt')
                torch.save({
                    'data': data,
                    # 'traj': traj_batch,
                    'final_x': final_x, 'final_c': final_c, 'final_bond': final_bond,
                    'gen_mols': gen_mols
                }, save_path)
                logger.info(f'dump sampling vis to {save_path}!')
                recon_rate, complete_rate = eval_success_rate(gen_mols)
                logger.info(f'recon rate:    {recon_rate:.4f}')
                logger.info(f'complete rate: {complete_rate:.4f}')
                writer.add_scalar('sampling/recon_rate', recon_rate, it)
                writer.add_scalar('sampling/complete_rate', complete_rate, it)
                writer.flush()

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'configs': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    test(it)
                model.train()

    except KeyboardInterrupt:
        logger.info('Terminating...')
