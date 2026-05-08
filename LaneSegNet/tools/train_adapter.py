#!/usr/bin/env python
"""
Adapter Fine-tuning Script (Production Version)

This is the CLEAN implementation for paper publication.

Stage 2-(b): Fine-tune learnable affinity matrix while freezing vision model

Steps:
1. Build model with AdapterRelationshipHead
2. Load baseline checkpoint (vision model + base LSTE)
3. Initialize text guidance from semantic checkpoint
4. Freeze ALL components except affinity matrix (~118 params)
5. Train for 1 epoch on OpenLane-V2 train set

Usage:
    python tools/train_adapter.py projects/configs/adapter_finetune_clean.py
"""

import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger

try:
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Adapter fine-tuning (clean)')
    parser.add_argument('config', help='adapter config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def freeze_vision_model(model, logger):
    """
    Freeze all components except LSTE head's affinity modulator.

    Components to freeze:
    - img_backbone (ResNet-50)
    - img_neck
    - bev_constructor
    - pts_bbox_head (lane detection)
    - bbox_head (TE detection)
    - lsls_head, area_head (auxiliary heads)
    - lste_head base components (MLP_o1, MLP_o2, classifier)
    - lste_head text guidance (text_proj, lane_proj, temperature)

    Only trainable:
    - lste_head.affinity_modulator (118 params)
    """
    logger.info("\n" + "="*80)
    logger.info("Freezing vision model...")
    logger.info("="*80)

    # Get actual model (unwrap if in DataParallel)
    actual_model = model.module if hasattr(model, 'module') else model

    # Freeze all parameters first
    for param in actual_model.parameters():
        param.requires_grad = False

    # Unfreeze only affinity modulator
    for param in actual_model.lste_head.affinity_modulator.parameters():
        param.requires_grad = True

    # Count frozen/trainable params
    frozen_params = sum(p.numel() for p in actual_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actual_model.parameters())

    logger.info(f"\nParameter Summary:")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Frozen parameters:    {frozen_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio:      {trainable_params/total_params*100:.6f}%")

    # Verify only affinity is trainable
    affinity_params = sum(p.numel() for p in actual_model.lste_head.affinity_modulator.parameters())

    if trainable_params == affinity_params:
        logger.info(f"\n✅ SUCCESS! Only affinity matrix ({affinity_params} params) is trainable")
    else:
        logger.warning(f"\n⚠ WARNING: Expected {affinity_params} trainable, got {trainable_params}")
        logger.warning("Trainable components:")
        for name, param in actual_model.named_parameters():
            if param.requires_grad:
                logger.warning(f"  - {name}: {param.numel()} params")

    logger.info("="*80 + "\n")

    return frozen_params, trainable_params


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Setup multi-processing
    setup_multi_processes(cfg)

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # Autoscale learning rate
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # Init distributed env
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # Init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    # Log config
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seed
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    # Build model
    logger.info("\nBuilding model with AdapterRelationshipHead...")
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    logger.info(f"✓ Model built: {type(model).__name__}")
    logger.info(f"✓ LSTE Head type: {type(model.lste_head).__name__}")

    # Load baseline checkpoint
    if cfg.get('load_from', None):
        logger.info(f'\nLoading baseline checkpoint: {cfg.load_from}')
        checkpoint = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights (strict=False to allow new affinity params)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info('✓ Baseline checkpoint loaded')
        logger.info(f'  Missing keys: {len(missing)} (expected: affinity matrix)')
        if len(unexpected) > 0:
            logger.warning(f'  Unexpected keys: {len(unexpected)}')

        # Initialize text guidance from semantic checkpoint
        logger.info('\nInitializing text guidance...')
        model.lste_head.init_weights()

    # Freeze vision model (only keep affinity trainable)
    frozen_params, trainable_params = freeze_vision_model(model, logger)

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    # Add metadata
    model.CLASSES = datasets[0].CLASSES

    # Train model
    logger.info("\nStarting adapter fine-tuning...")
    logger.info(f"  Training epochs: {cfg.total_epochs}")
    logger.info(f"  Optimizer: {cfg.optimizer['type']}")
    logger.info(f"  Learning rate: {cfg.optimizer['lr']}")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_params/frozen_params*100:.6f}% of frozen)")

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp)


if __name__ == '__main__':
    main()
