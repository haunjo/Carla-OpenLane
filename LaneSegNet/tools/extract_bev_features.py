#!/usr/bin/env python
"""
Extract BEV Features for Domain Gap Analysis

This script extracts scene-level BEV features from the encoder for UMAP visualization.

Usage:
    # Extract from CARLA-OpenLane (synthetic)
    python tools/extract_bev_features.py \
        projects/configs/lanesegnet_r50_1x2_8e_carla_subset_A.py \
        work_dirs/carla_pretrain/epoch_8.pth \
        --out features_carla.pkl \
        --domain synthetic

    # Extract from OpenLane-V2 (real)
    python tools/extract_bev_features.py \
        projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py \
        work_dirs/baseline/epoch_24.pth \
        --out features_real.pkl \
        --domain real

    # Extract from adapted model
    python tools/extract_bev_features.py \
        projects/configs/adapter_finetune_clean.py \
        work_dirs/adapter_finetune/epoch_4.pth \
        --out features_adapted.pkl \
        --domain adapted
"""

import argparse
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from tqdm import tqdm

import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract BEV features')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', required=True, help='output pickle file')
    parser.add_argument('--domain', required=True,
                       choices=['synthetic', 'real', 'adapted'],
                       help='domain label for this dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='maximum number of samples to extract')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='batch size for extraction')
    args = parser.parse_args()
    return args


def extract_bev_feature_hook(module, input, output):
    """Hook to capture BEV features from bev_constructor"""
    # Handle different output formats
    if isinstance(output, (list, tuple)):
        # If multiple outputs, use the first one
        output = output[0]

    if not isinstance(output, torch.Tensor):
        print(f"Warning: Unexpected output type: {type(output)}")
        return

    # Debug: print shape on first call
    if not hasattr(module, '_hook_called'):
        print(f"[DEBUG] BEV constructor output shape: {output.shape}")
        module._hook_called = True

    # BEVFormer output is typically [bev_h*bev_w, bs, embed_dims]
    # We need to convert to [bs, embed_dims]

    if output.dim() == 3:
        # Assume [seq_len, bs, embed_dims] or [bs, seq_len, embed_dims]
        # Check which dimension is batch size (usually smallest or second)
        if output.size(1) < output.size(0) and output.size(1) < output.size(2):
            # [seq_len, bs, embed_dims] format
            bev_pooled = output.mean(dim=0)  # Average over sequence → [bs, embed_dims]
        else:
            # [bs, seq_len, embed_dims] format
            bev_pooled = output.mean(dim=1)  # Average over sequence → [bs, embed_dims]
    elif output.dim() == 4:
        # [B, C, H, W] → [B, C]
        bev_pooled = output.mean(dim=[2, 3])
    elif output.dim() == 2:
        # Already [B, C]
        bev_pooled = output
    else:
        print(f"Warning: Unexpected tensor shape: {output.shape}")
        # Fallback: flatten all but first dimension
        bev_pooled = output.view(output.size(0), -1).mean(dim=1, keepdim=True)

    module._captured_bev_feature = bev_pooled.detach().cpu()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Build dataset
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # Limit samples if specified
    if args.max_samples is not None:
        print(f"Limiting to {args.max_samples} samples (total: {len(dataset)})")
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))

    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # Build model
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Wrap model
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Register hook to capture BEV features
    # BEV constructor is typically model.module.bev_constructor
    bev_module = model.module.bev_constructor
    hook_handle = bev_module.register_forward_hook(extract_bev_feature_hook)

    print(f"\n{'='*80}")
    print(f"Extracting BEV features from {args.checkpoint}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Domain: {args.domain}")
    print(f"{'='*80}\n")

    # Extract features
    all_features = []
    all_metadata = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Extracting features")):
            # Forward pass
            _ = model(return_loss=False, rescale=True, **data)

            # Get captured BEV feature
            if hasattr(bev_module, '_captured_bev_feature'):
                bev_feat = bev_module._captured_bev_feature.numpy()  # [B, C]

                # Store features
                for b in range(bev_feat.shape[0]):
                    all_features.append(bev_feat[b])  # [C,]

                    # Store metadata
                    sample_idx = i * args.batch_size + b
                    all_metadata.append({
                        'domain': args.domain,
                        'sample_idx': sample_idx,
                        'checkpoint': args.checkpoint,
                        'config': args.config
                    })

    # Remove hook
    hook_handle.remove()

    # Convert to numpy array
    all_features = np.stack(all_features, axis=0)  # [N, C]

    print(f"\n{'='*80}")
    print(f"Extraction complete!")
    print(f"  Features shape: {all_features.shape}")
    print(f"  Feature dim: {all_features.shape[1]}")
    print(f"  Num samples: {all_features.shape[0]}")
    print(f"{'='*80}\n")

    # Save results
    results = {
        'features': all_features,
        'metadata': all_metadata,
        'domain': args.domain,
        'checkpoint': args.checkpoint,
        'config': args.config
    }

    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.out)))
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved to: {args.out}\n")


if __name__ == '__main__':
    main()
