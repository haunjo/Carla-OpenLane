"""
Text-guided 2-stage config for Map Element Bucket with fair hyperparameters.

Stage 2 (OpenLane-V2 fine-tuning):
1. Frozen text projection for domain-invariant topology reasoning
2. Matched hyperparameters with naive baseline for fair comparison
"""

_base_ = ['./lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

# Use base optimizer (no custom learning rates for bbox_head)

model = dict(
    # Keep text-guided topology reasoning from Stage 1
    lsls_head=dict(
        type='TextEnhancedRelationshipHead',
        freeze_text_proj=True,  # Freeze text projection in Stage 2
        text_embed_path='text_embeddings_siglip.pt',
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5),
        loss_text=dict(
            type='CrossEntropyLoss',
            loss_weight=0.0)),  # No text loss in Stage 2
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,  # Match naive baseline
            alpha=0.25,
            loss_weight=1.0),  # Match naive baseline (was 2.0)
    ),
    # Match train_cfg cls_cost with loss_cls weight
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),  # Match loss_weight
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))))
)

# Load from Carla Stage 1 checkpoint
# load_from = 'work_dirs/lanesegnet_1stage_text_guided_mapele_bucket/epoch_8.pth'
# resume_from = None