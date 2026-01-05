"""
Phase 2 (Epoch 9-24): Full fine-tuning with lower learning rate
after new classes are learned.

Strategy:
- Unfreeze all layers
- Lower learning rate for stable convergence
- Balanced fine-tuning of all components
"""

_base_ = ['./lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

# Total epochs for phase 2
total_epochs = 16

# Optimizer: Lower LR, unfreeze all
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # Lower than phase 1 (2e-4 → 1e-4)
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),

            # Moderate learning for fine-tuning
            'bbox_head.transformer.encoder': dict(lr_mult=0.5),
            'bbox_head.transformer.decoder': dict(lr_mult=0.7),
            # fc_cls uses default lr
        }),
    weight_decay=0.01)

# Moderate loss weight
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
            gamma=2.5,
            alpha=0.25,
            loss_weight=2.0),  # Balanced
    ),
    # Match train_cfg cls_cost with loss_cls weight
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),  # Match loss_weight
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))))
)

# Evaluation
evaluation = dict(interval=2, start=4, pipeline=_base_.test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=2, max_keep_ckpts=6)

# Resume from phase 1
load_from = None
resume_from = 'work_dirs/lanesegnet_8e_carla_8e_olv2_mapele_phase1_1103/epoch_8.pth'

work_dir = 'work_dirs/lanesegnet_8e_carla_24e_olv2_mapele_phase2_1103'
