"""
Phase 1 (Epoch 1-8): Focus on learning new traffic element classes
while preserving traffic light knowledge.

Naive version without text enhancement.

Strategy:
- Freeze most of transformer (preserve traffic light features)
- Only train classification head (learn new classes)
- High learning rate for fast adaptation
"""

_base_ = ['./lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

# Total epochs for phase 1
total_epochs = 8

# Optimizer: Focus on classification head
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),

            # Freeze transformer (preserve traffic light knowledge)
            'bbox_head.transformer.encoder': dict(lr_mult=0.0),
            'bbox_head.transformer.decoder': dict(lr_mult=0.05),

            # Fast learning for new classes
            # Note: fc_cls will use default lr (2e-4)
        }),
    weight_decay=0.01)

# Stronger loss for traffic elements
model = dict(
    # Use naive RelationshipHead (no text enhancement)
    lsls_head=dict(
        type='RelationshipHead',
        in_channels_o1=256,
        in_channels_o2=256,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.5,
            alpha=0.25,
            loss_weight=3.0),  # Higher weight for phase 1
    ),
    # Match train_cfg cls_cost with loss_cls weight
    train_cfg=dict(
        bbox=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=3.0),  # Match loss_weight
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0))))
)

# Evaluation
evaluation = dict(interval=2, start=4, pipeline=_base_.test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=2, max_keep_ckpts=6)

# Load from Carla Stage 1 checkpoint (naive version)
load_from = 'work_dirs/lanesegnet_8e_carla_naive_mapele_bucket/epoch_8.pth'
resume_from = None

work_dir = 'work_dirs/lanesegnet_8e_carla_8e_olv2_mapele_phase1_naive'
