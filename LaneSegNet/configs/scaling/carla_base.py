"""
Base configuration for CARLA-OpenLane scaling experiments.
This config will be inherited by scale-specific configs.
"""

_base_ = [
    '../_base_/models/lanesegnet_r50.py',
    '../_base_/default_runtime.py'
]

# Model configuration
model = dict(
    type='LaneSegNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    head=dict(
        type='LaneSegNetHead',
        in_channels=256,
        hidden_dim=256,
        num_classes=4,  # Background, lane, lane boundary, other
        num_queries=50,
        num_topology_queries=100,
        loss_cfg=dict(
            loss_ce=dict(type='CrossEntropyLoss', weight=1.0),
            loss_dice=dict(type='DiceLoss', weight=2.0),
            loss_topology=dict(type='TopologyLoss', weight=3.0)
        )
    )
)

# Dataset base configuration (will be overridden by scale-specific configs)
dataset_type = 'CARLAOpenLaneDataset'
data_root = '/path/to/carla_openlane'
split_root = './data/scaling_splits'

# Base data pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='Resize', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels_3d', 'gt_bboxes_3d'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

# Training configuration
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5
)

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['DET_l', 'TOP_ll', 'TOP_lt', 'OLS'])

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Work directory will be set by scale-specific configs
work_dir = None