"""
Simple Adapter Fine-tuning Config
- Load from lanesegnet_2stage_naive_mapele_bucket/epoch_23.pth
- Load text guidance from work_dirs/lanesegnet_text_guided_lste_1stage/epoch_12.pth
- Only train adapter weights in lste_head (117 params = 9x13)
- Freeze everything else
- 1 epoch training
"""

_base_ = ['./lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

# Model: Replace LSTE head with simple adapter version
model = dict(
    lste_head=dict(
        type='SimpleAdapterHead',
        in_channels_o1=256,
        in_channels_o2=256,
        shared_param=False,
        text_checkpoint_path='work_dirs/lanesegnet_text_guided_lste_1stage/epoch_12.pth',
        adapter_scale=0.5,  # Moderate adapter influence
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25
        )
    )
)

# Load baseline checkpoint
load_from = 'lanesegnet_2stage_naive_mapele_bucket/epoch_23.pth'

# Training: 1 epoch only
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Optimizer: Only for adapter weights
# Use a custom optimizer_constructor to freeze everything except adapter_weights
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # Moderate learning rate for adapter
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr_ratio=0.01
)

# Evaluation
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# Work directory
work_dir = 'work_dirs/simple_adapter_1ep'

# Custom hooks
custom_hooks = [
    dict(type='AdapterOptimizerHook', priority='HIGHEST')
]

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)