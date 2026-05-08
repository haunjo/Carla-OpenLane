"""
Configuration for 25% CARLA-OpenLane scaling experiment.
"""

_base_ = ['./carla_base.py']

# Scale-specific parameters
scale_percentage = 25
scale_fraction = 0.25
sampling_method = 'stratified'  # 'random' or 'stratified'
seed = 42

# Dataset configuration
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='${dataset_type}',
        data_root='${data_root}',
        split_file=f'{split_root}/carla_{scale_percentage}pct_{sampling_method}_seed{seed}.json',
        pipeline='${train_pipeline}',
        test_mode=False
    ),
    val=dict(
        type='${dataset_type}',
        data_root='${data_root}',
        split_file=f'{split_root}/carla_{scale_percentage}pct_{sampling_method}_seed{seed}_val.json',
        pipeline='${test_pipeline}',
        test_mode=True
    ),
    test=dict(
        type='${dataset_type}',
        data_root='${data_root}',
        split_file=f'{split_root}/carla_{scale_percentage}pct_{sampling_method}_seed{seed}_test.json',
        pipeline='${test_pipeline}',
        test_mode=True
    )
)

# Training schedule
runner = dict(type='EpochBasedRunner', max_epochs=24)

# Optimizer
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

# Work directory
work_dir = f'./work_dirs/scaling/carla_{scale_percentage}pct_{sampling_method}_seed{seed}'

# Add scale info to checkpoint name
checkpoint_config = dict(
    interval=1,
    filename_tmpl=f'carla_{scale_percentage}pct_epoch_{{:02d}}.pth'
)