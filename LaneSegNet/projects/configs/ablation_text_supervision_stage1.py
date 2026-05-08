"""
Ablation Study (c): Single-head w/ text supervision - Stage 1

Stage 1: Pre-train on Carla-OLV2 with text supervision
- TextEnhancedRelationshipHead (n_heads=1)
- Text supervision loss enabled (loss_text weight=0.1)
- Ground truth text labels from Carla-OLV2 (template_id)
- 8 epochs training

After Stage 1, use checkpoint for Stage 2 (ablation_text_supervision_stage2.py)
"""

_base_ = ['lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

# Override dataset to Carla-OLV2
dataset_type = 'OpenLaneV2_subset_A_MapElementBucket_Dataset'
data_root = 'data/Carla-OLV2/'

# Model - only lsls_head differs from baseline
lsls_head = dict(
    type='TextEnhancedRelationshipHead',
    in_channels_o1=256,
    in_channels_o2=256,
    shared_param=False,
    text_embed_path='text_embeddings_siglip.pt',
    text_embed_dim=1152,
    proj_dim=256,
    temperature_init=0.07,
    freeze_text_proj=False,  # Stage 1: trainable
    use_self_attn=False,
    n_heads=1,  # Single-head
    loss_rel=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=5),
    loss_text=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=0.1))  # Text supervision enabled

# Data configuration
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        collection='data_dict_carla_train_argoverse2_ls',
        meta_file='data_dict_carla_train_argoverse2_ls.pkl',
        with_lane_text=True),  # Enable text label loading
    val=dict(
        type=dataset_type,
        data_root=data_root,
        collection='data_dict_carla_val_argoverse2_ls',
        meta_file='data_dict_carla_val_argoverse2_ls.pkl',
        with_lane_text=True))

# Training settings
total_epochs = 8
evaluation = dict(interval=8)
checkpoint_config = dict(interval=1, max_keep_ckpts=8)

# No checkpoint loading for Stage 1
load_from = None
resume_from = None

# Work directory
work_dir = './work_dirs/ablation_text_supervision_stage1'
