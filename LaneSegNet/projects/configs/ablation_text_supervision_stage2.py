"""
Ablation Study (c): Single-head w/ text supervision - Stage 2

Stage 2: Fine-tune on OpenLane-V2 with frozen text components
- TextEnhancedRelationshipHead (n_heads=1)
- Text components frozen (freeze_text_proj=True)
- No text supervision loss (loss_text=None)
- Load from Stage 1 checkpoint
- 24 epochs fine-tuning

This demonstrates the effectiveness of text supervision during pre-training.
"""

_base_ = ['lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py']

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
    freeze_text_proj=True,  # Stage 2: freeze text components
    use_self_attn=False,
    n_heads=1,  # Single-head
    loss_rel=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=5),
    loss_text=None)  # No text supervision in Stage 2

# Data configuration - OpenLane-V2
dataset_type = 'OpenLaneV2_subset_A_MapElementBucket_Dataset'
data_root = 'data/OpenLane-V2/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        collection='data_dict_subset_A_train_ls',
        meta_file='data_dict_subset_A_train_ls.pkl',
        with_lane_text=False),  # No text labels in OpenLane-V2
    val=dict(
        type=dataset_type,
        data_root=data_root,
        collection='data_dict_subset_A_val_ls',
        meta_file='data_dict_subset_A_val_ls.pkl',
        with_lane_text=False))

# Training settings
total_epochs = 24
evaluation = dict(interval=24)
checkpoint_config = dict(interval=1, max_keep_ckpts=8)

# Load from Stage 1 checkpoint (already trained!)
load_from = 'work_dirs/lanesegnet_1stage_text_guided_mapele_bucket/epoch_8.pth'
resume_from = None

# Work directory
work_dir = './work_dirs/ablation_text_supervision_stage2'
