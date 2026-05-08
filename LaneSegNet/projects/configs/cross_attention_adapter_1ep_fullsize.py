"""
Cross-Attention Adapter with Full-Size Images
- Full resolution images for proper GPU memory usage
- Larger batch size option
"""

_base_ = ['./cross_attention_adapter_1ep.py']

# Override train pipeline to use full-size images
train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3DLaneSegment',
         with_lane_3d=True, with_lane_label_3d=True, with_lane_adj=True, with_lane_type=True,
         with_bbox=True, with_label=True, with_lane_lste_adj=True, with_area=True),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[1.0]),  # FULL SIZE
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='GridMaskMultiViewImage'),
    dict(type='LaneSegmentParameterize3D', method='fix_pts_interp', method_para=dict(n_points=10)),
    dict(type='GenerateLaneSegmentMask', points_num=10, bev_h=100, bev_w=200),
    dict(type='CustomFormatBundle3DLane', class_names=['lane_segment', 'ped_crossing', 'road_boundary']),
    dict(type='CustomCollect3D', keys=[
        'img', 'gt_lanes_3d', 'gt_lane_labels_3d', 'gt_lane_adj',
        'gt_instance_masks', 'gt_lane_left_type', 'gt_lane_right_type',
        'gt_labels', 'gt_bboxes', 'gt_lane_lste_adj', 'gt_areas_3d', 'gt_area_labels_3d'])
]

# Optionally increase batch size if GPU memory allows
data = dict(
    samples_per_gpu=1,  # Reduce if OOM with full images
    train=dict(pipeline=train_pipeline)
)

# Adjust work directory
work_dir = 'work_dirs/cross_attention_adapter_1ep_fullsize'