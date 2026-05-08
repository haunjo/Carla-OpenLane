# Baseline Visualization Guide

## 1. Generate Baseline Test Results

When GPU is available, run:

```bash
cd /home/user/LaneSegNet
./tools/visualize_baseline.sh 1
```

Or manually:
```bash
export PYTHONPATH="/home/user/LaneSegNet/":$PYTHONPATH
WORK_DIR=work_dirs/lanesegnet_baseline_mapele_bucket
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py
CHECKPOINT=${WORK_DIR}/latest.pth

python -m torch.distributed.run --nproc_per_node=1 --master_port=28510 \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 --show --show-dir ${WORK_DIR}/vis_results
```

## 2. Visualize Specific Samples

After test results are generated, you can visualize specific samples:

```python
import mmcv
import numpy as np
from projects.lanesegnet.utils.visualize_topology import visualize_bev_topology

# Load test results
results = mmcv.load('work_dirs/lanesegnet_baseline_mapele_bucket/test/results.pkl')

# Select a sample (e.g., sample 100)
sample_idx = 100
result = results[sample_idx]

# Extract predictions
pred_lanes = result['lane_results'][0]  # [N_lanes, points*3]
pred_scores = result['lane_results'][1]  # [N_lanes]
pred_topology = result['lsls_results']   # [N_lanes, N_lanes]

# Visualize
img = visualize_bev_topology(
    pred_lanes=pred_lanes,
    pred_topology=pred_topology,
    mode='pred',
    map_size=[-51.2, 51.2, -25.6, 25.6],
    scale=10
)

# Save
import cv2
cv2.imwrite('baseline_vis_sample100.png', img)
```

## 3. Compare Multiple Methods

To generate comparison figures for paper (Fig. 5):

```python
import matplotlib.pyplot as plt
from projects.lanesegnet.utils.visualize_topology import visualize_bev_topology

# Load all results
baseline_results = mmcv.load('work_dirs/lanesegnet_baseline_mapele_bucket/test/results.pkl')
carla_results = mmcv.load('work_dirs/lanesegnet_2stage_naive_mapele_bucket/test/results.pkl')
textguided_results = mmcv.load('work_dirs/lanesegnet_2stage_text_guided_mapele_bucket_v2/test/results.pkl')

# Select same sample from all methods
sample_idx = 100  # Choose challenging intersection scene

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

methods = [
    ('Baseline', baseline_results[sample_idx]),
    ('Carla-OpenLane', carla_results[sample_idx]),
    ('Text-guided (Ours)', textguided_results[sample_idx])
]

for ax, (method_name, result) in zip(axes, methods):
    pred_lanes = result['lane_results'][0]
    pred_topology = result['lsls_results']

    img = visualize_bev_topology(
        pred_lanes=pred_lanes,
        pred_topology=pred_topology,
        mode='pred',
        map_size=[-51.2, 51.2, -25.6, 25.6],
        scale=10
    )

    ax.imshow(img)
    ax.set_title(method_name, fontsize=14, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('fig/comparison_baseline_carla_textguided.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

## 4. Files Location

- **Baseline checkpoint**: `work_dirs/lanesegnet_baseline_mapele_bucket/latest.pth`
- **Carla-OpenLane checkpoint**: `work_dirs/lanesegnet_2stage_naive_mapele_bucket/latest.pth`
- **Text-guided checkpoint**: `work_dirs/lanesegnet_2stage_text_guided_mapele_bucket_v2/latest.pth`

## 5. Config Files

- All methods use the same config: `projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_mapele_bucket.py`
- Only difference is the checkpoint used

## 6. Notes

- Make sure to use `--show --show-dir` flags for visualization during test
- Results will be saved in `${WORK_DIR}/vis_results/`
- For paper figures, select challenging scenes (intersections, merges, complex topology)
