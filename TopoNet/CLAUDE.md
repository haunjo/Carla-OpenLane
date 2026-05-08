# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoNet is a graph-based topology reasoning model for autonomous driving (CVPR 2023). It detects lane centerlines and traffic elements, then reasons about their topological relationships using a Scene Graph Neural Network (SGNN). Built on the MMDetection3D ecosystem, targeting the OpenLane-V2 dataset.

- **Paper**: [Graph-based Topology Reasoning for Driving Scenes](https://arxiv.org/abs/2304.05277)
- **Original repo**: [OpenDriveLab/TopoNet](https://github.com/OpenDriveLab/TopoNet)

## Commands

### Training
```bash
# Train with 4 GPUs (default config: subset B, 24 epochs)
./tools/dist_train.sh 4

# Config is hardcoded in tools/dist_train.sh — edit CONFIG variable to switch
```

### Evaluation
```bash
# Evaluate (default: subset B, latest checkpoint in work_dirs/)
./tools/dist_test.sh 4
```

### Data Preparation
```bash
mkdir -p data
ln -s /path/to/OpenLane-V2 data/OpenLane-V2
```

## Architecture

```
Input (multi-view images)
  → ResNet-50 backbone + FPN
  → BEVFormer (bird's eye view feature construction)
  → TopoNetHead
      ├── Lane detection (transformer decoder)
      ├── Traffic element detection (DeformableDETR)
      └── SGNN (Scene Graph Neural Network)
            ├── lane-to-lane topology (lclc)
            └── lane-to-traffic-element topology (lcte)
```

**Key files:**
- `projects/toponet/models/detectors/toponet.py` — main detector
- `projects/toponet/models/dense_heads/toponet_head.py` — combined detection + topology head
- `projects/toponet/models/modules/sgnn_decoder.py` — graph reasoning module
- `projects/configs/toponet_r50_8x1_24e_olv2_subset_B.py` — default config

## Environment

- Python 3.8, PyTorch 1.9.1, CUDA 11.1
- Same MMDetection stack as LaneSegNet and TopoLogic

```bash
pip install -r requirements.txt
```

## Notes

- Default config targets **Subset B** (nuScenes 6-camera setup)
- `projects/configs/toponet_r50_8x1_24e_olv2_subset_A.py` available for Subset A (ArgoVerse2 7-camera)
- `work_dirs/` and `data/` are gitignored
