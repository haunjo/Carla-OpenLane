# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TopoLogic is a NeurIPS 2024 lane topology reasoning system for autonomous driving that uses an interpretable pipeline combining geometric distance and semantic similarity. The project is built on the MMDetection ecosystem and works with the OpenLane-V2 dataset.

## Commands

### Training
```bash
# Train with 8 GPUs (recommended)
./tools/dist_train.sh 8 [work_dir_name] [--autoscale-lr]

# Examples:
./tools/dist_train.sh 8 topologic_experiment
./tools/dist_train.sh 8 topologic_experiment --autoscale-lr
```

### Testing/Evaluation
```bash
# Test with 8 GPUs
./tools/dist_test.sh 8 [work_dir_name] [--show]

# Examples:
./tools/dist_test.sh 8 topologic_experiment
./tools/dist_test.sh 8 topologic_experiment --show  # with visualization
```

### Environment Setup
```bash
# Create conda environment
conda create -n topologic python=3.8 -y
conda activate topologic

# Install dependencies
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Architecture

### Core Model Structure
- **Main Detector**: `TopoLogic` class in `projects/topologic/models/detectors/topologic.py`
- **BEV Constructor**: Uses BEVFormer architecture for bird's eye view feature construction
- **Lane Head**: `TopoLogicHead` for lane detection and topology reasoning
- **Bbox Head**: Custom deformable DETR head for traffic element detection

### Key Components
1. **Image Feature Extraction**: ResNet-50 backbone with FPN neck
2. **BEV Feature Construction**: BEVFormer-based temporal and spatial attention
3. **Lane Detection**: Transformer-based decoder with SGNN layers
4. **Topology Reasoning**: Combines geometric distance and semantic similarity
5. **Relationship Heads**: Separate heads for lane-to-lane (lclc) and lane-to-traffic-element (lcte) relationships

### Dataset Integration
- Built for OpenLane-V2 dataset (subset A and B)
- Supports both with and without SDMap data
- Custom data loading pipelines in `projects/topologic/datasets/`

### Configuration System
- Configs located in `projects/configs/`
- Main config: `topologic_r50_1x2_24e_carla_subset_A.py`
- Uses MMDetection's config inheritance system
- Point cloud range: `[-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]`

## Data Requirements

### Expected Data Structure
```
data/
├── Carla-OLV2/
│   ├── data_dict_carla_train_argoverse2.pkl
│   ├── data_dict_carla_val_argoverse2.pkl
│   └── [other data files]
```

### Environment Variables
The training/test scripts automatically set PYTHONPATH to the repo root using a relative path.

## Dependencies

### Key Requirements
- Python 3.8.x
- PyTorch 1.9.1 + CUDA 11.1
- MMDetection ecosystem: mmcv-full==1.5.2, mmdet==2.26.0, mmdet3d==1.0.0rc6
- OpenLane-V2 devkit: openlanev2==2.1.0
- Additional: numpy, scipy, ortools for optimization

### GPU Requirements
- Recommended: 8 GPUs for training
- Supports distributed training with automatic learning rate scaling

## Working Directories

### Standard Structure
- `work_dirs/[experiment_name]/`: Training outputs, logs, checkpoints
- Training logs: `work_dirs/[experiment_name]/train.[timestamp].log`
- Test logs: `work_dirs/[experiment_name]/test.[timestamp].log`
- Checkpoints saved every 4 epochs, max 4 kept

## Code Organization

### Projects Structure
```
projects/topologic/
├── models/
│   ├── detectors/topologic.py          # Main model
│   ├── dense_heads/topologic_head.py   # Lane detection head
│   └── modules/                        # Custom modules
├── datasets/                           # Data loading
├── core/lane/                          # Lane-specific utilities
└── utils/                              # Helper functions
```

### Key Inheritance
- Extends `MVXTwoStageDetector` from MMDet3D
- Uses custom Hungarian assigners for lane matching
- Implements custom loss functions for topology reasoning