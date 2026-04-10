# Complete Workflow: From CARLA to Trained Model

This document describes the end-to-end pipeline for creating and using the Carla-OpenLane dataset.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Complete Pipeline                            │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Data Generation (Optional - Use CARLA Simulator)
   │
   │  CARLA 0.9.15 + Custom Scripts
   │  → Multi-camera images
   │  → Ego pose data
   │  → Traffic element metadata
   │  → OpenDRIVE maps
   ▼
[Raw CARLA Data]
   │
   │
Step 2: Data Annotation (OpenLane-V2-HDmap-Converter)
   │
   │  Docker Container + Annotation Pipeline
   │  → Lane segment extraction
   │  → Topology matrix generation
   │  → Traffic element association
   ▼
[Annotated Dataset (OpenLane-V2 Format)]
   │
   │
Step 3: Dataset Preparation
   │
   │  Split generation (train/val/test)
   │  → JSON manifests
   │  → PKL preprocessing
   ▼
[Training-Ready Dataset]
   │
   │
Step 4: Model Training (LaneSegNet)
   │
   │  Distributed training
   │  → Checkpoints
   │  → Evaluation metrics
   ▼
[Trained Model]
```

---

## Step 1: Data Generation (Optional)

### Option A: Use Pre-Generated Dataset (Recommended)

**Skip this step** if you want to use our pre-annotated dataset.

Download the dataset:

```bash
cd Carla-OpenLane
./scripts/download_dataset.sh
```

**Available datasets:**
- **Carla-OpenLane-subset-A** (\~36GB): 200 scenes, ArgoVerse2 camera setup
- **Carla-OpenLane-subset-B**: 200 scenes, nuScenes camera setup

### Option B: Generate Custom Data from CARLA

If you need custom scenarios (different towns, weather, traffic):

#### 1.1 Install CARLA Simulator

```bash
# Download CARLA 0.9.15 (recommended) or 0.9.16
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15
```

**Documentation:** [CARLA 0.9.15 Docs](https://carla.readthedocs.io/en/0.9.15/)

#### 1.2 Use Reference Scripts

We provide reference scripts in `tools/data-generation/`:

```bash
cd Carla-OpenLane/tools/data-generation

# Example: Run data capture for specific towns
./run.sh Town01 Town03 Town10
```

**Configuration:**
- Camera setups: ArgoVerse2 (7 cameras) or nuScenes (6 cameras)
- Towns: Town01-10 (HD maps available)
- Weather presets: Clear, Cloudy, Wet, Night
- Traffic density: Low, Medium, High

**Output structure:**
```
datasets/raw/
├── train/
│   ├── 0000/
│   │   ├── info/
│   │   │   ├── 0.json         # Metadata (ego pose, cameras, traffic)
│   │   │   └── 1.json
│   │   ├── image_2/
│   │   │   ├── 0.jpg          # Camera images
│   │   │   └── 1.jpg
│   │   └── maps/
│   │       └── Town01.xodr    # OpenDRIVE map
│   └── 0001/
└── val/
```

**Note:** CARLA data generation is **not** part of the core Carla-OpenLane repository. Refer to [CARLA official documentation](https://carla.readthedocs.io/) for simulator setup and data collection.

---

## Step 2: Data Annotation

Use the **OpenLane-V2-HDmap-Converter** tool to convert raw CARLA data into OpenLane-V2 format.

### 2.1 Quick Setup

```bash
# Clone annotation tool (separate repository)
git clone https://github.com/haunjo/OpenLane-V2-HDmap-Converter.git
cd OpenLane-V2-HDmap-Converter

# Download and load Docker image
# [Download OLV2xHDmap.tar from Google Drive]
docker load -i OLV2xHDmap.tar
```

### 2.2 Configure Dataset Path

Edit `docker/run_docker.sh`:

```bash
DATASET=/path/to/Carla-OpenLane/datasets/raw
```

### 2.3 Run Annotation

```bash
cd docker
./run_docker.sh

# Inside container:
cd /home/developer/workspace/scripts
./annotation.sh --split train --subset argoverse2 --task segment
```

**Parameters:**
- `--split train`: Process training data
- `--subset argoverse2`: Scenes 0-199 (or `nuscenes` for 1000-1199)
- `--task segment`: Full annotation (or `centerline` for lightweight)

**Output:** `datasets/raw/{split}/{scene}/info/{frame}-ls.json`

### 2.4 Validate Annotations

```bash
# Exit Docker
exit

# Run validation
python OpenLane-V2-HDmap-Converter/src/checksum.py \
  --root datasets/raw/train \
  --root datasets/raw/val
```

**Detailed guide:** [docs/ANNOTATION.md](ANNOTATION.md)

---

## Step 3: Dataset Preparation

### 3.1 Generate Data Splits

Create JSON manifests for train/val/test splits:

```bash
cd Carla-OpenLane/scripts

# Generate split files
python generate_splits.py \
  --data-root ../datasets/raw \
  --output-dir ../datasets/splits
```

**Output:**
```
datasets/splits/
├── train.json      # Training split metadata
├── val.json        # Validation split
└── test.json       # Test split
```

### 3.2 Preprocess for Training

Convert to PKL format for LaneSegNet:

```bash
cd LaneSegNet/data

# Generate training data PKL
python gt_generator.py \
  --data-json ../../datasets/splits/train.json \
  --output carla_train_ls.pkl

# Generate validation data PKL
python gt_generator.py \
  --data-json ../../datasets/splits/val.json \
  --output carla_val_ls.pkl
```

**Detailed guide:** [docs/DATASET.md](DATASET.md)

---

## Step 4: Model Training

### 4.1 Setup LaneSegNet

```bash
cd Carla-OpenLane/LaneSegNet

# Install dependencies
pip install -r requirements.txt

# Configure data paths in config file
vim configs/carla_openlanev2.py
```

Update config:
```python
data_root = '../datasets/splits/'
train_pkl = 'carla_train_ls.pkl'
val_pkl = 'carla_val_ls.pkl'
```

### 4.2 Train Model

```bash
# Single GPU
python tools/train.py configs/carla_openlanev2.py

# Multi-GPU (distributed)
./tools/dist_train.sh configs/carla_openlanev2.py 4  # 4 GPUs
```

**Training options:**
- `--autoscale-lr`: Automatically scale learning rate by GPU count
- `--resume-from`: Resume from checkpoint
- `--work-dir`: Output directory for logs/checkpoints

### 4.3 Evaluate Model

```bash
# Evaluate on validation set
python tools/test.py \
  configs/carla_openlanev2.py \
  work_dirs/latest.pth \
  --eval

# Visualize predictions
./tools/dist_test.sh configs/carla_openlanev2.py work_dirs/latest.pth 1 --show
```

---

## Quick Start (Pre-Annotated Dataset)

If you want to skip Steps 1-2:

```bash
# 1. Clone main repository
git clone https://github.com/haunjo/Carla-OpenLane.git
cd Carla-OpenLane

# 2. Download pre-annotated dataset
./scripts/download_dataset.sh

# 3. Generate PKL files
cd LaneSegNet/data
python gt_generator.py --data-json ../../datasets/splits/train.json

# 4. Train model
cd ..
./tools/dist_train.sh configs/carla_openlanev2.py 4 --autoscale-lr
```

---

## Directory Structure Reference

```
Carla-OpenLane/
├── datasets/
│   ├── raw/                      # CARLA raw data (Step 1 output)
│   ├── splits/                   # Train/val/test splits (Step 3)
│   └── statistics/               # Dataset statistics
├── models/
│   └── LaneSegNet/               # Model training code
├── scripts/
│   ├── download_dataset.sh       # Download pre-generated data
│   ├── generate_splits.py        # Create train/val splits
│   └── analyze_scene_distribution.py
├── tools/
│   └── data-generation/          # CARLA data generation reference
└── docs/
    ├── FULL_WORKFLOW.md          # This file
    ├── ANNOTATION.md             # Annotation details
    └── DATASET.md                # Dataset specification

OpenLane-V2-HDmap-Converter/      # Separate annotation tool repo
├── src/
│   ├── carla2openlanev2.py
│   └── checksum.py
├── docker/
└── scripts/
```

---

## Troubleshooting

### Common Issues

**Issue:** Path errors in annotation or training

**Solution:**
- Check all paths are absolute or correctly relative
- Verify symlinks point to existing locations
- Update paths in config files

**Issue:** Out of memory during training

**Solution:**
```python
# Reduce batch size in config
samples_per_gpu = 2  # Default: 4
```

**Issue:** Poor model performance

**Solution:**
- Increase dataset size (generate more scenes)
- Augment data (weather, lighting variations)
- Tune hyperparameters (learning rate, epochs)

---

## Performance Benchmarks

**Annotation speed:**
- 200 scenes (6000 frames): 2-4 hours (16-core CPU)
- With map caching: 10-50x faster for same-scene frames

**Training time (LaneSegNet):**
- 4× RTX 3090 (24GB): ~12 hours for 24 epochs
- Batch size: 4 per GPU

**Dataset statistics:**
- ArgoVerse2 subset: 200 scenes, ~6000 frames
- Avg lanes per frame: 26.3
- Avg traffic elements: 3.2

---

## Next Steps

- **Custom scenarios:** Generate data with specific weather/traffic
- **Domain adaptation:** Train on CARLA, fine-tune on real data
- **Model improvements:** Experiment with LaneSegNet variants
- **Visualization:** Use provided tools to inspect predictions

---

## References

- [CARLA Simulator Documentation](https://carla.readthedocs.io/)
- [OpenLane-V2 Dataset Paper](https://github.com/OpenDriveLab/OpenLane-V2)
- [OpenLane-V2-HDmap-Converter](https://github.com/haunjo/OpenLane-V2-HDmap-Converter)
- [LaneSegNet Paper](https://github.com/OpenDriveLab/LaneSegNet)
