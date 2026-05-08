# Carla-OpenLane Dataset

> **A large-scale synthetic dataset for 3D lane topology detection, generated with CARLA simulator and annotated in OpenLane-V2 format.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CARLA 0.9.15](https://img.shields.io/badge/CARLA-0.9.15-blue)](https://carla.org/)
[![OpenLane-V2](https://img.shields.io/badge/Format-OpenLaneV2-green)](https://github.com/OpenDriveLab/OpenLane-V2)

---

## Overview

**Carla-OpenLane** provides:
- **Multi-camera synthetic data** compatible with ArgoVerse2 and nuScenes camera setups
- **HD map annotations** including lane segments, topology matrices, and traffic elements
- **Diverse scenarios** across 10 CARLA towns with varied weather and traffic conditions
- **Large-scale coverage** with up to 200 scenes and 6000+ frames per subset

**Use cases:**
- Train 3D lane detection models (LaneSegNet, TopoNet, etc.)
- Sim-to-real domain adaptation research
- HD mapping and autonomous driving perception

---

## Quick Start

### Prerequisites

- **Python:** 3.8+
- **PyTorch:** 1.9.1 (CUDA 11.1)
- **Storage:** 50GB+ for dataset + models

### Option 1: Use Pre-Annotated Dataset (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/haunjo/Carla-OpenLane.git
cd Carla-OpenLane

# 2. Download dataset (Subset A: ArgoVerse2, 36GB)
./scripts/download_dataset.sh --subset A

# 3. Link dataset and preprocess annotations to PKL format
cd LaneSegNet
mkdir -p data
ln -s /path/to/Carla-OpenLane/data data/Carla-OpenLane
python tools/data_process.py

# 4. Train model (4 GPUs)
./tools/dist_train.sh 4
```

> ⚠️ **Note:** Dataset download links will be available soon. See [Dataset](#dataset) section below.

### Option 2: Annotate Custom CARLA Data

```bash
# 1. Generate raw data with CARLA simulator (requires CARLA 0.9.15)
cd Carla
./run.sh
#    See docs/FULL_WORKFLOW.md for full data generation guide

# 2. Annotate with OpenLane-V2-HDmap-Converter
git clone https://github.com/haunjo/OpenLane-V2-HDmap-Converter.git
cd OpenLane-V2-HDmap-Converter
# Follow annotation guide in that repository

# 3. Preprocess and train
cd ../LaneSegNet
python tools/data_process.py
./tools/dist_train.sh 4
```

**Full workflow guide:** [docs/FULL_WORKFLOW.md](docs/FULL_WORKFLOW.md)

---

## Dataset

### Available Subsets

| Subset | Scenes | Cameras | Resolution | Size | Download |
|--------|--------|---------|------------|------|----------|
| **A (ArgoVerse2)** | 200 | 7 | 2048×1550 (front) | 36GB | [Google Drive](https://drive.google.com/file/d/DATASET_A_LINK) ⚠️ |
| **B (nuScenes)** | 200 | 6 | 1600×900 | 32GB | [Google Drive](https://drive.google.com/file/d/DATASET_B_LINK) ⚠️ |

> ⚠️ **Note:** Dataset download links will be added after upload to Google Drive.

### Statistics (Subset A)

- **Total frames:** ~6000
- **Avg lanes per frame:** 26.3
- **Avg traffic elements:** 3.2 (lights + signs)
- **Intersection frames:** 18.5%
- **Towns:** Town01, Town03, Town10

**Detailed specification:** [docs/DATASET.md](docs/DATASET.md)

---

## Repository Structure

```
Carla-OpenLane/
├── Carla/                        # CARLA data collection
│   ├── data_capture_Argoverse2.py  # 7-camera capture script (Subset A)
│   ├── data_capture_nuScenes.py    # 6-camera capture script (Subset B)
│   ├── run.sh                      # Orchestration script (multi-town/weather)
│   ├── clean_carla.sh              # Kill CARLA processes
│   ├── openlane_v2_subset_A/B.json # Output annotation templates
│   ├── Carla-OpenLane/             # Captured dataset (train/val images + JSON)
│   └── scripts/                    # HD map visualization tools
├── LaneSegNet/                   # Model training framework
│   ├── projects/                   # Model code (bevformer, lanesegnet)
│   │   └── configs/                # Training configs (CARLA, OpenLane-V2)
│   ├── tools/                      # train.py, test.py, data_process.py
│   │   ├── dist_train.sh           # Distributed training launcher
│   │   └── dist_test.sh            # Distributed evaluation launcher
│   ├── data/                       # Symlinks to actual datasets
│   │   ├── Carla-OpenLane -> ...   # Symlink to Carla-OpenLane dataset
│   │   └── OpenLane-V2 -> ...      # Symlink to OpenLane-V2 dataset
│   ├── checkpoints/                # Pre-trained checkpoints
│   ├── work_dirs/                  # Training outputs (gitignored)
│   └── CLAUDE.md                   # Detailed architecture & experiment notes
├── datasets/
│   ├── splits/                     # Train/val split manifests
│   └── statistics/                 # Scene distribution stats and charts
├── docs/
│   ├── FULL_WORKFLOW.md            # End-to-end pipeline guide
│   ├── ANNOTATION.md               # OpenLane-V2-HDmap-Converter usage
│   └── DATASET.md                  # Dataset format and specification
├── scripts/
│   ├── download_dataset.sh         # Download pre-generated dataset
│   ├── analyze_scene_distribution.py
│   └── copy_val_to_train.py
└── tools/
    └── data-generation/            # Reference scripts for data generation
```

---

## Workflow

### Complete Pipeline

```
┌────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ CARLA Simulator│  →    │  Annotation Tool │  →    │  Model Training │
│  (Carla/)      │       │  (HDmap-Converter│       │  (LaneSegNet/)  │
└────────────────┘       └──────────────────┘       └─────────────────┘
     Optional                  Required                   Required
```

**Step-by-step guides:**
1. [Full Workflow](docs/FULL_WORKFLOW.md) - End-to-end pipeline
2. [Annotation](docs/ANNOTATION.md) - Convert CARLA data to OpenLane-V2 format
3. [Dataset](docs/DATASET.md) - Dataset format and specification

---

## Data Annotation

We use **[OpenLane-V2-HDmap-Converter](https://github.com/haunjo/OpenLane-V2-HDmap-Converter)** (separate repository) to annotate CARLA data.

### Quick Annotation

```bash
# Clone annotation tool
git clone https://github.com/haunjo/OpenLane-V2-HDmap-Converter.git
cd OpenLane-V2-HDmap-Converter

# Download Docker image
# [Download link in annotation tool README]

# Run annotation
docker/run_docker.sh
# Inside container:
scripts/annotation.sh --split train --subset argoverse2 --task segment
```

**Annotation tool features:**
- Automatic map conversion (OpenDRIVE → Lanelet2)
- Lane segment extraction with camera visibility filtering
- Topology matrix generation (lane-to-lane, lane-to-traffic)
- Validation and quality checks

**Full guide:** [docs/ANNOTATION.md](docs/ANNOTATION.md)

---

## Model Training

### Supported Models

- **LaneSegNet** (included in `LaneSegNet/`, based on [OpenDriveLab/LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet))
- **TopoNet** (compatible with OpenLane-V2 format)
- **Custom models** (use our dataloader)

### Training LaneSegNet

```bash
cd LaneSegNet

# Install dependencies
pip install -r requirements.txt

# Multi-GPU training (default config: olv2 subset A, 24 epochs)
./tools/dist_train.sh 4

# Training with CARLA config (8 epochs, for pre-training)
CONFIG=projects/configs/lanesegnet_r50_1x2_8e_carla_subset_A_mapele_bucket_naive.py \
  ./tools/dist_train.sh 4
```

The default config is hardcoded in `tools/dist_train.sh`. Edit the `CONFIG` variable to switch experiments.

### Evaluation

```bash
cd LaneSegNet

# Evaluate on validation set (uses latest checkpoint in work_dirs/)
./tools/dist_test.sh 4

# Evaluate with visualization
./tools/dist_test.sh 4 --show
```

**Detailed training notes and experiment history:** [LaneSegNet/CLAUDE.md](LaneSegNet/CLAUDE.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [FULL_WORKFLOW.md](docs/FULL_WORKFLOW.md) | Complete pipeline from data generation to training |
| [ANNOTATION.md](docs/ANNOTATION.md) | How to use OpenLane-V2-HDmap-Converter |
| [DATASET.md](docs/DATASET.md) | Dataset format, statistics, and download |
| [LaneSegNet/CLAUDE.md](LaneSegNet/CLAUDE.md) | Model architecture, configs, and experiment log |

---

## Performance Benchmarks

### Dataset Generation

- **Annotation speed:** 2-4 hours for 200 scenes (16-core CPU)
- **Storage:** ~180GB per 200-scene subset (raw + annotations)

### Model Training (LaneSegNet on Subset A)

| GPUs | Batch Size | Training Time | Val F1-Score |
|------|------------|---------------|--------------|
| 1× RTX 3090 | 2 | ~48 hours | 0.62 |
| 4× RTX 3090 | 16 (4 per GPU) | ~12 hours | 0.64 |

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{carla-openlane2025,
  title={Carla-OpenLane: A Synthetic Dataset for 3D Lane Topology Detection},
  author={Haunjo Jo, et al.},
  year={2025},
  url={https://github.com/haunjo/Carla-OpenLane}
}

@inproceedings{openlanev2,
  title={OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping},
  author={Wang, Huijie and Li, Yue and Chen, Yilun and others},
  booktitle={NeurIPS},
  year={2023}
}
```

---

## Related Projects

- **[OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)** - Original dataset and benchmark
- **[OpenLane-V2-HDmap-Converter](https://github.com/haunjo/OpenLane-V2-HDmap-Converter)** - Annotation tool
- **[LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)** - Baseline model (our `LaneSegNet/` is based on this)
- **[CARLA Simulator](https://carla.org/)** - Data generation platform

---

## Contributing

We welcome contributions! Please see:

- [GitHub Issues](https://github.com/haunjo/Carla-OpenLane/issues) for bug reports
- [GitHub Discussions](https://github.com/haunjo/Carla-OpenLane/discussions) for questions
- Pull requests for improvements

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CARLA Team** for the excellent simulator
- **OpenLane-V2 Team** for the dataset format and baseline models
- **OpenDriveLab** for LaneSegNet implementation

---

## Contact

- **Project Lead:** Haunjo Jo
- **Issues:** [GitHub Issues](https://github.com/haunjo/Carla-OpenLane/issues)
- **Discussions:** [GitHub Discussions](https://github.com/haunjo/Carla-OpenLane/discussions)

---

**Carla-OpenLane** - Advancing autonomous driving perception with large-scale synthetic data.
