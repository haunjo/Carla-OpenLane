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
- **PyTorch:** 1.10+
- **Storage:** 50GB+ for dataset + models

### Option 1: Use Pre-Annotated Dataset (Recommended)

```bash
# Clone repository
git clone https://github.com/haunjo/Carla-OpenLane.git
cd Carla-OpenLane

# Download pre-annotated dataset (Subset A)
# Note: Dataset will be available soon
# ./scripts/download_dataset.sh --subset A

# For now, prepare with your own annotated data
# Follow Option 2 to annotate CARLA data first

# Prepare training data
cd LaneSegNet/data
python gt_generator.py --data-json ../../datasets/splits/train.json

# Train model
cd ..
./tools/dist_train.sh configs/carla_openlanev2.py 4 --autoscale-lr
```

### Option 2: Full Pipeline (Custom Data)

```bash
# 1. Generate CARLA data (requires CARLA 0.9.15)
#    See docs/FULL_WORKFLOW.md for details

# 2. Annotate with OpenLane-V2-HDmap-Converter
git clone https://github.com/haunjo/OpenLane-V2-HDmap-Converter.git
cd OpenLane-V2-HDmap-Converter
# Follow annotation guide: docs/ANNOTATION.md

# 3. Return to Carla-OpenLane and train
cd ../Carla-OpenLane/LaneSegNet
./tools/dist_train.sh configs/carla_openlanev2.py 4
```

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
├── datasets/
│   ├── raw/                      # CARLA raw data + annotations
│   ├── splits/                   # Train/val/test manifests
│   └── statistics/               # Dataset analysis
├── models/
│   └── LaneSegNet/               # Model training code
├── scripts/
│   ├── download_dataset.sh       # Download pre-generated data
│   ├── generate_splits.py        # Create data splits
│   └── analyze_scene_distribution.py
├── tools/
│   └── data-generation/          # CARLA data generation reference
├── docs/
│   ├── FULL_WORKFLOW.md          # Complete pipeline guide
│   ├── ANNOTATION.md             # Annotation tool usage
│   ├── DATASET.md                # Dataset specification
│   └── TRAINING.md               # Model training guide
└── README.md                     # This file
```

---

## Workflow

### Complete Pipeline

```
┌────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ CARLA Simulator│  →    │  Annotation Tool │  →    │  Model Training │
│  (Data Gen)    │       │  (OpenLane-V2)   │       │   (LaneSegNet)  │
└────────────────┘       └──────────────────┘       └─────────────────┘
     Optional                  Required                   Required
```

**Step-by-step guides:**
1. [Full Workflow](docs/FULL_WORKFLOW.md) - End-to-end pipeline
2. [Annotation](docs/ANNOTATION.md) - Convert CARLA data to OpenLane-V2 format
3. [Training](docs/TRAINING.md) - Train lane detection models

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

- **LaneSegNet** (Included in `models/LaneSegNet/`)
- **TopoNet** (Compatible with OpenLane-V2 format)
- **Custom models** (Use our dataloader)

### Training LaneSegNet

```bash
cd LaneSegNet

# Install dependencies
pip install -r requirements.txt

# Single GPU
python tools/train.py configs/carla_openlanev2.py

# Multi-GPU (4 GPUs, auto-scale learning rate)
./tools/dist_train.sh configs/carla_openlanev2.py 4 --autoscale-lr
```

### Evaluation

```bash
# Evaluate on validation set
python tools/test.py configs/carla_openlanev2.py work_dirs/latest.pth --eval

# Visualize predictions
./tools/dist_test.sh configs/carla_openlanev2.py work_dirs/latest.pth 1 --show
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [FULL_WORKFLOW.md](docs/FULL_WORKFLOW.md) | Complete pipeline from data generation to training |
| [ANNOTATION.md](docs/ANNOTATION.md) | How to use OpenLane-V2-HDmap-Converter |
| [DATASET.md](docs/DATASET.md) | Dataset format, statistics, and download |
| [TRAINING.md](docs/TRAINING.md) | Model training and evaluation guide |

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
- **[LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)** - Baseline model
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
