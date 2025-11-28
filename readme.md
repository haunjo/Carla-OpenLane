# Carla-OpenLane Dataset

A CARLA-based synthetic dataset generation pipeline for autonomous driving research with multi-camera sensor arrays and traffic light annotations compatible with Argoverse2 and nuScenes formats.

## Installation

### 1. CARLA Simulator Setup

Install CARLA 0.9.15 following the official documentation:
- [CARLA Documentation](https://carla.readthedocs.io/en/0.9.15/)
- Setup includes: simulator installation, Python environment configuration, and dependency installation

### 2. Project Setup

```bash
# Download project folder from release
# (Release URL to be added)

# Download Docker image tar file from NAS
# (NAS path to be added)

# Load Docker image
docker load < olv2Xhdmap.tar

docker images
```

## Data Generation

Run data collection with `./run.sh`:

```bash
# Default settings (see run.sh for configuration)
./run.sh

# Run for specific towns
./run.sh Town01 Town03
# or
TOWNS="1,3,7,10" ./run.sh
```

Refer to `run.sh` for detailed generation pipeline including:
- CARLA server lifecycle management
- Town iteration and weather preset rotation
- Traffic level configuration
- Spawn offset calculation

## Data Annotation

Run annotation pipeline in Docker container:

```bash
# Navigate to annotation tool directory
cd OpenLane-V2-HDmap-Converter

# Run Docker container with annotation script
./run_docker.sh

# Inside container, run annotation
./annotation.sh
```

Refer to `annotation.sh` for annotation pipeline details.

## Data Visualization

*(To be updated)*

## Training Model

*(To be updated)*

## Citation

*(To be added)*

## License

*(To be added)*
