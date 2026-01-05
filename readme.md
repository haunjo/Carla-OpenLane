# Carla-OpenLane Dataset

A CARLA-based synthetic dataset generation pipeline for autonomous driving research with multi-camera sensor arrays and traffic light annotations compatible with Argoverse2 and nuScenes formats.

## Installation

### 1. CARLA Simulator Setup

carla 시뮬레이터를 작업환경에 설치하세요. 0.10.0 버전은 호환이 잘 안되니 0.9.15 혹은 0.9.16 버전 권장

Install CARLA 0.9.15 following the official documentation:
- [CARLA Documentation](https://carla.readthedocs.io/en/0.9.15/)
- Setup includes: simulator installation, Python environment configuration, and dependency installation

### 2. Project Setup


carla 데이터 어노테이션 툴을 설치합니다. 도커 tar 이미지를 다운로드해 주세요. 

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

run.sh 이 데이터 생성을 위한 통합 스크립트입니다. 실행해 주세요

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

생성한 데이터에 OpenLane-V2 형식의 주석을 입힙니다. 도커 기반으로 실행되고 argument가 많으니 참고해주세요

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

./LaneSegNet/data 의 gt_generator.py 와 gt_generator_centerline.py 를 적절히 활용

## Training Model

cd LaneSegNet/

./tools/dist_train 1 --autoscale-lr

## Test Model

cd LaneSegNet/

./tools/dist_test 1 --show