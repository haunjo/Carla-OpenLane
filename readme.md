# Carla-OpenLane Dataset

A CARLA-based synthetic dataset generation pipeline for autonomous driving research with multi-camera sensor arrays and traffic light annotations compatible with Argoverse2 and nuScenes formats.


## Warning

### 1. 데이터 생성 및 주석 시 주의할 점
본 레포지토리는 데이터 생성(Carla), 데이터 주석(OpenLane_V2-HDmap-Converter), 모델 학습(LaneSegNet)이 개별적으로 있지만 각 파일마다 경로 설정이 아직 USER가 쓰기에는 불편하게 설정되어 있습니다.
스크립트에서 오류가 난다면 높은 확률로 경로설정 문제이니 참고해주세요.

### 2. 데이터셋
데이터 생성&주석의 샘플이 NAS에 있습니다. NAS를 통해서 먼저 데이터셋 구조를 파악하세요.

## Installation

### 1. CARLA Simulator Setup

carla 시뮬레이터를 작업환경에 설치하세요. 0.10.0 버전은 호환이 잘 안되니 0.9.15 혹은 0.9.16 버전 권장

Install CARLA 0.9.15 following the official documentation:
- [CARLA Documentation](https://carla.readthedocs.io/en/0.9.15/)
- Setup includes: simulator installation, Python environment configuration, and dependency installation

### 2. Model Setup

- OpenLane-V2 : https://github.com/OpenDriveLab/OpenLane-V2
    - warning : OpenLane-V2 설치 도중 오류가 나오는데, 의존성 패키지 중 ortools 버전이 삭제되었기 때문에 그 다음 버전 설치를 권장드립니다. 
- LaneSegNet : https://github.com/OpenDriveLab/LaneSegNet 


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

먼저 GitHub Release의 v1.0 tag가 붙은 버전에서 .tar 파일을 다운로드하고 압축해제 해 주세요. 

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

```
cd LaneSegNet/

./tools/dist_train 1 --autoscale-lr
```

## Test Model

```
cd LaneSegNet/

./tools/dist_test 1 --show
```

## Git 협업 방법

main branch : 공동 작업 공간

개인 작업 공간 만드는 방법

git checkout -b bonghun

브랜치 새로 생성됨

git commit -a -m " 변경 내용을 텍스트로 정리 "

git push origin bonghun
