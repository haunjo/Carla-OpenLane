# Data Annotation Guide

This guide explains how to annotate CARLA-generated raw data into OpenLane-V2 format using the **OpenLane-V2-HDmap-Converter** tool.

---

## Overview

The annotation process converts:
- **Input:** CARLA raw data (JSON metadata + images + OpenDRIVE maps)
- **Output:** OpenLane-V2 format annotations (lane segments, topology, traffic elements)

**Annotation Tool Repository:** [OpenLane-V2-HDmap-Converter](https://github.com/haunjo/OpenLane-V2-HDmap-Converter)

### Annotation Pipeline

For detailed pipeline architecture, see the [converter's documentation](https://github.com/haunjo/OpenLane-V2-HDmap-Converter#pipeline-architecture).

---

## Prerequisites

- **Docker:** Version 20.10+
- **CARLA raw data:** Generated from CARLA simulator (see [FULL_WORKFLOW.md](FULL_WORKFLOW.md))
- **Storage:** 10GB+ free space for annotation output

---

## Installation

### Step 1: Clone Annotation Tool

```bash
git clone https://github.com/haunjo/OpenLane-V2-HDmap-Converter.git
cd OpenLane-V2-HDmap-Converter
```

### Step 2: Download Docker Image

Download the pre-built Docker image (~15 GB):

**[📦 Download lanelet2.tar.gz (Google Drive)](https://drive.google.com/file/d/YOUR_DOCKER_IMAGE_LINK)**

### Step 3: Load Docker Image

```bash
docker load < lanelet2.tar.gz
docker images  # Verify image loaded
```

---

## Quick Start

### 1. Configure Dataset Path

Edit `docker/run_docker.sh` to point to your CARLA raw data:

```bash
# Example: Point to your dataset location
DATASET=/path/to/Carla-OpenLane/datasets/raw
```

The expected raw data structure:
```
datasets/raw/
├── train/
│   ├── 0000/
│   │   ├── info/
│   │   │   ├── 0.json
│   │   │   └── 1.json
│   │   └── image_2/
│   │       ├── 0.jpg
│   │       └── 1.jpg
│   └── 0001/
└── val/
```

### 2. Run Docker Container

```bash
cd docker
chmod +x run_docker.sh
./run_docker.sh
```

This mounts your dataset into the container and starts an interactive shell.

### 3. Run Annotation

Inside the Docker container:

```bash
cd /home/developer/workspace/scripts

# Full annotation (recommended)
./annotation.sh --split train --subset argoverse2 --task segment

# Centerline-only (faster, less detailed)
./annotation.sh --split train --subset argoverse2 --task centerline
```

**Arguments:**
- `--split`: `train` or `val`
- `--subset`: `argoverse2` (scenes 0-199) or `nuscenes` (scenes 1000-1199)
- `--task`: `segment` (full) or `centerline` (lightweight)
- `--segment-scope`: `missing` (skip existing) or `all` (reprocess everything)

### 4. Monitor Progress

The annotation process logs:
```
Processing scene 0000/0199...
  Frame 0/30: ✓ 26 lanes, 3 traffic elements
  Frame 1/30: ✓ 24 lanes, 2 traffic elements
...
```

**Output location:** `datasets/raw/train/{scene}/info/{frame}-ls.json`

---

## Validation

After annotation, validate the output:

```bash
# Exit Docker container
exit

# Run validation script
cd OpenLane-V2-HDmap-Converter
python src/checksum.py --root /path/to/datasets/raw/train
```

**Example output:**
```
Scanning: 100%|████████████| 200/200 scenes
✓ Total frames: 6000
✓ Valid annotations: 5987 (99.8%)
✗ Missing/Invalid: 13 frames

Statistics:
  - Avg lanes per frame: 26.3
  - Avg traffic elements: 3.2
  - Intersection ratio: 18.5%

Missing files logged to: projects/missing_ls.log
```

---

## Output Format

Each annotated frame produces a `*-ls.json` file:

```json
{
  "lane_segment": [
    {
      "id": 0,
      "centerline": [[x1, y1, z1], [x2, y2, z2], ...],
      "left_laneline": {
        "points": [[x1, y1, z1], ...],
        "type": "solid"
      },
      "right_laneline": { ... },
      "is_intersection": false
    }
  ],
  "area": [
    {
      "id": 0,
      "category": 2,  // 1: crosswalk, 2: road border
      "points": [[x1, y1, z1], ...]
    }
  ],
  "topology_lsls": [[0, 1, 0], [0, 0, 1], ...],  // N×N lane connectivity
  "topology_lste": [[1, 0], [0, 1], ...]         // N×M lane-traffic association
}
```

---

## Advanced Usage

### Custom Annotation Parameters

Modify `src/carla2openlanev2.py` for custom behavior:

```python
# Line 303: Adjust camera visibility threshold
VISIBILITY_DISTANCE = 5.0  # meters

# Line 384: Filter minimum lane points
MIN_LANE_POINTS = 2
```

### Batch Processing

Process multiple splits in parallel:

```bash
# Inside Docker container
./annotation.sh --split train --subset argoverse2 --task segment &
./annotation.sh --split val --subset argoverse2 --task segment &
wait
```

### Resume Interrupted Annotation

Use `--segment-scope missing` to skip already-processed frames:

```bash
./annotation.sh --split train --subset argoverse2 --task segment --segment-scope missing
```

---

## Troubleshooting

### Issue: "No such file or directory" errors

**Cause:** Incorrect dataset path in `run_docker.sh`

**Solution:**
1. Verify your dataset path exists
2. Check `docker/run_docker.sh` mounts correct directory
3. Ensure read/write permissions

### Issue: Missing lane segments in output

**Cause:** Lanes filtered by camera visibility

**Solution:**
1. Check camera calibration in input JSON
2. Increase `VISIBILITY_DISTANCE` threshold
3. Verify ego pose is correct

### Issue: Docker container fails to start

**Cause:** Insufficient resources

**Solution:**
```bash
# Check Docker resources
docker system df

# Prune unused images/containers
docker system prune -a
```

---

## Performance Tips

- **Map caching:** Processing frames from the same scene is 10-50x faster (reuses converted maps)
- **Subset filtering:** Use `--subset` to process only specific scenes
- **Parallel processing:** Run multiple annotation jobs on different splits

**Estimated time:**
- 200 scenes (6000 frames) ≈ 2-4 hours on 16-core CPU

---

## Next Steps

After annotation:

1. **Validate annotations** (see above)
2. **Prepare training splits** (see [DATASET.md](DATASET.md))
3. **Visualize results** (see examples in annotation tool repo)
4. **Train model** (see [../README.md](../README.md))

---

## References

- [OpenLane-V2-HDmap-Converter GitHub](https://github.com/haunjo/OpenLane-V2-HDmap-Converter)
- [OpenLane-V2-HDmap-Converter Pipeline Details](https://github.com/haunjo/OpenLane-V2-HDmap-Converter/blob/main/docs/PIPELINE.md)
- [OpenLane-V2 Dataset Specification](https://github.com/OpenDriveLab/OpenLane-V2)
