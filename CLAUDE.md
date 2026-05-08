# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sub-projects

- **CARLA Data Collection** (`Carla/`): Generates synthetic driving datasets in Argoverse2/nuScenes formats. See below.
- **LaneSegNet** (`LaneSegNet/`): Lane segment perception baseline. See @LaneSegNet/CLAUDE.md.
- **TopoLogic** (`TopoLogic/`): NeurIPS 2024 lane topology reasoning baseline. See @TopoLogic/CLAUDE.md.
- **TopoNet** (`TopoNet/`): CVPR 2023 graph-based topology reasoning baseline. See @TopoNet/CLAUDE.md.

## Project Overview

This is a **CARLA-based data collection system** for generating autonomous driving datasets compatible with **Argoverse2** and **nuScenes** formats. The project generates synthetic driving data with multi-camera sensor arrays, traffic light annotations, and scene descriptions for lane topology reasoning research.

## Running Data Collection

### Prerequisites
- CARLA simulator must be installed and accessible via `./CarlaUE4.sh`
- Conda environment named `carla` with required dependencies (carla, numpy, opencv-python)
- Python API utilities in `./PythonAPI/util/config.py`

### Primary Commands

**Run data collection with default settings:**
```bash
./run.sh
```

**Run for specific towns:**
```bash
./run.sh Town01 Town03
# or
./run.sh 1 3
# or
TOWNS="1,3,7,10" ./run.sh
```

**Run single data capture script manually:**
```bash
# Argoverse2 format (7 ring cameras)
python data_capture_Argoverse2.py --scene 15 --sample 10 --spawn-offset 0 \
  --traffic-level 1 --weather ClearNoon --dir train --pose-format openlanev2

# nuScenes format (6 cameras)
python data_capture_nuScenes.py --scene 15 --sample 10 --spawn-offset 0 \
  --traffic-level 1 --weather ClearNoon --dir train --pose-format openlanev2
```

**Clean up CARLA processes:**
```bash
./clean_carla.sh
```

### Key Arguments
- `--scene N`: Number of scenes per cycle (default: 15)
- `--sample N`: Number of waypoint samples per scene (default: 10)
- `--spawn-offset N`: Deterministic spawn point selection offset
- `--traffic-level {1,2}`: 1=low traffic, 2=high traffic (~50% of spawn points)
- `--weather PRESET`: One of 17 weather presets (ClearNoon, HardRainSunset, etc.)
- `--dir {train,val}`: Output directory for collected data
- `--pose-format {openlanev2,carla-matrix,carla-rpy}`: Pose encoding format

## Architecture

### Data Capture Scripts

Two parallel implementations for different sensor configurations:

1. **`data_capture_Argoverse2.py`** - Argoverse2-style 7-camera ring setup
   - Outputs to `openlane_v2_subset_A.json`
   - Cameras: ring_front_center, ring_front_left/right, ring_rear_left/right, ring_side_left/right

2. **`data_capture_nuScenes.py`** - nuScenes-style 6-camera setup
   - Outputs to `openlane_v2_subset_B.json`
   - Cameras: CAM_FRONT, CAM_FRONT_LEFT/RIGHT, CAM_BACK, CAM_BACK_LEFT/RIGHT

### Core Workflow

Both scripts follow the same pipeline:

1. **Initialization**: Connect to CARLA, set synchronous mode, configure weather
2. **Scene Generation**: Generate scene description tokens (time_of_day, weather, road_type)
3. **Spawn Vehicle**: Place hero vehicle at deterministic spawn point using `--spawn-offset`
4. **Waypoint Sampling**: Generate N random waypoints along drivable roads
5. **Sensor Setup**: Attach camera sensors with dataset-specific configurations
6. **Traffic Spawning**: Add NPC vehicles based on `--traffic-level`
7. **Data Capture Loop**: For each waypoint:
   - Teleport vehicle to waypoint
   - Clear nearby NPC vehicles (radius=3m)
   - Wait for sensor stabilization
   - Capture all camera images synchronously
   - Detect and annotate traffic lights in front camera FOV
   - Save sensor data and annotations to disk
8. **Cleanup**: Destroy sensors, vehicle, and NPCs

### Key Functions

**`generate_scene_tokens(world, map_name, weather_preset)`** (lines 154-252 in Argoverse2, 146-244 in nuScenes)
- Generates categorical scene descriptors from weather preset and map name
- Returns: `{categories: {time_of_day, weather, road_type}, tokens: [...], text: "..."}`

**`filter_traffic_lights(vehicle, K, world_2_camera, image_w, image_h)`** (lines 503-554 in Argoverse2, 495-546 in nuScenes)
- Projects traffic light bounding boxes into front camera image space
- Filters by: distance (3-50m), FOV, facing direction (75° threshold), projection validity
- Returns list of traffic element annotations with affected lane waypoints

**`save_annotations(vehicle, frame, save_path, traffics, name, pose_format, scene_tokens)`** (lines 357-414 in Argoverse2, 349-406 in nuScenes)
- Converts CARLA pose to OpenLane coordinate frame when `pose_format='openlanev2'`
- Saves per-frame JSON with pose, sensor metadata, traffic annotations, scene tokens

**`carla_to_openlane_pose(x, y, z, roll, pitch, yaw)`** (lines 330-355 in Argoverse2, 322-347 in nuScenes)
- Transforms CARLA right-handed coordinate system to OpenLane format
- Applies Y-axis flip and ensures proper rotation matrix via SVD

### Output Structure

```
{train,val}/
  ├── 0001/
  │   ├── ring_front_center/{frame}.jpg  (or CAM_FRONT/)
  │   ├── ring_front_left/{frame}.jpg    (or CAM_FRONT_LEFT/)
  │   ├── ...
  │   └── info/{frame}.json              # Annotations with pose, traffic, scene tokens
  ├── 0002/
  └── ...
```

### Synchronization Model

- **Synchronous mode** enabled: `settings.fixed_delta_seconds = 1.0 / fps`
- Traffic manager synchronized to CARLA world ticks
- Sensor data validated by frame ID matching: `if frame_id != frame: continue`
- Multi-threaded disk I/O (max 7 threads) with join barriers

### Scene Description System

Scene tokens encode environmental conditions for conditional generation:
- **time_of_day**: {noon, sunset, night} - extracted from weather preset or sun altitude
- **weather**: {clear, cloudy, soft_rain, mid_rain, hard_rain, wet, wet_cloudy}
- **road_type**: {urban, suburban, downtown, highway} - mapped from Town ID

### Traffic Light Detection

Detection pipeline:
1. Find all traffic lights in world via `'traffic_light' in a.type_id`
2. Filter by distance (3-50m) and forward-facing direction
3. Iterate through light boxes per traffic light actor
4. Check facing angle (vehicle must see light face, not back)
5. Project 3D bounding box vertices to 2D image plane
6. Validate projection and bounding box sanity (min size, max area ratio)
7. Store with state attribute (Red=1, Green=2, Yellow=3) and affected lane waypoints

### Coordinate Systems

- **CARLA**: Right-handed, X-forward, Y-right, Z-up
- **OpenLane**: Y-axis flipped relative to CARLA
- **Camera**: [Y, -Z, X] permutation for image projection (lines 423 in Argoverse2, 415 in nuScenes)

## Orchestration Script (run.sh)

- Manages CARLA server lifecycle (start, wait, kill)
- Iterates over towns, traffic levels, and scene cycles
- Calculates deterministic spawn offsets per configuration
- Rotates through 17 weather presets cyclically
- Supports flexible town selection via args or `TOWNS` environment variable
