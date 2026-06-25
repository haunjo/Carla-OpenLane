#!/usr/bin/env python3
# autopilot_data_capture_unified_BH_v2.py
"""
🔀 UNIFIED DATA CAPTURE SCRIPT - Supports Both Argoverse2 and nuScenes Formats

This is a SINGLE script that dynamically supports TWO multi-camera configurations via --subset:

📷 SUBSET A (Argoverse2): 7-camera ring setup
   python data_capture_unified_BH_0222.py --subset argoverse2 ...
   Output: openlane_v2_subset_A.json + 7 camera streams

📷 SUBSET B (nuScenes): 6-camera setup
   python data_capture_unified_BH_0222.py --subset nuscenes ...
   Output: openlane_v2_subset_B.json + 6 camera streams

⚡ UNIFIED FEATURES:
   - Single codebase handles both sensor configs (Argoverse2/nuScenes)
   - Dynamic weather system: presets or extreme/realistic sampling
   - Advanced traffic control: local+distant vehicle distribution
   - Improved traffic light detection: facing angle, oncoming filtering
   - SVD-based pose conversion for numerical stability

===============================
🔄 CHANGES FROM ORIGINAL (data_capture_Argoverse2.py)
===============================

1️⃣ SENSOR CONFIGURATION (CRITICAL CHANGE)
   - Split SENSORS into SENSORS_ARGOVERSE2 (7-cam ring) and SENSORS_NUSCENES (6-cam)
   - Dynamic sensor selection via --subset parameter (default: argoverse2)
   - Enables SINGLE script for multiple formats (no need for separate files)

2️⃣ WEATHER SYSTEM (MAJOR ENHANCEMENT)
   - Added 'HardRainyNight' preset (18 total instead of 17)
   - New apply_custom_weather(severity, sun_alt): continuous weather control (0.0-1.0 scale)
   - New sample_extreme_real_weather(): realistic weather sampling from 10 weather types
   - Support for --severity and --sun-alt arguments for custom weather control
   - Weather types: clear, fog_light/mid/heavy, rain_light/mid/heavy, morning, night, afternoon
   - "extreme" weather mode samples realistic weather distributions

3️⃣ TRAFFIC SYSTEM (IMPROVED)
   - New spawn_mixed_traffic(): spawn NPC vehicles with local vs distant distribution
   - Arguments: --total-traffic, --local-traffic, --radius for flexible traffic control
   - Replaces simple traffic_level binary (1/2) with continuous control

4️⃣ TRAFFIC LIGHT DETECTION (REWRITTEN)
   - Complete refactor of filter_traffic_lights() function
   - Extended distance threshold: 50m → 80m for better TL coverage
   - Added facing detection: distinguishes visible (facing=True) vs obscured (facing=False) signals
   - Added oncoming lane filtering: excludes opposite-direction lanes (dot < -0.5)
   - Keeps bbox for all visible TLs even when signal not facing (attribute=0/unknown)

5️⃣ POSE CONVERSION (IMPROVED)
   - carla_to_openlane_pose() now uses SVD orthogonalization
   - Ensures rotation matrix is valid (det=+1) after frame transformation
   - More numerically stable than previous flip_Y sandwich conjugation

6️⃣ DATA SAVING (SIMPLIFIED)
   - save_data_to_disk() now saves RGB images only (removed Lidar, Semantic Lidar, RADAR, GNSS, IMU)
   - Adds SENSORS parameter to save_annotations() for dynamic sensor list
   - Adds subset parameter to distinguish Argoverse2 vs nuScenes format

7️⃣ WAYPOINT GENERATION (NEW)
   - New build_teleport_waypoints(): generates diverse waypoint paths
   - Supports: prefer_complex (junctions/turns), min_dist_m spacing, sample_num count
   - More controllable than original random waypoint selection

8️⃣ ADDITIONAL UTILITIES
   - flush_sensor_queue(): clear accumulated sensor queue frames
   - _safe_is_junction(), _is_turn_candidate(): robust waypoint filtering
   - spawn_mixed_traffic(): improved NPC spawning with local/distant distribution

9️⃣ NEW ARGUMENTS
   - --subset: argoverse2 (default) or nuscenes ← MAIN SELECTOR FOR FORMAT
   - --weather: preset name, 'auto'/'random', or 'extreme' for realistic sampling
   - --severity, --sun-alt: custom weather parameters
   - --total-traffic, --local-traffic: fine-grained traffic control
   - --radius, --step, --min-dist, --prefer-complex: waypoint customization
   - --tm-port: traffic manager port override

🎯 USAGE EXAMPLES:

   # SubsetA (Argoverse2, 7-cam) with diverse weather and traffic
   python data_capture_unified_BH_0222.py \\
     --subset argoverse2 --weather extreme \\
     --total-traffic 50 --local-traffic 20 --sample 10 --dir train

   # SubsetB (nuScenes, 6-cam) with specific weather (fog), 15 waypoints
   python data_capture_unified_BH_0222.py \\
     --subset nuscenes --weather fog_heavy \\
     --sample 15 --dir train

   # Random weather sampling (will pick clear/fog/rain/night variants)
   python data_capture_unified_BH_0222.py --subset nuscenes --weather auto --dir val
"""
import zlib
import os
import sys
import gc
import time
import math
import json
import argparse
import numpy as np
import cv2
from queue import Queue, Empty
import threading

import carla
import numpy.random as random

# CHANGE: Added multi-subset support - separate sensor configurations for Argoverse2 vs nuScenes
# =========================
# Argoverse2 7-camera (MODIFIED: now supports dynamic subset selection)
# =========================
SENSORS_ARGOVERSE2 = [
    ['ring_front_center', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1550, 'image_size_y': 2048, 'fov': 50,
        'x': 0.53, 'y': 0.00, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    }],
    ['ring_front_left', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.45, 'y': -0.2, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0
    }],
    ['ring_front_right', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.45, 'y': 0.2, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0
    }],
    ['ring_rear_left', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.0, 'y': -0.12, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -153.0
    }],
    ['ring_rear_right', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.0, 'y': 0.12, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 153.0
    }],
    ['ring_side_left', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.21, 'y': -0.27, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -99.2
    }],
    ['ring_side_right', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
        'x': 0.21, 'y': 0.27, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 99.2
    }],
]

# =========================
# nuScenes 6-camera
# =========================
SENSORS_NUSCENES = [
    ['CAM_FRONT', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
        'x': 1.70, 'y': 0.00, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    }],
    ['CAM_FRONT_LEFT', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
        'x': 1.35, 'y': -0.49, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0
    }],
    ['CAM_FRONT_RIGHT', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
        'x': 1.35, 'y': 0.49, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0
    }],
    ['CAM_BACK', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 110,
        'x': -0.50, 'y': 0.00, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
    }],
    ['CAM_BACK_LEFT', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
        'x': -0.32, 'y': -0.49, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0
    }],
    ['CAM_BACK_RIGHT', {
        'bp': 'sensor.camera.rgb',
        'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
        'x': -0.32, 'y': 0.49, 'z': 1.51, 'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0
    }],
]

WEATHER_PRESETS = [
    'ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
    'SoftRainNoon', 'SoftRainSunset', 'MidRainyNoon', 'WetNoon',
    'WetSunset', 'WetCloudyNoon', 'MidRainSunset', 'WetCloudySunset',
    'HardRainNoon', 'HardRainSunset', 'ClearNight', 'CloudyNight', 'MidRainyNight',
    'HardRainyNight'  # CHANGE: Added night rain for more weather diversity (was 17, now 18 presets)
]

TL_ATTRIBUTE = {
    carla.TrafficLightState.Off: 0,
    carla.TrafficLightState.Unknown: 0,
    carla.TrafficLightState.Red: 1,
    carla.TrafficLightState.Green: 2,
    carla.TrafficLightState.Yellow: 3,
    'go_straight': 4, 'turn_left': 5, 'turn_right': 6,
    'no_left_turn': 7, 'no_right_turn': 8, 'u_turn': 9,
    'no_u_turn': 10, 'slight_left': 11, 'slight_right': 12
}


def get_next_folder_path(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    existing = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    next_number = int(sorted(existing)[-1]) + 1 if existing else 1
    next_path = os.path.join(root_dir, f"{next_number:04d}")
    os.makedirs(next_path, exist_ok=False)
    return next_path


def generate_scene_tokens(world, map_name, weather_preset=None):
    scene_dict = {}
    if weather_preset:
        preset_lower = weather_preset.lower()
        if 'noon' in preset_lower:
            scene_dict['time_of_day'] = 'noon'
        elif 'sunset' in preset_lower:
            scene_dict['time_of_day'] = 'sunset'
        elif 'night' in preset_lower:
            scene_dict['time_of_day'] = 'night'
        else:
            scene_dict['time_of_day'] = 'noon'
        if 'hardrain' in preset_lower:
            scene_dict['weather'] = 'hard_rain'
        elif 'midrain' in preset_lower:
            scene_dict['weather'] = 'mid_rain'
        elif 'softrain' in preset_lower:
            scene_dict['weather'] = 'soft_rain'
        elif 'wetcloudy' in preset_lower:
            scene_dict['weather'] = 'wet_cloudy'
        elif 'wet' in preset_lower:
            scene_dict['weather'] = 'wet'
        elif 'cloudy' in preset_lower:
            scene_dict['weather'] = 'cloudy'
        elif 'clear' in preset_lower:
            scene_dict['weather'] = 'clear'
        elif 'fog' in preset_lower:  # CHANGE: Added fog condition support
            scene_dict['weather'] = 'fog'
        else:
            scene_dict['weather'] = 'clear'
    else:
        weather = world.get_weather()
        sun_altitude = weather.sun_altitude_angle
        if sun_altitude > 40:
            scene_dict['time_of_day'] = 'noon'
        elif sun_altitude > -10:
            scene_dict['time_of_day'] = 'sunset'
        else:
            scene_dict['time_of_day'] = 'night'
        if weather.precipitation > 50:
            scene_dict['weather'] = 'hard_rain'
        elif weather.precipitation > 20:
            scene_dict['weather'] = 'mid_rain'
        elif weather.precipitation > 5:
            scene_dict['weather'] = 'soft_rain'
        elif weather.wetness > 50:
            scene_dict['weather'] = 'wet'
        elif weather.cloudiness > 50:
            scene_dict['weather'] = 'cloudy'
        else:
            scene_dict['weather'] = 'clear'

    map_lower = map_name.lower()
    if "town01" in map_lower or "town02" in map_lower:
        scene_dict['road_type'] = 'urban'
    elif "town03" in map_lower or "town05" in map_lower or "town10" in map_lower:
        scene_dict['road_type'] = 'downtown'
    elif "town04" in map_lower:
        scene_dict['road_type'] = 'highway'
    elif "town06" in map_lower or "town07" in map_lower:
        scene_dict['road_type'] = 'suburban'
    else:
        scene_dict['road_type'] = 'urban'

    return {
        "categories": scene_dict,
        "tokens": list(scene_dict.values()),
        "text": " | ".join([f"{k}:{v}" for k, v in scene_dict.items()])
    }


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def modify_vehicle_physics(actor):
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass


def save_data_to_disk(sensor_id, frame, data, endpoint, imu_data=None):
    # CHANGE: Simplified to only save RGB camera images (removed Lidar, Semantic Lidar, RADAR, GNSS, IMU)
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.jpg"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        cv2.imwrite(sensor_endpoint, img[:, :, :3])  # CHANGE: RGB only (drop alpha channel)
    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', unknown type '{type(data)}'.")


def carla_to_openlane_pose(x, y, z, roll_deg, pitch_deg, yaw_deg):
    # CHANGE: Use SVD to ensure rotation matrix orthogonality (was: direct flip_Y sandwich conjugation)
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    R_carla = Rz @ Ry @ Rx
    flip_Y = np.diag([1, -1, 1])
    R_openlane = flip_Y @ R_carla
    # SVD orthogonalization: ensures R_openlane is a valid rotation matrix (det=+1)
    U, _, Vt = np.linalg.svd(R_openlane)
    R_openlane_fixed = U @ Vt
    if np.linalg.det(R_openlane_fixed) < 0:
        Vt[-1, :] *= -1
        R_openlane_fixed = U @ Vt
    t_openlane = flip_Y @ np.array([x, y, z], dtype=float)
    return R_openlane_fixed.tolist(), t_openlane.tolist()


def save_annotations(vehicle, frame, save_path, traffics, name, SENSORS,
                     pose_format='openlanev2', scene_tokens=None, subset='argoverse2'):
    # CHANGE: Added SENSORS parameter (passed in) and subset parameter for multi-format support
    pose_path = "openlane_v2_subset_A.json" if subset == 'argoverse2' else "openlane_v2_subset_B.json"
    source_label = "Carla_Argoverse2" if subset == 'argoverse2' else "Carla_nuScenes"

    os.makedirs(save_path, exist_ok=True)
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation

    R, t = carla_to_openlane_pose(
        round(location.x, 6), round(location.y, 6), round(location.z, 6),
        rotation.roll, rotation.pitch, rotation.yaw
    )
    euler_deg = [rotation.roll, rotation.pitch, rotation.yaw]

    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            anno = json.load(f)
    else:
        anno = {}

    anno.setdefault('pose', {})
    anno.setdefault('sensor', {})
    anno.setdefault('annotation', {})
    anno['annotation'].setdefault('traffic_element', [])
    anno.setdefault('meta_data', {})

    anno['meta_data']['source'] = source_label
    anno['meta_data']['source_id'] = f"{name}/frame_{frame}"
    if scene_tokens is not None:
        anno['meta_data']['scene_description'] = scene_tokens

    anno['pose']['rotation'] = R
    anno['pose']['translation'] = t
    anno['pose']['euler_deg'] = euler_deg
    anno['timestamp'] = frame
    anno['segment_id'] = os.path.basename(save_path)

    # CHANGE: Use SENSORS parameter passed from caller (dynamic based on subset)
    for sensor_id, _cfg in SENSORS:
        anno['sensor'].setdefault(sensor_id, {})
        anno['sensor'][sensor_id]['image_path'] = f"{save_path}/{sensor_id}/{frame}.jpg"

    if traffics is not None:
        anno['annotation']['traffic_element'] = [
            {
                'id': bb['id'],
                'points': bb['points'],
                'category': bb['category'],
                'attribute': bb['attribute'],
                'affected_lanelet': bb.get('affected_lanelet', [])
            }
            for bb in traffics
        ]

    os.makedirs(os.path.join(save_path, "info"), exist_ok=True)
    with open(os.path.join(save_path, "info", f"{frame}.json"), 'w') as f:
        json.dump(anno, f, indent=2)


def sensor_callback(sensor_id, queue):
    def _callback(data):
        queue.put((sensor_id, data.frame, data))
    return _callback


def get_image_point(loc, K, w2c, debug=False, actor_id=None):
    point = np.array([loc.x, loc.y, loc.z, 1.0], dtype=float)
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    if point_camera[2] <= 0.1:
        return None
    img_point = np.dot(K, point_camera)
    img_point /= img_point[2]
    return np.array([img_point[0], img_point[1]], dtype=float)


def get_rotation_matrix(yaw, pitch, roll):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    R_yaw = np.array([[math.cos(yaw_rad),-math.sin(yaw_rad),0],[math.sin(yaw_rad),math.cos(yaw_rad),0],[0,0,1]])
    R_pitch = np.array([[math.cos(pitch_rad),0,math.sin(pitch_rad)],[0,1,0],[-math.sin(pitch_rad),0,math.cos(pitch_rad)]])
    R_roll = np.array([[1,0,0],[0,math.cos(roll_rad),-math.sin(roll_rad)],[0,math.sin(roll_rad),math.cos(roll_rad)]])
    return R_yaw @ R_pitch @ R_roll


def get_bbox_facing(bbox_rot):
    R = get_rotation_matrix(bbox_rot.yaw, bbox_rot.pitch, bbox_rot.roll)
    return R @ np.array([0, 1, 0], dtype=float)


def bbox_sanity_check(x_min, x_max, y_min, y_max, image_w, image_h, min_wh=10, max_area_ratio=0.5):
    w = x_max - x_min
    h = y_max - y_min
    if w < min_wh or h < min_wh:
        return False
    if w * h > max_area_ratio * image_w * image_h:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True


def get_projected_bbox(verts, K, world_2_camera, image_w, image_h):
    x_max, x_min, y_max, y_min = -1e6, 1e6, -1e6, 1e6
    for vert in verts:
        p = get_image_point(vert, K, world_2_camera)
        if p is None or np.isnan(p[0]) or np.isnan(p[1]):
            return None, None, None, None, False
        x_min = min(x_min, p[0]); x_max = max(x_max, p[0])
        y_min = min(y_min, p[1]); y_max = max(y_max, p[1])
    x_min = int(max(0, min(image_w - 1, x_min)))
    x_max = int(max(0, min(image_w - 1, x_max)))
    y_min = int(max(0, min(image_h - 1, y_min)))
    y_max = int(max(0, min(image_h - 1, y_max)))
    return x_min, x_max, y_min, y_max, True


def filter_traffic_lights(vehicle, front_camera):
    """
    # CHANGE: Complete rewrite of traffic light detection pipeline
    Improved logic:
      1) Project TL bounding boxes only to front camera frame space
      2) Distinguish between facing (TL signal visible) vs non-facing (TL visible but signal obscured)
      3) Filter affected lanes by removing oncoming vehicles (dot product > -0.5)
      4) Store bbox even when TL is not facing (attribute=0/unknown) for completeness

    front_camera: (cam_actor, K_matrix, image_w, image_h)

    Rules:
      1) Include TL only if projected into front camera frame
      2) If TL signal faces camera (facing=True)
           → Record affected_lanelet for topology connection (filtering oncoming)
      3) If TL visible but signal faces away (facing=False)
           → Record bbox only, affected_lanelet=[] (no topology)
      4) If TL outside front camera frame → exclude entirely
    """
    cam_actor, K_mat, im_w, im_h = front_camera
    traffics = []
    tls = [a for a in vehicle.get_world().get_actors() if 'traffic_light' in a.type_id]

    ego_fwd = vehicle.get_transform().get_forward_vector()
    ego_dir = np.array([ego_fwd.x, ego_fwd.y])
    ve_loc  = vehicle.get_transform().location

    cam_fwd = cam_actor.get_transform().get_forward_vector()
    c_fwd   = np.array([cam_fwd.x, cam_fwd.y, cam_fwd.z], dtype=float)
    w2c     = np.array(cam_actor.get_transform().get_inverse_matrix(), dtype=float)

    for tl in tls:
        tl_id = tl.get_opendrive_id()
        state = tl.state
        category_val = TL_ATTRIBUTE.get(state, 0)

        tl_loc = tl.get_location()
        dist = math.hypot(tl_loc.x - ve_loc.x, tl_loc.y - ve_loc.y)
        # CHANGE: Extended distance threshold from 50m to 80m for better coverage
        if not (3 < dist < 80):
            continue

        for idx, bbox in enumerate(tl.get_light_boxes()):
            bbox_center = bbox.location
            if bbox_center.z - ve_loc.z < 1:
                continue

            bbox_facing = get_bbox_facing(bbox.rotation)
            verts = bbox.get_local_vertices()

            # Check if TL projects into front camera frame (regardless of facing)
            x_min, x_max, y_min, y_max, valid = get_projected_bbox(verts, K_mat, w2c, im_w, im_h)
            if not valid:
                continue
            if not bbox_sanity_check(x_min, x_max, y_min, y_max, im_w, im_h):
                continue

            best_bbox = [x_min, y_min, x_max, y_max]

            # Determine if TL signal is facing the camera (within 75° cone)
            facing_camera = np.dot(bbox_facing, c_fwd) <= -math.cos(math.radians(75))

            if facing_camera:
                # Signal is visible → include affected lanes but filter oncoming vehicles
                affected = []
                for wp in tl.get_affected_lane_waypoints():
                    wp_fwd = wp.transform.get_forward_vector()
                    wp_dir = np.array([wp_fwd.x, wp_fwd.y])
                    # CHANGE: Added oncoming filter (dot < -0.5 = opposite direction lanes)
                    if np.dot(wp_dir, ego_dir) < -0.5:  # oncoming lanes only excluded
                        continue
                    affected.append([wp.transform.location.x, -1 * wp.transform.location.y])
            else:
                # Signal not visible → include bbox but no lane topology
                affected = []

            traffics.append({
                'id':               int(tl_id) * 10 + idx,
                'points':           [best_bbox[:2], best_bbox[2:]],
                'category':         1,
                # CHANGE: Set attribute=0 (unknown) when signal not facing camera
                'attribute':        category_val if facing_camera else 0,  # 신호 안 보이면 unknown
                'affected_lanelet': affected,
            })
    return traffics


def clear_nearby_vehicles(world, target_location, radius=5.0):
    destroyed_count = 0
    for actor in world.get_actors():
        if (actor.type_id.startswith('vehicle.') and
                actor.attributes.get("role_name") == "autopilot"):
            if target_location.distance(actor.get_location()) < radius:
                try:
                    actor.destroy()
                    destroyed_count += 1
                except RuntimeError:
                    pass
    return destroyed_count


def apply_custom_weather(world, severity: float, sun_alt: float):
    # CHANGE: New function for precise weather control (severity 0.0-1.0)
    # Used by sample_extreme_real_weather() for realistic weather variation
    severity = float(np.clip(severity, 0.0, 1.0))
    w = carla.WeatherParameters(
        cloudiness=severity * 100.0,
        precipitation=severity * 100.0,
        precipitation_deposits=severity * 100.0,
        wetness=severity * 100.0,
        wind_intensity=severity * 50.0,
        sun_altitude_angle=float(sun_alt),
        sun_azimuth_angle=180.0,
        fog_density=max(0.0, (severity - 0.4) * 160.0) if severity > 0.4 else 0.0,
        fog_distance=max(5.0, 100.0 - (severity * 95.0)),
    )
    world.set_weather(w)
    print(f"🌦️ CustomWeather severity={severity:.2f}, sun_alt={sun_alt:.1f}")


def sample_extreme_real_weather(rng: np.random.RandomState):
    # CHANGE: New function for realistic weather variety (replaces fixed presets when used)
    # Samples from 10 weather types with realistic probability distribution
    weather_type = rng.choice(
        ["clear", "fog_light", "fog_mid", "fog_heavy",
         "rain_light", "rain_mid", "rain_heavy",
         "morning", "night", "afternoon"],
        p=[0.13, 0.09, 0.09, 0.09, 0.11, 0.11, 0.10, 0.09, 0.09, 0.10]
    )
    cloud=0.0; rain=0.0; deposits=0.0; wet=0.0; wind=0.0
    fog_density=0.0; fog_dist=100.0; fog_falloff=1.0
    sun_alt=45.0; sun_az=float(rng.uniform(0, 360))

    if weather_type == "clear":
        cloud=float(rng.uniform(0,10)); rain=0.0; deposits=0.0; wet=0.0
        wind=float(rng.uniform(0,10)); sun_alt=float(rng.uniform(40,75))
    elif weather_type == "fog_light":
        cloud=float(rng.uniform(20,50)); rain=0.0; wet=float(rng.uniform(0,10))
        wind=float(rng.uniform(0,8)); fog_density=float(rng.uniform(10,30))
        fog_dist=float(rng.uniform(40,80)); fog_falloff=float(rng.uniform(1.0,2.0))
        sun_alt=float(rng.uniform(20,60))
    elif weather_type == "fog_mid":
        cloud=float(rng.uniform(40,70)); rain=0.0; wet=float(rng.uniform(0,20))
        wind=float(rng.uniform(0,5)); fog_density=float(rng.uniform(40,65))
        fog_dist=float(rng.uniform(15,40)); fog_falloff=float(rng.uniform(1.5,2.5))
        sun_alt=float(rng.uniform(10,40))
    elif weather_type == "fog_heavy":
        cloud=float(rng.uniform(70,100)); rain=0.0; wet=float(rng.uniform(0,15))
        wind=float(rng.uniform(0,5)); fog_density=float(rng.uniform(75,100))
        fog_dist=float(rng.uniform(5,15)); fog_falloff=float(rng.uniform(1.2,2.0))
        sun_alt=float(rng.uniform(0,20))
    elif weather_type == "rain_light":
        cloud=float(rng.uniform(40,70)); rain=float(rng.uniform(10,35))
        deposits=float(rng.uniform(10,40)); wet=float(rng.uniform(30,60))
        wind=float(rng.uniform(5,20)); fog_density=0.0; fog_dist=100.0
        sun_alt=float(rng.uniform(20,50))
    elif weather_type == "rain_mid":
        cloud=float(rng.uniform(60,90)); rain=float(rng.uniform(40,65))
        deposits=float(rng.uniform(40,70)); wet=float(rng.uniform(60,90))
        wind=float(rng.uniform(15,35)); fog_density=0.0; fog_dist=100.0
        sun_alt=float(rng.uniform(5,35))
    elif weather_type == "rain_heavy":
        cloud=float(rng.uniform(85,100)); rain=float(rng.uniform(75,100))
        deposits=float(rng.uniform(75,100)); wet=100.0
        wind=float(rng.uniform(30,70)); fog_density=0.0; fog_dist=100.0
        sun_alt=float(rng.uniform(0,15))
    elif weather_type == "morning":
        cloud=float(rng.uniform(0,30)); rain=0.0; wet=0.0
        wind=float(rng.uniform(0,10)); sun_alt=float(rng.uniform(5,25))
        sun_az=float(rng.uniform(60,120))
    elif weather_type == "night":
        cloud=float(rng.uniform(0,40)); rain=0.0; wet=0.0
        wind=float(rng.uniform(0,15)); sun_alt=float(rng.uniform(-45,-15))
        sun_az=float(rng.uniform(0,360))
    elif weather_type == "afternoon":
        cloud=float(rng.uniform(10,40)); rain=0.0; wet=0.0
        wind=float(rng.uniform(0,10)); sun_alt=float(rng.uniform(5,20))
        sun_az=float(rng.uniform(240,300)); fog_density=0.0; fog_dist=100.0

    w = carla.WeatherParameters(
        cloudiness=cloud, precipitation=rain, precipitation_deposits=deposits,
        wetness=wet, wind_intensity=wind, sun_azimuth_angle=sun_az,
        sun_altitude_angle=sun_alt, fog_density=fog_density,
        fog_distance=fog_dist, fog_falloff=fog_falloff,
    )
    meta = {"type": weather_type, "cloudiness": cloud, "precipitation": rain,
            "precipitation_deposits": deposits, "wetness": wet, "wind_intensity": wind,
            "sun_azimuth_angle": sun_az, "sun_altitude_angle": sun_alt,
            "fog_density": fog_density, "fog_distance": fog_dist, "fog_falloff": fog_falloff}
    return w, meta


def spawn_mixed_traffic(client, world, tm_port, total_count, local_count,
                        center_loc, radius, rng):
    total_count = max(0, int(total_count))
    local_count = max(0, int(local_count))
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points or total_count == 0:
        return []
    rng.shuffle(spawn_points)
    nearby_sp = [sp for sp in spawn_points if sp.location.distance(center_loc) < radius]
    far_sp = [sp for sp in spawn_points if sp.location.distance(center_loc) >= radius]
    bps = [bp for bp in world.get_blueprint_library().filter('vehicle.*')
           if bp.has_attribute('number_of_wheels') and int(bp.get_attribute('number_of_wheels')) == 4]
    if not bps:
        bps = list(world.get_blueprint_library().filter('vehicle.*'))
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    for i in range(min(local_count, len(nearby_sp), total_count)):
        bp = rng.choice(bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, nearby_sp[i]).then(SetAutopilot(FutureActor, True, tm_port)))
    for i in range(min(total_count - len(batch), len(far_sp))):
        bp = rng.choice(bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
        bp.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(bp, far_sp[i]).then(SetAutopilot(FutureActor, True, tm_port)))
    responses = client.apply_batch_sync(batch, True)
    for _ in range(5):
        world.tick()
    spawned_ids = [r.actor_id for r in responses if not r.error]
    print(f"🚗 Traffic requested={total_count} (local={local_count}), spawned={len(spawned_ids)}")
    return spawned_ids


def flush_sensor_queue(sensor_queue, max_pop=99999):
    popped = 0
    while popped < max_pop:
        try:
            sensor_queue.get_nowait()
            popped += 1
        except Exception:
            break
    return popped


def _safe_is_junction(wp):
    try:
        return bool(wp.is_junction)
    except Exception:
        return False


def _is_turn_candidate(wp, step=8.0, yaw_thr_deg=25.0):
    try:
        nxts = wp.next(float(step))
        if not nxts:
            return False
        y0 = wp.transform.rotation.yaw
        for n in nxts:
            dy = (n.transform.rotation.yaw - y0 + 180.0) % 360.0 - 180.0
            if abs(dy) >= float(yaw_thr_deg):
                return True
        return False
    except Exception:
        return False


def build_teleport_waypoints(wl_map, rng, sample_num, step=8.0, min_dist_m=35.0,
                             extra_probe_per_spawn=2, max_tries=10000, prefer_complex=False):
    spawn_points = wl_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in this map.")
    pool = []
    for sp in spawn_points:
        wp = wl_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is not None:
            pool.append(wp)
    for sp in spawn_points[::max(1, len(spawn_points)//200)]:
        base = wl_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if base is None:
            continue
        cur = base
        for _ in range(int(extra_probe_per_spawn)):
            nxts = cur.next(float(step))
            if not nxts:
                break
            cur = nxts[int(rng.randint(0, len(nxts)))]
            if cur is None:
                break
            if _safe_is_junction(cur) or _is_turn_candidate(cur, step=step):
                pool.append(cur)
    if not pool:
        raise RuntimeError("Waypoint pool is empty.")
    complex_pool = [wp for wp in pool if wp and (_safe_is_junction(wp) or _is_turn_candidate(wp, step=step))]
    sample_pool = complex_pool if (prefer_complex and len(complex_pool) > 0) else pool

    chosen = []
    def far_enough(loc):
        return all(loc.distance(w.transform.location) >= float(min_dist_m) for w in chosen)

    tries = 0
    while len(chosen) < int(sample_num) and tries < int(max_tries):
        tries += 1
        wp = sample_pool[int(rng.randint(0, len(sample_pool)))]
        if wp and far_enough(wp.transform.location):
            chosen.append(wp)
    while len(chosen) < int(sample_num):
        wp = sample_pool[int(rng.randint(0, len(sample_pool)))]
        if wp:
            chosen.append(wp)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--spawn-offset', type=int, default=0)
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--dir', default='train')
    parser.add_argument('--pose-format', choices=['openlanev2'], default='openlanev2')
    parser.add_argument('--traffic-level', default=1, type=int)
    parser.add_argument('--weather', default=None, type=str)
    parser.add_argument('--severity', type=float, default=None)
    parser.add_argument('--sun-alt', type=float, default=None)
    parser.add_argument('--total-traffic', type=int, default=None)
    parser.add_argument('--local-traffic', type=int, default=None)
    parser.add_argument('--radius', type=float, default=70.0)
    parser.add_argument('--tm-port', type=int, default=None)
    parser.add_argument('--step', type=float, default=8.0)
    parser.add_argument('--min-dist', type=float, default=35.0)
    parser.add_argument('--prefer-complex', action='store_true')
    # ✅ NEW: subset 선택
    parser.add_argument('--subset', choices=['argoverse2', 'nuscenes'], default='argoverse2',
                        help='argoverse2: 7-cam SubsetA, nuscenes: 6-cam SubsetB')

    args = parser.parse_args()

    # ✅ subset에 따라 카메라 설정 선택
    if args.subset == 'nuscenes':
        SENSORS = SENSORS_NUSCENES
        print("📷 SubsetB (nuScenes 6-cam)")
    else:
        SENSORS = SENSORS_ARGOVERSE2
        print("📷 SubsetA (Argoverse2 7-cam)")

    THREADS = len(SENSORS)

    endpoint = get_next_folder_path(args.dir)
    os.makedirs(endpoint, exist_ok=True)

    seg_id = int(os.path.basename(endpoint))
    seed_str = (
        f"{seg_id}|{args.spawn_offset}|{args.weather}|{args.severity}|{args.sun_alt}|"
        f"{args.total_traffic}|{args.local_traffic}|{args.radius}|{args.step}|{args.min_dist}|{args.subset}"
    )
    base_seed = zlib.crc32(seed_str.encode("utf-8")) & 0x7fffffff
    rng = np.random.RandomState(base_seed)
    print(f"[SEG] endpoint={endpoint} seg_id={seg_id} seed={base_seed}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    wl_map = world.get_map()
    map_name = wl_map.name

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(1e-6, float(args.fps))
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    tm_port = args.tm_port if args.tm_port is not None else client.get_trafficmanager().get_port()
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(1.5)
    traffic_manager.global_percentage_speed_difference(30.0)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)
    traffic_manager.set_respawn_dormant_vehicles(True)
    traffic_manager.set_random_device_seed(0)

    weather_meta = None
    if args.weather is None or str(args.weather).lower() in ["auto", "random"]:
        args.weather = WEATHER_PRESETS[int(rng.randint(0, len(WEATHER_PRESETS)))]

    if str(args.weather).lower() == "extreme":
        wparam, weather_meta = sample_extreme_real_weather(rng)
        world.set_weather(wparam)
        applied_weather_preset = f"EXTREME:{weather_meta['type']}"
        print(f"🌦️ ExtremeWeather: {weather_meta}")
    else:
        applied_weather_preset = args.weather
        if args.weather in WEATHER_PRESETS:
            if args.weather == "ClearNight":
                wp = getattr(carla.WeatherParameters, "ClearSunset")
                wp.sun_altitude_angle = -30.0
                applied_weather_preset = "ClearSunset"
            elif args.weather == "CloudyNight":
                wp = getattr(carla.WeatherParameters, "CloudySunset")
                wp.sun_altitude_angle = -30.0
                applied_weather_preset = "CloudySunset"
            elif args.weather == "MidRainyNight":
                wp = getattr(carla.WeatherParameters, "MidRainSunset")
                wp.sun_altitude_angle = -30.0
                applied_weather_preset = "MidRainSunset"
            elif args.weather == "HardRainyNight":
                wp = getattr(carla.WeatherParameters, "HardRainSunset")
                wp.sun_altitude_angle = -30.0
                applied_weather_preset = "HardRainSunset"
            else:
                wp = getattr(carla.WeatherParameters, args.weather)
            world.set_weather(wp)
            print(f"🌦️ Weather preset: {args.weather}")
        elif args.weather in ("fog_light", "fog_mid", "fog_heavy"):
            # Pure fog presets: deterministic parameters, precipitation=0 (no rain)
            fog_cfgs = {
                "fog_light": dict(cloudiness=35.0, precipitation=0.0, precipitation_deposits=0.0,
                                  wetness=0.0, wind_intensity=5.0, sun_azimuth_angle=180.0,
                                  sun_altitude_angle=25.0, fog_density=20.0, fog_distance=60.0, fog_falloff=1.5),
                "fog_mid":   dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0,
                                  wetness=0.0, wind_intensity=3.0, sun_azimuth_angle=180.0,
                                  sun_altitude_angle=12.0, fog_density=52.0, fog_distance=25.0, fog_falloff=2.0),
                "fog_heavy": dict(cloudiness=90.0, precipitation=0.0, precipitation_deposits=0.0,
                                  wetness=0.0, wind_intensity=2.0, sun_azimuth_angle=180.0,
                                  sun_altitude_angle=5.0,  fog_density=85.0, fog_distance=10.0, fog_falloff=1.5),
            }
            cfg = fog_cfgs[args.weather]
            wp = carla.WeatherParameters(**cfg)
            world.set_weather(wp)
            print(f"🌫️ Pure fog: {args.weather} density={cfg['fog_density']} dist={cfg['fog_distance']} precip=0")
        else:
            print(f"⚠️ Unknown weather preset: {args.weather}")

    if args.severity is not None:
        sun_alt = float(args.sun_alt) if args.sun_alt is not None else (
            -30.0 if args.weather and "night" in args.weather.lower() else 45.0
        )
        apply_custom_weather(world, args.severity, sun_alt)

    scene_tokens = generate_scene_tokens(world, map_name, weather_preset=applied_weather_preset)
    print(f"📝 Scene: {scene_tokens['text']}")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.micro.microlino')[0]
    vehicle_bp.set_attribute('role_name', 'hero')

    vehicle = None
    spawn_points = wl_map.get_spawn_points()
    if not spawn_points:
        print('No spawn points.')
        sys.exit(1)

    base_idx = int(args.spawn_offset) % len(spawn_points)
    rand_shift = int(rng.randint(0, len(spawn_points)))
    selected_spawn_point = spawn_points[(base_idx + rand_shift) % len(spawn_points)]
    while vehicle is None:
        vehicle = world.try_spawn_actor(vehicle_bp, selected_spawn_point)
        if not vehicle:
            args.spawn_offset = (int(args.spawn_offset) + 1) % len(spawn_points)
            selected_spawn_point = spawn_points[int(args.spawn_offset) % len(spawn_points)]
        else:
            modify_vehicle_physics(vehicle)

    # Sequential waypoints: follow road step-m at a time (OpenLane-V2 style)
    # start_wp: 맵 전체 도로망에서 랜덤 선택 (seq마다 완전히 다른 위치)
    # 50% → junction/curve 우선, 50% → 전체 general
    _prefer_complex = args.prefer_complex or bool(rng.rand() < 0.5)
    _all_road_wps = wl_map.generate_waypoints(10.0)
    _complex_wps = [wp for wp in _all_road_wps
                    if _safe_is_junction(wp) or _is_turn_candidate(wp, step=float(args.step))]
    _start_pool = _complex_wps if (_prefer_complex and len(_complex_wps) >= 5) else _all_road_wps
    start_wp = _start_pool[int(rng.randint(0, len(_start_pool)))]
    print(f"🎯 start_wp: prefer_complex={_prefer_complex} pool={len(_start_pool)}")

    waypoints = [start_wp]
    curr_wp = start_wp
    for _ in range(int(args.sample) - 1):
        next_wps = curr_wp.next(float(args.step))
        if not next_wps:
            break
        curr_wp = next_wps[int(rng.randint(0, len(next_wps)))]
        waypoints.append(curr_wp)

    sensor_queue = Queue()
    sensors = []
    for sensor_id, config in SENSORS:
        bp = blueprint_library.find(config['bp'])
        for attr, value in config.items():
            if attr in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                continue
            bp.set_attribute(attr, str(value))
        transform = carla.Transform(
            carla.Location(x=config['x'], y=config['y'], z=config['z']),
            carla.Rotation(pitch=config['pitch'], roll=config['roll'], yaw=config['yaw'])
        )
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        sensor.listen(sensor_callback(sensor_id, sensor_queue))
        sensors.append(sensor)

    front_cam = sensors[0]
    image_w = int(front_cam.attributes['image_size_x'])
    image_h = int(front_cam.attributes['image_size_y'])

    # front camera 전용 tuple (TL bbox는 항상 front camera 좌표계)
    front_cam_tuple = (
        front_cam,
        build_projection_matrix(image_w, image_h, float(front_cam.attributes['fov'])),
        image_w,
        image_h,
    )

    autopilot_ids = []
    if args.total_traffic is not None:
        autopilot_ids = spawn_mixed_traffic(
            client, world, tm_port,
            total_count=args.total_traffic,
            local_count=(args.local_traffic if args.local_traffic is not None else int(args.total_traffic * 0.25)),
            center_loc=vehicle.get_location(), radius=args.radius, rng=rng
        )
        world.tick(); world.tick()

    for _ in range(20):
        world.tick()

    try:
        for i, wp in enumerate(waypoints):
            clear_nearby_vehicles(world, wp.transform.location, radius=3.0)
            world.tick()

            t = wp.transform
            vehicle.set_transform(carla.Transform(
                t.location, carla.Rotation(pitch=0.0, roll=0.0, yaw=t.rotation.yaw)
            ))

            settle = 6 if i == 0 else 4
            for _ in range(settle):
                time.sleep(0.01)
                world.tick()

            flush_sensor_queue(sensor_queue)
            world.tick()
            frame = world.get_snapshot().frame

            missing_sensors = len(SENSORS)
            results = []
            annotation_saved = False
            t0 = time.time()

            while True:
                if (time.time() - t0) > 3.0:
                    print(f"[WARN] timeout frame={frame}, missing={missing_sensors}")
                    break
                try:
                    sensor_id, frame_id, data = sensor_queue.get(True, timeout=2.0)
                    if frame_id != frame:
                        continue
                    missing_sensors -= 1
                except Empty:
                    time.sleep(0.005)
                    continue

                th = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_id, data, endpoint))
                results.append(th)
                th.start()

                if len(results) >= THREADS:
                    for th2 in results:
                        th2.join()
                    results = []

                if not annotation_saved:
                    traffics = filter_traffic_lights(vehicle, front_cam_tuple)
                    save_annotations(
                        vehicle, frame_id, endpoint, traffics, map_name,
                        SENSORS=SENSORS, pose_format=args.pose_format,
                        scene_tokens=scene_tokens, subset=args.subset
                    )
                    annotation_saved = True

                if missing_sensors <= 0:
                    break

            for th in results:
                th.join()
            gc.collect()

        for _ in range(10):
            world.tick()

    except KeyboardInterrupt:
        pass

    finally:
        for s in sensors:
            try: s.stop()
            except Exception: pass
            try: s.destroy()
            except Exception: pass
        try:
            if vehicle: vehicle.destroy()
        except Exception: pass
        try:
            if autopilot_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in autopilot_ids])
        except Exception: pass
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        settings.substepping = False
        world.apply_settings(settings)


if __name__ == '__main__':
    main()