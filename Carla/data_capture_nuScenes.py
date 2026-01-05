# autopilot_data_capture.py

"""
Modified version of the scenario-based data capture script to instead run a live
CARLA simulation with an autopilot agent (e.g., BehaviorAgent) and record
sensor data (e.g., RGB camera, instance segmentation, etc.) directly.
"""
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

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

# ========== Constants ==========

# nuScenes : https://www.nuscenes.org/
SENSORS = [
    [
        'CAM_FRONT',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
            'x': 0.80, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        }
    ],
    [
        'CAM_FRONT_LEFT',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
            'x': 0.27, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
        }
    ],
    [
        'CAM_FRONT_RIGHT',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
            'x': 0.27, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
        }
    ],
    [
        'CAM_BACK',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 110,
            'x': -2.0, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
        }
    ],
    [
        'CAM_BACK_LEFT',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
            'x': -0.32, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
        }
    ],
    [
        'CAM_BACK_RIGHT',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1600, 'image_size_y': 900, 'fov': 70,
            'x': -0.32, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
        }
    ]
]

# Weather presets pulled from CARLA's built-in configurations.
WEATHER_PRESETS = [
    'ClearNoon',
    'ClearSunset',
    'CloudyNoon',
    'CloudySunset',
    'SoftRainNoon',
    'SoftRainSunset',
    "MidRainyNoon",
    'WetNoon',
    'WetSunset',
    'WetCloudyNoon',
    "MidRainSunset",
    'WetCloudySunset',
    'HardRainNoon',
    'HardRainSunset',
    "ClearNight",
    "CloudyNight",
    "MidRainyNight"
]



THREADS = 7
CURRENT_THREADS = 0

TL_ATTRIBUTE = {
    carla.TrafficLightState.Off : 0,
    carla.TrafficLightState.Unknown: 0,
    carla.TrafficLightState.Red: 1,
    carla.TrafficLightState.Green: 2,
    carla.TrafficLightState.Yellow: 3,
    'go_straight': 4,
    'turn_left': 5,
    'turn_right': 6,
    'no_left_turn': 7,
    'no_right_turn': 8,
    'u_turn': 9,
    'no_u_turn': 10,
    'slight_left': 11,
    'slight_right': 12
}


def get_next_folder_path(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    existing = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    next_number = int(sorted(existing)[-1]) + 1 if existing else 1
    next_path = os.path.join(root_dir, f"{next_number:04d}")
    os.makedirs(next_path, exist_ok=False)
    return next_path


def are_waypoints_connected(prev_wp, curr_wp):
    """
    두 waypoint가 연결된 도로상에 있거나 인접한 경우 True 반환
    """
    next_wps = prev_wp.next(10.0)
    for next_wp in next_wps:
        if next_wp.road_id == curr_wp.road_id:
            return True

    return False

def generate_scene_tokens(world, map_name, weather_preset=None):
    """
    Generate scene-level description tokens for the driving environment.

    Categories:
    - time_of_day: noon, sunset, night
    - weather: clear, soft_rain, mid_rain, hard_rain, wet, wet_cloudy
    - road_type: urban, suburban, downtown, highway

    Args:
        world: carla.World object
        map_name: Name of the CARLA map (e.g., "Town01")
        weather_preset: Weather preset name (e.g., "ClearNoon", "HardRainSunset")

    Returns:
        dict: Scene description with categorical tokens
    """
    scene_dict = {}

    # === 1. Weather preset mapping ===
    # Extract time_of_day and weather from preset name
    if weather_preset:
        preset_lower = weather_preset.lower()

        # Time of day
        if 'noon' in preset_lower:
            scene_dict['time_of_day'] = 'noon'
        elif 'sunset' in preset_lower:
            scene_dict['time_of_day'] = 'sunset'
        elif 'night' in preset_lower:
            scene_dict['time_of_day'] = 'night'
        else:
            scene_dict['time_of_day'] = 'noon'

        # Weather condition
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
        else:
            scene_dict['weather'] = 'clear'
    else:
        # Fallback to weather object if no preset
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

    # === 2. Road Type (based on map) ===
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

    # Create flattened token list and text description
    tokens = list(scene_dict.values())
    text_description = " | ".join([f"{k}:{v}" for k, v in scene_dict.items()])

    return {
        "categories": scene_dict,
        "tokens": tokens,
        "text": text_description
    }

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def is_within_fov(cam_forward, cam_location, target_location, fov_deg):
    ray = target_location - cam_location
    ray_norm = ray.make_unit_vector()
    dot = cam_forward.dot(ray_norm)
    if dot <= 0:
        return False
    angle = math.degrees(math.acos(dot))
    return angle < (fov_deg / 2.0)


def modify_vehicle_physics(actor):
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass

def save_data_to_disk(sensor_id, frame, data, endpoint, imu_data=None):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """

    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.jpg"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        cv2.imwrite(sensor_endpoint, img)
        # data.save_to_disk(sensor_endpoint, color_converter=carla.ColorConverter.Raw)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.SemanticLidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.RadarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
        for point_data in data:
            data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
        with open(sensor_endpoint, 'w') as data_file:
            data_file.write(data_txt)

    elif isinstance(data, carla.GnssMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        with open(sensor_endpoint, 'a') as data_file:   
            data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
            data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{imu_data[0][0]},{imu_data[0][1]},{imu_data[0][2]},{data.compass},{imu_data[1][0]},{imu_data[1][1]},{imu_data[1][2]}\n"
            data_file.write(data_txt)

    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

def carla_to_openlane_pose(x, y, z, roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_carla = Rz @ Ry @ Rx
    flip_Y = np.diag([1, -1, 1])
    R_openlane = flip_Y @ R_carla
    U, _, Vt = np.linalg.svd(R_openlane)
    R_openlane_fixed = U @ Vt
    if np.linalg.det(R_openlane_fixed) < 0:
        Vt[-1, :] *= -1
        R_openlane_fixed = U @ Vt
    t_openlane = flip_Y @ np.array([x, y, z])
    return R_openlane_fixed.tolist(), t_openlane.tolist()

def save_annotations(vehicle, frame, save_path, traffics, name, pose_format='carla-rpy', scene_tokens=None):
    pose_path = "openlane_v2_subset_B.json"
    os.makedirs(save_path, exist_ok=True)
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    # Build pose depending on requested format
    if pose_format == 'openlanev2':
        R, t = carla_to_openlane_pose(round(location.x, 6), round(location.y, 6), round(location.z, 6),
                                      round(rotation.roll, 6), round(rotation.pitch, 6), round(rotation.yaw, 6))
        pose_rotation = R
        pose_translation = t
        euler_deg = [
            rotation.roll,
            rotation.pitch,
            rotation.yaw
        ]
    else:
        raise ValueError(f"Unsupported pose_format: {pose_format}")

    anno = {}
    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            anno = json.load(f)
    else:
        anno['pose'] = {}
        anno['sensor'] = {}
        anno['annotation'] = {'traffic_element': []}


    anno['meta_data']['source'] = "Carla_nuScenes"
    anno['meta_data']['source_id'] = name + str(frame)

    # Populate pose according to format
    if pose_format == 'openlanev2':
        # Ensure pose dict exists
        anno['pose'] = anno.get('pose', {})
        # Remove matrix if it exists to avoid confusion
        anno['pose']['rotation'] = pose_rotation
        anno['pose']['euler_deg'] = euler_deg
        anno['pose']['translation'] = pose_translation
        
    # Add scene-level description tokens
    if scene_tokens is not None:
        anno['meta_data']['scene_description'] = scene_tokens

    anno['timestamp'] = frame
    anno['segment_id'] = os.path.basename(save_path)
    for i in anno.get('sensor', {}).keys():
        anno['sensor'][i]['image_path'] = f"{save_path}/{i}/{frame}.jpg"
    if traffics is not None:
        anno['annotation']['traffic_element'] = [
            {'id': bb['id'], 'points': bb['points'], 'category': bb['category'], 'attribute': bb['attribute'], 'affected_lanelet' : bb['affected_lanelet']}
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
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    if point_camera[2] <= 0.1:
        if debug:
            print(f"[⚠️] Actor {actor_id} is behind camera. Camera z: {point_camera[2]:.2f}")
        return None

    img_point = np.dot(K, point_camera)
    img_point /= img_point[2]
    x, y = img_point[0], img_point[1]

    if debug and (x < 0 or y < 0 or x > K[0, 2] * 2 or y > K[1, 2] * 2):
        print(f"[❌] Actor {actor_id} projected outside image bounds: ({x:.1f}, {y:.1f})")
    return np.array([x, y])

def get_rotation_matrix(yaw, pitch, roll):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_pitch = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad),  math.cos(roll_rad)]
    ])
    return R_yaw @ R_pitch @ R_roll

def get_bbox_facing(bbox_rot):
    R = get_rotation_matrix(bbox_rot.yaw, bbox_rot.pitch, bbox_rot.roll)
    y_axis = np.array([0, 1, 0])
    return R @ y_axis

def bbox_sanity_check(x_min, x_max, y_min, y_max, image_w, image_h, min_wh=10, max_area_ratio=0.5):
    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    img_area = image_w * image_h
    if w < min_wh or h < min_wh:
        return False
    if area > max_area_ratio * img_area:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True

def get_projected_bbox(verts, K, world_2_camera, image_w, image_h):
    x_max = -1e6
    x_min = 1e6
    y_max = -1e6
    y_min = 1e6
    valid_projection = True
    for vert in verts:
        p = get_image_point(vert, K, world_2_camera)
        if p is None or np.isnan(p[0]) or np.isnan(p[1]):
            valid_projection = False
            break
        x, y = p
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    if not valid_projection:
        return None, None, None, None, False
    x_min = int(max(0, min(image_w - 1, x_min)))
    x_max = int(max(0, min(image_w - 1, x_max)))
    y_min = int(max(0, min(image_h - 1, y_min)))
    y_max = int(max(0, min(image_h - 1, y_max)))
    return x_min, x_max, y_min, y_max, True

def filter_traffic_lights(vehicle, K, world_2_camera, image_w, image_h):
    traffics = []
    id = 0
    # 모든 traffic light actor 탐색 (bounding_box_set이 아닌 traffic_light actor 기준)
    tls = [a for a in vehicle.get_world().get_actors() if 'traffic_light' in a.type_id]
    for tl in tls:
        id = tl.get_opendrive_id()
        state = tl.state
        category_val = TL_ATTRIBUTE.get(state, 0)  # default 4: Unknown

        tl_loc = tl.get_location()
        ve_loc = vehicle.get_transform().location
        dist = math.hypot(tl_loc.x - ve_loc.x, tl_loc.y - ve_loc.y)
        if not (3 < dist < 50):
            continue

        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = tl.get_transform().location - ve_loc
        if forward_vec.dot(ray) <= 0:
            continue
        
        for idx, bbox in enumerate(tl.get_light_boxes()):
            bbox_center = bbox.location
            if  bbox_center.z - ve_loc.z < 1:
                continue
            bbox_facing = get_bbox_facing(bbox.rotation)
            vehicle_forward = vehicle.get_transform().get_forward_vector()
            v_forward = np.array([vehicle_forward.x, vehicle_forward.y, vehicle_forward.z])
            dot = np.dot(bbox_facing, v_forward)
            # vehicle 기준 불빛이 정면을 향하는 경우만 (마주봄 75도 이내)
            if dot > -math.cos(math.radians(75)):
                continue

            verts = bbox.get_local_vertices()
            x_min, x_max, y_min, y_max, valid_projection = get_projected_bbox(
                verts, K, world_2_camera, image_w, image_h
            )
            if not valid_projection:
                continue
            if not bbox_sanity_check(x_min, x_max, y_min, y_max, image_w, image_h):
                continue
            
            affected = [[wp.transform.location.x,-1*wp.transform.location.y]  for wp in tl.get_affected_lane_waypoints()]
            
            traffics.append({
                'id': f"tl_{str(id)}_{idx}",
                'points': [[x_min, y_min], [x_max, y_max]],
                'category': 0,
                'attribute': category_val,
                'affected_lanelet' : affected 
            })
    return traffics


def clear_nearby_vehicles(world, target_location, radius=5.0):
    """
    지정된 위치 주변의 차량들을 미리 정리
    """
    all_actors = world.get_actors()
    destroyed_count = 0

    for actor in all_actors:
        if (actor.type_id.startswith('vehicle.') and
            actor.attributes.get("role_name") == "autopilot"):

            distance = target_location.distance(actor.get_location())
            if distance < radius:
                try:
                    actor.destroy()
                    destroyed_count += 1
                except RuntimeError:
                    pass

    return destroyed_count

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--duration', type=int, default=120, help='Duration to run (sec)')
    parser.add_argument('--sample', type=int, default=10, help='Number of sample waypoints to capture (default: 15)')
    parser.add_argument('--scene', type=int, default=15, help='Scene number for folder naming (default: 1)')
    parser.add_argument('--spawn-offset', type=int, default=0, help='Spawn point offset for deterministic selection')
    parser.add_argument('--fps', type=int, default=1, help='fps')
    parser.add_argument('--dir', default='train', help='Random seed for waypoint generation')
    parser.add_argument('--pose-format', choices=['openlanev2', 'carla-matrix', 'carla-rpy'], default='carla-rpy',
                        help='Pose encoding: OpenLane (matrix in OpenLane frame), CARLA rotation matrix, or CARLA roll/pitch/yaw degrees')
    parser.add_argument(
        '--traffic-level',
        default=1,
        type=int,
        help='TCP port to listen to (default: 2000)')
    parser.add_argument(
        '--weather',
        default=None,
        type=str,
        help='Weather preset name (e.g., ClearNoon, HardRainNoon). If not specified, uses default weather.')



    args = parser.parse_args()


    # spawn-offset을 사용하므로 seed는 제거
    if args.fps is not None:
        fps = 0.5
    if args.dir is not None:
        dir = args.dir
    if args.sample is not None:
        sample_num = args.sample

    endpoint = get_next_folder_path(dir)
    os.makedirs(endpoint, exist_ok=True)
        
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    wl_map = world.get_map()
    map_name = wl_map.name
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    traffic_manager = client.get_trafficmanager()
    tm_port = traffic_manager.get_port()

    traffic_manager.set_synchronous_mode(True)
    world.apply_settings(settings)

    # Set weather if specified
    if args.weather:
        if args.weather in WEATHER_PRESETS:
            if args.weather == "ClearNight":
                args.weather = "ClearSunset"
                weather_param = getattr(carla.WeatherParameters, args.weather)
                weather_param.sun_altitude_angle = -30.0
            elif args.weather == "CloudyNight":
                args.weather = "CloudySunset"
                weather_param = getattr(carla.WeatherParameters, args.weather)
                weather_param.sun_altitude_angle = -30.0
            elif args.weather == "MidRainyNight":   
                args.weather = "MidRainSunset"
                weather_param = getattr(carla.WeatherParameters, args.weather)
                weather_param.sun_altitude_angle = -30.0
            else:
                weather_param = getattr(carla.WeatherParameters, args.weather)
            world.set_weather(weather_param)
            print(f"🌦️  Weather set to: {args.weather}")
        else:
            print(f"⚠️  Unknown weather preset: {args.weather}. Available: {', '.join(WEATHER_PRESETS)}")
            print("Using default weather.")

    # Generate scene tokens once for this sequence
    scene_tokens = generate_scene_tokens(world, map_name, weather_preset=args.weather)
    print(f"📝 Scene description: {scene_tokens['text']}")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.micro.microlino')[0]
    vehicle_bp.set_attribute('role_name', 'hero')
    
    vehicle = None
    
    spawn_points = wl_map.get_spawn_points()
    if not spawn_points:
        print('No spawn points available in map.')
        sys.exit(1)

    # Offset 기반 spawn point 선택 (순환)
    spawn_offset = (len(spawn_points) // (args.scene*2)) * args.spawn_offset
    selected_spawn_point = spawn_points[spawn_offset]

    while vehicle is None:
        vehicle = world.try_spawn_actor(vehicle_bp, selected_spawn_point)
        if vehicle:
            modify_vehicle_physics(vehicle)
        else:
            spawn_offset = (spawn_offset + 1) % len(spawn_points)
            selected_spawn_point = spawn_points[spawn_offset]

    spawn_point = selected_spawn_point

    start_waypoint =  wl_map.get_waypoint(spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    waypoints = [start_waypoint]
    curr_wp = start_waypoint
    for _ in range(sample_num - 1):
        next_wps = curr_wp.next(5.0)
        if not next_wps:
            break
        curr_wp = random.choice(next_wps)
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
    image_w, image_h, fov = int(front_cam.attributes['image_size_x']), int(front_cam.attributes['image_size_y']), float(front_cam.attributes['fov'])
    K = build_projection_matrix(image_w, image_h, fov)


    traffics = {1 : 0, 2 : (len(spawn_points))//2}
    num_vehicles = traffics[args.traffic_level]
    max_threads = THREADS

    if num_vehicles > 0:
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [bp for bp in vehicle_blueprints if not bp.id.endswith('isetta')]

        spawn_points = wl_map.get_spawn_points()
        random.shuffle(spawn_points)

        # spawn 수 제한

        if num_vehicles > len(spawn_points):
            num_vehicles = len(spawn_points)
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for i in range(num_vehicles):
            bp = random.choice(vehicle_blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            transform = spawn_points[i]
            bp.set_attribute('role_name', 'autopilot')
            batch.append(
                SpawnActor(bp, transform)
                .then(SetAutopilot(FutureActor, True, tm_port))
            )

        responses = client.apply_batch_sync(batch, True)
        world.tick()
    try:
        for i, wp in enumerate(waypoints):

    
            if i > 0:
                clear_nearby_vehicles(world, wp.transform.location, radius=3.0)
                world.tick()
                world.tick()

            vehicle.set_transform(wp.transform)

            if i == 0:
                for _ in range(10):
                    time.sleep(0.02)
                    world.tick()
            else:
                for _ in range(5):
                    time.sleep(0.02)
                    world.tick()

            while True:
                try:
                    sensor_queue.get_nowait()
                    time.sleep(0.02)
                except:
                    break

            world.tick()
            frame = world.get_snapshot().frame
            missing_sensors = len(SENSORS)
            results = []
            annotation_saved = False

            while True:
                
                
                frame = world.get_snapshot().frame

                try:
                    sensor_id, frame_id, data = sensor_queue.get(True, timeout=2.0)
                    if frame_id != frame: continue  # Ignore previous frame data
                    missing_sensors -= 1
                except Empty:
                    time.sleep(0.01)

                res = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_id, data, endpoint))
                results.append(res)
                res.start()
                if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []
                        
                if not annotation_saved:
                    world_2_camera = np.array(front_cam.get_transform().get_inverse_matrix())
                    traffics = []
                    traffics = filter_traffic_lights(vehicle, K, world_2_camera, image_w, image_h)

                    save_annotations(vehicle, frame_id, endpoint, traffics, map_name,
                                   pose_format=args.pose_format,
                                   scene_tokens=scene_tokens)
                    annotation_saved = True
                if missing_sensors <= 0:
                    break
                
            gc.collect()
        for _ in range(10):
            world.tick()
    except KeyboardInterrupt:
        pass
    finally:
        for s in sensors:
            s.stop()
            s.destroy()
        vehicle.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        all_actors = world.get_actors()
        for actor in all_actors:
            if actor.type_id.startswith("vehicle.") and actor.attributes.get("role_name") == "autopilot":
                actor.destroy()
if __name__ == '__main__':
    main()
