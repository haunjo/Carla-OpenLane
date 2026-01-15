#!/usr/bin/env python3
"""
data_capture_Argoverse2_BH.py  (Drop-in replacement, based on your working autopilot_data_capture.py)

- Compatible args:
  --host --port --sample --step --scene --spawn-offset --fps --dir --pose-format(openlanev2)
  --traffic-level --weather
  --hard-scenario --hard-occluders N --hard-jaywalkers N

- Output:
  <dir>/<####>/[sensor_id]/<frame>.jpg
  <dir>/<####>/info/<frame>.json

PATCH:
- ONLY when --hard-scenario:
    use custom extreme weather from weather_four_classes()
"""

import os
import sys
import gc
import time
import math
import json
import argparse
import threading
from queue import Queue, Empty

import numpy as np
import numpy.random as npr
import cv2

import carla


# =========================
# Sensors (same as your working script)
# =========================
SENSORS = [
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

WEATHER_PRESETS = [
    'ClearNoon', 'ClearSunset', 'CloudyNoon', 'CloudySunset',
    'SoftRainNoon', 'SoftRainSunset', 'MidRainyNoon',
    'WetNoon', 'WetSunset', 'WetCloudyNoon', 'MidRainSunset',
    'WetCloudySunset', 'HardRainNoon', 'HardRainSunset',
    'ClearNight', 'CloudyNight', 'MidRainyNight'
]

# =========================
# Custom HARD weather (your 4-class)
# =========================
def weather_four_classes():
    """
    현실적인(하지만 어려운) 4-class 악천후 세트.
    - 너무 극단적인 값(안개 거리 5m, 밀도 100 등) 제거
    - 카메라가 완전 하얗게/검게 날아가거나 시뮬이 불안정해지는 케이스를 줄임
    """
    return {
        # 1) 맑은 날 + 약간의 글레어 (실제 주행에서 흔함)
        "SUNNY_GLARE_DAY": carla.WeatherParameters(
            cloudiness=5.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wetness=0.0,
            wind_intensity=8.0,
            sun_azimuth_angle=120.0,
            sun_altitude_angle=55.0,   # 85 -> 55 (너무 직광이면 하늘/차선 날아감)
            fog_density=0.0,
            fog_distance=0.0
        ),

        # 2) 짙은 안개(현실적인 수준) - 가시거리 30~60m 느낌
        "SUPER_FOG": carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wetness=20.0,
            wind_intensity=5.0,
            sun_azimuth_angle=45.0,
            sun_altitude_angle=8.0,
            fog_density=65.0,          # 100 -> 65 (완전 백화 방지)
            fog_distance=45.0,         # 5 -> 45 (5m는 거의 벽)
            fog_falloff=1.8            # 5 -> 1.8 (5는 급격히 뿌옇게, 너무 비현실적)
        ),

        # 3) 강우 + 젖은 노면 (낮) - 현실적인 폭우/시야 저하
        "HARD_RAIN_WET_DAY": carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=85.0,        # 100 -> 85 (100은 너무 과장된 경우 많음)
            precipitation_deposits=80.0,
            wetness=95.0,
            wind_intensity=35.0,       # 60 -> 35 (너무 강풍이면 이상해 보일 때가 있음)
            sun_azimuth_angle=60.0,
            sun_altitude_angle=18.0,
            fog_density=20.0,          # 비로 인한 헤이즈 약간
            fog_distance=120.0,
            fog_falloff=0.6
        ),

        # 4) 강우 + 젖은 노면 (밤) - 야간 난이도 + 헤드라이트 난반사 느낌
        "HARD_RAIN_WET_NIGHT": carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=85.0,
            precipitation_deposits=85.0,
            wetness=100.0,
            wind_intensity=30.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-20.0,  # -30 -> -20 (너무 어두워져서 학습/가시성 망가지는 경우 감소)
            fog_density=25.0,
            fog_distance=110.0,
            fog_falloff=0.7
        ),
    }



# =========================
# Utils
# =========================

def settle_after_scenario_change(world, sensor_queue: Queue, settle_ticks=12, flush=True, sleep_s=0.0):
    """
    시나리오 변화(ego teleport / hard actor spawn / weather change) 직후
    물리/센서가 안정화될 시간을 주고, old frame 센서 데이터를 버린다.
    """
    for _ in range(int(settle_ticks)):
        if sleep_s > 0:
            time.sleep(sleep_s)
        world.tick()

    if flush:
        flush_queue(sensor_queue)

def get_next_folder_path(root_dir: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    existing = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    next_number = int(sorted(existing)[-1]) + 1 if existing else 1
    next_path = os.path.join(root_dir, f"{next_number:04d}")
    os.makedirs(next_path, exist_ok=False)
    return next_path


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


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
        elif 'fog' in preset_lower:
            scene_dict['weather'] = 'fog'
        else:
            # custom labels like HARD_RAIN_WET_NIGHT etc.
            if 'rain' in preset_lower:
                scene_dict['weather'] = 'hard_rain'
            elif 'fog' in preset_lower:
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
        elif getattr(weather, "fog_density", 0.0) > 20:
            scene_dict['weather'] = 'fog'
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

    tokens = list(scene_dict.values())
    text_description = " | ".join([f"{k}:{v}" for k, v in scene_dict.items()])

    return {"categories": scene_dict, "tokens": tokens, "text": text_description}


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


def sensor_callback(sensor_id, queue: Queue):
    def _callback(data):
        queue.put((sensor_id, data.frame, data))
    return _callback


def flush_queue(q: Queue):
    while True:
        try:
            q.get_nowait()
        except Exception:
            break


def save_image_jpg(path, carla_image: carla.Image):
    img = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    img = img.reshape((carla_image.height, carla_image.width, 4))
    bgr = img[:, :, :3]  # BGRA -> BGR
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)


def save_data_to_disk(sensor_id, frame, data, endpoint):
    if isinstance(data, carla.Image):
        out = f"{endpoint}/{sensor_id}/{frame}.jpg"
        save_image_jpg(out, data)
    else:
        pass


def save_annotations(vehicle, frame, save_path, map_name, pose_format='openlanev2', scene_tokens=None):
    os.makedirs(save_path, exist_ok=True)

    transform = vehicle.get_transform()
    loc = transform.location
    rot = transform.rotation

    if pose_format != 'openlanev2':
        raise ValueError("Only openlanev2 is supported in this BH script.")

    R, t = carla_to_openlane_pose(
        round(loc.x, 6), round(loc.y, 6), round(loc.z, 6),
        round(rot.roll, 6), round(rot.pitch, 6), round(rot.yaw, 6)
    )

    anno = {
        "meta_data": {
            "source": "Carla_Argoverse2",
            "source_id": f"{map_name}_{frame}",
        },
        "timestamp": frame,
        "segment_id": os.path.basename(save_path),
        "pose": {
            "rotation": R,
            "translation": t,
            "euler_deg": [rot.roll, rot.pitch, rot.yaw],
        },
        "sensor": {},
        "annotation": {"traffic_element": []}
    }

    if scene_tokens is not None:
        anno["meta_data"]["scene_description"] = scene_tokens

    for sid, _cfg in SENSORS:
        anno["sensor"][sid] = {
            "image_path": f"{save_path}/{sid}/{frame}.jpg"
        }

    os.makedirs(os.path.join(save_path, "info"), exist_ok=True)
    with open(os.path.join(save_path, "info", f"{frame}.json"), "w") as f:
        json.dump(anno, f, indent=2)


# =========================
# "Hard scenario" helpers (occluders + jaywalkers)
# =========================
def spawn_occluder_vehicles(world, blueprint_library, wl_map, center_loc, tm_port, count, rng):
    spawned = []
    if count <= 0:
        return spawned

    vehicle_bps = blueprint_library.filter('vehicle.*')
    vehicle_bps = [bp for bp in vehicle_bps if bp.has_attribute('number_of_wheels')]

    spawn_points = wl_map.get_spawn_points()
    if not spawn_points:
        return spawned

    near = []
    for sp in spawn_points:
        if sp.location.distance(center_loc) < 60.0:
            near.append(sp)
    rng.shuffle(near)

    traffic_manager = carla.Client("127.0.0.1", 2000).get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)

    for sp in near[:count * 3]:
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', 'occluder')
        if bp.has_attribute('color'):
            try:
                color = rng.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            except Exception:
                pass

        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue

        try:
            actor.set_autopilot(True, tm_port)
        except Exception:
            pass

        spawned.append(actor)
        if len(spawned) >= count:
            break

    return spawned


def spawn_jaywalkers(world, blueprint_library, center_loc, count, rng):
    walkers = []
    controllers = []
    if count <= 0:
        return walkers, controllers

    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    ctrl_bp = blueprint_library.find("controller.ai.walker")

    for _ in range(count * 5):
        if len(walkers) >= count:
            break

        spawn_loc = world.get_random_location_from_navigation()
        if spawn_loc is None:
            continue
        if spawn_loc.distance(center_loc) > 40.0:
            continue

        wp = carla.Transform(spawn_loc)
        wbp = rng.choice(walker_bps)
        walker = world.try_spawn_actor(wbp, wp)
        if walker is None:
            continue

        controller = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        if controller is None:
            try:
                walker.destroy()
            except Exception:
                pass
            continue

        try:
            controller.start()
            controller.set_max_speed(float(rng.uniform(1.2, 2.0)))
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
        except Exception:
            pass

        walkers.append(walker)
        controllers.append(controller)

    return walkers, controllers


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--sample', type=int, default=5)
    parser.add_argument('--step', type=float, default=12.0, help='Waypoint step distance (meters)')
    parser.add_argument('--scene', type=int, default=1, help='For compatibility (not used for indexing here)')
    parser.add_argument('--spawn-offset', type=int, default=0)

    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--dir', required=True)

    parser.add_argument('--pose-format', choices=['openlanev2'], default='openlanev2')
    parser.add_argument('--traffic-level', type=int, default=1)
    parser.add_argument('--weather', type=str, default=None)

    parser.add_argument('--hard-scenario', action='store_true')
    parser.add_argument('--hard-occluders', type=int, default=0)
    parser.add_argument('--hard-jaywalkers', type=int, default=0)

    args = parser.parse_args()

    rng = npr.RandomState(abs(int(args.spawn_offset)) + 12345)

    endpoint = get_next_folder_path(args.dir)
    os.makedirs(endpoint, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    wl_map = world.get_map()
    map_name = wl_map.name

    # Sync settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(1e-6, float(args.fps))
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    tm_port = traffic_manager.get_port()
    traffic_manager.set_synchronous_mode(True)

    # =========================================================
    # Weather (PATCHED): hard-scenario => custom 4-class
    # =========================================================
    weather_for_scene_tokens = args.weather  # default (normal)

    if args.hard_scenario:
        hard_weathers = weather_four_classes()
        keys = list(hard_weathers.keys())
        # deterministic pick by spawn-offset so reproducible
        key = keys[abs(int(args.spawn_offset)) % len(keys)]
        world.set_weather(hard_weathers[key])
        weather_for_scene_tokens = key
        print(f"🌩️  [HARD] Weather set to: {key}")
    else:
        if args.weather:
            if args.weather in WEATHER_PRESETS:
                if args.weather == "ClearNight":
                    preset = "ClearSunset"
                    w = getattr(carla.WeatherParameters, preset)
                    w.sun_altitude_angle = -30.0
                elif args.weather == "CloudyNight":
                    preset = "CloudySunset"
                    w = getattr(carla.WeatherParameters, preset)
                    w.sun_altitude_angle = -30.0
                elif args.weather == "MidRainyNight":
                    preset = "MidRainSunset"
                    w = getattr(carla.WeatherParameters, preset)
                    w.sun_altitude_angle = -30.0
                else:
                    w = getattr(carla.WeatherParameters, args.weather)

                world.set_weather(w)
                print(f"🌦️  Weather set to: {args.weather}")
            else:
                print(f"⚠️  Unknown weather preset: {args.weather} (using default)")
        else:
            # if not specified, keep current world weather
            pass

    scene_tokens = generate_scene_tokens(world, map_name, weather_preset=weather_for_scene_tokens)
    print(f"📝 Scene description: {scene_tokens['text']}")

    blueprint_library = world.get_blueprint_library()

    # Ego vehicle
    vehicle_bp = blueprint_library.filter('vehicle.micro.microlino')[0]
    vehicle_bp.set_attribute('role_name', 'hero')

    spawn_points = wl_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in this map.")

    sp_idx = int(args.spawn_offset) % len(spawn_points)
    selected_sp = spawn_points[sp_idx]

    vehicle = None
    for _ in range(50):
        vehicle = world.try_spawn_actor(vehicle_bp, selected_sp)
        if vehicle is not None:
            break
        sp_idx = (sp_idx + 1) % len(spawn_points)
        selected_sp = spawn_points[sp_idx]
    if vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    # Attach sensors
    sensor_queue = Queue()
    sensors = []
    for sensor_id, cfg in SENSORS:
        bp = blueprint_library.find(cfg['bp'])
        for attr, val in cfg.items():
            if attr in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                continue
            bp.set_attribute(attr, str(val))

        transform = carla.Transform(
            carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
            carla.Rotation(pitch=cfg['pitch'], roll=cfg['roll'], yaw=cfg['yaw'])
        )
        s = world.spawn_actor(bp, transform, attach_to=vehicle)
        s.listen(sensor_callback(sensor_id, sensor_queue))
        sensors.append(s)

    autopilots = []
    try:
        if args.traffic_level == 2:
            max_spawn = min(len(spawn_points) // 4, 80)
        else:
            max_spawn = 0

        if max_spawn > 0:
            v_bps = blueprint_library.filter('vehicle.*')
            v_bps = [bp for bp in v_bps if not bp.id.endswith('isetta')]
            rng.shuffle(spawn_points)

            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            batch = []
            for i in range(min(max_spawn, len(spawn_points))):
                bp = rng.choice(v_bps)
                if bp.has_attribute('color'):
                    try:
                        color = rng.choice(bp.get_attribute('color').recommended_values)
                        bp.set_attribute('color', color)
                    except Exception:
                        pass
                bp.set_attribute('role_name', 'autopilot')
                batch.append(
                    SpawnActor(bp, spawn_points[i]).then(SetAutopilot(FutureActor, True, tm_port))
                )

            responses = client.apply_batch_sync(batch, True)
            for r in responses:
                if not r.error:
                    autopilots.append(r.actor_id)

        start_wp = wl_map.get_waypoint(selected_sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        waypoints = [start_wp]
        cur = start_wp
        for _ in range(max(0, args.sample - 1)):
            nxt = cur.next(float(args.step))
            if not nxt:
                break
            cur = rng.choice(nxt)
            waypoints.append(cur)

        hard_occluders = []
        hard_walkers = []
        hard_controllers = []

        for _ in range(10):
            world.tick()

        for i, wp in enumerate(waypoints):
            t = wp.transform
            fixed = carla.Transform(
                t.location,
                carla.Rotation(pitch=0.0, roll=0.0, yaw=t.rotation.yaw)
            )
            vehicle.set_transform(fixed)

            # ✅ 시나리오/자세 변경 직후 안정화
            settle_after_scenario_change(world, sensor_queue, settle_ticks=12, flush=True)


            if args.hard_scenario:
                for c in hard_controllers:
                    try:
                        c.stop()
                    except Exception:
                        pass
                    try:
                        c.destroy()
                    except Exception:
                        pass
                for w in hard_walkers:
                    try:
                        w.destroy()
                    except Exception:
                        pass
                for v in hard_occluders:
                    try:
                        v.destroy()
                    except Exception:
                        pass
                hard_occluders, hard_walkers, hard_controllers = [], [], []

                if args.hard_occluders > 0:
                    hard_occluders = spawn_occluder_vehicles(
                        world, blueprint_library, wl_map, wp.transform.location, tm_port,
                        int(args.hard_occluders), rng
                    )
                if args.hard_jaywalkers > 0:
                    hw, hc = spawn_jaywalkers(
                        world, blueprint_library, wp.transform.location,
                        int(args.hard_jaywalkers), rng
                    )
                    hard_walkers, hard_controllers = hw, hc

                settle_after_scenario_change(world, sensor_queue, settle_ticks=15, flush=True)

            world.tick()
            frame = world.get_snapshot().frame

            missing = len(SENSORS)
            threads = []

            t0 = time.time()
            while missing > 0:
                if time.time() - t0 > 5.0:
                    print(f"[WARN] sensor timeout at frame={frame} (missing={missing})")
                    break
                try:
                    sensor_id, frame_id, data = sensor_queue.get(True, timeout=1.0)
                except Empty:
                    continue
                if frame_id != frame:
                    continue

                th = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_id, data, endpoint))
                th.start()
                threads.append(th)
                missing -= 1

            for th in threads:
                th.join()

            save_annotations(
                vehicle, frame, endpoint, map_name,
                pose_format=args.pose_format,
                scene_tokens=scene_tokens
            )

            gc.collect()

        for _ in range(5):
            world.tick()

    finally:
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass

        try:
            vehicle.destroy()
        except Exception:
            pass

        try:
            all_actors = world.get_actors()
            for a in all_actors:
                if a.type_id.startswith("vehicle.") and a.attributes.get("role_name") in ("autopilot", "occluder"):
                    try:
                        a.destroy()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception:
            pass


if __name__ == "__main__":
    main()
