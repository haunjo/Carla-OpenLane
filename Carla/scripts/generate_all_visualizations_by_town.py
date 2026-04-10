#!/usr/bin/env python3
"""
CARLA-OLV2 Train/Val Dataset Town별 시각화 생성 스크립트

Train과 Val 데이터셋을 Town별로 분리하여 각각의 시각화를 생성합니다.
meta_data의 source_id에서 Town 이름을 추출하여 분류합니다.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
import glob
from collections import defaultdict
import time

# CARLA 모듈 임포트
# try:
#     import glob as glob_module
#     carla_egg = glob_module.glob('/home/user/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-*.egg')
#     if carla_egg:
#         sys.path.insert(0, carla_egg[0])
#     import carla
#     CARLA_AVAILABLE = True
#     print("CARLA Python API 모듈 로드 성공")
# except ImportError as e:
#     CARLA_AVAILABLE = False
#     print(f"CARLA Python API 없음: {e}. 기존 저장된 HD map 데이터를 사용합니다.")

def extract_town_from_source_id(source_id):
    """source_id에서 Town 이름을 추출

    예시: "Carla/Maps/Town03/frame_31468" -> "Town03" (새 형식, 슬래시 4개)
         "Carla/Maps/Town10HD/frame_31468" -> "Town10" (새 형식 Town10HD)
         "Carla/Maps/Town10HD14686" -> "Town10" (구 형식, 슬래시 3개)
    """
    if not source_id or 'Town' not in source_id:
        return "Unknown"

    try:
        parts = source_id.split('/')

        # 슬래시 개수로 형식 판단
        if len(parts) == 4:
            # 새 형식: "Carla/Maps/Town03/frame_31468"
            town_part = parts[2]  # "Town03" or "Town10HD"
            if town_part.startswith('Town'):
                # Town10HD를 Town10으로 통합
                if town_part == 'Town10HD':
                    return 'Town10'
                return town_part
        elif len(parts) == 3:
            # 구 형식: "Carla/Maps/Town10HD14686"
            town_part = parts[2]  # "Town10HD14686"
            if town_part.startswith('Town') and len(town_part) >= 6:
                town_num = town_part[4:6]  # "10"
                return f"Town{town_num}"

        return "Unknown"
    except:
        return "Unknown"

def extract_pose_coordinates_by_town(dataset_path):
    """데이터셋에서 Town별로 pose 좌표 추출"""
    coordinates_by_town = defaultdict(list)
    metadata_by_town = defaultdict(lambda: defaultdict(list))

    segment_dirs = glob.glob(os.path.join(dataset_path, '*'))
    segment_dirs = [d for d in segment_dirs if os.path.isdir(d)]
    print(segment_dirs)

    print(f"발견된 segment 디렉토리 수: {len(segment_dirs)}")

    town_stats = defaultdict(int)

    for segment_dir in tqdm(segment_dirs, desc="Processing segments"):
        segment_id = os.path.basename(segment_dir)
        info_dir = os.path.join(segment_dir, 'info')

        if not os.path.exists(info_dir):
            continue

        json_files = glob.glob(os.path.join(info_dir, '*.json'))
        json_files = [f for f in json_files if not f.endswith('-ls.json')]

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'pose' in data and 'translation' in data['pose']:
                    x, y, z = data['pose']['translation']


                    # Town 정보 추출
                    town_name = "Unknown"
                    if 'meta_data' in data and 'source_id' in data['meta_data']:
                        source_id = data['meta_data']['source_id']
                        town_name = extract_town_from_source_id(source_id)

                    coordinates_by_town[town_name].append((x, y))
                    town_stats[town_name] += 1

                    metadata_by_town[town_name][segment_id].append({
                        'x': x, 'y': y, 'z': z,
                        'town': town_name,
                        'timestamp': data.get('timestamp', 0),
                        'source_id': data.get('meta_data', {}).get('source_id', '')
                    })

            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue

    print(f"\n발견된 Town별 데이터:")
    for town, count in sorted(town_stats.items()):
        print(f"   • {town}: {count:,}개 pose")

    return dict(coordinates_by_town), dict(metadata_by_town)

def load_or_extract_hd_map_data(town_name):
    """Town별 HD Map 데이터 로드 (JSON 우선, OpenDRIVE는 fallback)"""
    if town_name == "Town10HD":
        town_name = "Town10"
    # 1순위: CARLA 서버에서 추출한 waypoint 기반 JSON (가장 정확)
    map_file = f"{town_name.lower()}_hd_map_data.json"
    if os.path.exists(map_file):
        try:
            with open(map_file, 'r') as f:
                map_data = json.load(f)
            print(f"{town_name} HD map JSON 로드 (waypoint-based)")
            return map_data['road_segments'], map_data['spawn_points']
        except Exception as e:
            print(f"[WARNING]  {town_name} JSON 로드 실패: {e}")


def create_town_visualization(town_name, coordinates, metadata, road_segments, spawn_points,
                            output_dir, dataset_type):
    """특정 Town에 대한 시각화 생성"""

    os.makedirs(output_dir, exist_ok=True)

    if not coordinates:
        print(f"[ERROR] {town_name}: 좌표 데이터 없음")
        return

    print(f"[INFO] {town_name} 시각화 생성 중... ({len(coordinates)}개 pose)")

    # 논문용 고해상도 플롯 설정
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)

    # 1. HD Map 도로 네트워크 그리기 (centerline only)
    if road_segments:
        print(f"   - {town_name} HD Map 도로 네트워크 렌더링...")

        # Junction lines (노란색)
        junction_lines = []
        # Driving lane centerlines (회색)
        driving_lines = []

        for segment in road_segments:
            points = segment['points']
            if len(points) < 2:
                continue

            # Centerline points
            line_points = [(p['x'], p['y']) for p in points]

            # Line segments
            for i in range(len(line_points) - 1):
                line_segment = [line_points[i], line_points[i + 1]]

                if segment['is_junction']:
                    junction_lines.append(line_segment)
                elif 'Driving' in segment['lane_type']:
                    driving_lines.append(line_segment)
                # Skip non-driving lanes (sidewalks, etc.)

        # Draw driving lanes (gray, thin, background)
        if driving_lines:
            lc_driving = LineCollection(driving_lines, colors='gray', alpha=0.3, linewidths=1.0, zorder=1)
            ax.add_collection(lc_driving)

        # Draw junctions (orange, slightly thicker)
        if junction_lines:
            lc_junction = LineCollection(junction_lines, colors='orange', alpha=0.5, linewidths=1.5, zorder=2)
            ax.add_collection(lc_junction)

    # 2. Spawn points 표시 (생략 - 너무 복잡함)
    # if spawn_points:
    #     spawn_x = [sp['x'] for sp in spawn_points]
    #     spawn_y = [sp['y'] for sp in spawn_points]
    #     ax.scatter(spawn_x, spawn_y, c='green', s=20, marker='s', alpha=0.4,
    #               edgecolors='darkgreen', linewidth=0.3, zorder=2)

    # 3. 세그먼트별 trajectory 표시 (scene별로 선명하게)
    print(f"   - {town_name} 세그먼트별 trajectories 렌더링...")

    segment_keys = list(metadata.keys())
    n_segments = len(segment_keys)

    # 색상 설정
    if n_segments <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_segments))
    else:
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, min(12, n_segments - 20)))
        colors3 = plt.cm.Pastel1(np.linspace(0, 1, min(9, max(0, n_segments - 32))))
        colors = np.vstack([colors1, colors2, colors3])[:n_segments]

    for i, segment_id in enumerate(segment_keys):
        segment_coords = metadata[segment_id]
        if not segment_coords:
            continue

        # timestamp로 정렬
        segment_coords_sorted = sorted(segment_coords, key=lambda x: x.get('timestamp', 0))

        seg_x = [coord['x'] for coord in segment_coords_sorted]
        seg_y = [coord['y'] for coord in segment_coords_sorted]

        color = colors[i] if i < len(colors) else plt.cm.tab20(i % 20)

        # 세그먼트별 선으로 표시 (논문용 - 선명하고 굵게)
        if len(seg_x) > 1:
            # 메인 경로 선 (논문용으로 조금 더 굵게)
            ax.plot(seg_x, seg_y, color=color, alpha=0.9, linewidth=2.5,
                   linestyle='-', zorder=4, solid_capstyle='round')

            # 시작점과 끝점 (선명하게)
            ax.scatter(seg_x[0], seg_y[0], color=color, s=50, marker='o',
                      edgecolors='darkgreen', linewidth=2.5, zorder=5, alpha=1.0)
            ax.scatter(seg_x[-1], seg_y[-1], color=color, s=50, marker='s',
                      edgecolors='darkred', linewidth=2.5, zorder=5, alpha=1.0)
        else:
            ax.scatter(seg_x[0], seg_y[0], color=color, s=40, marker='o',
                      edgecolors='black', linewidth=2, zorder=5, alpha=1.0)

    # 4. 통계 정보 계산
    pose_x = np.array([coord[0] for coord in coordinates])
    pose_y = np.array([coord[1] for coord in coordinates])

    # Town 면적 계산 (width x height)
    town_width = max(pose_x) - min(pose_x)
    town_height = max(pose_y) - min(pose_y)

    # Scene 개수 (고유 segment 수)
    scene_count = len(metadata)

    # Frame 개수 (총 pose 수)
    frame_count = len(coordinates)

    # 5. 스타일링 (논문용)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'CARLA {town_name} {dataset_type.title()} Dataset',
                 fontsize=24, fontweight='bold', pad=15)

    ax.set_xticks([])
    ax.set_yticks([])

    # 맵 경계 설정 (논문용 - 타이트한 패딩)
    padding = 20
    ax.set_xlim(min(pose_x) - padding, max(pose_x) + padding)
    ax.set_ylim(min(pose_y) - padding, max(pose_y) + padding)

    # 6. Summary 정보 박스 (왼쪽 위)
    summary_text = f"""{town_name}
Area: {town_width:.0f}m × {town_height:.0f}m
Scenes: {scene_count:,}
Frames: {frame_count:,}"""

    # 텍스트 박스 스타일
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)

    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=18,
           verticalalignment='top', horizontalalignment='left', bbox=bbox_props,
           fontweight='bold', zorder=10)

    # 논문용 - 깔끔한 grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout(pad=0.5)

    # 저장 (논문용 고해상도)
    output_file = os.path.join(output_dir, f'{town_name.lower()}_{dataset_type}_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[SUCCESS] {town_name} 시각화 저장: {output_file}")

    plt.close(fig)  # 메모리 정리

def main():
    """메인 함수"""
    print("🎨 CARLA-OLV2 Town별 시각화 생성 시작...")
    print("=" * 80)

    # 데이터셋 경로 설정 (Train만)
    datasets = [
        {
            'name': 'Train Dataset',
            'path': '../Carla-OpenLane/train/',
            'type': 'train'
        },
        # {
        #     'name': 'Val Dataset',
        #     'path': '../Carla-OpenLane/subset_A/val/',
        #     'type': 'val'
        # }
    ]

    # 각 데이터셋 처리
    for dataset in datasets:
        print(f"\n[PROCESSING] {dataset['name']} 처리 시작...")

        if not os.path.exists(dataset['path']):
            print(f"[ERROR] {dataset['path']} 경로 없음")
            continue

        # Town별 데이터 추출
        coordinates_by_town, metadata_by_town = extract_pose_coordinates_by_town(dataset['path'])

        if not coordinates_by_town:
            print(f"[ERROR] {dataset['name']}: 추출된 데이터 없음")
            continue

        # 각 Town별 시각화 생성
        for town_name in sorted(coordinates_by_town.keys()):
            if town_name == "Unknown":
                continue

            print(f"\n[PROCESSING] {town_name} 처리 중...")

            coordinates = coordinates_by_town[town_name]
            metadata = metadata_by_town[town_name]

            if len(coordinates) < 10:  # 데이터가 너무 적으면 건너뛰기
                print(f"[WARNING] {town_name}: 데이터 부족 ({len(coordinates)}개) - 건너뛰기")
                continue

            # HD Map 데이터 로드/추출
            road_segments, spawn_points = load_or_extract_hd_map_data(town_name)

            # 출력 디렉토리 설정
            output_dir = f"town_visualizations_{dataset['type']}"

            # 시각화 생성
            create_town_visualization(
                town_name, coordinates, metadata,
                road_segments, spawn_points,
                output_dir, dataset['type']
            )

            # 메모리 정리를 위한 잠시 대기
            time.sleep(1)

    # 결과 요약
    print(f"\n{'='*80}")
    print("[COMPLETE] Town별 시각화 생성 완료!")
    print(f"{'='*80}")

    # 생성된 결과물 확인
    output_dirs = ['town_visualizations_train', 'town_visualizations_val']

    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            print(f"[DIR] {output_dir}/: {len(files)}개 시각화")

            # Town별 파일 목록
            for file in sorted(files):
                print(f"   • {file}")
        else:
            print(f"[ERROR] {output_dir}/: 생성되지 않음")

if __name__ == "__main__":
    main()