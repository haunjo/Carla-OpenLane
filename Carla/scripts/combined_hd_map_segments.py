#!/usr/bin/env python3
"""
CARLA Town07 HD Map + Segment별 Pose Distribution 통합 시각화

pose_by_segment.png의 세그먼트별 색상 구분과 Town07 HD map을 합성하여
하나의 통합된 이미지를 생성합니다.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from tqdm import tqdm
import glob
from collections import defaultdict
import time

# CARLA 모듈 임포트
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'PythonAPI/carla'))
    import carla
    CARLA_AVAILABLE = True
    print("✅ CARLA Python API 모듈 로드 성공")
except ImportError as e:
    CARLA_AVAILABLE = False
    print(f"⚠️ CARLA Python API 없음. 기존 저장된 HD map 데이터를 사용합니다.")

class CombinedMapVisualizer:
    """HD Map + Segment별 Pose 데이터 통합 시각화 클래스"""

    def __init__(self):
        self.road_segments = None
        self.spawn_points = None
        self.pose_coordinates = []
        self.metadata = {}

    def extract_carla_map_data(self, host='127.0.0.1', port=2000):
        """CARLA에서 실시간 HD map 데이터 추출"""
        if not CARLA_AVAILABLE:
            return False

        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            world = client.get_world()

            print("🗺️ Town07으로 맵 전환 중...")
            world = client.load_world('Town07')
            carla_map = world.get_map()
            time.sleep(2)

            print("📊 HD map 데이터 추출 중...")

            # 도로 네트워크 추출
            topology = carla_map.get_topology()
            road_segments = []

            for waypoint_pair in tqdm(topology, desc="Extracting road topology"):
                start_wp, end_wp = waypoint_pair
                waypoints = [start_wp]
                current_wp = start_wp

                while current_wp.transform.location.distance(end_wp.transform.location) > 1.0:
                    next_waypoints = current_wp.next(1.0)
                    if next_waypoints:
                        current_wp = next_waypoints[0]
                        waypoints.append(current_wp)
                    else:
                        break

                segment_points = []
                for wp in waypoints:
                    segment_points.append({
                        'x': wp.transform.location.x,
                        'y': wp.transform.location.y,
                        'z': wp.transform.location.z,
                        'lane_width': wp.lane_width,
                        'is_junction': wp.is_junction
                    })

                road_segments.append({
                    'points': segment_points,
                    'lane_type': str(start_wp.lane_type),
                    'is_junction': start_wp.is_junction
                })

            # Spawn points 추출
            spawn_points = []
            for sp in carla_map.get_spawn_points():
                spawn_points.append({
                    'x': sp.location.x,
                    'y': sp.location.y,
                    'yaw': sp.rotation.yaw
                })

            self.road_segments = road_segments
            self.spawn_points = spawn_points

            # 데이터 저장
            map_data = {
                'road_segments': road_segments,
                'spawn_points': spawn_points,
                'map_name': 'Town07'
            }
            with open('town07_hd_map_data.json', 'w') as f:
                json.dump(map_data, f, indent=2)

            print(f"✅ HD map 데이터 추출 완료: {len(road_segments)}개 도로 구간, {len(spawn_points)}개 spawn point")
            return True

        except Exception as e:
            print(f"❌ CARLA HD map 추출 실패: {e}")
            return False

    def load_existing_map_data(self):
        """기존 저장된 HD map 데이터 로드"""
        try:
            with open('town07_hd_map_data.json', 'r') as f:
                map_data = json.load(f)

            self.road_segments = map_data['road_segments']
            self.spawn_points = map_data['spawn_points']

            print(f"✅ 기존 HD map 데이터 로드: {len(self.road_segments)}개 도로 구간, {len(self.spawn_points)}개 spawn point")
            return True
        except Exception as e:
            print(f"❌ HD map 데이터 로드 실패: {e}")
            return False

    def extract_pose_coordinates(self, val_dataset_path):
        """Val 데이터셋에서 pose 좌표 추출 (기존 함수 개선)"""
        coordinates = []
        metadata = defaultdict(list)

        segment_dirs = glob.glob(os.path.join(val_dataset_path, '*'))
        segment_dirs = [d for d in segment_dirs if os.path.isdir(d)]

        print(f"📁 발견된 segment 디렉토리 수: {len(segment_dirs)}")

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
                        # Y 좌표에 마이너스 적용 (CARLA 좌표계 정렬)
                        y = -y
                        coordinates.append((x, y))

                        town_info = "Unknown"
                        if 'meta_data' in data and 'source_id' in data['meta_data']:
                            town_info = data['meta_data']['source_id']

                        metadata[segment_id].append({
                            'x': x, 'y': y, 'z': z,
                            'town': town_info,
                            'timestamp': data.get('timestamp', 0)
                        })

                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    continue

        self.pose_coordinates = coordinates
        self.metadata = dict(metadata)

        print(f"✅ 총 {len(coordinates)}개의 pose 좌표를 {len(metadata)}개 세그먼트에서 추출했습니다.")

    def create_combined_visualization(self, output_dir='combined_visualizations'):
        """HD Map + 세그먼트별 Pose 데이터 통합 시각화"""

        os.makedirs(output_dir, exist_ok=True)

        if not self.road_segments or not self.pose_coordinates:
            print("❌ 필요한 데이터가 없습니다.")
            return

        # 고해상도 플롯 설정
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(20, 16))

        print("🎨 통합 시각화 생성 중...")

        # 1. HD Map 도로 네트워크 그리기
        print("   - HD Map 도로 네트워크 렌더링...")
        road_lines = []
        road_colors = []

        for segment in self.road_segments:
            points = segment['points']
            if len(points) < 2:
                continue

            # 도로 포인트를 LineCollection용 형태로 변환
            line_points = [(p['x'], p['y']) for p in points]

            # 연속된 점들을 선분으로 연결
            for i in range(len(line_points) - 1):
                road_lines.append([line_points[i], line_points[i + 1]])

                # 도로 타입별 색상 결정
                if segment['is_junction']:
                    road_colors.append('orange')
                elif 'Driving' in segment['lane_type']:
                    road_colors.append('gray')
                else:
                    road_colors.append('lightgray')

        # 도로 네트워크를 LineCollection으로 그리기 (배경으로 약하게)
        if road_lines:
            lc = LineCollection(road_lines, colors=road_colors, alpha=0.3, linewidths=1.5)
            ax.add_collection(lc)

        # 2. Spawn points 표시 (배경으로 약하게)
        if self.spawn_points:
            spawn_x = [sp['x'] for sp in self.spawn_points]
            spawn_y = [sp['y'] for sp in self.spawn_points]
            ax.scatter(spawn_x, spawn_y, c='green', s=20, marker='s', alpha=0.4,
                      edgecolors='darkgreen', linewidth=0.3, zorder=2)

        # 3. 세그먼트별 색상으로 Pose 데이터 표시
        print("   - 세그먼트별 Pose 데이터 렌더링...")

        segment_keys = list(self.metadata.keys())
        n_segments = len(segment_keys)

        # 더 많은 색상을 위한 색상 맵 조합
        if n_segments <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_segments))
        else:
            # 여러 색상맵 조합
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, min(12, n_segments - 20)))
            colors3 = plt.cm.Pastel1(np.linspace(0, 1, min(9, max(0, n_segments - 32))))
            colors = np.vstack([colors1, colors2, colors3])[:n_segments]

        for i, segment_id in enumerate(segment_keys):
            segment_coords = self.metadata[segment_id]
            if not segment_coords:
                continue

            # timestamp로 정렬하여 올바른 순서로 연결
            segment_coords_sorted = sorted(segment_coords, key=lambda x: x.get('timestamp', 0))

            seg_x = [coord['x'] for coord in segment_coords_sorted]
            seg_y = [coord['y'] for coord in segment_coords_sorted]

            color = colors[i] if i < len(colors) else plt.cm.tab20(i % 20)

            # 세그먼트별 선으로 표시 (경로 연결) - 더 선명하고 강조
            if len(seg_x) > 1:
                # 배경 선 (더 두껍고 어두운 테두리 효과)
                ax.plot(seg_x, seg_y, color='black', alpha=0.3, linewidth=5.5,
                       linestyle='-', zorder=3, solid_capstyle='round')

                # 메인 선 (더 굵고 선명하게)
                ax.plot(seg_x, seg_y, color=color, alpha=0.95, linewidth=4.0,
                       linestyle='-', zorder=4, solid_capstyle='round')

                # 시작점과 끝점 표시 (더 크고 선명하게)
                ax.scatter(seg_x[0], seg_y[0], c=color, s=80, marker='o',
                          edgecolors='darkgreen', linewidth=3, zorder=6, alpha=1.0)  # 시작점
                ax.scatter(seg_x[-1], seg_y[-1], c=color, s=80, marker='s',
                          edgecolors='darkred', linewidth=3, zorder=6, alpha=1.0)    # 끝점
            else:
                # 단일 포인트인 경우 (더 크고 선명하게)
                ax.scatter(seg_x[0], seg_y[0], c=color, s=60, marker='o',
                          edgecolors='black', linewidth=2, zorder=5, alpha=1.0)

            # 범례 제거됨 (깔끔한 시각화를 위해)

        # 4. 도로별 통계 표시 (옵션)
        print("   - 통계 정보 계산...")

        # Pose 데이터의 도로 커버리지 분석
        pose_x = np.array([coord[0] for coord in self.pose_coordinates])
        pose_y = np.array([coord[1] for coord in self.pose_coordinates])

        # 맵 경계 설정
        all_road_x = []
        all_road_y = []
        for segment in self.road_segments:
            for point in segment['points']:
                all_road_x.append(point['x'])
                all_road_y.append(point['y'])

        if all_road_x and pose_x.size > 0:
            map_bounds = {
                'x_min': min(min(all_road_x), min(pose_x)) - 50,
                'x_max': max(max(all_road_x), max(pose_x)) + 50,
                'y_min': min(min(all_road_y), min(pose_y)) - 50,
                'y_max': max(max(all_road_y), max(pose_y)) + 50
            }
        else:
            map_bounds = {
                'x_min': -500, 'x_max': 500,
                'y_min': -500, 'y_max': 500
            }

        # 5. 통계 정보 계산
        pose_x = np.array([coord[0] for coord in self.pose_coordinates])
        pose_y = np.array([coord[1] for coord in self.pose_coordinates])

        # Town 면적 계산
        town_width = max(pose_x) - min(pose_x)
        town_height = max(pose_y) - min(pose_y)

        # Scene 개수 (고유 segment 수)
        scene_count = len(self.metadata)

        # Frame 개수 (총 pose 수)
        frame_count = len(self.pose_coordinates)

        # 6. 스타일링 및 레이블
        ax.set_xlabel('')  # X축 레이블 제거
        ax.set_ylabel('')  # Y축 레이블 제거
        ax.set_title('CARLA Town 07 Dataset', fontsize=16, fontweight='bold')

        # X, Y 축 눈금 값 제거
        ax.set_xticks([])
        ax.set_yticks([])

        # 7. Summary 정보 박스 (왼쪽 위)
        summary_text = f"""📍 Town 07
📐 Area: {town_width:.0f}m × {town_height:.0f}m
🎬 Scenes: {scene_count:,}
🎞️ Frames: {frame_count:,}"""

        # 텍스트 박스 스타일
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)

        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', horizontalalignment='left', bbox=bbox_props,
               fontweight='bold', zorder=10)

        # 그리드 및 축 설정
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(map_bounds['x_min'], map_bounds['x_max'])
        ax.set_ylim(map_bounds['y_min'], map_bounds['y_max'])
        ax.set_aspect('equal')

        plt.tight_layout()

        # 저장
        output_file = os.path.join(output_dir, 'town07_hd_map_segments_combined.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 통합 시각화 저장 완료: {output_file}")

        plt.show()

        # 6. 추가 분석 시각화 - 밀도 히트맵 + HD Map
        self.create_density_overlay(ax, output_dir)

    def create_density_overlay(self, ax, output_dir):
        """밀도 히트맵 + HD Map 오버레이 생성"""

        plt.figure(figsize=(18, 14))

        pose_x = np.array([coord[0] for coord in self.pose_coordinates])
        pose_y = np.array([coord[1] for coord in self.pose_coordinates])

        # 히트맵 생성
        plt.hist2d(pose_x, pose_y, bins=60, cmap='Reds', alpha=0.7)
        plt.colorbar(label='Pose Density')

        # HD Map 오버레이 (얇은 선으로)
        for segment in self.road_segments:
            points = segment['points']
            if len(points) < 2:
                continue

            xs = [p['x'] for p in points]
            ys = [p['y'] for p in points]

            if segment['is_junction']:
                plt.plot(xs, ys, 'white', alpha=0.8, linewidth=2, linestyle='-')
            else:
                plt.plot(xs, ys, 'black', alpha=0.5, linewidth=1, linestyle='-')

        plt.xlabel('X Coordinate (m)', fontsize=14)
        plt.ylabel('Y Coordinate (m)', fontsize=14)
        plt.title('CARLA Town07: Pose Density Heatmap + Road Network', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()

        density_file = os.path.join(output_dir, 'town07_density_heatmap_overlay.png')
        plt.savefig(density_file, dpi=300, bbox_inches='tight')
        print(f"✅ 밀도 히트맵 오버레이 저장: {density_file}")

        plt.show()

def main():
    """메인 함수"""
    print("🎨 CARLA Town07 HD Map + Segment별 Pose 통합 시각화 시작...")

    visualizer = CombinedMapVisualizer()

    # 1. HD Map 데이터 획득
    print("\n🗺️ HD Map 데이터 준비...")
    if not visualizer.extract_carla_map_data():
        print("   실시간 추출 실패, 기존 데이터 로드 시도...")
        if not visualizer.load_existing_map_data():
            print("❌ HD Map 데이터를 가져올 수 없습니다.")
            print("   CARLA 서버를 실행하거나 기존 town07_hd_map_data.json 파일이 필요합니다.")
            return

    # 2. Pose 데이터 추출
    print("\n📊 Pose 데이터 추출...")
    val_dataset_path = 'Carla-OLV2/val/'

    if not os.path.exists(val_dataset_path):
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {val_dataset_path}")
        return

    visualizer.extract_pose_coordinates(val_dataset_path)

    if not visualizer.pose_coordinates:
        print("❌ 추출된 pose 좌표가 없습니다.")
        return

    # 3. 통합 시각화 생성
    print("\n🎨 통합 시각화 생성...")
    visualizer.create_combined_visualization()

    print("\n✅ 모든 시각화 완료!")
    print("결과 파일:")
    print("  • town07_hd_map_segments_combined.png - HD Map + 세그먼트별 pose")
    print("  • town07_density_heatmap_overlay.png - 밀도 히트맵 + 도로 네트워크")

if __name__ == "__main__":
    main()