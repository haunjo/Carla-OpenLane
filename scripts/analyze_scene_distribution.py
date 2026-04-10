#!/usr/bin/env python3
"""
Train 데이터셋에서 날씨, 지역, 시간대 분포를 분석하는 스크립트
"""

import os
import json
import glob
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def extract_scene_info_from_dataset(dataset_path):
    """데이터셋에서 scene description 정보 추출"""

    # 통계를 위한 카운터
    time_of_day_counter = Counter()
    weather_counter = Counter()
    road_type_counter = Counter()
    town_counter = Counter()  # Town 카운터 추가

    # 모든 조합 추적
    combined_counter = Counter()

    # 세그먼트별 통계
    segment_scenes = defaultdict(set)

    # 모든 세그먼트 디렉토리 찾기
    segment_dirs = glob.glob(os.path.join(dataset_path, '*'))
    segment_dirs = [d for d in segment_dirs if os.path.isdir(d)]

    print(f"발견된 segment 디렉토리 수: {len(segment_dirs)}")

    total_files = 0
    missing_scene_desc = 0

    for segment_dir in tqdm(segment_dirs, desc="Processing segments"):
        segment_id = os.path.basename(segment_dir)
        info_dir = os.path.join(segment_dir, 'info')

        if not os.path.exists(info_dir):
            continue

        # JSON 파일 찾기 (-ls.json 제외)
        json_files = glob.glob(os.path.join(info_dir, '*.json'))
        json_files = [f for f in json_files if not f.endswith('-ls.json')]

        for json_file in json_files:
            total_files += 1
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # scene_description 추출
                if 'meta_data' in data and 'scene_description' in data['meta_data']:
                    scene_desc = data['meta_data']['scene_description']

                    if 'tokens' in scene_desc and len(scene_desc['tokens']) >= 3:
                        tokens = scene_desc['tokens']
                        time_of_day = tokens[0]

                        # 토큰 개수에 따라 파싱 방법 변경
                        if len(tokens) == 3:
                            # 3개 토큰: [time, weather, road]
                            weather = tokens[1]
                            road_type = tokens[2]
                        elif len(tokens) >= 4:
                            # 4개 이상 토큰: [time, weather1, weather2, road]
                            # weather2는 무시하고 weather1만 사용
                            weather = tokens[1]
                            road_type = tokens[3]
                        else:
                            # 예상치 못한 경우
                            continue

                        # 날씨 카테고리 단순화
                        weather_mapping = {
                            'clear': 'clear',
                            'cloudy': 'cloudy',
                            'overcast': 'cloudy',  # overcast도 cloudy로
                            'soft_rain': 'soft_rain',
                            'mid_rain': 'mid_rain',
                            'rain': 'mid_rain',  # rain을 mid_rain으로
                            'hard_rain': 'hard_rain',
                            'heavy_rain': 'hard_rain',  # heavy_rain을 hard_rain으로
                            'fog': 'fog',
                            'wet': 'cloudy',  # wet을 cloudy로
                            'wet_cloudy': 'cloudy'  # wet_cloudy를 cloudy로
                        }
                        weather = weather_mapping.get(weather, weather)

                        # Town 정보 추출 (source_id에서)
                        town_name = "Unknown"
                        if 'source_id' in data['meta_data']:
                            source_id = data['meta_data']['source_id']
                            # "Carla/Maps/Town01124" 또는 "Carla/Maps/Town03/frame_31468" 형식
                            if 'Town' in source_id:
                                parts = source_id.split('/')
                                for part in parts:
                                    if part.startswith('Town'):
                                        if len(part) == 6 and part[4:6].isdigit():  # Town03 형식
                                            town_name = part
                                            break
                                        elif len(part) > 6:  # Town01124 형식
                                            town_num = part[4:6]
                                            town_name = f"Town{town_num}"
                                            break
                                # Town10HD를 Town10으로 통합
                                if town_name == 'Town10HD':
                                    town_name = 'Town10'

                        # 개별 카운터 업데이트
                        time_of_day_counter[time_of_day] += 1
                        weather_counter[weather] += 1
                        road_type_counter[road_type] += 1
                        town_counter[town_name] += 1

                        # 조합 카운터 업데이트
                        combination = f"{time_of_day}_{weather}_{road_type}"
                        combined_counter[combination] += 1

                        # 세그먼트별 scene 추적
                        segment_scenes[segment_id].add(combination)
                    else:
                        missing_scene_desc += 1
                else:
                    missing_scene_desc += 1

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                # print(f"Error processing {json_file}: {e}")
                continue

    return {
        'time_of_day': dict(time_of_day_counter),
        'weather': dict(weather_counter),
        'road_type': dict(road_type_counter),
        'town': dict(town_counter),
        'combined': dict(combined_counter),
        'total_files': total_files,
        'missing_scene_desc': missing_scene_desc,
        'segment_count': len(segment_dirs),
        'unique_segments_per_scene': {scene: len([s for s, scenes in segment_scenes.items() if scene in scenes])
                                      for scene in combined_counter.keys()}
    }

def print_statistics(stats):
    """통계 출력"""
    print("\n" + "="*80)
    print("CARLA-OpenLane Train Dataset Scene Distribution Analysis")
    print("="*80)

    print(f"\n[전체 통계]")
    print(f"   • 총 세그먼트 수: {stats['segment_count']:,}")
    print(f"   • 총 프레임 수: {stats['total_files']:,}")
    print(f"   • Scene description 누락: {stats['missing_scene_desc']:,}")

    # 시간대 분포
    print(f"\n[시간대 분포] (Time of Day):")
    total_time = sum(stats['time_of_day'].values())
    for time, count in sorted(stats['time_of_day'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_time) * 100
        print(f"   • {time:<15} : {count:7,} ({percentage:5.1f}%)")

    # 날씨 분포
    print(f"\n[날씨 분포] (Weather):")
    total_weather = sum(stats['weather'].values())
    for weather, count in sorted(stats['weather'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_weather) * 100
        print(f"   • {weather:<15} : {count:7,} ({percentage:5.1f}%)")

    # 도로 유형 분포
    print(f"\n[도로 유형 분포] (Road Type):")
    total_road = sum(stats['road_type'].values())
    for road, count in sorted(stats['road_type'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_road) * 100
        print(f"   • {road:<15} : {count:7,} ({percentage:5.1f}%)")

    # Town 분포
    print(f"\n[Town 분포]:")
    total_town = sum(stats['town'].values())
    for town, count in sorted(stats['town'].items(), key=lambda x: x[1], reverse=True):
        if town != "Unknown":  # Unknown 제외
            percentage = (count / total_town) * 100
            print(f"   • {town:<15} : {count:7,} ({percentage:5.1f}%)")

    # 상위 10개 조합
    print(f"\n[상위 10개 조합] (Time_Weather_Road):")
    combined_sorted = sorted(stats['combined'].items(), key=lambda x: x[1], reverse=True)
    total_combined = sum(stats['combined'].values())
    for i, (combo, count) in enumerate(combined_sorted[:10], 1):
        percentage = (count / total_combined) * 100
        parts = combo.split('_')
        print(f"   {i:2}. {combo:<40} : {count:7,} ({percentage:5.1f}%)")

    print(f"\n[다양성 통계]")
    print(f"   • 고유 시간대 수: {len(stats['time_of_day'])}")
    print(f"   • 고유 날씨 수: {len(stats['weather'])}")
    print(f"   • 고유 도로 유형 수: {len(stats['road_type'])}")
    print(f"   • 고유 Town 수: {len([t for t in stats['town'] if t != 'Unknown'])}")
    print(f"   • 고유 조합 수: {len(stats['combined'])}")

def create_pie_charts(stats, output_dir='scene_distribution_charts'):
    """파이 차트 생성"""
    os.makedirs(output_dir, exist_ok=True)

    # 스타일 설정
    plt.style.use('default')

    # 3개의 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 색상 팔레트
    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    # 1. 시간대 파이 차트
    time_data = sorted(stats['time_of_day'].items(), key=lambda x: x[1], reverse=True)
    time_labels = [k for k, v in time_data]  # 개수 제거
    time_values = [v for k, v in time_data]

    wedges, texts, autotexts = axes[0].pie(time_values, labels=time_labels, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(time_data)], textprops={'fontsize': 14})
    axes[0].set_title('Time of Day Distribution', fontsize=18, fontweight='bold', pad=20)

    # 2. 날씨 파이 차트
    weather_data = sorted(stats['weather'].items(), key=lambda x: x[1], reverse=True)
    weather_labels = [k for k, v in weather_data]  # 개수 제거
    weather_values = [v for k, v in weather_data]

    wedges, texts, autotexts = axes[1].pie(weather_values, labels=weather_labels, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(weather_data)], textprops={'fontsize': 14})
    axes[1].set_title('Weather Distribution', fontsize=18, fontweight='bold', pad=20)

    # 3. Town 파이 차트 (도로 유형 대신)
    town_data = [(k, v) for k, v in stats['town'].items() if k != "Unknown"]  # Unknown 제외
    town_data = sorted(town_data, key=lambda x: x[1], reverse=True)
    town_labels = [k for k, v in town_data]  # 개수 제거
    town_values = [v for k, v in town_data]

    # Town이 많을 수 있으므로 색상 팔레트 확장
    if len(town_data) > 12:
        colors_town = plt.cm.tab20(np.linspace(0, 1, len(town_data)))
    else:
        colors_town = colors[:len(town_data)]

    wedges, texts, autotexts = axes[2].pie(town_values, labels=town_labels, autopct='%1.1f%%',
                startangle=90, colors=colors_town, textprops={'fontsize': 14})
    axes[2].set_title('Town Distribution', fontsize=18, fontweight='bold', pad=20)

    # 전체 제목
    fig.suptitle('CARLA-OpenLane Train Dataset Scene Distribution',
                 fontsize=20, fontweight='bold', y=1.02)

    # 퍼센트 텍스트 폰트 크기 조정
    for ax in axes:
        for autotext in ax.texts:
            if '%' in autotext.get_text():
                autotext.set_fontsize(12)
                autotext.set_weight('bold')

    plt.tight_layout()

    # 저장
    output_file = os.path.join(output_dir, 'scene_distribution_pie_charts.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[파이 차트 저장]: {output_file}")

    # plt.show()  # GUI 환경이 아닌 경우 주석처리

def main():
    """메인 함수"""

    # 데이터셋 경로 (train)
    train_path = '/home/user/dataset/Carla-OpenLane/train'

    if not os.path.exists(train_path):
        print(f"❌ 경로를 찾을 수 없음: {train_path}")
        return

    print(f"[데이터셋 분석 시작]: {train_path}")

    # Scene 정보 추출
    stats = extract_scene_info_from_dataset(train_path)

    # 통계 출력
    print_statistics(stats)

    # 파이 차트 생성
    create_pie_charts(stats)

    # JSON으로 저장 (나중에 활용)
    import json
    with open('scene_distribution_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n[통계 데이터 저장]: scene_distribution_stats.json")

if __name__ == "__main__":
    main()