"""
Generate Figure 4: Dataset distribution comparison (Carla-OpenLane vs OpenLane-V2, train set only)

Three panels:
(a) Lane geometry: length & curvature distribution
(b) Environmental diversity: weather & time-of-day (Carla-OpenLane only)
(c) Topology complexity: intersection ratio & connection density
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Paths
CARLA_TRAIN = Path("/home/user/LaneSegNet/data/Carla-OLV2/data_dict_carla_train_argoverse2_ls.pkl")
OLV2_TRAIN = Path("/home/user/OpenLane-V2/data/OpenLane-V2/data_dict_subset_A_train_ls.pkl")

def load_data(pkl_path):
    """Load pickle data - returns list of sample dicts"""
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Data is dict with keys as tuples (split, segment, timestamp)
    samples = list(data.values())
    print(f"  Found {len(samples)} samples")
    return samples

def analyze_lane_geometry(samples, dataset_name):
    """Extract lane length and curvature statistics"""
    lengths = []
    curvatures = []

    for sample in samples:
        ann = sample.get('annotation', {})
        for lane in ann.get('lane_segment', []):
            # Lane centerline is in 'centerline' field
            centerline = lane.get('centerline', [])
            if len(centerline) < 2:
                continue

            # Calculate length
            points = np.array(centerline)
            diffs = np.diff(points, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(segment_lengths)
            lengths.append(total_length)

            # Calculate curvature (total angle change / length)
            if len(centerline) >= 3:
                vectors = diffs / (segment_lengths[:, None] + 1e-6)
                angles = np.arccos(np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1))
                curvature = np.sum(angles) / (total_length + 1e-6)
                curvatures.append(curvature)

    print(f"{dataset_name} lane geometry:")
    print(f"  Total lanes: {len(lengths)}")
    if lengths:
        print(f"  Lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}")
    if curvatures:
        print(f"  Curvatures: mean={np.mean(curvatures):.4f}, std={np.std(curvatures):.4f}")

    return np.array(lengths), np.array(curvatures)

def analyze_weather_time(samples, dataset_name):
    """Extract weather and time-of-day distribution (Carla-OpenLane only)"""
    weather_counts = {}
    time_counts = {}

    for sample in samples:
        meta = sample.get('meta_data', {})
        scene_desc = meta.get('scene_description', {})
        tokens = scene_desc.get('tokens', [])

        # Parse tokens for weather and time
        # Time: afternoon, dusk, night
        # Weather: clear, rain, heavy_rain, cloudy, etc.
        for token in tokens:
            token_lower = token.lower()
            if token_lower in ['afternoon', 'noon', 'dusk', 'night', 'morning']:
                time_counts[token] = time_counts.get(token, 0) + 1
            elif 'rain' in token_lower or token_lower in ['clear', 'cloudy', 'wet', 'overcast']:
                weather_counts[token] = weather_counts.get(token, 0) + 1

    print(f"{dataset_name} environmental diversity:")
    print(f"  Weather: {weather_counts}")
    print(f"  Time: {time_counts}")

    return weather_counts, time_counts

def analyze_topology(samples, dataset_name):
    """Extract topology complexity statistics"""
    intersection_frames = 0
    total_frames = len(samples)
    connection_counts = []

    for sample in samples:
        ann = sample.get('annotation', {})
        # Count topology connections
        ll_connections = len(ann.get('topology_lsls', []))
        lt_connections = len(ann.get('topology_lste', []))
        total_connections = ll_connections + lt_connections
        connection_counts.append(total_connections)

        # Check if intersection (heuristic: high connection density or has traffic elements)
        has_traffic = len(ann.get('traffic_element', [])) > 0
        high_density = total_connections > 5
        if has_traffic or high_density:
            intersection_frames += 1

    intersection_ratio = intersection_frames / total_frames if total_frames > 0 else 0
    avg_connections = np.mean(connection_counts) if connection_counts else 0

    print(f"{dataset_name} topology complexity:")
    print(f"  Intersection ratio: {intersection_ratio:.2%}")
    print(f"  Avg connections per frame: {avg_connections:.2f}")

    return intersection_ratio, np.array(connection_counts)

def plot_comparison(carla_data, olv2_data):
    """Generate 3-panel comparison figure"""
    fig = plt.figure(figsize=(18, 5))

    # Analyze data
    carla_lengths, carla_curvatures = analyze_lane_geometry(carla_data, "Carla-OpenLane")
    olv2_lengths, olv2_curvatures = analyze_lane_geometry(olv2_data, "OpenLane-V2")
    weather_counts, time_counts = analyze_weather_time(carla_data, "Carla-OpenLane")
    carla_int_ratio, carla_connections = analyze_topology(carla_data, "Carla-OpenLane")
    olv2_int_ratio, olv2_connections = analyze_topology(olv2_data, "OpenLane-V2")

    # (a) Lane geometry - Histograms for length & curvature
    # Create 2x1 subplot grid for histograms
    ax1 = plt.subplot(2, 6, (1, 2))
    ax2 = plt.subplot(2, 6, (7, 8))

    # Length distribution
    bins_len = np.linspace(0, 100, 30)
    ax1.hist(carla_lengths, bins=bins_len, alpha=0.6, label='Carla-OpenLane',
             color='skyblue', density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(olv2_lengths, bins=bins_len, alpha=0.6, label='OpenLane-V2',
             color='coral', density=True, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Lane Length (m)', fontsize=9)
    ax1.set_ylabel('Density', fontsize=9)
    ax1.set_title('Lane Length Distribution', fontsize=10, pad=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Curvature distribution
    bins_curv = np.linspace(0, 0.15, 30)
    ax2.hist(carla_curvatures, bins=bins_curv, alpha=0.6, label='Carla-OpenLane',
             color='skyblue', density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(olv2_curvatures, bins=bins_curv, alpha=0.6, label='OpenLane-V2',
             color='coral', density=True, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Curvature (rad/m)', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.set_title('Lane Curvature Distribution', fontsize=10, pad=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add main title for (a)
    fig.text(0.13, 0.95, '(a) Lane Geometry', fontsize=11, fontweight='bold')

    # (b) Environmental diversity - Pie charts
    ax5 = plt.subplot(2, 6, (3, 9))

    # Process weather data
    weather_map = {
        'clear': 'Clear', 'cloudy': 'Clear',
        'heavy_rain': 'Rain', 'rain': 'Rain'
    }
    simplified_weather = {}
    for w, count in weather_counts.items():
        category = weather_map.get(w, 'Other')
        simplified_weather[category] = simplified_weather.get(category, 0) + count

    # Process time data
    time_map = {
        'afternoon': 'Afternoon', 'noon': 'Afternoon',
        'dusk': 'Dusk', 'night': 'Night'
    }
    simplified_time = {}
    for t, count in time_counts.items():
        category = time_map.get(t, 'Other')
        simplified_time[category] = simplified_time.get(category, 0) + count

    # Create pie charts side by side
    total_weather = sum(simplified_weather.values())
    total_time = sum(simplified_time.values())

    weather_labels = list(simplified_weather.keys())
    weather_sizes = [simplified_weather[k]/total_weather*100 for k in weather_labels]
    time_labels = list(simplified_time.keys())
    time_sizes = [simplified_time[k]/total_time*100 for k in time_labels]

    # Weather pie
    colors_weather = ['#87ceeb', '#ff7f50']  # skyblue, coral
    wedges1, texts1, autotexts1 = ax5.pie(weather_sizes, labels=weather_labels, autopct='%1.1f%%',
                                            colors=colors_weather, startangle=90,
                                            textprops={'fontsize': 9})
    ax5.set_title('Weather Distribution\n(Carla-OpenLane)', fontsize=10, pad=10)

    # Time pie
    ax6 = plt.subplot(2, 6, (4, 10))
    # Match colors to time: Afternoon(gold), Dusk(orange), Night(steelblue)
    time_color_map = {'Afternoon': '#ffd700', 'Dusk': '#daa520', 'Night': '#4682b4'}
    colors_time = [time_color_map.get(label, '#cccccc') for label in time_labels]
    wedges2, texts2, autotexts2 = ax6.pie(time_sizes, labels=time_labels, autopct='%1.1f%%',
                                            colors=colors_time, startangle=90,
                                            textprops={'fontsize': 9})
    ax6.set_title('Time Distribution\n(Carla-OpenLane)', fontsize=10, pad=10)

    fig.text(0.47, 0.95, '(b) Environmental Diversity', fontsize=11, fontweight='bold')

    # (c) Topology complexity
    ax7 = plt.subplot(1, 6, (5, 6))
    x = np.arange(2)
    width = 0.35

    carla_stats = [carla_int_ratio * 100, np.mean(carla_connections)]
    olv2_stats = [olv2_int_ratio * 100, np.mean(olv2_connections)]

    ax7.bar(x - width/2, carla_stats, width, label='Carla-OpenLane', color='skyblue')
    ax7.bar(x + width/2, olv2_stats, width, label='OpenLane-V2', color='coral')

    ax7.set_ylabel('Value', fontsize=10)
    ax7.set_title('(c) Topology Complexity', fontsize=11, fontweight='bold', pad=15)
    ax7.set_xticks(x)
    ax7.set_xticklabels(['Intersection\nRatio (%)', 'Avg Connections\nper Frame'], fontsize=9)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path_pdf = Path("/home/user/LaneSegNet/fig/fig04.pdf")
    output_path_png = Path("/home/user/LaneSegNet/fig/fig04.png")
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path_pdf}")
    print(f"Figure saved to {output_path_png}")

    plt.show()

if __name__ == "__main__":
    # Load data
    carla_data = load_data(CARLA_TRAIN)
    olv2_data = load_data(OLV2_TRAIN)

    # Generate comparison figure
    plot_comparison(carla_data, olv2_data)
