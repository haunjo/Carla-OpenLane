"""
Generate Figure 4 (FINAL): Dataset comparison - Geometric Realism only

Single panel:
(a) Geometric Realism: lane length & curvature with KDE (comparable to OLV2)
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
            centerline = lane.get('centerline', [])
            if len(centerline) < 2:
                continue

            # Calculate length
            points = np.array(centerline)
            diffs = np.diff(points, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(segment_lengths)
            lengths.append(total_length)

            # Calculate curvature (in radians)
            if len(centerline) >= 3:
                vectors = diffs / (segment_lengths[:, None] + 1e-6)
                angles = np.arccos(np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1))
                curvature_rad = np.sum(angles) / (total_length + 1e-6)
                curvatures.append(curvature_rad)

    print(f"{dataset_name} lane geometry:")
    print(f"  Total lanes: {len(lengths)}")
    if lengths:
        print(f"  Lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}")
    if curvatures:
        print(f"  Curvatures: mean={np.mean(curvatures):.4f}, std={np.std(curvatures):.4f}")

    return np.array(lengths), np.array(curvatures)

def plot_comparison(carla_data, olv2_data):
    """Generate geometric comparison figure for single column layout"""
    # Increase figure size and make it suitable for single column
    fig = plt.figure(figsize=(7, 6))

    # Analyze data
    carla_lengths, carla_curvatures = analyze_lane_geometry(carla_data, "Carla-OpenLane")
    olv2_lengths, olv2_curvatures = analyze_lane_geometry(olv2_data, "OpenLane-V2")

    # Two subplots stacked vertically for better readability in single column
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    # Length distribution with KDE
    from scipy.stats import gaussian_kde

    # Filter outliers for better KDE
    carla_lengths_clip = carla_lengths[carla_lengths <= 100]
    olv2_lengths_clip = olv2_lengths[olv2_lengths <= 100]

    carla_len_kde = gaussian_kde(carla_lengths_clip, bw_method='scott')
    olv2_len_kde = gaussian_kde(olv2_lengths_clip, bw_method='scott')

    x_len = np.linspace(0, 100, 500)
    carla_len_density = carla_len_kde(x_len)
    olv2_len_density = olv2_len_kde(x_len)

    # Plot with strong, distinct colors
    ax1.plot(x_len, carla_len_density, color='#1E88E5', linewidth=4, alpha=0.95, label='Carla-OpenLane')
    ax1.plot(x_len, olv2_len_density, color='#D84315', linewidth=4, alpha=0.95, label='OpenLane-V2')
    ax1.fill_between(x_len, carla_len_density, alpha=0.2, color='#1E88E5')
    ax1.fill_between(x_len, olv2_len_density, alpha=0.2, color='#D84315')

    # Add legend only
    ax1.legend(fontsize=14, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)

    ax1.set_xlim([0, 100])
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.tick_params(labelsize=16, width=2, length=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    # Simple grid
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=1)

    # Curvature distribution with KDE (in radians/m)
    carla_curv_clip = carla_curvatures[carla_curvatures <= 0.15]
    olv2_curv_clip = olv2_curvatures[olv2_curvatures <= 0.15]

    carla_curv_kde = gaussian_kde(carla_curv_clip, bw_method='scott')
    olv2_curv_kde = gaussian_kde(olv2_curv_clip, bw_method='scott')

    x_curv = np.linspace(0, 0.15, 500)
    carla_curv_density = carla_curv_kde(x_curv)
    olv2_curv_density = olv2_curv_kde(x_curv)

    # Plot with strong, distinct colors
    ax2.plot(x_curv, carla_curv_density, color='#1E88E5', linewidth=4, alpha=0.95, label='Carla-OpenLane')
    ax2.plot(x_curv, olv2_curv_density, color='#D84315', linewidth=4, alpha=0.95, label='OpenLane-V2')
    ax2.fill_between(x_curv, carla_curv_density, alpha=0.2, color='#1E88E5')
    ax2.fill_between(x_curv, olv2_curv_density, alpha=0.2, color='#D84315')

    # Add legend only
    ax2.legend(fontsize=14, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)

    ax2.set_xlim([0, 0.15])
    ax2.set_xticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
    ax2.tick_params(labelsize=16, width=2, length=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    # Simple grid
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=1)

    plt.tight_layout()

    # Save figure
    output_path_pdf = Path("/home/user/LaneSegNet/fig/fig04.pdf")
    output_path_png = Path("/home/user/LaneSegNet/fig/fig04.png")
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path_pdf}")
    print(f"Figure saved to {output_path_png}")

    plt.show()

if __name__ == "__main__":
    carla_data = load_data(CARLA_TRAIN)
    olv2_data = load_data(OLV2_TRAIN)
    plot_comparison(carla_data, olv2_data)
