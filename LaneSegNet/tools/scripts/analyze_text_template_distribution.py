"""
Analyze text template distribution in Carla-OLV2 dataset.

Checks template_id field in lane segments and reports statistics.
"""

import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Text template definitions
TEXT_TEMPLATES = [
    "straight lane on regular road",                  # 0
    "straight lane connecting to intersection",       # 1
    "straight lane within intersection area",         # 2
    "left-turning lane on regular road",              # 3
    "left-turning lane connecting to intersection",   # 4
    "left-turning lane within intersection area",     # 5
    "right-turning lane on regular road",             # 6
    "right-turning lane connecting to intersection",  # 7
    "right-turning lane within intersection area"     # 8
]

CURVATURE_NAMES = {0: "straight", 1: "left-turn", 2: "right-turn"}
INTERSECTION_NAMES = {0: "regular", 1: "connector", 2: "intersection"}


def analyze_distribution(pkl_path, split_name="train"):
    """Analyze template_id distribution in dataset."""

    print(f"\n{'='*80}")
    print(f"Analyzing Carla-OLV2 {split_name.upper()} Dataset")
    print(f"File: {pkl_path}")
    print(f"{'='*80}\n")

    # Load pickle
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Collect template_ids
    template_ids = []
    curvature_types = []
    intersection_types = []
    total_frames = 0
    frames_with_template = 0

    # Iterate through all samples
    for key, sample in data_dict.items():
        if not isinstance(sample, dict) or 'annotation' not in sample:
            continue

        total_frames += 1
        annotation = sample['annotation']

        if 'lane_segment' not in annotation:
            continue

        lane_segments = annotation['lane_segment']
        frame_has_template = False

        for lane in lane_segments:
            if 'template_id' in lane:
                template_id = lane['template_id']
                template_ids.append(template_id)

                # Decompose template_id into curvature and intersection
                curvature = template_id // 3
                intersection = template_id % 3
                curvature_types.append(curvature)
                intersection_types.append(intersection)

                frame_has_template = True

        if frame_has_template:
            frames_with_template += 1

    # Check if template_id exists
    if len(template_ids) == 0:
        print("❌ No template_id found in lane segments!")
        print("   Make sure carla2openlanev2.py was run with the updated version.")
        return

    print(f"✓ Found template_id in {len(template_ids)} lane segments")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with template_id: {frames_with_template}")
    print(f"  Average lanes per frame: {len(template_ids) / frames_with_template:.2f}")

    # Overall template distribution
    print(f"\n{'='*80}")
    print("Template Distribution (9 classes)")
    print(f"{'='*80}")

    counter = Counter(template_ids)
    total = len(template_ids)

    for template_id in range(9):
        count = counter.get(template_id, 0)
        percentage = (count / total) * 100
        curvature = template_id // 3
        intersection = template_id % 3

        bar = '█' * int(percentage / 2)

        print(f"Template {template_id}: {TEXT_TEMPLATES[template_id]}")
        print(f"  Count: {count:5d} ({percentage:5.2f}%) {bar}")

    # Curvature distribution
    print(f"\n{'='*80}")
    print("Curvature Distribution (3 classes)")
    print(f"{'='*80}")

    curv_counter = Counter(curvature_types)
    for curv_id in range(3):
        count = curv_counter.get(curv_id, 0)
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 2)
        print(f"{CURVATURE_NAMES[curv_id]:12s}: {count:5d} ({percentage:5.2f}%) {bar}")

    # Intersection distribution
    print(f"\n{'='*80}")
    print("Intersection Context Distribution (3 classes)")
    print(f"{'='*80}")

    inter_counter = Counter(intersection_types)
    for inter_id in range(3):
        count = inter_counter.get(inter_id, 0)
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 2)
        print(f"{INTERSECTION_NAMES[inter_id]:12s}: {count:5d} ({percentage:5.2f}%) {bar}")

    # Statistical analysis
    print(f"\n{'='*80}")
    print("Statistical Analysis")
    print(f"{'='*80}")

    # Check balance
    template_counts = [counter.get(i, 0) for i in range(9)]
    min_count = min(template_counts)
    max_count = max(template_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"Min count: {min_count} (Template {template_counts.index(min_count)})")
    print(f"Max count: {max_count} (Template {template_counts.index(max_count)})")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio < 5:
        print("✓ Distribution is reasonably balanced")
    elif imbalance_ratio < 10:
        print("⚠ Distribution is moderately imbalanced")
    else:
        print("❌ Distribution is highly imbalanced")

    # Rare classes warning
    rare_threshold = total * 0.02  # 2% threshold
    rare_classes = [i for i in range(9) if counter.get(i, 0) < rare_threshold]
    if rare_classes:
        print(f"\n⚠ Warning: {len(rare_classes)} rare classes (< 2%):")
        for cls in rare_classes:
            count = counter.get(cls, 0)
            print(f"  - Template {cls}: {count} ({count/total*100:.2f}%)")

    # Generate histogram
    print(f"\n{'='*80}")
    print("Generating histogram...")
    print(f"{'='*80}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Template distribution
    ax1 = axes[0]
    template_counts = [counter.get(i, 0) for i in range(9)]
    bars1 = ax1.bar(range(9), template_counts, color='steelblue')
    ax1.set_xlabel('Template ID', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Template Distribution (9 classes)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(9))
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/total*100:.1f}%',
                ha='center', va='bottom', fontsize=9)

    # Curvature distribution
    ax2 = axes[1]
    curv_counts = [curv_counter.get(i, 0) for i in range(3)]
    bars2 = ax2.bar(range(3), curv_counts, color=['green', 'orange', 'red'])
    ax2.set_xlabel('Curvature Type', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Curvature Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Straight', 'Left-turn', 'Right-turn'])
    ax2.grid(axis='y', alpha=0.3)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/total*100:.1f}%',
                ha='center', va='bottom', fontsize=10)

    # Intersection distribution
    ax3 = axes[2]
    inter_counts = [inter_counter.get(i, 0) for i in range(3)]
    bars3 = ax3.bar(range(3), inter_counts, color=['skyblue', 'gold', 'tomato'])
    ax3.set_xlabel('Intersection Context', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Intersection Context Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['Regular', 'Connector', 'Intersection'])
    ax3.grid(axis='y', alpha=0.3)

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/total*100:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = f'text_template_distribution_{split_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved histogram to: {output_path}")

    return {
        'total_lanes': total,
        'total_frames': total_frames,
        'frames_with_template': frames_with_template,
        'template_distribution': counter,
        'curvature_distribution': curv_counter,
        'intersection_distribution': inter_counter,
        'imbalance_ratio': imbalance_ratio
    }


if __name__ == "__main__":
    import sys

    # Default paths
    train_pkl = "data/Carla-OLV2/data_dict_carla_train_argoverse2_ls.pkl"
    val_pkl = "data/Carla-OLV2/data_dict_carla_val_argoverse2_ls.pkl"

    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
        split_name = "custom"
    else:
        pkl_path = train_pkl
        split_name = "train"

    # Analyze train set
    try:
        stats = analyze_distribution(pkl_path, split_name)

        # Also analyze val set if train was analyzed
        if split_name == "train":
            print("\n" + "="*80)
            print("Also analyzing validation set...")
            print("="*80)
            val_stats = analyze_distribution(val_pkl, "val")

    except FileNotFoundError as e:
        print(f"❌ Error: File not found")
        print(f"   {e}")
        print(f"\nUsage: python {sys.argv[0]} [path_to_pickle]")
        print(f"Default: {train_pkl}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
