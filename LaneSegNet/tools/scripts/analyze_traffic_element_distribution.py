"""
Analyze traffic element distribution in Carla-OLV2 dataset.
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

# OpenLane-V2 traffic element categories
TRAFFIC_ELEMENT_CLASSES = {
    1: "traffic_light",
    2: "road_sign",
    3: "traffic_sign",
    4: "pedestrian_crossing_sign",
    5: "speed_limit_sign",
    6: "stop_sign",
    7: "construction_sign",
    8: "other_sign",
    9: "warning_sign",
    10: "direction_sign",
    11: "parking_sign",
    12: "lane_plate",
    13: "pole"
}

def analyze_distribution(data_root, split='train', limit=None):
    """Analyze traffic element distribution."""

    # Find all JSON files
    json_pattern = f"{data_root}/{split}/*/info/*-ls.json"
    json_files = sorted(glob.glob(json_pattern))

    if limit:
        json_files = json_files[:limit]

    print(f"\n{'='*80}")
    print(f"Analyzing {split} split: {len(json_files)} files")
    print(f"{'='*80}\n")

    # Statistics
    te_class_counts = defaultdict(int)
    total_te = 0
    samples_with_te = 0
    samples_without_te = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        traffic_elements = data['annotation'].get('traffic_element', [])

        if len(traffic_elements) > 0:
            samples_with_te += 1
            total_te += len(traffic_elements)

            for te in traffic_elements:
                category = te['category']
                te_class_counts[category] += 1
        else:
            samples_without_te += 1

    # Print statistics
    print(f"Total samples: {len(json_files)}")
    print(f"Samples with traffic elements: {samples_with_te} ({100*samples_with_te/len(json_files):.2f}%)")
    print(f"Samples without traffic elements: {samples_without_te} ({100*samples_without_te/len(json_files):.2f}%)")
    print(f"Total traffic elements: {total_te}")
    print(f"Average TE per sample: {total_te/len(json_files):.2f}")
    print()

    # Class distribution
    print("Class distribution:")
    print(f"{'Class ID':<10} {'Class Name':<30} {'Count':<10} {'Percentage'}")
    print("-" * 70)

    for class_id in sorted(te_class_counts.keys()):
        count = te_class_counts[class_id]
        percentage = 100 * count / total_te if total_te > 0 else 0
        class_name = TRAFFIC_ELEMENT_CLASSES.get(class_id, f"Unknown({class_id})")
        print(f"{class_id:<10} {class_name:<30} {count:<10} {percentage:6.2f}%")

    # Find missing classes
    missing_classes = set(TRAFFIC_ELEMENT_CLASSES.keys()) - set(te_class_counts.keys())
    if missing_classes:
        print("\nMissing classes (0 samples):")
        for class_id in sorted(missing_classes):
            class_name = TRAFFIC_ELEMENT_CLASSES.get(class_id, f"Unknown({class_id})")
            print(f"  {class_id}: {class_name}")

    return te_class_counts, total_te, samples_with_te

if __name__ == "__main__":
    data_root = "data/OpenLane-V2"

    # Analyze train split
    print("\n" + "="*80)
    print("CARLA-OLV2 DATASET ANALYSIS")
    print("="*80)

    train_counts, train_total, train_with_te = analyze_distribution(data_root, 'train')

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal classes with samples: {len(train_counts)}/13")
    print(f"Total traffic elements: {train_total}")
    print(f"Samples with TE: {train_with_te}")

    # Calculate class imbalance ratio
    if len(train_counts) > 0:
        max_count = max(train_counts.values())
        min_count = min(train_counts.values())
        print(f"\nClass imbalance ratio: {max_count/min_count:.2f}x")
        print(f"Most frequent class: {max(train_counts, key=train_counts.get)} ({max_count} samples)")
        print(f"Least frequent class: {min(train_counts, key=train_counts.get)} ({min_count} samples)")
