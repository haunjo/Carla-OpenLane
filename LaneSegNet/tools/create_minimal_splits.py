"""
Create data splits for minimal scaling experiments.
Generates 15K, 37K, 50K, and 80K (augmented) datasets.
"""

import os
import json
import pickle
import random
import numpy as np
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


def load_carla_data(data_root: str) -> Dict:
    """Load CARLA-OpenLane dataset information."""
    meta_file = os.path.join(data_root, 'data_dict.pkl')
    if os.path.exists(meta_file):
        with open(meta_file, 'rb') as f:
            return pickle.load(f)
    else:
        meta_file = os.path.join(data_root, 'data_dict.json')
        with open(meta_file, 'r') as f:
            return json.load(f)


def create_15k_split(full_data: Dict, seed: int = 42) -> List[str]:
    """
    Create 15K sample split (30% of full dataset).
    Prioritize diversity in this small dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Categorize by importance
    categories = {
        'intersection': [],
        'multi_lane': [],
        'weather_variant': [],
        'standard': []
    }

    for sample_id, info in full_data.items():
        # Categorize based on complexity
        if 'intersection' in info.get('scene_type', ''):
            categories['intersection'].append(sample_id)
        elif len(info.get('lanes', [])) > 3:
            categories['multi_lane'].append(sample_id)
        elif info.get('weather', 'clear') != 'clear':
            categories['weather_variant'].append(sample_id)
        else:
            categories['standard'].append(sample_id)

    # Sample proportionally with emphasis on diversity
    selected = []
    target_total = 15000

    # Distribution: 30% complex, 25% weather, 20% multi-lane, 25% standard
    distributions = {
        'intersection': 0.30,
        'weather_variant': 0.25,
        'multi_lane': 0.20,
        'standard': 0.25
    }

    for category, ratio in distributions.items():
        n_samples = min(
            int(target_total * ratio),
            len(categories[category])
        )
        if n_samples > 0:
            selected.extend(
                random.sample(categories[category], n_samples)
            )

    # Fill remaining if needed
    remaining = target_total - len(selected)
    if remaining > 0:
        all_unselected = [
            s for s in full_data.keys() if s not in selected
        ]
        selected.extend(random.sample(all_unselected, min(remaining, len(all_unselected))))

    return selected[:target_total]


def create_37k_split(full_data: Dict, seed: int = 42) -> List[str]:
    """
    Create 37K sample split (74% of full dataset).
    This is the main configuration used in the paper.
    """
    random.seed(seed)
    all_samples = list(full_data.keys())
    n_samples = min(37000, int(len(all_samples) * 0.74))
    return random.sample(all_samples, n_samples)


def create_50k_split(full_data: Dict) -> List[str]:
    """
    Create 50K sample split (full CARLA dataset).
    """
    all_samples = list(full_data.keys())
    return all_samples[:50000]  # Use all available up to 50K


def create_80k_augmented(full_data: Dict, base_50k: List[str], seed: int = 42) -> List[dict]:
    """
    Create 80K dataset by augmenting 50K base dataset.

    Strategies:
    1. Repeat complex scenes (30K additional)
    2. Add temporal neighbors from sequences
    3. Include augmentation markers for online augmentation
    """
    random.seed(seed)
    np.random.seed(seed)

    augmented_data = []

    # Start with base 50K
    for sample_id in base_50k:
        augmented_data.append({
            'id': sample_id,
            'augmentation': 'none',
            'source': 'original'
        })

    # Strategy 1: Repeat complex scenes (15K)
    complex_scenes = []
    for sample_id in base_50k:
        info = full_data.get(sample_id, {})
        # Complex if: intersection, >4 lanes, or bad weather
        if ('intersection' in info.get('scene_type', '') or
            len(info.get('lanes', [])) > 4 or
            info.get('weather', 'clear') in ['rain', 'fog', 'night']):
            complex_scenes.append(sample_id)

    # Sample 15K from complex scenes with replacement
    repeated_complex = np.random.choice(
        complex_scenes,
        min(15000, len(complex_scenes) * 2),
        replace=True
    ).tolist()

    for sample_id in repeated_complex:
        augmented_data.append({
            'id': sample_id,
            'augmentation': 'heavy',  # Apply heavy augmentation
            'source': 'repeated_complex'
        })

    # Strategy 2: Temporal neighbors (15K)
    # Assuming sequences have temporal structure
    temporal_samples = []
    for sample_id in random.sample(base_50k, min(15000, len(base_50k) // 2)):
        # Create temporal variant marker
        temporal_samples.append({
            'id': sample_id,
            'augmentation': 'temporal_shift',
            'source': 'temporal_neighbor',
            'shift_frames': random.randint(-2, 2)  # ±2 frames
        })

    augmented_data.extend(temporal_samples[:15000])

    return augmented_data[:80000]


def save_split(split_data, output_path: str, metadata: dict = None):
    """Save split data to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        'samples': split_data,
        'num_samples': len(split_data),
        'metadata': metadata or {}
    }

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # Save as pickle for faster loading
    pickle_path = output_path.replace('.json', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Saved {len(split_data)} samples to {output_path}")


def create_all_minimal_splits(data_root: str, output_dir: str, seed: int = 42):
    """Create all splits for minimal experiment."""

    print("Loading CARLA-OpenLane data...")
    full_data = load_carla_data(data_root)
    print(f"Loaded {len(full_data)} samples")

    os.makedirs(output_dir, exist_ok=True)

    # Create 15K split
    print("\nCreating 15K split (30%)...")
    split_15k = create_15k_split(full_data, seed)
    save_split(
        split_15k,
        os.path.join(output_dir, 'carla_15k.json'),
        {'scale': '15K', 'ratio': 0.3, 'seed': seed}
    )

    # Create 37K split
    print("\nCreating 37K split (74%)...")
    split_37k = create_37k_split(full_data, seed)
    save_split(
        split_37k,
        os.path.join(output_dir, 'carla_37k.json'),
        {'scale': '37K', 'ratio': 0.74, 'seed': seed}
    )

    # Create 50K split
    print("\nCreating 50K split (100%)...")
    split_50k = create_50k_split(full_data)
    save_split(
        split_50k,
        os.path.join(output_dir, 'carla_50k.json'),
        {'scale': '50K', 'ratio': 1.0, 'seed': seed}
    )

    # Create 80K augmented split
    print("\nCreating 80K augmented split...")
    split_80k = create_80k_augmented(full_data, split_50k, seed)
    save_split(
        split_80k,
        os.path.join(output_dir, 'carla_80k_augmented.json'),
        {'scale': '80K', 'ratio': 1.6, 'seed': seed, 'augmented': True}
    )

    # Generate summary report
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)

    report = []
    report.append("# Minimal Scaling Experiment Data Splits\n\n")
    report.append("| Scale | Samples | Ratio | Type | File |\n")
    report.append("|-------|---------|-------|------|------|\n")
    report.append(f"| 15K | {len(split_15k)} | 30% | Diverse subset | carla_15k.json |\n")
    report.append(f"| 37K | {len(split_37k)} | 74% | Main paper | carla_37k.json |\n")
    report.append(f"| 50K | {len(split_50k)} | 100% | Full dataset | carla_50k.json |\n")
    report.append(f"| 80K | {len(split_80k)} | 160% | Augmented | carla_80k_augmented.json |\n")

    # Analyze overlap
    report.append("\n## Dataset Overlap Analysis\n\n")
    s15k = set(split_15k)
    s37k = set(split_37k)
    s50k = set(split_50k)

    report.append(f"- 15K ∩ 37K: {len(s15k & s37k)} samples ({len(s15k & s37k)/len(s15k)*100:.1f}% of 15K)\n")
    report.append(f"- 37K ∩ 50K: {len(s37k & s50k)} samples ({len(s37k & s50k)/len(s37k)*100:.1f}% of 37K)\n")
    report.append(f"- 15K ⊆ 37K ⊆ 50K: {s15k.issubset(s37k) and s37k.issubset(s50k)}\n")

    # Save report
    report_path = os.path.join(output_dir, 'MINIMAL_SPLITS_REPORT.md')
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create minimal scaling experiment splits'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to CARLA-OpenLane dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/minimal_splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    create_all_minimal_splits(args.data_root, args.output_dir, args.seed)


if __name__ == '__main__':
    main()