"""
Create data splits for scaling analysis experiments.
Generates different scales of CARLA-OpenLane data for pre-training.
"""

import os
import json
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def load_carla_metadata(data_root: str) -> Dict:
    """Load CARLA-OpenLane metadata for sampling."""
    meta_file = os.path.join(data_root, 'data_dict.pkl')
    if os.path.exists(meta_file):
        with open(meta_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Fallback to JSON if pickle not available
        meta_file = os.path.join(data_root, 'data_dict.json')
        with open(meta_file, 'r') as f:
            return json.load(f)


def categorize_samples(metadata: Dict) -> Dict[str, List]:
    """
    Categorize samples based on diversity criteria.

    Returns:
        Dict with categories as keys and sample lists as values
    """
    categories = defaultdict(list)

    for sample_id, sample_info in metadata.items():
        # Weather condition
        weather = sample_info.get('weather', 'clear')

        # Scene complexity (number of lanes)
        num_lanes = len(sample_info.get('lanes', []))
        if num_lanes <= 2:
            complexity = 'simple'
        elif num_lanes <= 4:
            complexity = 'medium'
        else:
            complexity = 'complex'

        # Topology pattern
        topology = sample_info.get('topology_type', 'straight')

        # Create category key
        category = f"{weather}_{complexity}_{topology}"
        categories[category].append(sample_id)

    return categories


def create_random_split(
    metadata: Dict,
    scale: float,
    seed: int = 42
) -> List[str]:
    """
    Create random split of specified scale.

    Args:
        metadata: Full dataset metadata
        scale: Fraction of data to sample (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample IDs
    """
    random.seed(seed)
    all_samples = list(metadata.keys())
    n_samples = int(len(all_samples) * scale)

    selected = random.sample(all_samples, n_samples)
    return selected


def create_stratified_split(
    metadata: Dict,
    scale: float,
    seed: int = 42
) -> List[str]:
    """
    Create diversity-aware stratified split.

    Args:
        metadata: Full dataset metadata
        scale: Fraction of data to sample (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of selected sample IDs
    """
    random.seed(seed)
    np.random.seed(seed)

    # Categorize samples
    categories = categorize_samples(metadata)

    # Sample proportionally from each category
    selected = []
    for category, samples in categories.items():
        n_samples = max(1, int(len(samples) * scale))
        if n_samples <= len(samples):
            category_selected = random.sample(samples, n_samples)
        else:
            category_selected = samples
        selected.extend(category_selected)

    return selected


def save_split(
    sample_ids: List[str],
    output_path: str,
    metadata: Dict = None
):
    """Save split to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    split_data = {
        'sample_ids': sample_ids,
        'num_samples': len(sample_ids),
    }

    if metadata:
        # Add statistics
        split_data['statistics'] = compute_split_statistics(
            sample_ids, metadata
        )

    # Save as JSON for readability
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)

    # Also save as pickle for faster loading
    pickle_path = output_path.replace('.json', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(split_data, f)

    print(f"Saved split with {len(sample_ids)} samples to {output_path}")


def compute_split_statistics(
    sample_ids: List[str],
    metadata: Dict
) -> Dict:
    """Compute statistics for the split."""
    stats = {
        'total_samples': len(sample_ids),
        'weather_distribution': defaultdict(int),
        'complexity_distribution': defaultdict(int),
        'topology_distribution': defaultdict(int),
    }

    for sample_id in sample_ids:
        sample_info = metadata.get(sample_id, {})

        # Weather
        weather = sample_info.get('weather', 'clear')
        stats['weather_distribution'][weather] += 1

        # Complexity
        num_lanes = len(sample_info.get('lanes', []))
        if num_lanes <= 2:
            complexity = 'simple'
        elif num_lanes <= 4:
            complexity = 'medium'
        else:
            complexity = 'complex'
        stats['complexity_distribution'][complexity] += 1

        # Topology
        topology = sample_info.get('topology_type', 'straight')
        stats['topology_distribution'][topology] += 1

    return stats


def create_all_scaling_splits(
    data_root: str,
    output_dir: str,
    scales: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    seeds: List[int] = [42, 123, 456]
):
    """Create all scaling splits for experiments."""

    # Load metadata
    print(f"Loading metadata from {data_root}")
    metadata = load_carla_metadata(data_root)
    print(f"Total samples in dataset: {len(metadata)}")

    # Create splits for each scale and sampling method
    for scale in scales:
        scale_pct = int(scale * 100)

        for seed in seeds:
            # Random sampling
            random_split = create_random_split(metadata, scale, seed)
            random_path = os.path.join(
                output_dir,
                f'carla_{scale_pct}pct_random_seed{seed}.json'
            )
            save_split(random_split, random_path, metadata)

            # Stratified sampling
            stratified_split = create_stratified_split(metadata, scale, seed)
            stratified_path = os.path.join(
                output_dir,
                f'carla_{scale_pct}pct_stratified_seed{seed}.json'
            )
            save_split(stratified_split, stratified_path, metadata)

    print(f"\nAll splits created in {output_dir}")

    # Generate summary report
    generate_summary_report(output_dir, scales, seeds)


def generate_summary_report(
    output_dir: str,
    scales: List[float],
    seeds: List[int]
):
    """Generate summary report of all splits."""
    report = []
    report.append("# Scaling Splits Summary Report\n")
    report.append(f"Total configurations: {len(scales)} scales × "
                  f"{len(seeds)} seeds × 2 methods = "
                  f"{len(scales) * len(seeds) * 2} splits\n")

    for scale in scales:
        scale_pct = int(scale * 100)
        report.append(f"\n## {scale_pct}% Scale\n")

        for method in ['random', 'stratified']:
            report.append(f"\n### {method.capitalize()} Sampling\n")

            for seed in seeds:
                split_file = os.path.join(
                    output_dir,
                    f'carla_{scale_pct}pct_{method}_seed{seed}.json'
                )

                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        data = json.load(f)

                    report.append(f"- Seed {seed}: {data['num_samples']} samples\n")

                    if 'statistics' in data:
                        stats = data['statistics']
                        report.append(f"  - Weather: {dict(stats['weather_distribution'])}\n")
                        report.append(f"  - Complexity: {dict(stats['complexity_distribution'])}\n")
                        report.append(f"  - Topology: {dict(stats['topology_distribution'])}\n")

    report_path = os.path.join(output_dir, 'SCALING_SPLITS_SUMMARY.md')
    with open(report_path, 'w') as f:
        f.writelines(report)

    print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create scaling splits for CARLA-OpenLane'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/path/to/carla_openlane',
        help='Path to CARLA-OpenLane dataset root'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/scaling_splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--scales',
        nargs='+',
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help='Data scales to create (as fractions)'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[42, 123, 456],
        help='Random seeds for reproducibility'
    )

    args = parser.parse_args()

    create_all_scaling_splits(
        args.data_root,
        args.output_dir,
        args.scales,
        args.seeds
    )


if __name__ == '__main__':
    main()