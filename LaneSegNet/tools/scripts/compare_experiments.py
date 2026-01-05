"""
Compare performance between two experiments.
"""

import json
import sys
from pathlib import Path

def extract_val_metrics(log_path):
    """Extract validation metrics from log file."""
    metrics_by_epoch = {}

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Look for validation results
                if data.get('mode') == 'val':
                    epoch = data.get('epoch')

                    # Extract key metrics
                    metrics = {
                        'mAP': data.get('mAP', 0),
                        'AP_ls': data.get('AP_ls', 0),
                        'AP_ped': data.get('AP_ped', 0),
                        'TOP_lsls': data.get('TOP_lsls', 0),
                    }

                    if epoch is not None:
                        metrics_by_epoch[epoch] = metrics

            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue

    return metrics_by_epoch

def print_comparison(exp1_name, exp1_metrics, exp2_name, exp2_metrics):
    """Print comparison table."""

    print(f"\n{'='*100}")
    print(f"Experiment Comparison")
    print(f"{'='*100}\n")

    # Get latest epochs
    exp1_epochs = sorted(exp1_metrics.keys())
    exp2_epochs = sorted(exp2_metrics.keys())

    if not exp1_epochs or not exp2_epochs:
        print("❌ No validation results found!")
        return

    exp1_latest = exp1_epochs[-1]
    exp2_latest = exp2_epochs[-1]

    exp1_final = exp1_metrics[exp1_latest]
    exp2_final = exp2_metrics[exp2_latest]

    print(f"Experiment 1: {exp1_name} (Epoch {exp1_latest})")
    print(f"Experiment 2: {exp2_name} (Epoch {exp2_latest})")
    print()

    # Print table header
    print(f"{'Metric':<15} {'Exp1':<12} {'Exp2':<12} {'Diff':<12} {'Status'}")
    print(f"{'-'*70}")

    metric_names = ['mAP', 'AP_ls', 'AP_ped', 'TOP_lsls']

    for metric in metric_names:
        v1 = exp1_final.get(metric, 0)
        v2 = exp2_final.get(metric, 0)
        diff = v2 - v1

        if diff > 0:
            status = f"✓ +{diff:.4f}"
        elif diff < 0:
            status = f"✗ {diff:.4f}"
        else:
            status = "="

        print(f"{metric:<15} {v1:<12.4f} {v2:<12.4f} {diff:+12.4f} {status}")

    print()

    # Summary
    print(f"{'='*100}")
    print(f"Summary")
    print(f"{'='*100}\n")

    improvements = sum(1 for m in metric_names if exp2_final.get(m, 0) > exp1_final.get(m, 0))
    degradations = sum(1 for m in metric_names if exp2_final.get(m, 0) < exp1_final.get(m, 0))

    print(f"Improvements: {improvements}/{len(metric_names)} metrics")
    print(f"Degradations: {degradations}/{len(metric_names)} metrics")

    map_diff = exp2_final.get('mAP', 0) - exp1_final.get('mAP', 0)
    top_diff = exp2_final.get('TOP_lsls', 0) - exp1_final.get('TOP_lsls', 0)

    if map_diff > 0:
        print(f"✓ Detection: Exp2 is BETTER by {map_diff:.4f} mAP")
    elif map_diff < 0:
        print(f"✗ Detection: Exp2 is WORSE by {abs(map_diff):.4f} mAP")
    else:
        print(f"= Detection: Equal performance")

    if top_diff > 0:
        print(f"✓ Topology: Exp2 is BETTER by {top_diff:.4f} TOP_lsls")
    elif top_diff < 0:
        print(f"✗ Topology: Exp2 is WORSE by {abs(top_diff):.4f} TOP_lsls")
    else:
        print(f"= Topology: Equal performance")

    print()

    # Per-epoch progress for both
    print(f"{'='*100}")
    print(f"Training Progress")
    print(f"{'='*100}\n")

    print(f"Exp1 ({exp1_name}) - Validation per epoch:")
    for epoch in exp1_epochs:
        metrics = exp1_metrics[epoch]
        print(f"  Epoch {epoch:2d}: mAP={metrics['mAP']:.4f}, TOP_lsls={metrics['TOP_lsls']:.4f}")

    print()
    print(f"Exp2 ({exp2_name}) - Validation per epoch:")
    for epoch in exp2_epochs:
        metrics = exp2_metrics[epoch]
        print(f"  Epoch {epoch:2d}: mAP={metrics['mAP']:.4f}, TOP_lsls={metrics['TOP_lsls']:.4f}")

    print()

if __name__ == "__main__":
    # Define experiments to compare
    exp1_path = "work_dirs/lanesegnet_8e_carla_24e_olv2_subset_A_ls/20251030_165356.log.json"
    exp2_path = "work_dirs/lanesegnet_8e_carla_laneseg_text_guided_1031/20251101_010347.log.json"

    exp1_name = "2-stage (subset_A_ls)"
    exp2_name = "Text-guided (1031)"

    print("Loading experiment logs...")
    exp1_metrics = extract_val_metrics(exp1_path)
    exp2_metrics = extract_val_metrics(exp2_path)

    print_comparison(exp1_name, exp1_metrics, exp2_name, exp2_metrics)
