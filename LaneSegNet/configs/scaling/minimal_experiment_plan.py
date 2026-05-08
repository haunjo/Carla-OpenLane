"""
Minimal Scaling Experiment Configuration
Focus on 4 key data points: 15K, 37K, 50K, 80K samples
"""

import os
import json

# Data scale configuration
SCALE_CONFIG = {
    "15K": {
        "num_samples": 15000,
        "scale_ratio": 0.30,
        "description": "Minimal baseline"
    },
    "37K": {
        "num_samples": 37000,
        "scale_ratio": 0.74,
        "description": "Main paper configuration"
    },
    "50K": {
        "num_samples": 50000,
        "scale_ratio": 1.00,
        "description": "Full CARLA dataset"
    },
    "80K": {
        "num_samples": 80000,
        "scale_ratio": 1.60,
        "description": "Augmented dataset (with repetition/augmentation)"
    }
}

# Training configuration for each scale
TRAINING_CONFIG = {
    "pretraining": {
        "epochs": 24,
        "batch_size": 8,
        "lr": 2e-4,
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "checkpoint_interval": 1,
        "eval_interval": 1
    },
    "finetuning": {
        "epochs": 12,
        "batch_size": 4,
        "lr": 5e-5,
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "frozen_stages": 2,  # Freeze first 2 backbone stages
        "unfreeze_epoch": 3,  # Unfreeze after 3 epochs
        "eval_interval": 1
    }
}

# Metrics to track
METRICS = {
    "pretraining": [
        "loss",
        "loss_ce",
        "loss_dice",
        "loss_topology"
    ],
    "finetuning": [
        "DET_l",
        "TOP_ll",
        "TOP_lt",
        "OLS"
    ]
}

# Experiment settings
EXPERIMENT = {
    "seeds": [42],  # Single seed for minimal experiment
    "sampling": "stratified",  # Use stratified sampling only
    "gpus": 1,  # Single GPU setup
    "save_best_metric": "OLS",  # For fine-tuning
    "early_stopping_patience": 3  # Optional early stopping
}

def generate_configs():
    """Generate configuration files for each scale."""

    for scale_name, scale_info in SCALE_CONFIG.items():
        config = {
            "scale": scale_name,
            "num_samples": scale_info["num_samples"],
            "scale_ratio": scale_info["scale_ratio"],
            "description": scale_info["description"],
            "training": TRAINING_CONFIG,
            "metrics": METRICS,
            "experiment": EXPERIMENT
        }

        # Save config
        config_path = f"configs/scaling/{scale_name.lower()}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Generated config: {config_path}")

def create_data_augmentation_strategy():
    """
    Strategy for creating 80K dataset from 50K CARLA data.
    """
    augmentation = {
        "base_samples": 50000,
        "target_samples": 80000,
        "strategies": [
            {
                "method": "repeat_complex",
                "description": "Repeat complex scenes (intersections, multi-lane)",
                "ratio": 0.3
            },
            {
                "method": "augment_weather",
                "description": "Apply weather augmentations to existing samples",
                "ratio": 0.3
            },
            {
                "method": "temporal_shifts",
                "description": "Use temporal neighbors from sequences",
                "ratio": 0.4
            }
        ]
    }
    return augmentation

def estimate_training_time():
    """Estimate total training time for minimal experiment."""

    times = {}

    for scale_name, scale_info in SCALE_CONFIG.items():
        # Approximate time based on data size
        base_time_per_epoch = 1.5  # hours for 50K samples
        scale_factor = scale_info["scale_ratio"]

        if scale_name == "80K":
            # Augmented data takes longer due to online augmentation
            scale_factor = 1.3

        pretrain_time = TRAINING_CONFIG["pretraining"]["epochs"] * base_time_per_epoch * scale_factor
        finetune_time = TRAINING_CONFIG["finetuning"]["epochs"] * 0.5  # Subset-A is smaller

        times[scale_name] = {
            "pretraining": pretrain_time,
            "finetuning": finetune_time,
            "total": pretrain_time + finetune_time
        }

    return times

def get_scaling_curve_config():
    """Configuration for scaling curve visualization."""

    return {
        "primary_curve": "finetuning",  # Main curve to show
        "metrics_to_plot": {
            "finetuning": ["OLS", "DET_l", "TOP_ll", "TOP_lt"],
            "pretraining": ["loss", "loss_topology"]  # For supplementary
        },
        "visualization": {
            "figure_size": (10, 6),
            "marker_size": 10,
            "line_width": 2,
            "error_bars": True,  # If multiple seeds
            "grid": True,
            "legend_location": "lower right"
        },
        "analysis": {
            "fit_curve": "logarithmic",  # Fit log curve to data
            "compute_saturation": True,
            "efficiency_metric": True,  # Performance per sample
            "marginal_gain": True  # Gain per additional sample
        }
    }

if __name__ == "__main__":
    # Generate configs
    generate_configs()

    # Print time estimates
    print("\n=== Training Time Estimates ===")
    times = estimate_training_time()
    total_time = 0
    for scale, time_info in times.items():
        print(f"{scale}: {time_info['total']:.1f} hours")
        total_time += time_info['total']
    print(f"Total: {total_time:.1f} hours")

    # Print augmentation strategy
    print("\n=== 80K Dataset Augmentation Strategy ===")
    aug = create_data_augmentation_strategy()
    for strategy in aug['strategies']:
        print(f"- {strategy['method']}: {strategy['ratio']*100:.0f}%")