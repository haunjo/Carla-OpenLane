"""
Select 48 diverse images from Carla-OpenLane dataset
- Weather diversity: rain, clear, cloudy, etc.
- Time diversity: afternoon, dusk, night, morning
- Scene diversity: suburban, downtown, urban, highway
"""

import pickle
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict

# Paths
CARLA_TRAIN = Path("/home/user/LaneSegNet/data/Carla-OLV2/data_dict_carla_train_argoverse2_ls.pkl")
CARLA_IMAGE_DIR = Path("/home/user/LaneSegNet/data/Carla-OLV2")
OUTPUT_DIR = Path("/home/user/LaneSegNet/image")

def load_data(pkl_path):
    """Load pickle data"""
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def categorize_samples(data):
    """Categorize samples by weather, time, and scene type"""
    categories = defaultdict(list)

    for key, sample in data.items():
        meta = sample.get('meta_data', {})
        scene_desc = meta.get('scene_description', {})
        tokens = scene_desc.get('tokens', [])

        # Extract categorization
        weather = None
        time = None
        scene = None

        for token in tokens:
            token_lower = token.lower()
            # Weather (priority: rain > overcast > clear/cloudy)
            if 'rain' in token_lower and weather not in ['rain']:
                weather = 'rain'
            elif token_lower == 'overcast' and weather not in ['rain']:
                weather = 'overcast'
            elif token_lower in ['clear', 'cloudy'] and weather is None:
                weather = token_lower

            # Time
            if token_lower in ['afternoon', 'noon']:
                time = 'afternoon'
            elif token_lower == 'dusk':
                time = 'dusk'
            elif token_lower == 'night':
                time = 'night'
            elif token_lower == 'morning':
                time = 'morning'

            # Scene type
            if token_lower in ['suburban', 'downtown', 'urban', 'highway']:
                scene = token_lower

        # Store with categories
        category_key = (weather or 'unknown', time or 'unknown', scene or 'unknown')
        categories[category_key].append({
            'key': key,
            'sample': sample,
            'weather': weather,
            'time': time,
            'scene': scene
        })

    return categories

def select_diverse_samples(categories, target_count=128):
    """Select diverse samples ensuring good coverage"""

    # First, print distribution
    print("\nCategory distribution:")
    for cat_key, samples in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        if len(samples) > 0:
            print(f"  {cat_key}: {len(samples)} samples")

    # Strategy: Select samples to maximize diversity
    # Target: 64 images covering weather x time x scene combinations

    selected = []

    # Define target combinations
    weather_types = ['rain', 'clear', 'cloudy', 'overcast']
    time_types = ['afternoon', 'dusk', 'night']
    scene_types = ['suburban', 'downtown', 'urban', 'highway']

    # Target: 64 images with diverse coverage
    # First pass: get at least 1 sample from each unique combination
    for weather in weather_types:
        for time in time_types:
            for scene in scene_types:
                cat_key = (weather, time, scene)
                if cat_key in categories and len(categories[cat_key]) > 0:
                    # Randomly select one from this category
                    idx = np.random.randint(0, len(categories[cat_key]))
                    selected.append(categories[cat_key][idx])
                else:
                    # If exact combination doesn't exist, try to find close match
                    # Priority: weather > time > scene
                    found = False
                    # Try relaxing scene constraint
                    for alt_scene in scene_types:
                        alt_key = (weather, time, alt_scene)
                        if alt_key in categories and len(categories[alt_key]) > 0:
                            idx = np.random.randint(0, len(categories[alt_key]))
                            selected.append(categories[alt_key][idx])
                            found = True
                            break
                    if not found:
                        # Try relaxing time constraint
                        for alt_time in time_types:
                            for alt_scene in scene_types:
                                alt_key = (weather, alt_time, alt_scene)
                                if alt_key in categories and len(categories[alt_key]) > 0:
                                    idx = np.random.randint(0, len(categories[alt_key]))
                                    selected.append(categories[alt_key][idx])
                                    found = True
                                    break
                            if found:
                                break

    # Second pass: add more samples to reach target, cycling through categories for balance
    already_selected_keys = {s['key'] for s in selected}

    # Calculate how many more samples we need
    remaining = target_count - len(selected)

    if remaining > 0:
        # Get all categories sorted by size
        all_categories = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)

        # Round-robin through categories to maintain diversity
        while len(selected) < target_count:
            added_this_round = False
            for cat_key, samples in all_categories:
                if len(selected) >= target_count:
                    break
                # Get available samples from this category
                available = [s for s in samples if s['key'] not in already_selected_keys]
                if available:
                    # Add one sample from this category
                    idx = np.random.randint(0, len(available))
                    selected.append(available[idx])
                    already_selected_keys.add(available[idx]['key'])
                    added_this_round = True

            # If no samples were added this round, we've exhausted all categories
            if not added_this_round:
                break

    # Limit to target count
    selected = selected[:target_count]

    print(f"\nSelected {len(selected)} samples")

    # Print distribution of selected samples
    weather_dist = defaultdict(int)
    time_dist = defaultdict(int)
    scene_dist = defaultdict(int)

    for s in selected:
        weather_dist[s['weather']] += 1
        time_dist[s['time']] += 1
        scene_dist[s['scene']] += 1

    print("\nSelected distribution:")
    print(f"  Weather: {dict(weather_dist)}")
    print(f"  Time: {dict(time_dist)}")
    print(f"  Scene: {dict(scene_dist)}")

    return selected

def copy_images(selected_samples, output_dir):
    """Copy selected images to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying images to {output_dir}...")

    copied = 0
    failed = 0

    for idx, s in enumerate(selected_samples):
        sample = s['sample']
        meta = sample.get('meta_data', {})
        key = s['key']

        # Parse source_id to get segment and timestamp
        # Format: 'Carla/Maps/Town07147444' -> segment from key tuple
        # Key is tuple: (split, segment, timestamp)
        # Example: ('train', '0001', '100')

        if isinstance(key, tuple) and len(key) >= 3:
            split, segment, timestamp = key[0], key[1], key[2]

            # Construct image path
            image_path = CARLA_IMAGE_DIR / split / segment / 'ring_front_center' / f"{timestamp}.jpg"

            if image_path.exists():
                # Create descriptive filename
                weather = s['weather'] or 'unknown'
                time = s['time'] or 'unknown'
                scene = s['scene'] or 'unknown'

                output_name = f"{idx+1:03d}_{weather}_{time}_{scene}{image_path.suffix}"
                output_path = output_dir / output_name

                shutil.copy2(image_path, output_path)
                copied += 1
                print(f"  [{idx+1}/128] Copied: {output_name}")
            else:
                failed += 1
                print(f"  [{idx+1}/128] FAILED: Image not found at {image_path}")
        else:
            failed += 1
            print(f"  [{idx+1}/128] FAILED: Invalid key format: {key}")

    print(f"\nCompleted: {copied} copied, {failed} failed")

if __name__ == "__main__":
    # Load data
    data = load_data(CARLA_TRAIN)

    # Categorize samples
    categories = categorize_samples(data)

    # Select diverse samples
    selected = select_diverse_samples(categories, target_count=128)

    # Copy images
    copy_images(selected, OUTPUT_DIR)
