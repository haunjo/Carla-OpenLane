"""
Generate pie charts for Carla-OpenLane dataset distribution
For CVPR Supplementary Material
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
from collections import Counter

# Recompute statistics with updated weather classification
pkl_path = '/home/user/LaneSegNet/data/Carla-OLV2/data_dict_carla_train_argoverse2_ls.pkl'
print(f"Loading {pkl_path}...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

times = []
weathers = []
towns = []

for key, sample in data.items():
    meta_data = sample['meta_data']
    scene_desc = meta_data.get('scene_description', {})
    tokens = scene_desc.get('tokens', [])
    source_id = meta_data.get('source_id', '')

    # Time classification
    for token in tokens:
        if token in ['afternoon', 'noon']:
            times.append('Noon')
            break
        elif token in ['dusk', 'sunset']:
            times.append('Sunset')
            break
        elif token == 'night':
            times.append('Night')
            break

    # Weather classification (NEW LOGIC)
    has_heavy_rain = 'heavy_rain' in tokens
    has_overcast = 'overcast' in tokens
    has_rain = 'rain' in tokens
    has_cloudy = 'cloudy' in tokens
    has_fog = 'fog' in tokens or 'heavy_fog' in tokens

    if has_heavy_rain and has_overcast:
        weathers.append('Heavy Rain')
    elif has_heavy_rain:  # heavy_rain without overcast
        weathers.append('Rain')
    elif has_rain:
        weathers.append('Soft Rain')
    elif has_fog:
        weathers.append('Foggy')
    elif has_cloudy:
        weathers.append('Cloudy')
    else:
        weathers.append('Clear')

    # Town extraction
    match = re.search(r'Town(\d{2})', source_id)
    if match:
        towns.append(f"Town{match.group(1)}")

# Add validation data
pkl_path_val = '/home/user/LaneSegNet/data/Carla-OLV2/data_dict_carla_val_argoverse2_ls.pkl'
print(f"Loading {pkl_path_val}...")
with open(pkl_path_val, 'rb') as f:
    data_val = pickle.load(f)

for key, sample in data_val.items():
    meta_data = sample['meta_data']
    scene_desc = meta_data.get('scene_description', {})
    tokens = scene_desc.get('tokens', [])
    source_id = meta_data.get('source_id', '')

    for token in tokens:
        if token in ['afternoon', 'noon']:
            times.append('Noon')
            break
        elif token in ['dusk', 'sunset']:
            times.append('Sunset')
            break
        elif token == 'night':
            times.append('Night')
            break

    has_heavy_rain = 'heavy_rain' in tokens
    has_overcast = 'overcast' in tokens
    has_rain = 'rain' in tokens
    has_cloudy = 'cloudy' in tokens
    has_fog = 'fog' in tokens or 'heavy_fog' in tokens

    if has_heavy_rain and has_overcast:
        weathers.append('Heavy Rain')
    elif has_heavy_rain:
        weathers.append('Rain')
    elif has_rain:
        weathers.append('Soft Rain')
    elif has_fog:
        weathers.append('Foggy')
    elif has_cloudy:
        weathers.append('Cloudy')
    else:
        weathers.append('Clear')

    match = re.search(r'Town(\d{2})', source_id)
    if match:
        towns.append(f"Town{match.group(1)}")

print(f"Total samples: {len(times)}")

# Create statistics
stats = {
    'weather': dict(Counter(weathers)),
    'time': dict(Counter(times)),
    'town': dict(Counter(towns)),
    'total_samples': len(times)
}

# Save updated statistics
with open('/home/user/LaneSegNet/doc/carla_distribution_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nUpdated statistics saved!")

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Color schemes - vibrant and distinguishable colors
weather_colors = ['#FFD700', '#87CEEB', '#98D8C8', '#FF8C69', '#DC143C']  # Clear, Cloudy, Soft Rain, Rain, Heavy Rain
time_colors = ['#FFA500', '#FF6347', '#4169E1']  # Noon, Sunset, Night
town_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#F38181', '#AA96DA', '#FCBAD3']  # 7 distinct colors

# 1. Weather Distribution
weather_data = stats['weather']
weather_labels = list(weather_data.keys())
weather_values = list(weather_data.values())
weather_total = sum(weather_values)
weather_pcts = [v/weather_total*100 for v in weather_values]

axes[0].pie(weather_values, labels=[f'{l}\n{p:.1f}%' for l, p in zip(weather_labels, weather_pcts)],
            autopct='', colors=weather_colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
axes[0].set_title('(a) Weather Distribution', fontsize=13, weight='bold', pad=15)

# 2. Time-of-Day Distribution
time_data = stats['time']
time_labels = list(time_data.keys())
time_values = list(time_data.values())
time_total = sum(time_values)
time_pcts = [v/time_total*100 for v in time_values]

axes[1].pie(time_values, labels=[f'{l}\n{p:.1f}%' for l, p in zip(time_labels, time_pcts)],
            autopct='', colors=time_colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
axes[1].set_title('(b) Time-of-Day Distribution', fontsize=13, weight='bold', pad=15)

# 3. CARLA Town Distribution
town_data = stats['town']
town_labels = sorted(town_data.keys())  # Town01, Town03, Town04, ...
town_values = [town_data[k] for k in town_labels]
town_total = sum(town_values)
town_pcts = [v/town_total*100 for v in town_values]

axes[2].pie(town_values, labels=[f'{l}\n{p:.1f}%' for l, p in zip(town_labels, town_pcts)],
            autopct='', colors=town_colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
axes[2].set_title('(c) CARLA Town Distribution', fontsize=13, weight='bold', pad=15)

plt.suptitle('Carla-OpenLane Environmental Diversity (Subset A)',
             fontsize=15, weight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_path = '/home/user/LaneSegNet/doc/figs/carla_distribution.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f'✓ Saved to {output_path}')

# Also save PNG for preview
output_path_png = '/home/user/LaneSegNet/doc/figs/carla_distribution.png'
plt.savefig(output_path_png, bbox_inches='tight', dpi=150)
print(f'✓ Saved to {output_path_png}')

plt.close()

# Print summary
print("\n=== Dataset Distribution Summary ===")
print(f"Total samples: {stats['total_samples']:,}")
print(f"\nWeather:")
for label, value in weather_data.items():
    print(f"  {label:10s}: {value:5,} ({value/weather_total*100:5.1f}%)")
print(f"\nTime-of-Day:")
for label, value in time_data.items():
    print(f"  {label:10s}: {value:5,} ({value/time_total*100:5.1f}%)")
print(f"\nCARLA Town:")
for label, value in zip(town_labels, town_values):
    print(f"  {label:10s}: {value:5,} ({value/town_total*100:5.1f}%)")
