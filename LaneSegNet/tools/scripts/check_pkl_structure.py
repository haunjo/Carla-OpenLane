#!/usr/bin/env python3
"""Check structure of Carla-OLV2 pkl files."""

import pickle
import sys

pkl_path = sys.argv[1] if len(sys.argv) > 1 else '/home/user/LaneSegNet/data/Carla-OLV2/data_dict_carla_argoverse2_train_lanesegnet.pkl'

print(f"Loading {pkl_path}...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Total samples: {len(data)}")

# Get first sample
sample_key = list(data.keys())[0]
sample = data[sample_key]

print(f"\nSample key: {sample_key}")
print(f"Top-level keys: {list(sample.keys())}")
print(f"\nAnnotation keys: {list(sample['annotation'].keys())}")

# Lane segment structure
print(f"\nNumber of lane segments: {len(sample['annotation']['lane_segment'])}")
if len(sample['annotation']['lane_segment']) > 0:
    print(f"Lane segment [0] keys: {list(sample['annotation']['lane_segment'][0].keys())}")

# Traffic element structure
print(f"\nNumber of traffic elements: {len(sample['annotation']['traffic_element'])}")
if len(sample['annotation']['traffic_element']) > 0:
    print(f"Traffic element [0] keys: {list(sample['annotation']['traffic_element'][0].keys())}")

# Check for text labels
has_lane_text = 'lane_text_labels' in sample['annotation']
print(f"\nHas lane_text_labels? {has_lane_text}")
if has_lane_text:
    print(f"  Shape: {sample['annotation']['lane_text_labels'].shape}")
    print(f"  Values: {sample['annotation']['lane_text_labels']}")
    print(f"  Num labels: {len(sample['annotation']['lane_text_labels'])}")
    print(f"  Num lane_segments: {len(sample['annotation']['lane_segment'])}")

has_te_text = 'traffic_element_text_labels' in sample['annotation']
print(f"\nHas traffic_element_text_labels? {has_te_text}")
if has_te_text:
    print(f"  Shape: {sample['annotation']['traffic_element_text_labels'].shape}")
    print(f"  Values: {sample['annotation']['traffic_element_text_labels']}")
