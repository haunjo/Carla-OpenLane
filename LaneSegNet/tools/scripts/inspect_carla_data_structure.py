#!/usr/bin/env python3
"""
Inspect Carla-OLV2 data structure to understand what information is available.
"""
import pickle
import numpy as np
from pprint import pprint

# Load data
data_path = 'data/Carla-OLV2/data_dict_carla_train_argoverse2_ls.pkl'
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

print("=" * 80)
print("Data Structure Overview")
print("=" * 80)

# Find a sample dict
sample = None
sample_key = None

print(f"Number of top-level keys: {len(data_dict.keys())}")

# Iterate to find first dict sample
for key in list(data_dict.keys())[:100]:
    value = data_dict[key]
    if isinstance(value, dict) and 'annotation' in value:
        sample = value
        sample_key = key
        break

if sample is None:
    print("Could not find sample dict in first 100 entries!")
    exit(1)

print(f"\nFound sample with key: {sample_key}")

# Check meta_data
if sample and isinstance(sample, dict):
    if 'meta_data' in sample:
        print(f"\nmeta_data keys: {sample['meta_data'].keys()}")

    # Check annotation
    if 'annotation' in sample:
        annotation = sample['annotation']
        print(f"\nannotation keys: {annotation.keys()}")

        # Lane segments
        if 'lane_segment' in annotation:
            lanes = annotation['lane_segment']
            print(f"\nNumber of lane segments: {len(lanes)}")
            if len(lanes) > 0:
                print(f"\nFirst lane segment:")
                pprint(lanes[0])

        # Topology
        if 'topology_lsls' in annotation:
            topo = annotation['topology_lsls']
            print(f"\ntopology_lsls shape: {topo.shape}")
            print(f"Number of connections: {np.sum(topo)}")

print("\n" + "=" * 80)
