#!/usr/bin/env python
"""
Convert checkpoint weight names from old version to new version.

Old version uses: lsls_head.text_guided_proj.weight/bias
New version uses: lsls_head.text_proj.weight/bias

Usage:
    python tools/convert_checkpoint_weight_names.py \
        --input work_dirs/ablation_multihead_h4_stage1/latest.pth \
        --output work_dirs/ablation_multihead_h4_stage1/latest_converted.pth
"""

import argparse
import torch


def convert_checkpoint(input_path, output_path):
    """Convert checkpoint weight names from old to new format."""

    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Get state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Convert weight names
    new_state_dict = {}
    converted_keys = []

    for key, value in state_dict.items():
        # Replace text_guided_proj with text_proj
        if 'text_guided_proj' in key:
            new_key = key.replace('text_guided_proj', 'text_proj')
            new_state_dict[new_key] = value
            converted_keys.append(f"{key} -> {new_key}")
        else:
            new_state_dict[key] = value

    # Update checkpoint
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict

    # Save converted checkpoint
    print(f"\nConverted {len(converted_keys)} keys:")
    for key_change in converted_keys:
        print(f"  {key_change}")

    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Convert checkpoint weight names')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output checkpoint')

    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()
