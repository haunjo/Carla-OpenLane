"""
Ablation Study: Single-head Cross-Attention (H=1, Baseline)

This is the baseline text-guided model with single-head cross-attention.

Changes from multi-head versions:
- n_heads=1 (single attention head)
- Simpler attention mechanism
- Same training schedule (8 epochs on Carla-OLV2)

Baseline for comparison with multi-head variants.
"""

_base_ = ['./ablation_multihead_h4.py']

# Override only the multi-head parameter
model = dict(
    lsls_head=dict(
        n_heads=1,  # Single-head attention (baseline)
    )
)

work_dir = 'work_dirs/ablation_multihead_h1'
