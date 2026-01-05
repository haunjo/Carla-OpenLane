"""
Ablation Study: Multi-head Cross-Attention (H=8)

This config uses 8-head cross-attention for maximum semantic diversity.

Changes from baseline:
- n_heads=8 (instead of 1)
- Many-head attention with 8 heads
- Same training schedule (8 epochs on Carla-OLV2)

Expected characteristics:
- Maximum semantic coverage
- Each head may capture very specific aspects
- Potential risk of overfitting or redundancy
"""

_base_ = ['./ablation_multihead_h4.py']

# Override only the multi-head parameter
model = dict(
    lsls_head=dict(
        n_heads=8,  # Many-head attention
    )
)

work_dir = 'work_dirs/ablation_multihead_h8'
