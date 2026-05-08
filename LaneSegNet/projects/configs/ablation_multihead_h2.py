"""
Ablation Study: Multi-head Cross-Attention (H=2)

This config extends the baseline with 2-head cross-attention.

Changes from baseline:
- n_heads=2 (instead of 1)
- Dual-head attention for semantic diversity
- Same training schedule (8 epochs on Carla-OLV2)

Expected improvements:
- Modest improvement over single-head
- Two complementary semantic aspects
"""

_base_ = ['./ablation_multihead_h4.py']

# Override only the multi-head parameter
model = dict(
    lsls_head=dict(
        n_heads=2,  # Dual-head attention
    )
)

work_dir = 'work_dirs/ablation_multihead_h2'
