# Cross-Attention Adapter Architecture

## Overview
The CrossAttentionAdapterHead implements a learnable gating mechanism to control text-guided feature fusion while keeping the cross-attention mechanism frozen.

## Architecture Flow

```
1. INPUT FEATURES
   lane_feats [B, N_lanes, 256] ──┐
   te_feats [B, N_tes, 256]       │
                                  ↓
2. CROSS-ATTENTION (FROZEN)
   text_embeddings [9, 1152] ──→ text_proj ──→ text_feats [9, 256]
                                                      ↓
   lane_feats ──→ lane_proj ──→ attention ──→ text_enhanced [B, N_lanes, 256]
                                                      ↓
3. MLP GATING (TRAINABLE)
   [lane_feats, text_enhanced] ──→ MLP (512→128→128→1) ──→ gate [B, N_lanes, 1]
                                                                    ↓
4. FEATURE FUSION
   enhanced = lane_feats + (gate * fusion_scale * text_enhanced)
                                                      ↓
5. RELATION PREDICTION
   enhanced_lanes + te_feats ──→ MLPs ──→ classifier ──→ lste_pred [B, N_lanes, N_tes, 1]
```

## Component Details

### Frozen Components (from epoch_12.pth)
- **text_embeddings**: [9, 1152] - 9 semantic templates
- **text_proj**: MLP(1152→512→256) - Projects text to feature space
- **lane_proj**: Linear(256→256) - Projects lanes for attention
- **temperature**: 0.0866 - Attention temperature

### Trainable Component (NEW)
- **gating_mlp**: Sequential MLP with 82,305 parameters
  - Layer 1: Linear(512→128) + ReLU + Dropout(0.1)
  - Layer 2: Linear(128→128) + ReLU + Dropout(0.1)
  - Layer 3: Linear(128→1) + Sigmoid

## Training Strategy

1. **Load frozen cross-attention**: All text guidance components from epoch_12.pth
2. **Load base model**: RelationshipHead MLPs and classifier from epoch_23.pth
3. **Train only gating MLP**: 82,305 parameters (0.01% of total)
4. **Optimize LSTE loss**: Learn optimal fusion amount

## Key Advantages

1. **Preserves learned representations**: Cross-attention stays frozen
2. **Adaptive fusion**: Learns how much text guidance to apply
3. **Efficient training**: Only 82K parameters to train
4. **Stable optimization**: Gating prevents catastrophic forgetting

## Usage

```bash
bash run_cross_attention_adapter.sh
```

This trains for 1 epoch, learning the optimal gating values to control text-guided feature enhancement.