#---------------------------------------------------------------------------------------#
# TextEnhancedRelationshipHead: Text-guided topology reasoning                         #
# Extends RelationshipHead with frozen SigLIP text embeddings for domain adaptation    #
#---------------------------------------------------------------------------------------#

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class TextEnhancedRelationshipHead(BaseModule):
    """
    RelationshipHead with text-guided enhancement for sim-to-real domain adaptation.

    Uses frozen SigLIP text embeddings to provide domain-invariant semantic guidance
    during topology reasoning. Text embeddings are aligned with lane features via
    cross-attention, with a learnable temperature parameter.

    Args:
        in_channels_o1 (int): Input channels for object 1 features
        in_channels_o2 (int, optional): Input channels for object 2. If None, uses in_channels_o1
        shared_param (bool): Whether to share parameters between o1 and o2 MLPs
        text_embed_path (str): Path to pre-generated text embeddings (.pt file)
        text_embed_dim (int): Dimension of text embeddings (default: 1152 for SigLIP-so400m)
        proj_dim (int): Projection dimension for text-lane alignment (default: 256)
        temperature_init (float): Initial temperature for attention scaling (default: 0.07)
        freeze_text_proj (bool): Whether to freeze text projection in Stage 2 (default: False)
        use_self_attn (bool): Whether to use lane self-attention before cross-attention (default: False)
        n_heads (int): Number of attention heads for cross-attention (default: 1)
        loss_rel (dict): Loss config for relationship prediction
        loss_text (dict): Loss config for text alignment supervision (Stage 1 only)
        init_cfg (dict, optional): Initialization config
    """

    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=True,
                 text_embed_path='text_embeddings_siglip.pt',
                 text_embed_dim=1152,
                 proj_dim=256,
                 temperature_init=0.07,
                 freeze_text_proj=False,
                 use_self_attn=False,
                 n_heads=1,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25),
                 loss_text=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                 init_cfg=None):
        super().__init__()

        # Store config
        self.in_channels_o1 = in_channels_o1
        self.in_channels_o2 = in_channels_o2 if in_channels_o2 is not None else in_channels_o1
        self.shared_param = shared_param
        self.text_embed_dim = text_embed_dim
        self.proj_dim = proj_dim
        self.freeze_text_proj = freeze_text_proj
        self.use_self_attn = use_self_attn
        self.n_heads = n_heads

        # ========== Text Embeddings (Frozen) ==========
        # Load pre-generated SigLIP embeddings
        if not os.path.isabs(text_embed_path):
            # Relative path from project root
            text_embed_path = os.path.join(
                os.path.dirname(__file__), '../../../..', text_embed_path
            )

        print(f"[TextEnhancedRelationshipHead] Loading text embeddings from: {text_embed_path}")
        text_data = torch.load(text_embed_path, map_location='cpu')
        text_embeddings = text_data['embeddings']  # [9, text_embed_dim]

        print(f"  ✓ Text embeddings shape: {text_embeddings.shape}")
        print(f"  ✓ Templates: {len(text_data['templates'])}")
        print(f"  ✓ Model: {text_data.get('model_name', 'unknown')}")

        # Register as buffer (frozen, non-trainable)
        self.register_buffer('text_embeddings', text_embeddings)
        self.num_templates = text_embeddings.size(0)

        # ========== Lane Self-Attention (Optional) ==========
        if use_self_attn:
            print(f"[TextEnhancedRelationshipHead] Using lane self-attention (n_heads={n_heads})")
            self.self_attn = nn.MultiheadAttention(
                embed_dim=in_channels_o1,
                num_heads=n_heads,
                batch_first=True
            )
            self.self_attn_norm = nn.LayerNorm(in_channels_o1)

        # ========== Learnable Projections ==========
        # For multi-head attention, we need to ensure proj_dim is divisible by n_heads
        if n_heads > 1:
            assert proj_dim % n_heads == 0, f"proj_dim ({proj_dim}) must be divisible by n_heads ({n_heads})"
            print(f"[TextEnhancedRelationshipHead] Using multi-head cross-attention (n_heads={n_heads})")

        # Text projection: text_embed_dim -> proj_dim
        self.text_proj = MLP(text_embed_dim, proj_dim * 2, proj_dim, num_layers=3)

        # Lane projection: in_channels_o1 -> proj_dim
        self.lane_proj = nn.Linear(in_channels_o1, proj_dim, bias=False)

        # Multi-head projections for cross-attention
        if n_heads > 1:
            # Q, K, V projections for multi-head attention
            self.Q_proj = nn.Linear(proj_dim, proj_dim)
            self.K_proj = nn.Linear(proj_dim, proj_dim)
            self.V_proj = nn.Linear(proj_dim, proj_dim)
            self.out_proj = nn.Linear(proj_dim, proj_dim)

        # Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # Projection back to original feature dimension
        self.text_guided_proj = nn.Linear(proj_dim, in_channels_o1, bias=False)

        # ========== Original RelationshipHead Components ==========
        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(self.in_channels_o2, self.in_channels_o2, 128, 3)

        self.classifier = MLP(256, 256, 1, 3)

        # ========== Losses ==========
        self.loss_rel = build_loss(loss_rel)
        self.loss_text = build_loss(loss_text) if loss_text is not None else None

        # Apply freezing if specified (for Stage 2)
        if freeze_text_proj:
            self._freeze_text_components()

    def _freeze_text_components(self):
        """Freeze text-related parameters for Stage 2 (domain adaptation)"""
        print("[TextEnhancedRelationshipHead] Freezing text components for Stage 2")

        # Freeze text projection
        for param in self.text_proj.parameters():
            param.requires_grad = False

        # Freeze lane projection
        for param in self.lane_proj.parameters():
            param.requires_grad = False

        # Freeze text_guided_proj
        for param in self.text_guided_proj.parameters():
            param.requires_grad = False

        # Freeze multi-head projections if present
        if self.n_heads > 1:
            for param in self.Q_proj.parameters():
                param.requires_grad = False
            for param in self.K_proj.parameters():
                param.requires_grad = False
            for param in self.V_proj.parameters():
                param.requires_grad = False
            for param in self.out_proj.parameters():
                param.requires_grad = False
            print("  ✓ Multi-head projections: frozen")

        # Freeze self-attention if present
        if self.use_self_attn:
            for param in self.self_attn.parameters():
                param.requires_grad = False
            for param in self.self_attn_norm.parameters():
                param.requires_grad = False
            print("  ✓ Self-attention: frozen")

        # Freeze temperature
        self.temperature.requires_grad = False

        print("  ✓ Text projection: frozen")
        print("  ✓ Lane projection: frozen")
        print("  ✓ Text-guided projection: frozen")
        print("  ✓ Temperature: frozen")

    def _apply_text_guidance(self, lane_feats):
        """
        Apply text-guided enhancement to lane features via cross-attention.

        Supports:
        1. Optional lane self-attention (if use_self_attn=True)
        2. Single-head or multi-head cross-attention (controlled by n_heads)

        Args:
            lane_feats: Lane features [B, N, in_channels_o1]

        Returns:
            enhanced_feats: Text-enhanced lane features [B, N, in_channels_o1]
            attention_weights: Attention weights [B, N, 9] for loss computation
        """
        B, N, D = lane_feats.shape

        # ========== Step 1: Lane Self-Attention (Optional) ==========
        if self.use_self_attn:
            # Lane-to-lane context aggregation
            lane_contextual, _ = self.self_attn(
                lane_feats, lane_feats, lane_feats
            )  # [B, N, D]

            # Residual + LayerNorm
            lane_feats = self.self_attn_norm(lane_feats + lane_contextual)

        # ========== Step 2: Text Cross-Attention ==========
        # Project text embeddings [9, text_embed_dim] -> [9, proj_dim]
        text_proj = self.text_proj(self.text_embeddings)  # [9, proj_dim]

        # Project lane features [B, N, D] -> [B, N, proj_dim]
        lane_proj = self.lane_proj(lane_feats)  # [B, N, proj_dim]

        if self.n_heads == 1:
            # ========== Single-head Cross-Attention ==========
            # Compute attention logits [B, N, 9]
            attention_logits = torch.matmul(
                lane_proj, text_proj.t()
            ) / self.temperature.clamp(min=0.01)  # Prevent division by zero

            # Softmax over text templates
            attention_weights = F.softmax(attention_logits, dim=-1)  # [B, N, 9]

            # Get text-guided features [B, N, proj_dim]
            text_guided = torch.matmul(attention_weights, text_proj)
        else:
            # ========== Multi-head Cross-Attention ==========
            d_k = self.proj_dim // self.n_heads
            T = text_proj.size(0)  # 9 templates

            # Project and split into heads
            Q = self.Q_proj(lane_proj).view(B, N, self.n_heads, d_k).transpose(1, 2)  # [B, H, N, d_k]
            K = self.K_proj(text_proj).view(1, T, self.n_heads, d_k).transpose(1, 2)  # [1, H, 9, d_k]
            V = self.V_proj(text_proj).view(1, T, self.n_heads, d_k).transpose(1, 2)  # [1, H, 9, d_k]

            # Scaled dot-product attention per head
            attn_logits = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, N, 9]
            attn_logits = attn_logits / self.temperature.clamp(min=0.01)

            # Softmax over text templates
            attn_weights_multihead = F.softmax(attn_logits, dim=-1)  # [B, H, N, 9]

            # Apply attention
            attended = attn_weights_multihead @ V  # [B, H, N, d_k]

            # Concatenate heads
            attended = attended.transpose(1, 2).contiguous().view(B, N, self.proj_dim)  # [B, N, proj_dim]

            # Output projection
            text_guided = self.out_proj(attended)  # [B, N, proj_dim]

            # For loss computation, average attention weights across heads
            attention_weights = attn_weights_multihead.mean(dim=1)  # [B, N, 9]

        # ========== Step 3: Project back and residual connection ==========
        # Project back to original dimension [B, N, D]
        text_guided_full = self.text_guided_proj(text_guided)

        # Residual connection
        enhanced_feats = lane_feats + text_guided_full

        return enhanced_feats, attention_weights

    def forward(self, o1_feats, o2_feats):
        """
        Forward pass with text-guided enhancement.

        Args:
            o1_feats: List of object 1 features at different layers.
                      o1_feats[-1] has shape [B, N1, in_channels_o1]
            o2_feats: List of object 2 features.
                      o2_feats[-1] has shape [B, N2, in_channels_o2]

        Returns:
            relationship_pred: Relationship predictions [B, N1, N2, 1]
            attention_weights_o1: Attention weights for o1 [B, N1, 9] (optional)
            attention_weights_o2: Attention weights for o2 [B, N2, 9] (optional)
        """
        # Apply text-guided enhancement to lane features
        o1_feats_last = o1_feats[-1]  # [B, N1, D]
        o1_enhanced, attention_o1 = self._apply_text_guidance(o1_feats_last)

        # For o2: apply same enhancement if shared_param, else separate
        if self.shared_param:
            o2_feats_last = o2_feats[-1]
            o2_enhanced, attention_o2 = self._apply_text_guidance(o2_feats_last)
        else:
            # If not shared, o2 might have different dimension
            # For now, assume same dimension (typical for lane-to-lane)
            o2_feats_last = o2_feats[-1]
            o2_enhanced, attention_o2 = self._apply_text_guidance(o2_feats_last)

        # Apply original MLPs to enhanced features
        o1_embeds = self.MLP_o1(o1_enhanced)  # [B, N1, 128]
        o2_embeds = self.MLP_o2(o2_enhanced)  # [B, N2, 128]

        # Pairwise relationship prediction (same as original)
        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)  # [B, N1, N2, 128]
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)  # [B, N1, N2, 128]

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)  # [B, N1, N2, 256]
        relationship_pred = self.classifier(relationship_tensor)  # [B, N1, N2, 1]

        # Return attention weights for text loss computation
        return relationship_pred, attention_o1, attention_o2

    def forward_train(self, o1_feats, o1_assign_results, o2_feats, o2_assign_results,
                      gt_adj, gt_text_labels=None):
        """
        Training forward with optional text supervision.

        Args:
            o1_feats: Object 1 features
            o1_assign_results: Hungarian assignment results for o1
            o2_feats: Object 2 features
            o2_assign_results: Hungarian assignment results for o2
            gt_adj: Ground truth adjacency matrix
            gt_text_labels: Ground truth template IDs [B, N] (optional, for Stage 1 only)

        Returns:
            losses: Dict of losses (loss_rel, loss_text if applicable)
        """
        rel_pred, attention_o1, attention_o2 = self.forward(o1_feats, o2_feats)

        # Topology loss
        losses = self.loss(rel_pred, gt_adj, o1_assign_results, o2_assign_results)

        # Text alignment loss (only in Stage 1 with Carla data)
        if gt_text_labels is not None and self.loss_text is not None:
            loss_text = self._compute_text_loss(
                attention_o1, gt_text_labels, o1_assign_results
            )
            losses['loss_text'] = loss_text

        return losses

    def _compute_text_loss(self, attention_weights, gt_text_labels, assign_results):
        """
        Compute text alignment loss using GT template IDs.

        Only applied to matched queries (via Hungarian assignment).

        Args:
            attention_weights: Attention weights [B, N, 9]
            gt_text_labels: GT template IDs [B, max_lanes] (may have padding)
            assign_results: Hungarian assignment results

        Returns:
            loss_text: Cross-entropy loss on attention distribution
        """
        B, N, num_templates = attention_weights.shape
        assign = assign_results[-1]
        pos_inds = assign['pos_inds']
        pos_assigned_gt_inds = assign['pos_assigned_gt_inds']

        losses = []
        for i in range(B):
            # Get matched queries and their GT indices
            matched_query_inds = pos_inds[i]  # [num_matched]
            matched_gt_inds = pos_assigned_gt_inds[i]  # [num_matched]

            if len(matched_query_inds) == 0:
                continue

            # Get GT text labels for matched lanes
            gt_labels = gt_text_labels[i][matched_gt_inds]  # [num_matched]

            # Filter out padding (-1 or >= num_templates)
            valid_mask = (gt_labels >= 0) & (gt_labels < num_templates)
            if valid_mask.sum() == 0:
                continue

            # Get attention for matched queries
            matched_attention = attention_weights[i][matched_query_inds][valid_mask]  # [num_valid, 9]
            valid_gt_labels = gt_labels[valid_mask]  # [num_valid]

            # Cross-entropy loss (attention weights vs GT template ID)
            loss = F.cross_entropy(
                matched_attention, valid_gt_labels, reduction='mean'
            )
            losses.append(loss)

        if len(losses) == 0:
            # No valid supervision in this batch
            return attention_weights.sum() * 0

        return torch.stack(losses).mean()

    def get_relationship(self, o1_feats, o2_feats):
        """Inference-time relationship prediction"""
        rel_pred, _, _ = self.forward(o1_feats, o2_feats)
        rel_results = rel_pred.squeeze(-1).sigmoid()
        rel_results = [_ for _ in rel_results]
        return rel_results

    def loss(self, rel_preds, gt_adjs, o1_assign_results, o2_assign_results):
        """
        Compute relationship loss (same as original RelationshipHead).

        Note: rel_preds is now a tuple (rel_pred, attention_o1, attention_o2)
        from forward(), so we extract the first element.
        """
        # Extract relationship predictions if tuple
        if isinstance(rel_preds, tuple):
            rel_preds = rel_preds[0]

        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results[-1]
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results[-1]
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        targets = []
        masked_rel_preds = []
        for i in range(B):
            gt_adj = gt_adjs[i]
            len_o1 = gt_adj.size(0)
            len_o2 = gt_adj.size(1)
            o1_pos_mask = o1_pos_assigned_gt_inds[i] < len_o1
            o2_pos_mask = o2_pos_assigned_gt_inds[i] < len_o2

            masked_rel_pred = rel_preds[i][o1_pos_inds[i]][:, o2_pos_inds[i]][o1_pos_mask][:, o2_pos_mask]
            masked_rel_preds.append(masked_rel_pred.view(-1, 1))

            target = gt_adj[o1_pos_assigned_gt_inds[i][o1_pos_mask]][:, o2_pos_assigned_gt_inds[i][o2_pos_mask]]
            targets.append(1 - target.view(-1).long())

        targets = torch.cat(targets, dim=0)
        rel_preds = torch.cat(masked_rel_preds, dim=0)

        if torch.numel(targets) == 0:
            return dict(loss_rel=rel_preds.sum() * 0)

        loss_rel = self.loss_rel(rel_preds, targets, avg_factor=targets.sum())

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        return dict(loss_rel=loss_rel)
