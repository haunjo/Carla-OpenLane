"""
Cross-Attention Adapter Head with MLP Gating
Loads frozen cross-attention from epoch 12 and adds learnable gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .relationship_head import RelationshipHead, MLP


@HEADS.register_module()
class CrossAttentionAdapterHead(RelationshipHead):
    """
    Cross-attention adapter that:
    1. Uses frozen text-guided cross-attention from pretrained model
    2. Adds learnable MLP gating to control feature fusion
    3. Only the gating MLP is trained
    """

    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=True,
                 text_checkpoint_path=None,
                 gating_hidden_dim=128,
                 gating_dropout=0.1,
                 fusion_scale=1.0,
                 loss_rel=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25),
                 init_cfg=None):

        # Initialize base RelationshipHead
        super().__init__(
            in_channels_o1=in_channels_o1,
            in_channels_o2=in_channels_o2,
            shared_param=shared_param,
            loss_rel=loss_rel,
            init_cfg=init_cfg
        )

        self.text_checkpoint_path = text_checkpoint_path
        self.fusion_scale = fusion_scale

        # Frozen text guidance components (loaded from checkpoint)
        self.text_embeddings = nn.Parameter(torch.zeros(9, 1152), requires_grad=False)
        self.text_proj = MLP(1152, 512, 256, 3)
        self.lane_proj = nn.Linear(256, 256)
        self.temperature = nn.Parameter(torch.tensor(0.0866), requires_grad=False)

        # Freeze text components
        for param in self.text_proj.parameters():
            param.requires_grad = False
        for param in self.lane_proj.parameters():
            param.requires_grad = False

        # Learnable MLP gating mechanism (ONLY THIS IS TRAINABLE)
        self.gating_mlp = nn.Sequential(
            nn.Linear(256 + 256, gating_hidden_dim),  # lane_feat + text_enhanced
            nn.ReLU(),
            nn.Dropout(gating_dropout),
            nn.Linear(gating_hidden_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gating_dropout),
            nn.Linear(gating_hidden_dim, 1),
            nn.Sigmoid()  # Gate value between 0 and 1
        )

        # Initialize gating to be moderately open
        for m in self.gating_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def init_weights(self):
        """Initialize weights and load frozen text guidance"""
        super().init_weights()

        if self.text_checkpoint_path:
            print(f"\n[CrossAttentionAdapter] Loading frozen cross-attention from: {self.text_checkpoint_path}")
            checkpoint = torch.load(self.text_checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            loaded_components = []

            # Load text embeddings
            if 'lste_head.text_embeddings' in state_dict:
                self.text_embeddings.data = state_dict['lste_head.text_embeddings']
                loaded_components.append('text_embeddings')

            # Load text projection MLP
            text_proj_state = {}
            for k, v in state_dict.items():
                if 'lste_head.text_proj' in k:
                    new_key = k.replace('lste_head.text_proj.', '')
                    text_proj_state[new_key] = v
            if text_proj_state:
                self.text_proj.load_state_dict(text_proj_state)
                loaded_components.append('text_proj')

            # Load lane projection
            if 'lste_head.lane_proj.weight' in state_dict:
                self.lane_proj.weight.data = state_dict['lste_head.lane_proj.weight']
                if 'lste_head.lane_proj.bias' in state_dict and self.lane_proj.bias is not None:
                    self.lane_proj.bias.data = state_dict['lste_head.lane_proj.bias']
                loaded_components.append('lane_proj')

            # Load temperature
            if 'lste_head.temperature' in state_dict:
                self.temperature.data = state_dict['lste_head.temperature']
                loaded_components.append('temperature')

            print(f"  ✓ Loaded frozen components: {loaded_components}")
            print(f"  ✓ Temperature: {self.temperature.item():.4f}")

            # Verify everything is frozen except gating
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"  ✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
            print(f"  ✓ Gating MLP parameters: {sum(p.numel() for p in self.gating_mlp.parameters()):,}")

    def compute_text_enhanced_features(self, lane_feats):
        """
        Compute text-enhanced lane features using frozen cross-attention
        Args:
            lane_feats: [B, N_lanes, 256]
        Returns:
            enhanced_feats: [B, N_lanes, 256]
            gate_values: [B, N_lanes, 1]
        """
        B, N_lanes = lane_feats.shape[:2]

        with torch.no_grad():
            # Get text features [9, 256]
            text_feats = self.text_proj(self.text_embeddings)
            text_feats = F.normalize(text_feats, dim=-1)

            # Project lane features for attention [B, N_lanes, 256]
            lane_query = self.lane_proj(lane_feats)
            lane_query = F.normalize(lane_query, dim=-1)

            # Compute cross-attention scores [B, N_lanes, 9]
            attention_scores = torch.matmul(lane_query, text_feats.T) / self.temperature
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Get text-enhanced features [B, N_lanes, 256]
            text_enhanced = torch.matmul(attention_weights, text_feats)

        # Compute gating values (TRAINABLE)
        # Concatenate original and enhanced features
        gate_input = torch.cat([lane_feats, text_enhanced], dim=-1)  # [B, N_lanes, 512]
        gate_values = self.gating_mlp(gate_input)  # [B, N_lanes, 1]

        # Apply gating
        enhanced_feats = lane_feats + self.fusion_scale * gate_values * text_enhanced

        return enhanced_feats, gate_values

    def forward(self, feat_o1, feat_o2):
        """Forward with gated cross-attention enhancement"""
        # Get base features from last decoder layer
        lane_feats = feat_o1[-1]  # [B, N_lanes, 256]
        te_feats = feat_o2[-1]  # [B, N_tes, 256]

        # Apply text enhancement with gating (only during training)
        if self.training:
            enhanced_lane_feats, gate_values = self.compute_text_enhanced_features(lane_feats)

            # Log gating statistics for monitoring
            if hasattr(self, 'gate_history'):
                self.gate_history.append(gate_values.mean().item())
            else:
                self.gate_history = [gate_values.mean().item()]
        else:
            # During inference, use original features or apply enhancement based on config
            enhanced_lane_feats = lane_feats

        # Process through MLPs (using enhanced lane features)
        o1_embeds = self.MLP_o1(enhanced_lane_feats)  # [B, N_lanes, 128]
        o2_embeds = self.MLP_o2(te_feats)  # [B, N_tes, 128]

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)

        # Create relationship tensor
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)  # [B, N_lanes, N_tes, 128]
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)  # [B, N_lanes, N_tes, 128]

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)  # [B, N_lanes, N_tes, 256]
        relationship_pred = self.classifier(relationship_tensor)  # [B, N_lanes, N_tes, 1]

        return relationship_pred

    def get_gate_statistics(self):
        """Get gating statistics for logging"""
        if hasattr(self, 'gate_history') and len(self.gate_history) > 0:
            history = self.gate_history[-100:]  # Last 100 values
            return {
                'gate_mean': sum(history) / len(history),
                'gate_min': min(history),
                'gate_max': max(history)
            }
        return {}