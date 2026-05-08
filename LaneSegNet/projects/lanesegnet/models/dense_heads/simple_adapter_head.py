"""
Simple Adapter for LSTE Head - Minimal Implementation
Only adds text guidance to existing RelationshipHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .relationship_head import RelationshipHead, MLP


@HEADS.register_module()
class SimpleAdapterHead(RelationshipHead):
    """
    Simple adapter that adds text guidance to RelationshipHead.
    Loads pretrained text features and applies them to LSTE predictions.
    """

    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=True,
                 text_checkpoint_path=None,
                 adapter_scale=1.0,
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
        self.adapter_scale = adapter_scale

        # Text guidance components (loaded from checkpoint, frozen)
        self.text_embeddings = nn.Parameter(torch.zeros(9, 1152), requires_grad=False)
        self.text_proj = MLP(1152, 512, 256, 3)  # input_dim, hidden_dim, output_dim, num_layers
        self.lane_proj = nn.Linear(256, 256)
        self.text_guided_proj = nn.Linear(256, 256)

        # Simple learnable adapter (only this is trainable)
        self.adapter_weights = nn.Parameter(torch.ones(9, 13))  # 9 templates x 13 TE classes

    def init_weights(self):
        """Initialize weights and load text guidance"""
        super().init_weights()

        if self.text_checkpoint_path:
            print(f"\n[SimpleAdapterHead] Loading text guidance from: {self.text_checkpoint_path}")
            checkpoint = torch.load(self.text_checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            # Load text guidance components (frozen)
            loaded_params = []

            # Load text embeddings
            if 'lste_head.text_embeddings' in state_dict:
                self.text_embeddings.data = state_dict['lste_head.text_embeddings']
                loaded_params.append('text_embeddings')

            # Load text projection
            text_proj_keys = [k for k in state_dict.keys() if 'lste_head.text_proj' in k]
            if text_proj_keys:
                text_proj_state = {k.replace('lste_head.text_proj.', ''): v
                                  for k, v in state_dict.items() if 'lste_head.text_proj' in k}
                self.text_proj.load_state_dict(text_proj_state)
                loaded_params.append('text_proj')

            # Load lane projection
            if 'lste_head.lane_proj.weight' in state_dict:
                self.lane_proj.weight.data = state_dict['lste_head.lane_proj.weight']
                if 'lste_head.lane_proj.bias' in state_dict:
                    self.lane_proj.bias.data = state_dict['lste_head.lane_proj.bias']
                loaded_params.append('lane_proj')

            # Load text guided projection
            if 'lste_head.text_guided_proj.weight' in state_dict:
                self.text_guided_proj.weight.data = state_dict['lste_head.text_guided_proj.weight']
                if 'lste_head.text_guided_proj.bias' in state_dict:
                    self.text_guided_proj.bias.data = state_dict['lste_head.text_guided_proj.bias']
                loaded_params.append('text_guided_proj')

            print(f"  Loaded components: {loaded_params}")

            # Freeze text guidance components
            for param in self.text_proj.parameters():
                param.requires_grad = False
            for param in self.lane_proj.parameters():
                param.requires_grad = False
            for param in self.text_guided_proj.parameters():
                param.requires_grad = False

            print("  Text guidance components frozen")

            # Initialize adapter weights
            nn.init.xavier_uniform_(self.adapter_weights)
            print(f"  Adapter weights initialized: {self.adapter_weights.shape}")

    def forward(self, feat_o1, feat_o2):
        """Forward with text guidance adapter"""
        # feats: D, B, num_query, num_embedding (following original RelationshipHead)
        o1_embeds = self.MLP_o1(feat_o1[-1])  # [B, N1, 128]
        o2_embeds = self.MLP_o2(feat_o2[-1])  # [B, N2, 128]

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)

        # Create relationship tensor
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)  # [B, N1, N2, 128]
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)  # [B, N1, N2, 128]

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)  # [B, N1, N2, 256]
        lste_pred = self.classifier(relationship_tensor)  # [B, N1, N2, 1]

        # Apply text guidance adapter
        if self.training:
            B = lste_pred.size(0)
            with torch.no_grad():
                # Get text features
                text_feats = self.text_proj(self.text_embeddings)  # [9, 256]
                text_feats = F.normalize(text_feats, dim=-1)

                # Get lane features and compute attention
                lane_feats = self.lane_proj(feat_o1[-1])  # [B, num_query_o1, 256]
                lane_feats = F.normalize(lane_feats, dim=-1)

                # Compute attention between lanes and text templates
                attention = torch.matmul(lane_feats, text_feats.T)  # [B, num_query_o1, 9]
                attention = F.softmax(attention / 0.07, dim=-1)  # temperature=0.07

            # Apply adapter weights based on attention
            # attention: [B, num_query_o1, 9], adapter_weights: [9, 13]
            adapter_modulation = torch.matmul(attention, self.adapter_weights)  # [B, num_query_o1, 13]

            # Modulate predictions
            # Assuming num_query_o2 matches the number of TE predictions
            if num_query_o2 <= 13:
                # Expand adapter modulation to match prediction shape
                adapter_modulation = adapter_modulation[:, :, :num_query_o2].unsqueeze(-1)  # [B, num_query_o1, num_query_o2, 1]
                lste_pred = lste_pred + self.adapter_scale * adapter_modulation

        return lste_pred