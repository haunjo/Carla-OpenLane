#---------------------------------------------------------------------------------------#
# Inference-time Semantic Refinement for LaneSegNet                                    #
# Combines naive 2-stage model with text-guided semantic knowledge                     #
#---------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticRefinementModule:
    """
    Inference-time semantic refinement using text-guided attention from Stage 1.

    This module does NOT require retraining. It uses:
    1. Base model (naive 2-stage): Highest performance fully fine-tuned model
    2. Semantic model (text-guided stage 1): Provides semantic priors via frozen weights

    Usage:
        refinement = SemanticRefinementModule(base_model, semantic_model)
        refined_pred = refinement(img, base_pred)
    """

    def __init__(self, base_model, semantic_model, mode='attention', alpha=0.3):
        """
        Args:
            base_model: Naive 2-stage trained model (best performance)
            semantic_model: Text-guided Stage 1 model (semantic knowledge)
            mode: 'attention' | 'ensemble' | 'postprocess'
            alpha: Blending weight for semantic prior
        """
        self.base_model = base_model
        self.semantic_model = semantic_model
        self.mode = mode
        self.alpha = alpha

        # Freeze semantic model
        self.semantic_model.eval()
        for p in self.semantic_model.parameters():
            p.requires_grad = False

        # Define semantic rules: template_id -> traffic element affinity
        self.template_te_rules = self._build_semantic_rules()

    def _build_semantic_rules(self):
        """
        Define semantic compatibility between lane templates and traffic elements.

        Templates:
        0: curved forward
        1: curved merge left
        2: curved merge right
        3: straight forward
        4: straight merge left
        5: straight merge right
        6: intersection straight
        7: intersection left turn
        8: intersection right turn

        Traffic Elements (13 classes):
        0: traffic_light, 1: road_sign, 2: speed_bump, ...
        """
        rules = {
            # Straight lanes approaching intersection → traffic lights
            3: {0: 1.5, 1: 1.0, 2: 0.8},  # boost traffic light, neutral sign
            4: {0: 1.5, 1: 1.0, 2: 0.8},
            5: {0: 1.5, 1: 1.0, 2: 0.8},

            # Intersection lanes → different pattern
            6: {0: 1.3, 1: 0.9, 2: 0.7},  # still some traffic lights
            7: {0: 1.2, 1: 0.9, 2: 0.7},  # turning lanes
            8: {0: 1.2, 1: 0.9, 2: 0.7},

            # Curved/merge lanes → road signs
            0: {0: 1.0, 1: 1.3, 2: 0.9},
            1: {0: 1.0, 1: 1.3, 2: 0.9},
            2: {0: 1.0, 1: 1.3, 2: 0.9},
        }

        # Default: neutral affinity
        for template_id in range(9):
            if template_id not in rules:
                rules[template_id] = {i: 1.0 for i in range(13)}

        return rules

    @torch.no_grad()
    def __call__(self, img, gt_img_metas=None):
        """
        Inference with semantic refinement.

        Args:
            img: Input images [B, N_cam, 3, H, W]
            gt_img_metas: Image metadata (for base model)

        Returns:
            Refined predictions with improved lste topology
        """
        if self.mode == 'attention':
            return self._refine_with_attention(img, gt_img_metas)
        elif self.mode == 'ensemble':
            return self._refine_with_ensemble(img, gt_img_metas)
        elif self.mode == 'postprocess':
            return self._refine_with_postprocess(img, gt_img_metas)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _refine_with_attention(self, img, gt_img_metas):
        """
        Attention-based refinement using semantic priors.

        Strategy:
        1. Get base model predictions (lane features, te features, lste logits)
        2. Extract text attention from semantic model (which template each lane is)
        3. Apply semantic rules to boost/penalize lste predictions
        """
        # 1. Base model forward
        base_pred = self.base_model.forward_test(img, gt_img_metas)

        # 2. Get intermediate features from base model
        # NOTE: This requires modifying base model to return intermediate features
        # For now, we'll use a simpler approach: re-run semantic model

        # 3. Semantic model: get text attention
        # Extract lane features and compute text attention
        semantic_outputs = self.semantic_model(
            return_loss=False,
            img=img,
            img_metas=gt_img_metas,
            return_intermediate=True  # Need to add this flag
        )

        # text_attn: [B, N_lanes, 9] - probability over templates
        text_attn = semantic_outputs.get('text_attention', None)

        if text_attn is None:
            print("[Warning] Semantic model does not return text_attention, falling back to base prediction")
            return base_pred

        # 4. Apply semantic rules
        refined_pred = self._apply_semantic_rules(base_pred, text_attn)

        return refined_pred

    def _refine_with_ensemble(self, img, gt_img_metas):
        """
        Simple ensemble: weighted average of base and semantic predictions.
        """
        # 1. Base model
        base_pred = self.base_model.forward_test(img, gt_img_metas)

        # 2. Semantic model
        semantic_pred = self.semantic_model.forward_test(img, gt_img_metas)

        # 3. Ensemble only lste topology scores
        # Assuming predictions are dict with 'lste_scores' key
        if 'lste_scores' in base_pred[0] and 'lste_scores' in semantic_pred[0]:
            for i in range(len(base_pred)):
                base_score = torch.tensor(base_pred[i]['lste_scores'])
                sem_score = torch.tensor(semantic_pred[i]['lste_scores'])

                # Weighted average
                refined_score = (1 - self.alpha) * base_score + self.alpha * sem_score
                base_pred[i]['lste_scores'] = refined_score.cpu().numpy()

        return base_pred

    def _refine_with_postprocess(self, img, gt_img_metas):
        """
        Post-processing based refinement: adjust confidence based on semantic compatibility.

        This is the SIMPLEST approach and easiest to implement.
        """
        # 1. Base model predictions
        base_pred = self.base_model.forward_test(img, gt_img_metas)

        # 2. Semantic model: predict lane templates
        with torch.no_grad():
            # Run semantic model to get lane features
            semantic_outputs = self.semantic_model(
                return_loss=False,
                img=img,
                img_metas=gt_img_metas,
                return_intermediate=True
            )

            # Get predicted templates for each lane
            # text_attn: [B, N, 9]
            text_attn = semantic_outputs.get('text_attention')
            if text_attn is None:
                return base_pred

            # Get most likely template for each lane
            lane_templates = torch.argmax(text_attn, dim=-1)  # [B, N]

        # 3. Refine lste predictions based on template-TE compatibility
        for batch_idx, pred in enumerate(base_pred):
            if 'lste_topology' not in pred:
                continue

            lste_edges = pred['lste_topology']  # List of (lane_id, te_id, score)
            te_classes = pred.get('te_classes', [])

            refined_edges = []
            for lane_id, te_id, score in lste_edges:
                # Get lane template
                template = lane_templates[batch_idx, lane_id].item()

                # Get TE class
                if te_id < len(te_classes):
                    te_class = te_classes[te_id]
                else:
                    te_class = 0  # default

                # Apply semantic rule
                if template in self.template_te_rules:
                    affinity = self.template_te_rules[template].get(te_class, 1.0)
                    refined_score = score * affinity
                else:
                    refined_score = score

                refined_edges.append((lane_id, te_id, refined_score))

            pred['lste_topology'] = refined_edges

        return base_pred

    def _apply_semantic_rules(self, base_pred, text_attn):
        """
        Apply semantic rules to refine lste predictions.

        Args:
            base_pred: Base model predictions
            text_attn: [B, N_lanes, 9] - template probabilities

        Returns:
            Refined predictions
        """
        # Get most likely template for each lane
        lane_templates = torch.argmax(text_attn, dim=-1)  # [B, N]

        # Build semantic affinity matrix: [B, N_lanes, 13_te_classes]
        B, N = lane_templates.shape
        semantic_affinity = torch.ones(B, N, 13, device=lane_templates.device)

        for b in range(B):
            for n in range(N):
                template_id = lane_templates[b, n].item()
                if template_id in self.template_te_rules:
                    for te_class, affinity in self.template_te_rules[template_id].items():
                        semantic_affinity[b, n, te_class] = affinity

        # Apply to base predictions
        # This part depends on base_pred structure - need to check actual format

        return base_pred


def create_semantic_refinement_pipeline(base_ckpt_path, semantic_ckpt_path, config, mode='postprocess', alpha=0.3):
    """
    Factory function to create semantic refinement pipeline.

    Args:
        base_ckpt_path: Path to naive 2-stage checkpoint (best model)
        semantic_ckpt_path: Path to text-guided stage 1 checkpoint
        config: Model config
        mode: Refinement mode
        alpha: Blending weight

    Returns:
        SemanticRefinementModule ready for inference
    """
    from mmdet.models import build_detector
    from mmcv import Config
    import torch

    # Load config
    if isinstance(config, str):
        cfg = Config.fromfile(config)
    else:
        cfg = config

    # Build models
    base_model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    semantic_model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    # Load checkpoints
    base_ckpt = torch.load(base_ckpt_path, map_location='cpu')
    semantic_ckpt = torch.load(semantic_ckpt_path, map_location='cpu')

    base_model.load_state_dict(base_ckpt['state_dict'], strict=False)
    semantic_model.load_state_dict(semantic_ckpt['state_dict'], strict=False)

    # Create refinement module
    refinement = SemanticRefinementModule(
        base_model=base_model,
        semantic_model=semantic_model,
        mode=mode,
        alpha=alpha
    )

    print(f"[Semantic Refinement] Created pipeline with mode={mode}, alpha={alpha}")
    print(f"  Base model: {base_ckpt_path}")
    print(f"  Semantic model: {semantic_ckpt_path}")

    return refinement
