"""
Custom optimizer hook for adapter training
Ensures only adapter_weights are trainable
"""

from mmcv.runner import HOOKS, Hook
import torch


@HOOKS.register_module()
class AdapterOptimizerHook(Hook):
    """
    Hook to ensure only adapter weights are being optimized
    """

    def before_run(self, runner):
        """Freeze all parameters except adapter_weights before training starts"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only adapter_weights or gating_mlp
        unfrozen_params = []
        if hasattr(model, 'lste_head'):
            if hasattr(model.lste_head, 'adapter_weights'):
                model.lste_head.adapter_weights.requires_grad = True
                unfrozen_params.append('lste_head.adapter_weights')
            if hasattr(model.lste_head, 'gating_mlp'):
                for param in model.lste_head.gating_mlp.parameters():
                    param.requires_grad = True
                unfrozen_params.append('lste_head.gating_mlp')

        # Report status
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        runner.logger.info(f"[AdapterOptimizerHook] Frozen all params except: {unfrozen_params}")
        runner.logger.info(f"[AdapterOptimizerHook] Trainable params: {trainable} / {total} ({trainable/total*100:.4f}%)")

        # Update optimizer to only include trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            runner.optimizer.param_groups[0]['params'] = trainable_params
            runner.logger.info(f"[AdapterOptimizerHook] Optimizer updated with {len(trainable_params)} trainable parameters")

    def before_train_epoch(self, runner):
        """Check gradient flow at the beginning of each epoch"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        # Check which parameters have gradients
        params_with_grad = []
        params_without_grad = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'adapter_weights' in name or 'gating_mlp' in name:
                    params_with_grad.append(name)
                else:
                    params_without_grad.append(name)

        if params_with_grad:
            runner.logger.info(f"[AdapterOptimizerHook] Parameters with gradients: {params_with_grad}")

        if params_without_grad:
            runner.logger.warning(f"[AdapterOptimizerHook] Unexpected trainable params: {params_without_grad}")