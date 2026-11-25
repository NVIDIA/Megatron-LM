# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep activation checkpointing differentiable when base weights are frozen."""

import torch


def _require_input_grad(module, args, kwargs):
    if not module.training or not torch.is_grad_enabled():
        return args, kwargs
    hidden = args[0] if args else kwargs.get("hidden_states")
    if torch.is_tensor(hidden) and hidden.is_floating_point() and not hidden.requires_grad:
        hidden = hidden.detach().requires_grad_(True)
        if args:
            args = (hidden, *args[1:])
        else:
            kwargs = {**kwargs, "hidden_states": hidden}
    return args, kwargs


def enable_recompute_input_grad(chunks):
    """Enable gradients on checkpointed block inputs without unfreezing base parameters."""
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.transformer.transformer_block import TransformerBlock

    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, (TransformerBlock, HybridStack)):
                config = module.config
                if config.recompute_granularity is not None:
                    module.register_forward_pre_hook(_require_input_grad, with_kwargs=True)
