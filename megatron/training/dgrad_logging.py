# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""dgrad logging using backward hooks."""

from collections import defaultdict
import torch
import torch.nn as nn

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from .checkpointing import save_grads
from .utils import unwrap_model


def _get_linear_types():
    """Build tuple of linear layer types to capture gradients from."""
    types = [nn.Linear, nn.Embedding, ColumnParallelLinear, RowParallelLinear]

    # Add Transformer Engine layers if available.
    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        types.extend([TELinear, TEColumnParallelLinear, TERowParallelLinear,
                      TELayerNormColumnParallelLinear])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        if TEGroupedLinear is not None:
            types.extend([TEGroupedLinear, TEColumnParallelGroupedLinear,
                          TERowParallelGroupedLinear])
    except ImportError:
        pass

    return tuple(types)


LINEAR_TYPES = _get_linear_types()


class DataGradLogger:
    """Captures and saves gradients from all linear layers using backward hooks.
    
    NOTE: Right now, we only save the dgrads for the last microbatch in a batch on DP replica 0.
    The code below would need to be extended to save dgrads for all microbatches in a batch."""

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._dgrads_state_dict = defaultdict(dict)
        self._hooks = []

    def _make_hook(self, model_chunk_name: str, module_name: str):
        """Create a backward hook for a named module."""
        def hook(_, grad_input, grad_output):
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    grad_name = f"{module_name}/output{idx}"
                    self._dgrads_state_dict[model_chunk_name][grad_name] = grad.detach().cpu()
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    grad_name = f"{module_name}/input{idx}"
                    self._dgrads_state_dict[model_chunk_name][grad_name] = grad.detach().cpu()
        return hook

    def save(self, iteration: int):
        """Save captured gradients to disk and clear the buffer."""
        if not self._dgrads_state_dict:
            return
        save_grads(self._save_dir, self._dgrads_state_dict, iteration, "dgrads")
        self._dgrads_state_dict.clear()

    def register_hooks(self, model: torch.nn.Module):
        """Find and register hooks on all linear layers."""
        assert len(self._hooks) == 0
        for model_chunk_id, model_chunk in enumerate(model):
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for module_name, module in unwrapped_model_chunk.named_modules():
                if isinstance(module, LINEAR_TYPES):
                    model_chunk_name = f"model_chunk{model_chunk_id}"
                    handle = module.register_full_backward_hook(
                        self._make_hook(model_chunk_name, module_name)
                    )
                    self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


_LOGGER = None


def enable_dgrad_logging(model: torch.nn.Module, save_dir: str):
    """Enable dgrad logging on a model."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = DataGradLogger(save_dir)
    _LOGGER.register_hooks(model)


def disable_dgrad_logging():
    """Disable dgrad logging on a model."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.remove_hooks()


def save_dgrads(iteration: int):
    """Save dgrads to disk."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.save(iteration)
