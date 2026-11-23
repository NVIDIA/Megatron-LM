# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared GLU checkpoint layout contract for model and optimizer tensors."""

from collections.abc import Mapping
from typing import Any

import torch


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        transformer = config.get('transformer')
    else:
        if config is None:
            return default
        if hasattr(config, name):
            return getattr(config, name)
        transformer = getattr(config, 'transformer', None)
    return _value(transformer, name, default)


def glu_runtime_layout(config: Any) -> dict:
    """Describe runtime row order and partitioning from resolved or saved configuration."""
    gated = (
        _value(config, 'gated_linear_unit', False)
        or _value(config, 'swiglu', False)
        or (_value(config, 'quick_geglu', False) and not _value(config, 'squared_relu', False))
        or _value(config, 'activation_func') == 'swiglu'
    )
    tp = _value(config, 'tensor_model_parallel_size', 1)
    layout = {
        'routed': _value(config, 'moe_mlp_glu_interleave_size') if gated else None,
        'shared': (
            _value(config, 'moe_shared_expert_glu_interleave_size')
            if gated and _value(config, 'use_grouped_gemm_for_shared_expert', False)
            else None
        ),
        'dense': (
            32
            if gated
            and tp == 1
            and (
                _value(config, 'use_grouped_gemm_for_dense_mlp', False)
                or _value(config, 'dense_grouped_gemm', False)
            )
            else None
        ),
        'tp': tp,
        'etp': _value(config, 'expert_tensor_parallel_size') or tp,
    }
    for kind in ('routed', 'shared', 'dense'):
        size = layout[kind]
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise ValueError(f'GLU interleave size must be a positive integer, got {size!r}')
    return layout


def glu_checkpoint_metadata(config: Any, *, optimizer_uses_model_layout: bool) -> dict:
    """Describe canonical model rows and either canonical or raw-buffer optimizer rows."""
    return {
        'glu_checkpoint_layout': 'contiguous',
        'glu_optimizer_layout': 'contiguous' if optimizer_uses_model_layout else 'runtime',
        'glu_runtime_layout': glu_runtime_layout(config),
    }


def validate_glu_checkpoint_layout(
    metadata: dict | None, *, source_config: Any, target_config: Any, loading_optimizer: bool
) -> None:
    """Validate layout before checkpoint loading can write into parameter storage.

    Unmarked checkpoints from a known noninterleaved run retain compatibility.
    Historical interleaved checkpoints are ambiguous: frameworks did not apply
    the same conversion to tensors and factories. Require verified conversion
    instead of inferring their stored row order from runtime settings.
    """
    metadata = metadata or {}
    source = metadata.get('glu_runtime_layout', glu_runtime_layout(source_config))
    target = glu_runtime_layout(target_config)
    model_layout = metadata.get('glu_checkpoint_layout')
    if model_layout not in (None, 'contiguous'):
        raise ValueError(f'Unsupported GLU checkpoint layout: {model_layout!r}')
    interleaved_source = any(source[k] is not None for k in ('routed', 'shared', 'dense'))
    interleaved_target = any(target[k] is not None for k in ('routed', 'shared', 'dense'))
    if model_layout is None and (
        interleaved_source or (source_config is None and interleaved_target)
    ):
        raise ValueError(
            'Cannot establish the layout of an unmarked interleaved GLU checkpoint. '
            'Verify and convert its model, master weights and optimizer moments before '
            'adding glu_checkpoint_layout metadata; do not assume contiguous rows.'
        )
    # Historical model-space checkpoints from a known noninterleaved run already
    # contain canonical moments. Raw DP buffers still require matching settings.
    legacy_optimizer_layout = (
        'contiguous'
        if source_config is not None
        and not interleaved_source
        and metadata.get('distrib_optim_sharding_type') == 'fully_reshardable'
        else 'runtime'
    )
    optimizer_layout = metadata.get('glu_optimizer_layout', legacy_optimizer_layout)
    if optimizer_layout not in ('runtime', 'contiguous'):
        raise ValueError(f'Unsupported GLU optimizer layout: {optimizer_layout!r}')
    if not loading_optimizer or optimizer_layout == 'contiguous':
        return
    for kind in ('routed', 'shared', 'dense'):
        partition = 'etp' if kind == 'routed' else 'tp'
        if source[kind] != target[kind] or (
            source[kind] is not None and source[partition] != target[partition]
        ):
            raise ValueError(
                f'Cannot restore {kind} GLU optimizer state with different runtime layout '
                'or TP/ETP partitioning. Raw optimizer buffers retain runtime row order. '
                'Keep the saved settings or save with fully_reshardable optimizer state '
                'before changing the layout; model-only loading is also supported.'
            )


def permute_glu_rows(
    tensor: torch.Tensor, size: int, axis: int = 0, *, interleave: bool
) -> torch.Tensor:
    """Convert complete gate/up blocks between canonical and runtime row order."""
    if hasattr(tensor, 'placements'):
        raise NotImplementedError('GLU checkpoint conversion does not support DTensor weights.')
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        tensor = tensor.dequantize().to(dtype=tensor.dtype)
    shape = tensor.shape
    if shape[axis] % (2 * size):
        raise ValueError(f'FC1 shape {tuple(shape)} is incompatible with GLU block size {size}')
    blocks = shape[axis] // (2 * size)
    split_shape = (2, blocks, size) if interleave else (blocks, 2, size)
    tensor = tensor.reshape(*shape[:axis], *split_shape, *shape[axis + 1 :])
    return tensor.transpose(axis, axis + 1).contiguous().reshape(shape)
