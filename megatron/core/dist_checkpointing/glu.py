# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared GLU checkpoint layout contract for model and optimizer tensors."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .mapping import ShardedTensorFactory


@dataclass
class GluTensorFactory(ShardedTensorFactory):
    """A canonical GLU factory, including when reused for optimizer state."""

    glu_interleave_size: int | None = None
    glu_tp_size: int = 1


def glu_factory_layouts(state) -> dict[str, tuple[int | None, int]]:
    """Collect the actual layouts before generic preprocessing expands factories."""
    result = {}

    def visit(value):
        if isinstance(value, GluTensorFactory):
            result[value.key] = (value.glu_interleave_size, value.glu_tp_size)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(state)
    return result


def glu_parameter_layouts(model) -> dict:
    """Return only FC1 parameters whose built implementation uses interleaved GLU."""
    result = {}
    for name, module in model.named_modules():
        size = getattr(module, '_glu_interleave_size', None)
        if size is not None:
            for param_name, _ in module.named_parameters(recurse=False):
                if param_name.startswith(('weight', 'bias')):
                    result[f'{name}.{param_name}' if name else param_name] = size
    return result


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
    models = None
    if isinstance(config, (list, tuple)):
        models = config
    elif isinstance(config, torch.nn.Module):
        models = [config]
    if models is not None:
        config = next(
            (m.config for model in models for m in model.modules() if hasattr(m, 'config')), None
        )
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
                _value(config, 'use_transformer_engine_op_fuser', False)
                or _value(config, 'use_te_op_fuser', False)
            )
            and (
                _value(config, 'use_grouped_gemm_for_dense_mlp', False)
                or _value(config, 'dense_grouped_gemm', False)
            )
            else None
        ),
        'tp': tp,
        'etp': _value(config, 'expert_tensor_parallel_size') or tp,
    }
    if models is not None:
        observed: dict[str, int | None] = {}
        for model in models:
            for name, module in model.named_modules():
                getter = getattr(module, '_checkpoint_glu_interleave_size', None)
                if not callable(getter) or not _value(
                    getattr(module, 'config', None), 'gated_linear_unit', False
                ):
                    continue
                kind = (
                    'shared'
                    if 'shared_experts' in name
                    else ('routed' if hasattr(module, 'num_local_experts') else 'dense')
                )
                size = getter()
                if kind in observed and observed[kind] != size:
                    raise ValueError(
                        f'Mixed {kind} GLU layouts require per-module optimizer layout metadata.'
                    )
                observed[kind] = size
        layout.update(observed)
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
        from torch.distributed.tensor import DTensor, distribute_tensor

        if hasattr(tensor._local_tensor, '__create_chunk_list__'):
            from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
                gather_uneven_dtensor_to_full_tensor,
            )

            full = gather_uneven_dtensor_to_full_tensor(tensor)._local_tensor
            converted = permute_glu_rows(full, size, axis, interleave=interleave)
            chunk = tensor._local_tensor.__create_chunk_list__()[0]
            slices = tuple(
                slice(offset, offset + length) for offset, length in zip(chunk.offsets, chunk.sizes)
            )
            return DTensor.from_local(
                converted[slices],
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
                shape=tensor.shape,
                stride=tensor.stride(),
            )
        full = permute_glu_rows(tensor.full_tensor(), size, axis, interleave=interleave)
        return distribute_tensor(full, tensor.device_mesh, tensor.placements)
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        tensor = tensor.dequantize().to(dtype=tensor.dtype)
    shape = tensor.shape
    if shape[axis] % (2 * size):
        raise ValueError(f'FC1 shape {tuple(shape)} is incompatible with GLU block size {size}')
    blocks = shape[axis] // (2 * size)
    split_shape = (2, blocks, size) if interleave else (blocks, 2, size)
    tensor = tensor.reshape(*shape[:axis], *split_shape, *shape[axis + 1 :])
    return tensor.transpose(axis, axis + 1).contiguous().reshape(shape)


def convert_glu_model_state(state_dict, model, *, interleave):
    """Convert legacy/FSDP model tensors using exact names from the built module.

    Native Core checkpoint factories own their conversion and must not use this helper.
    """

    def normalized(name):
        while name.startswith('module.'):
            name = name[len('module.') :]
        return name

    sizes = {normalized(key): size for key, size in glu_parameter_layouts(model).items()}
    if not sizes:
        return state_dict
    result = dict(state_dict)
    for key in sorted(state_dict):
        size = sizes.get(normalized(key))
        if size is not None:
            tensor = state_dict[key]
            if hasattr(tensor, 'placements') and not hasattr(
                tensor._local_tensor, '__create_chunk_list__'
            ):
                from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
                    update_uneven_dtensor_chunk_metadata,
                )

                update_uneven_dtensor_chunk_metadata(tensor)
            axis = tensor.ndim - (1 if key.rsplit('.', 1)[-1].startswith('bias') else 2)
            result[key] = permute_glu_rows(tensor, size, axis, interleave=interleave)
    return result
