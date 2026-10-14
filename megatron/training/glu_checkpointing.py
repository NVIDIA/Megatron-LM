# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Translate MoE FC1 checkpoint rows between contiguous and runtime GLU layouts.

Model weights are saved as [all gate rows, all up rows]. Indexed routed experts
use their Core factory for torch_dist model and per-parameter optimizer tensors.
Other supported model paths are converted here; their optimizer layout is unchanged.
"""

import re
from copy import copy
from dataclasses import replace

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

_LAYOUT_KEY = 'glu_checkpoint_layout'
_FC1_KEY = re.compile(r'(?:^|\.)(experts|shared_experts)\.linear_fc1\.(weight|bias)\d*$')


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _interleave_sizes(config):
    sizes = {
        'routed': _config_value(config, 'moe_mlp_glu_interleave_size'),
        'shared': (
            _config_value(config, 'moe_shared_expert_glu_interleave_size')
            if _config_value(config, 'use_grouped_gemm_for_shared_expert', False)
            else None
        ),
    }
    for size in sizes.values():
        if size is not None and (not isinstance(size, int) or size <= 0):
            raise ValueError(f'GLU interleave size must be a positive integer, got {size!r}')
    return sizes


def _checkpoint_config(state_dict):
    if 'args' in state_dict:
        return state_dict['args']
    # Bridge model checkpoints are canonical, while its optimizer uses runtime rows.
    return _config_value(state_dict.get('cfg'), 'model')


def _source_layouts(state_dict):
    layout = state_dict.get(_LAYOUT_KEY)
    if layout is not None:
        if layout['model'] != 'contiguous':
            raise ValueError(f'Unknown GLU model checkpoint layout: {layout["model"]!r}')
        return {'routed': None, 'shared': None}, layout['optimizer']
    # Model input is always canonical. Do not infer or migrate historical physical
    # interleaved weights from the training arguments stored in a checkpoint.
    return {'routed': None, 'shared': None}, {'routed': None, 'shared': None}


def _factory_handles_routed(args, ckpt_format=None):
    return (
        (ckpt_format or _config_value(args, 'ckpt_format')) == 'torch_dist'
        and _config_value(args, 'moe_grouped_gemm', False)
        and not _config_value(args, 'moe_single_grouped_weight', False)
    )


def validate_glu_optimizer_format(args, metadata=None):
    """Reject bucket/partial-flat state before constructing optimizer load/save requests."""
    if not _factory_handles_routed(args):
        return
    if not _config_value(args, 'use_distributed_optimizer', False):
        return
    sharding = (metadata or {}).get(
        'distrib_optim_sharding_type',
        (
            'fully_reshardable'
            if _config_value(args, 'dist_ckpt_optim_fully_reshardable', False)
            else 'dp_reshardable'
        ),
    )
    gated = any(
        _config_value(args, flag, False) for flag in ('swiglu', 'quick_geglu', 'gated_linear_unit')
    )
    affected = _config_value(args, 'moe_mlp_glu_interleave_size') is not None or (
        gated and _config_value(args, 'expert_gtp_weight_remat_size', 1) > 1
    )
    if affected and sharding != 'fully_reshardable':
        raise NotImplementedError(
            'Routed GLU/GTP optimizer checkpointing requires the existing '
            'fully_reshardable per-parameter format. Select '
            '--dist-ckpt-optim-fully-reshardable, or explicitly use '
            '--no-save-optim / --no-load-optim for model-only checkpointing.'
        )


def _parallel_sizes(config):
    tp = _config_value(config, 'tensor_model_parallel_size', 1)
    etp = _config_value(config, 'expert_tensor_parallel_size') or tp
    return {'routed': etp, 'shared': tp}


def validate_glu_optimizer_layout(state_dict, args, *, loading_optimizer):
    """Reject incompatible runtime optimizer rows before loading any tensor storage.

    Model checkpoints are canonical. Factory-owned routed optimizer tensors are
    canonical too; other paths retain their existing runtime optimizer layout.
    """
    _, source_optimizer = _source_layouts(state_dict)
    target = _interleave_sizes(args)
    if _factory_handles_routed(args):
        target['routed'] = None
    if loading_optimizer:
        validate_glu_optimizer_format(args)
    source_parallel = _parallel_sizes(state_dict.get(_LAYOUT_KEY, _checkpoint_config(state_dict)))
    target_parallel = _parallel_sizes(args)
    for kind in target:
        if not loading_optimizer:
            continue
        if source_optimizer[kind] != target[kind] or (
            source_optimizer[kind] is not None and source_parallel[kind] != target_parallel[kind]
        ):
            raise ValueError(
                f'Cannot restore {kind} GLU optimizer state with a different interleave '
                'size or TP/ETP partitioning: FP32 master weights and optimizer moments '
                'use the saved runtime layout. Keep the saved configuration or use '
                '--finetune / --no-load-optim to initialize a new optimizer.'
            )


def _permute_glu_rows(tensor, size, axis, *, interleave):
    """Permute channels independently for each expert, before weight quantization."""
    if hasattr(tensor, 'placements'):
        raise NotImplementedError('GLU checkpoint conversion does not support DTensor weights.')
    # TE quantized Tensor subclasses must be unpacked before reshaping. Calling
    # torch.Tensor.dequantize() on an ordinary BF16 tensor would promote it to FP32.
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        tensor = tensor.dequantize().to(dtype=tensor.dtype)
    shape = tensor.shape
    if shape[axis] % (2 * size):
        raise ValueError(
            f'FC1 shape {tuple(shape)} has an invalid GLU channel dimension for block size {size}'
        )
    blocks = shape[axis] // (2 * size)
    split_shape = (2, blocks, size) if interleave else (blocks, 2, size)
    tensor = tensor.reshape(*shape[:axis], *split_shape, *shape[axis + 1 :])
    return tensor.transpose(axis, axis + 1).contiguous().reshape(shape)


def _convert_model_rows(model_state_dict, source, target, *, factory_handles_routed=False):
    def convert(value, key):
        if isinstance(value, dict):
            result = copy(value)
            for name, entry in value.items():
                result[name] = convert(entry, f'{key}.{name}' if key else name)
            return result
        if isinstance(value, list):
            return [convert(entry, key) for entry in value]
        match = _FC1_KEY.search(key)
        if match is None:
            return value
        kind = 'routed' if match[1] == 'experts' else 'shared'
        if factory_handles_routed and kind == 'routed' and key[-1].isdigit():
            # Applies after factory merge as well, when the value is a plain tensor.
            # Single grouped tensors and shared experts still use the helper.
            return value
        if source[kind] == target[kind]:
            return value
        sharded = isinstance(value, (ShardedTensor, ShardedTensorFactory))
        tensor = value.data if sharded else value
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Expected a tensor for GLU FC1 checkpoint entry {key}')
        if sharded and value.flattened_range is not None:
            raise NotImplementedError(
                'GLU model checkpoint conversion requires unflattened weights.'
            )
        # Indexed weights are [2F, H]; a single grouped weight is [E, 2F, H].
        # Bias has the same channel axis but no final H dimension.
        axis = tensor.ndim - (2 if match[2] == 'weight' else 1)
        if axis not in (0, 1):
            raise ValueError(
                f'Unexpected GLU FC1 checkpoint shape for {key}: {tuple(tensor.shape)}'
            )
        if source[kind] is not None:
            tensor = _permute_glu_rows(tensor, source[kind], axis, interleave=False)
        if target[kind] is not None:
            tensor = _permute_glu_rows(tensor, target[kind], axis, interleave=True)
        # Never mutate the model, its parameter aliases, or the optimizer's factory.
        return replace(value, data=tensor) if sharded else tensor

    return convert(model_state_dict, '')


def _convert_model_sections(state_dict, source, target, *, factory_handles_routed=False):
    result = copy(state_dict)
    if source != target:
        for key, value in state_dict.items():
            if key == 'model' or (key.startswith('model') and key[5:].isdigit()):
                result[key] = _convert_model_rows(
                    value, source, target, factory_handles_routed=factory_handles_routed
                )
    return result


@torch.no_grad()
def prepare_glu_checkpoint_for_save(state_dict, args, *, ckpt_format=None):
    """Canonicalize model sections after optimizer state has been constructed.

    Indexed routed FC1 is already handled by the Core factory, including optimizer
    tensors. Other paths convert only model data after optimizer metadata creation.
    """
    sizes = _interleave_sizes(args)
    factory_handles_routed = _factory_handles_routed(args, ckpt_format)
    result = _convert_model_sections(
        state_dict,
        sizes,
        {'routed': None, 'shared': None},
        factory_handles_routed=factory_handles_routed,
    )
    optimizer_sizes = dict(sizes)
    if factory_handles_routed:
        optimizer_sizes['routed'] = None
    parallel_sizes = _parallel_sizes(args)
    result[_LAYOUT_KEY] = {
        'model': 'contiguous',
        'optimizer': optimizer_sizes,
        'tensor_model_parallel_size': parallel_sizes['shared'],
        'expert_tensor_parallel_size': parallel_sizes['routed'],
    }
    return result


@torch.no_grad()
def prepare_glu_checkpoint_for_load(state_dict, args, *, ckpt_format=None):
    """Convert full-precision model entries before model and master-weight loading."""
    source, _ = _source_layouts(state_dict)
    return _convert_model_sections(
        state_dict,
        source,
        _interleave_sizes(args),
        factory_handles_routed=_factory_handles_routed(args, ckpt_format),
    )


def validate_glu_checkpoint_backend(
    state_dict, args, *, ckpt_format, skip_load_to_model_and_opt=False
):
    """Require the explicit load-state-dict path used for GLU layout conversion."""
    source, _ = _source_layouts(state_dict)
    if not any(size is not None for size in (*source.values(), *_interleave_sizes(args).values())):
        return
    if ckpt_format not in ('torch', 'torch_dist') or skip_load_to_model_and_opt:
        raise NotImplementedError(
            'GLU checkpoint layout conversion requires torch or torch_dist checkpoints '
            'with explicit model loading; in-place FSDP checkpoint loading is not supported.'
        )
