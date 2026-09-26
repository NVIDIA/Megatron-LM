# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Translate MoE FC1 checkpoint rows between contiguous and runtime GLU layouts.

Model weights are saved as [all gate rows, all up rows]. Optimizer tensors keep
their runtime layout: changing that layout on resume would also require moving
FP32 master weights and moments across distributed-optimizer shard boundaries.
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
    sizes = _interleave_sizes(_checkpoint_config(state_dict))
    # Before this conversion was introduced, native MLM saved physical runtime rows.
    # Bridge/exported weights without native MLM args use contiguous gate/up rows.
    model_sizes = sizes if 'args' in state_dict else {'routed': None, 'shared': None}
    return model_sizes, sizes


def _parallel_sizes(config):
    tp = _config_value(config, 'tensor_model_parallel_size', 1)
    etp = _config_value(config, 'expert_tensor_parallel_size') or tp
    return {'routed': etp, 'shared': tp}


def validate_glu_optimizer_layout(state_dict, args, *, loading_optimizer):
    """Reject incompatible runtime optimizer rows before loading any tensor storage.

    Legacy noncanonical model checkpoints also require the original TP partitioning.
    New canonical model weights can be loaded with a different runtime layout or TP
    size when optimizer state is not restored.
    """
    source_model, source_optimizer = _source_layouts(state_dict)
    target = _interleave_sizes(args)
    source_parallel = _parallel_sizes(state_dict.get(_LAYOUT_KEY, _checkpoint_config(state_dict)))
    target_parallel = _parallel_sizes(args)
    for kind in target:
        if source_model[kind] is not None and source_parallel[kind] != target_parallel[kind]:
            raise ValueError(
                f'Cannot reshard a legacy interleaved {kind} GLU model checkpoint. '
                'Re-save it with its original TP/ETP configuration to produce contiguous weights.'
            )
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


def _convert_model_rows(model_state_dict, source, target):
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


def _convert_model_sections(state_dict, source, target):
    result = copy(state_dict)
    if source != target:
        for key, value in state_dict.items():
            if key == 'model' or (key.startswith('model') and key[5:].isdigit()):
                result[key] = _convert_model_rows(value, source, target)
    return result


@torch.no_grad()
def prepare_glu_checkpoint_for_save(state_dict, args):
    """Canonicalize model sections after optimizer state has been constructed.

    Keeping this outside generate_state_dict preserves Parameter identity while the
    optimizer matches model parameters to sharding metadata. A new factory is made
    for each converted model weight; optimizer factories retain their runtime data.
    """
    sizes = _interleave_sizes(args)
    result = _convert_model_sections(state_dict, sizes, {'routed': None, 'shared': None})
    parallel_sizes = _parallel_sizes(args)
    result[_LAYOUT_KEY] = {
        'model': 'contiguous',
        'optimizer': sizes,
        'tensor_model_parallel_size': parallel_sizes['shared'],
        'expert_tensor_parallel_size': parallel_sizes['routed'],
    }
    return result


@torch.no_grad()
def prepare_glu_checkpoint_for_load(state_dict, args):
    """Convert full-precision model entries before model and master-weight loading."""
    source, _ = _source_layouts(state_dict)
    return _convert_model_sections(state_dict, source, _interleave_sizes(args))


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
