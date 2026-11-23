# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Translate MoE FC1 checkpoint rows between contiguous and runtime GLU layouts.

Native distributed checkpoints use Core factories for model and model-space
optimizer tensors. The plain-tensor helpers below serve legacy torch checkpoints,
whose optimizer buffers retain runtime order.
"""

import re
from copy import copy
from dataclasses import replace

import torch

from megatron.core.dist_checkpointing.glu import (
    glu_runtime_layout,
    permute_glu_rows,
    validate_glu_checkpoint_layout,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

_FC1_KEY = re.compile(r'(?:^|\.)(?:(experts|shared_experts)\.)?linear_fc1\.(weight|bias)\d*$')


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _interleave_sizes(config):
    layout = glu_runtime_layout(config)
    return {kind: layout[kind] for kind in ('routed', 'shared', 'dense')}


def _checkpoint_config(state_dict):
    if 'args' in state_dict:
        return state_dict['args']
    # Bridge stores its model configuration under cfg.model instead of args.
    return _config_value(state_dict.get('cfg'), 'model')


def validate_glu_optimizer_layout(
    state_dict, args, *, loading_optimizer, metadata=None, source_config=None
):
    """Validate explicit GLU layout metadata before loading any tensor storage."""
    validate_glu_checkpoint_layout(
        metadata if metadata is not None else state_dict.get('glu_checkpoint_metadata'),
        source_config=(
            source_config if source_config is not None else _checkpoint_config(state_dict)
        ),
        target_config=args,
        loading_optimizer=loading_optimizer,
    )


def _convert_model_rows(model_state_dict, sizes, *, interleave):
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
        kind = {'experts': 'routed', 'shared_experts': 'shared', None: 'dense'}[match[1]]
        if sizes[kind] is None:
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
        tensor = permute_glu_rows(tensor, sizes[kind], axis, interleave=interleave)
        # Never mutate the model, its parameter aliases, or the optimizer's factory.
        return replace(value, data=tensor) if sharded else tensor

    return convert(model_state_dict, '')


def _convert_model_sections(state_dict, sizes, *, interleave):
    result = copy(state_dict)
    if any(size is not None for size in sizes.values()):
        for key, value in state_dict.items():
            if key == 'model' or (key.startswith('model') and key[5:].isdigit()):
                result[key] = _convert_model_rows(value, sizes, interleave=interleave)
    return result


@torch.no_grad()
def prepare_glu_checkpoint_for_save(state_dict, args, *, use_factories=False):
    """Canonicalize legacy model tensors; native factories own their conversion."""
    if use_factories:
        return state_dict
    sizes = _interleave_sizes(args)
    return _convert_model_sections(state_dict, sizes, interleave=False)


@torch.no_grad()
def prepare_glu_checkpoint_for_load(state_dict, args, *, use_factories=False):
    """Restore legacy model rows; native factory merges already return runtime rows."""
    if use_factories:
        return state_dict
    return _convert_model_sections(state_dict, _interleave_sizes(args), interleave=True)


def validate_glu_checkpoint_backend(args, *, ckpt_format, skip_load_to_model_and_opt=False):
    """Require the explicit load-state-dict path used for GLU layout conversion."""
    if not any(size is not None for size in _interleave_sizes(args).values()):
        return
    if ckpt_format not in ('torch', 'torch_dist') or skip_load_to_model_and_opt:
        raise NotImplementedError(
            'GLU checkpoint layout conversion requires torch or torch_dist checkpoints '
            'with explicit model loading; in-place FSDP checkpoint loading is not supported.'
        )
