# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""
from operator import itemgetter
from typing import Dict, Iterable, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor, StateDict
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
    )


def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    state_dict_prefix: str,
    sharded_key_prefix: str,
    tensor_parallel_layers_axis_map: Dict[str, int],
    sharded_offsets: Iterable[Tuple[int, int, int]],
    extra_state_suffix: str = '_extra_state',
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps all regular tensors with ShardedTensor
    sharded according to `tensor_parallel_layers_axis_map`

    Args:
        state_dict (StateDict): state_dict to convert
        state_dict_prefix (str): prefix appended to keys in final state dict
        sharded_key_prefix (str): prefix appended to ShardedTensor keys
        tensor_parallel_layers_axis_map (Dict[str, int]): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]]): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.
            The mapping for ShardedObjects is based on the mapping
             of the corresponding ShardedTensor.

    """
    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{state_dict_prefix}{layer_name}'
        sharded_key = f'{sharded_key_prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            # defer creating extra_state objects until all regular tensors are converted
            continue

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor, sharded_key, tp_axis, prepend_offsets=sharded_offsets,
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor, sharded_key, prepend_offsets=sharded_offsets,
            )

    # Extra states
    if sharded_offsets:
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)
        assert list(axis) == list(
            range(len(axis))
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)
        extra_state_offset = (0,)

    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{state_dict_prefix}{layer_name}'
        sharded_key = f'{sharded_key_prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            # Get replica_id from the base tensor. Extra state adds the TP replication
            base_layer_name = f'{layer_key[:-len(extra_state_suffix)]}weight'
            base_sharded_tensor = sharded_state_dict[base_layer_name]
            assert isinstance(
                base_sharded_tensor, ShardedTensor
            ), f'Expected already converted tensor for {base_layer_name}, got: {type(base_sharded_tensor)}'
            replica_id = base_sharded_tensor.replica_id
            assert (
                len(replica_id) == 3
            ), f'Expected replica_id for {base_layer_name} to be in (PP, TP, DP) format, got: {replica_id}'
            replica_id = (
                replica_id[0],
                parallel_state.get_tensor_model_parallel_rank(),
                replica_id[2],
            )

            sharded_state_dict[layer_key] = ShardedObject(
                sharded_key, tensor, extra_state_shape, extra_state_offset, replica_id,
            )

    return sharded_state_dict
