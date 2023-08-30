# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""

import torch

from megatron import get_args
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, make_sharded_tensor_for_checkpoint


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


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


def make_sharded_tensors_for_checkpoint(state_dict, state_dict_prefix, sharded_key_prefix,
                                        tensor_parallel_layers_axis_map, sharded_offsets,
                                        replica_id=None):
    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{state_dict_prefix}{layer_name}'
        sharded_key = f'{sharded_key_prefix}{layer_name}'

        if layer_name.endswith('._extra_state'):
            assert len(sharded_offsets) == 1, 'TODO'
            _, pp_offset, pp_num_layers = sharded_offsets[0]
            if replica_id is None:
                replica_id = (0, parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_data_parallel_rank())

            sharded_state_dict[layer_key] = ShardedObject(
                sharded_key, tensor,
                (pp_num_layers,), (pp_offset,),
                replica_id,
            )

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor, sharded_key, tp_axis,
                prepend_offsets=sharded_offsets,
                replica_id=replica_id,
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor, sharded_key,
                prepend_offsets=sharded_offsets,
                replica_id=replica_id,
            )
    return sharded_state_dict
