# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args
from megatron.model import LayerNorm, RMSNorm

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


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
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *

                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)


#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


def get_norm(config):
    args = get_args()
    if args.normalization == "LayerNorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)
    elif args.normalization == "RMSNorm":
        if args.apply_layernorm_1p:
            raise NotImplementedError('RMSNorm does not currently support the layernorm_1p formulation.')

        return RMSNorm(dim=config.hidden_size,
                       eps=config.layernorm_epsilon,
                       sequence_parallel=config.sequence_parallel)
    else:
        raise Exception(f"unsupported norm type '{args.normalization}'.")
