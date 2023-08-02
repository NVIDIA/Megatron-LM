# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""

import torch

from megatron import get_args

from deepspeed.runtime.zero import GatheredParameters

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        with GatheredParameters(layer.weight, modifier_rank=0, enable=gather_params_on_init):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(layer.weight, modifier_rank=0, enable=gather_params_on_init):
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
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
