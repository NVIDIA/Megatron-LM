# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from torch import nn

from megatron.core.transformer.torch_norm import WrappedTorchNorm


class ModelNorm(WrappedTorchNorm):
    def __new__(cls, config, hidden_size, eps, **kwargs):
        return (
            super().__new__(cls, config, hidden_size, eps, **kwargs).to(dtype=config.params_dtype)
        )


class FP32Projection(nn.Module):
    """FP32 input projections accumulate from FP32 input/weights."""

    def __init__(self, input_size, output_size, *, config, init_method, **unused):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                output_size, input_size, dtype=torch.float32, device=torch.cuda.current_device()
            )
        )
        if config.perform_initialization:
            init_method(self.weight)

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight), None

    def backward_dw(self):
        pass  # Normal torch autograd; delayed weight gradients are rejected.
