# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


@jit_fuser
def situlu(x: torch.Tensor, beta1: float = 4.0, beta2: float = 25.0) -> torch.Tensor:
    """Apply SiTU-GLU to contiguous gate/up halves of an FC1 output.

    This is the slow PyTorch reference and config marker until PyTorch provides
    a dedicated ``torch.nn.functional.situlu``-style operation. Unary
    ``F.silu`` is not equivalent because SiTU-GLU transforms both branches.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = beta1 * torch.tanh(gate / beta1) * torch.sigmoid(gate)
    up = beta2 * torch.tanh(up / beta2)
    return gate * up
