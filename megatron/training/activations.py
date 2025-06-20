# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)

@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
