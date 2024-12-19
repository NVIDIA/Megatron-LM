# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F

from ..core.jit import jit_fuser


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)
