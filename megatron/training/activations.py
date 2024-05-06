# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn.functional as F

try:
    jit_fuser = torch.compile
except:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)
