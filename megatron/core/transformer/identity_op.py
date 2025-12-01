# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Any

import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any):
        return super().forward
