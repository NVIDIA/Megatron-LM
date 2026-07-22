# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import TypeVar

import torch

T = TypeVar('T')


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args: object, **kwargs: object):
        super().__init__()

    def forward(self, x: T, *args: object, **kwargs: object) -> T:
        """Forward pass.

        Returns x unchanged.
        """
        return x


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    def __init__(self, *args: object, **kwargs: object):
        super().__init__()

    def forward(self, *args: object, **kwargs: object):
        """Forward pass.

        Returns a function which returns its first argument unchanged, and discards all others.
        """
        return super().forward
