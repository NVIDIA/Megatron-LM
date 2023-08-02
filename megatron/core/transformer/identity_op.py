# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch


class IdentityOp(torch.nn.Module):
    """
    This is a placeholder for IdentityOp (NoOp)
    """

    def __init__(self, *args, **kwargs):
        super(IdentityOp, self).__init__()

    def forward(self, x, *args, **kwargs):
        if isinstance(x, (tuple, list)):
            return x[0]
        else:
            return x
