# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn

class RMSNorm(torch.nn.Module):

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False,
                 config: dict = None):
        """RMS Normaliation module

        Args:
            hidden_size (int): The width of input
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
