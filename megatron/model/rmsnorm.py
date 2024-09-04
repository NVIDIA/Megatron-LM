# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

from deepspeed.accelerator import get_accelerator
from megatron import get_args

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

# Taken from facebookresearch/llama
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        init_device = None
        if get_accelerator().device_name() == 'hpu':
            init_device = get_accelerator().current_device_name() 
        self.weight = Parameter(torch.empty(dim,
                                device=init_device,
                                dtype=get_args().params_dtype))
        init.ones_(self.weight)
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
