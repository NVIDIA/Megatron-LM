# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init

try:
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction
    HAVE_FUSED_RMS_NORM = True
except Exception:
    HAVE_FUSED_RMS_NORM = False


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5,
                 sequence_parallel=False,
                 zero_centered_gamma=False):
        super().__init__()

        self.zero_centered_gamma = zero_centered_gamma

        if not HAVE_FUSED_RMS_NORM:
            # TODO: Add pytorch only RMS norm
            raise ValueError(
                'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*hidden_size))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # Set sequence parallelism flag on weight parameter.
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):
        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input):
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        return FusedRMSNormAffineFunction.apply(
            input, weight, self.hidden_size, self.eps)
