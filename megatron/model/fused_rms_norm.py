# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers

from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


class MixedFusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5,
                 sequence_parallel=False,
                 apply_layernorm_1p=False):
        super(MixedFusedRMSNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # Set sequence parallelism flag on weight parameter.
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):
        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input):
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight
        return FusedRMSNormAffineFunction.apply(
            input, weight, self.normalized_shape, self.eps)
