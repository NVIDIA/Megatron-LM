# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib
import inspect
import numbers

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    HAVE_FUSED_LAYER_NORM = True
except:
    HAVE_FUSED_LAYER_NORM = False

from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction, FusedRMSNormFunction


class FusedLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5,
                 elementwise_affine=True,
                 sequence_parallel=False,
                 **kwargs):
        super(FusedLayerNorm, self).__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*hidden_size))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        if self.elementwise_affine:
            return FusedRMSNormAffineFunction.apply(input, self.weight, self.hidden_size, self.eps)
        else:
            return FusedRMSNormFunction.apply(input, self.hidden_size, self.eps)