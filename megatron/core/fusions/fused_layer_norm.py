# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib
import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

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


class FusedLayerNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        persist_layer_norm=True,
        sequence_parallel=False,
        zero_centered_gamma=False,
    ):
        super().__init__()

        self.zero_centered_gamma = zero_centered_gamma

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must currently be installed to use megatron core.')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*hidden_size))
        self.bias = Parameter(torch.Tensor(*hidden_size))
        self.reset_parameters()
        self.persist_layer_norm = persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):

        if self.zero_centered_gamma:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if self.persist_layer_norm:
            output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

        else:
            output = FusedLayerNormAffineFunction.apply(
                input, weight, self.bias, self.hidden_size, self.eps
            )

        return output
