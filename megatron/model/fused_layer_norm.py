# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False

from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction, FusedRMSNormFunction


global fused_layer_norm_cuda
fused_layer_norm_cuda = None


class MixedFusedNorm:
    """
    A conditional wrapper to initialize an instance of
    `MixedFusedLayerNorm` or `MixedFusedRMSNorm` based on input
    """
    def __new__(
        cls,
        normalization: str,  # LayerNorm or RMSNorm
        hidden_size: int,
        eps: float = 1e-5,
        **kwargs
    ):
        if normalization == "LayerNorm":
            instance = MixedFusedLayerNorm(
                normalized_shape=hidden_size,
                eps=eps,
                **kwargs)
        elif normalization == "RMSNorm":
            instance = MixedFusedRMSNorm(
                normalized_shape=hidden_size,
                eps=eps,
                **kwargs)
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance


class MixedFusedLayerNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5,
               no_persist_layer_norm=True,
               sequence_parallel=False,
               apply_layernorm_1p=False):
        super(MixedFusedLayerNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096,
            5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
            24576, 25600, 30720, 32768, 40960, 49152, 65536]
        if normalized_shape not in persist_ln_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


  def reset_parameters(self):

    if self.apply_layernorm_1p:
        init.zeros_(self.weight)
        init.zeros_(self.bias)
    else:
        init.ones_(self.weight)
        init.zeros_(self.bias)

  def forward(self, input):

    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

    if self.no_persist_layer_norm:
        return FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.normalized_shape, self.eps)
    else:
        output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

        # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
        # a populated '_base' field). This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        output = make_viewless_tensor(inp = output,
                                      requires_grad = input.requires_grad,
                                      keep_graph = True)

        return output


class MixedFusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """
    def __init__(self, normalized_shape, eps=1e-5,
                 elementwise_affine=True,
                 sequence_parallel=False,
                 **kwargs):
        super(MixedFusedRMSNorm, self).__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
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
            return FusedRMSNormAffineFunction.apply(input, self.weight, self.normalized_shape, self.eps)
        else:
            return FusedRMSNormFunction.apply(input, self.normalized_shape, self.eps)
