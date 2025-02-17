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
except ImportError:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False

try: 
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction

    HAVE_FUSED_RMS_NORM = True
except:
    HAVE_FUSED_RMS_NORM = False


class FusedLayerNorm(torch.nn.Module):
    """Layer Norm, fused into a single CUDA kernel.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      persist_layer_norm (bool): Use persistent fused layer norm kernel.
      This kernel supports only a set of hidden sizes. Please
      check persist_ln_hidden_sizes if your hidden size is supported.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
    ):
        super().__init__()

        self.config = config

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma
        assert (
            self.config.normalization == "LayerNorm"
        ), f'({self.config.normalization}) is not supported in FusedLayerNorm'

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
        persist_layer_norm = self.config.persist_layer_norm
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must be installed to use FusedLayerNorm.')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps
        # Parameters need to be initialized with torch.empty rather than torch.Tensor for correct device placement with nemo2.
        self.weight = Parameter(torch.empty(*hidden_size))
        self.bias = Parameter(torch.empty(*hidden_size))
        self.reset_parameters()
        self.persist_layer_norm = persist_layer_norm
        self.sequence_parallel = self.config.sequence_parallel

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

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if self.persist_layer_norm:
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:
                output = FastLayerNormFN.apply(
                    input, weight, self.bias, self.eps, self.config.memory_efficient_layer_norm
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

        else:
            if (
                'memory_efficient'
                in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            ):
                return FusedLayerNormAffineFunction.apply(
                    input,
                    weight,
                    self.bias,
                    self.hidden_size,
                    self.eps,
                    self.config.memory_efficient_layer_norm,
                )
            else:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.hidden_size, self.eps
                )

        return output


class FusedRMSNorm(torch.nn.Module):
    """RMS Norm, fused into a single CUDA kernel. Note: so far there is no
    persistent kernel for RMSNorm in apex, so we use a non-persistent one.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'RMSNorm' here.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",  # included to match TE interface
    ):
        super().__init__()

        self.config = config

        self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma

        # If someone is trying to instantiate directly FusedRMSNorm but has
        # specified another normalization in the config, raise an error
        assert self.config.normalization == "RMSNorm", (
            f'({self.config.normalization}) was specified in the config, but '
            'FusedRMSNorm is trying to be instantiated here'
        )

        if not HAVE_FUSED_RMS_NORM:
            raise ValueError(f'Apex must be installed to use FusedRMSNorm.')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.eps = eps
        # Parameters need to be initialized with torch.empty rather than torch.Tensor for correct device placement with nemo2.
        self.weight = Parameter(torch.empty(*hidden_size))
        self.reset_parameters()
        self.sequence_parallel = self.config.sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):

        if self.zero_centered_gamma:
            init.zeros_(self.weight)
        else:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:

        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if (
            'memory_efficient'
            in inspect.getfullargspec(FusedRMSNormAffineFunction.forward).args
        ):
            return FusedRMSNormAffineFunction.apply(
                input,
                weight,
                self.hidden_size,
                self.eps,
                self.config.memory_efficient_layer_norm,
            )
        else:
            return FusedRMSNormAffineFunction.apply(
                input, weight, self.hidden_size, self.eps
            )


class FusedApexNorm:
    """
    A conditional wrapper to initialize an instance of Apex
    `LayerNorm` or `RMSNorm` based on input.
    """
    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            instance = FusedLayerNorm(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                persist_layer_norm=config.persist_layer_norm,
                zero_centered_gamma=config.layernorm_zero_centered_gamma
            )
        elif config.normalization == "RMSNorm":
            instance = FusedRMSNorm(
                config=config,
                hidden_size=hidden_size,
                eps=eps,
                zero_centered_gamma=config.layernorm_zero_centered_gamma
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
