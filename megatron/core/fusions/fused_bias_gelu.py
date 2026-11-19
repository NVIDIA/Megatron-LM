# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.jit import jit_fuser
from megatron.core.ops.linear.bias_grad import bias_with_main_grad

# BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@jit_fuser
def bias_gelu(bias, y):
    """Add bias and apply the tanh approximation of GeLU."""
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def bias_gelu_back(g, bias, y):
    """Return the per-position gradient of the biased GeLU input."""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return ff * g


class GeLUFunction(torch.autograd.Function):
    """Autograd implementation of fused bias addition and GeLU."""

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        """Save the inputs needed for the activation derivative."""
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        """Return input and bias gradients before broadcast reduction."""
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

    # This is required to make Sphinx happy :-(
    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the custom GeLU autograd function."""
        return super().apply(*args, **kwargs)


def bias_gelu_impl(
    input: torch.Tensor, bias: torch.Tensor, *, gradient_accumulation_fusion: bool = False
) -> torch.Tensor:
    """Apply fused bias and GeLU with optional direct FP32 bias accumulation."""
    if gradient_accumulation_fusion:
        bias = bias_with_main_grad(bias, input.shape)
    return GeLUFunction.apply(input, bias)
