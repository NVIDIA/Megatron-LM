# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.jit import jit_fuser

###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


@jit_fuser
def geglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))) * y_2


@jit_fuser
def bias_geglu(bias, y):
    y = y + bias
    return geglu(y)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def geglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    tanh_out = torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (
        1 + tanh_out
    )
    return torch.cat(((g * y_2) * ff, g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)


@jit_fuser
def bias_geglu_back(g, y, bias):
    y = y + bias
    return geglu_back(g, y)


class BiasGeGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_geglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_geglu_back(grad_output, input, bias)
        return tmp, tmp


class GeGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return geglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = geglu_back(grad_output, input[0])
        return tmp


def bias_geglu_impl(input, bias):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasGeGLUFunction.apply(input, bias)
    else:
        output = GeGLUFunction.apply(input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)



# ------------------------- QUICK GEGLU FUSION --------------------------


@jit_fuser
def quick_gelu(y: torch.Tensor) -> torch.Tensor:
    """Sigmoid approximation of gelu"""
    return y * torch.sigmoid(1.702 * y)


@jit_fuser
def quick_geglu(y: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Performs Quick-GELU-based GEGLU activation : quick_gelu(y1) * (y2 + offset).

    Args:
        y: Input tensor split into two halves on the last dimension.
        linear_offset: Optional linear offset added to the second half before gating.

    Returns:
        Tensor after applying the GEGLU activation.
    """
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return quick_gelu(y_1) * (y_2 + linear_offset)


@jit_fuser
def weighted_quick_geglu(y: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Token-wise-weighted Quick-GEGLU activation.

    The weights tensor is expected to have the same first-dimension length as ``y`` and a trailing
    singleton dimension so that it broadcasts over the feature dimension.
    """
    dtype = y.dtype
    res = quick_geglu(y, linear_offset) * weights
    return res.to(dtype)


# gradient of sigmoid approximation of gelu
@jit_fuser
def quick_geglu_back(g, y, linear_offset: float = 0.0) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    sigmoid_out = torch.sigmoid(1.702 * y_1)
    dy_1 = g * sigmoid_out * (1 + 1.702 * y_1 * (1 - sigmoid_out)) * (y_2 + linear_offset)
    dy_2 = g * y_1 * sigmoid_out
    return torch.cat((dy_1, dy_2), -1) 

@jit_fuser
def weighted_quick_geglu_back(g, y, weights, linear_offset: float = 0.0):
    """Backward helper for weighted Quick-GEGLU.
    Returns gradient w.r.t input `y` and `weights`.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype
    # Gradient w.r.t input uses the chain rule with weighting.
    input_grad = quick_geglu_back(g * weights, y, linear_offset)
    # Gradient w.r.t weights is the activation times upstream grad (cast to weight dtype).
    weights_grad = quick_geglu(y, linear_offset) * g.to(w_dtype)
    # Sum across the feature dimension to keep weights shape `[tokens, 1]`.
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedQuickGeGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted Quick-GEGLU (no bias)."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, fp8_input_store: bool, linear_offset: torch.Tensor):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights, linear_offset)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_quick_geglu(input, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, linear_offset = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, wgrad = weighted_quick_geglu_back(grad_output, input, weights, linear_offset)
        return input_grad, wgrad, None, None


def weighted_bias_quick_geglu_impl(input, bias, weights, fp8_input_store=False, linear_offset=0.0):
    """
    Token-wise-weighted bias quick_geglu fusion.
        input: [num_selected_experts * seq_len, hidden_size * 2]
        bias: None
        weights: [num_selected_experts * seq_len, 1]
        fp8_input_store: bool
        linear_offset: float
        output: [num_selected_experts * seq_len, hidden_size]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    x_glu, x_linear = input.chunk(2, -1)
    input = torch.cat((x_glu.clamp(min=None, max=7.0), x_linear.clamp(min=-7.0, max=7.0)), -1)
    input = input.view(-1, ori_shape[-1])
    linear_offset = torch.tensor(linear_offset, dtype=input.dtype, device=input.device)
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedQuickGeGLUFunction.apply(input, weights, fp8_input_store, linear_offset)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
