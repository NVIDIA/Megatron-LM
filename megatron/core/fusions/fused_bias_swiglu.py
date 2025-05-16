# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@jit_fuser
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_swiglu_impl(input, bias, fp8_input_store=False):
    """
    Bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
