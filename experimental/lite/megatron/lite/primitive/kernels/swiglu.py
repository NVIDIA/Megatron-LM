# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SwiGLU fused autograd helpers for MLite."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.lite.primitive.kernels.jit import jit_fuser


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
    result = swiglu(y) * weights
    return result.to(dtype)


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
    weight_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    weights_grad = swiglu(y) * g.to(weight_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(weight_dtype)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, fp8_input_store=False, cpu_offload_input=False):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            if bias is not None:
                bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        grad = bias_swiglu_back(grad_output, input, bias)
        return grad, grad, None, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fp8_input_store=False, cpu_offload_input=False):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        grad = swiglu_back(grad_output, input)
        return grad, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, fp8_input_store=False):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, weights_grad = weighted_swiglu_back(grad_output, input, weights)
        return input_grad, weights_grad, None


def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    original_shape = input.shape
    assert len(original_shape) in [2, 3]
    input = input.view(-1, original_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input)
    return (
        output
        if len(original_shape) == 2
        else output.view(original_shape[0], original_shape[1], -1)
    )


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
    original_shape = input.shape
    assert len(original_shape) in [2, 3]
    input = input.view(-1, original_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)
    return (
        output
        if len(original_shape) == 2
        else output.view(original_shape[0], original_shape[1], -1)
    )


__all__ = ["bias_swiglu_impl", "swiglu", "weighted_bias_swiglu_impl"]
