# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@torch.jit.script
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@torch.jit.script
def bias_swiglu(y, bias):
    y = y + bias
    return swiglu(y)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@torch.jit.script
def bias_swiglu_back(g, y, bias):
    y = y + bias
    return swiglu_back(g, y)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_swiglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return swiglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = swiglu_back(grad_output, input[0])
        return tmp


bias_swiglu_impl = BiasSwiGLUFunction.apply
swiglu_impl = SwiGLUFunction.apply
