# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))

@torch.jit.script
def swiglu(y, y_2):
    return F.silu(y) * y_2

@torch.jit.script
def bias_swiglu(y, bias, y_2, bias_2):
    x = bias + y
    x_2 = bias_2 + y_2
    return swiglu(x, x_2)

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def swiglu_back(g, y, y_2):
    return g * torch.sigmoid(y) * (1 + y * (1 - torch.sigmoid(y))) * y_2, g * F.silu(y)

@torch.jit.script
def bias_swiglu_back(g, y, bias, y_2, bias_2):
    x_1 = bias + y
    x_2 = bias_2 + y_2
    return swiglu_back(g, x_1, x_2)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias, input_2, bias_2):
        ctx.save_for_backward(input, bias, input_2, bias_2)
        return bias_swiglu(input, bias, input_2, bias_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, input_2, bias_2 = ctx.saved_tensors
        tmp, tmp2 = bias_swiglu_back(grad_output, input, bias, input_2, bias_2)
        return tmp, tmp, tmp2, tmp2

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, input_2):
        ctx.save_for_backward(input, input_2)
        return swiglu(input, input_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, input_2 = ctx.saved_tensors
        tmp, tmp2 = swiglu_back(grad_output, input, input_2)
        return tmp, tmp2

bias_swiglu_impl = BiasSwiGLUFunction.apply
swiglu_impl = SwiGLUFunction.apply
