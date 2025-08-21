# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

######################  WEIGHTED SQUARED ReLU FUSION  ######################


@jit_fuser
def weighted_squared_relu(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Element-wise weight applied after Squared-ReLU.

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor that will be broadcast-multiplied with the
            activation result. Typically of shape ``(B, 1)`` so it can be broadcast across
            the hidden dimension.

    Returns:
        torch.Tensor: ``squared_relu(x) * weights`` with original ``dtype`` preserved.
    """
    out_dtype = x.dtype
    res = torch.pow(F.relu(x), 2) * weights
    return res.to(out_dtype)


@jit_fuser
def _squared_relu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of Squared-ReLU.

    The derivative of ``(ReLU(x))^2`` w.r.t ``x`` is ``2 * ReLU(x)``.
    """
    return g * 2 * F.relu(x)


@jit_fuser
def weighted_squared_relu_back(g: torch.Tensor, x: torch.Tensor, weights: torch.Tensor):
    """Backward for weighted Squared-ReLU.

    Returns gradients w.r.t ``x`` and ``weights``.
    """
    input_dtype = x.dtype
    w_dtype = weights.dtype

    # Gradient w.r.t. the input.
    input_grad = _squared_relu_back(g * weights, x)

    # Gradient w.r.t. the weights.
    weights_grad = squared_relu(x) * g.to(w_dtype)
    # Sum across the hidden dimension so each token has a single scalar weight.
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)

    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedSquaredReLUFunction(torch.autograd.Function):
    """Autograd wrapper around the weighted Squared-ReLU fused kernels."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor):
        """forward method for `WeightedSquaredReLUFunction`

        Args:
            ctx : context object to store intermediate tensors.
            input (torch.Tensor): input tensor.
            weights (torch.Tensor): weight tensor.
            fp8_input_store (bool): a bool flag to indicate if storing input in fp8.
        """
        ctx.save_for_backward(input, weights)
        return weighted_squared_relu(input, weights)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output: torch.Tensor):
        """backward method for `WeightedSquaredReLUFunction`

        Args:
            ctx : context object to store intermediate tensors.
            grad_output (torch.Tensor): gradient of the output of the forward function.
        """
        input, weights = ctx.saved_tensors
        inp_grad, w_grad = weighted_squared_relu_back(grad_output, input, weights)
        return inp_grad, w_grad


def weighted_squared_relu_impl(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Token-wise weighted Squared-ReLU fusion with optional FP8 storage.

    Args:
        input (torch.Tensor): Input tensor of shape ``(B, *, hidden_size)`` where ``*`` can be
            the sequence dimension.
        weights (torch.Tensor): Per-token weights broadcastable to the output of
            ``squared_relu``.

    Returns:
        torch.Tensor: Output tensor with the same shape as ``input`` except that the hidden
            dimension remains unchanged.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])

    output = WeightedSquaredReLUFunction.apply(input, weights)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
