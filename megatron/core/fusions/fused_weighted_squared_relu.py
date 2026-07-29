# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

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


@jit_fuser
def _tanh_relu_over_scale(x: torch.Tensor, clamp_scale: float) -> torch.Tensor:
    """``tanh(ReLU(x) / clamp_scale)`` in fp32, the shared term of the clamped forward/backward.

    The ReLU is applied before the tanh rather than after the soft clamp. The two orderings agree
    exactly: ReLU commutes with any non-decreasing ``g`` satisfying ``g(0) == 0``, and
    ``s * tanh(x / s)`` is such a ``g``, so ``ReLU(s * tanh(x / s)) == s * tanh(ReLU(x) / s)``.
    The derivatives agree too, including at the kink, because the factor the chain rule pairs the
    ReLU subgradient with is ``sech^2(0) == 1`` under either ordering.
    """
    return torch.tanh(F.relu(x.float()) / clamp_scale)


@jit_fuser
def weighted_clamped_squared_relu(
    x: torch.Tensor, weights: torch.Tensor, clamp_scale: float
) -> torch.Tensor:
    """Element-wise weight applied after tanh soft-clamped Squared-ReLU.

    Computes ``squared_relu(s * tanh(x / s)) * weights``, bounding the activation output by
    ``s**2``. This is a faithful fusion of the unfused composition: the clamped pre-activation is
    rounded back to the input dtype exactly as ``tanh_soft_clamp`` does, and is then squared in
    that dtype exactly as ``squared_relu`` does, so the forward is bit-identical to the unfused
    path and toggling ``use_fused_weighted_squared_relu`` does not change results.

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor that will be broadcast-multiplied with the
            activation result.
        clamp_scale (float): The soft-clamp scale ``s``.

    Returns:
        torch.Tensor: The weighted activation with original ``dtype`` preserved.
    """
    out_dtype = x.dtype
    t = _tanh_relu_over_scale(x, clamp_scale)

    c = (clamp_scale * t).to(out_dtype)
    res = torch.pow(c, 2) * weights
    return res.to(out_dtype)


@jit_fuser
def weighted_clamped_squared_relu_back(
    g: torch.Tensor, x: torch.Tensor, weights: torch.Tensor, clamp_scale: float
):
    """Backward for weighted tanh soft-clamped Squared-ReLU.

    Returns gradients w.r.t ``x`` and ``weights``.
    """
    input_dtype = x.dtype
    w_dtype = weights.dtype

    t = _tanh_relu_over_scale(x, clamp_scale)
    c = (clamp_scale * t).to(input_dtype)
    act = torch.pow(c, 2)

    input_grad = (1 - torch.pow(t, 2)) * (2 * c) * g * weights

    weights_grad = act.float() * g.float()

    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)

    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedSquaredReLUFunction(torch.autograd.Function):
    """Autograd wrapper around the weighted Squared-ReLU fused kernels."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, clamp_scale: Optional[float]):
        """forward method for `WeightedSquaredReLUFunction`

        Args:
            ctx : context object to store intermediate tensors.
            input (torch.Tensor): input tensor.
            weights (torch.Tensor): weight tensor.
            clamp_scale (Optional[float]): if set, soft-clamp the input with
                ``clamp_scale * tanh(input / clamp_scale)`` before the activation.
        """
        ctx.save_for_backward(input, weights)
        ctx.clamp_scale = clamp_scale
        if clamp_scale is None:
            return weighted_squared_relu(input, weights)
        return weighted_clamped_squared_relu(input, weights, clamp_scale)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output: torch.Tensor):
        """backward method for `WeightedSquaredReLUFunction`

        Args:
            ctx : context object to store intermediate tensors.
            grad_output (torch.Tensor): gradient of the output of the forward function.
        """
        input, weights = ctx.saved_tensors
        if ctx.clamp_scale is None:
            inp_grad, w_grad = weighted_squared_relu_back(grad_output, input, weights)
        else:
            inp_grad, w_grad = weighted_clamped_squared_relu_back(
                grad_output, input, weights, ctx.clamp_scale
            )
        return inp_grad, w_grad, None


def weighted_squared_relu_impl(
    input: torch.Tensor, weights: torch.Tensor, clamp_scale: Optional[float] = None
) -> torch.Tensor:
    """Token-wise weighted Squared-ReLU fusion with optional FP8 storage.

    Args:
        input (torch.Tensor): Input tensor of shape ``(B, *, hidden_size)`` where ``*`` can be
            the sequence dimension.
        weights (torch.Tensor): Per-token weights broadcastable to the output of
            ``squared_relu``.
        clamp_scale (Optional[float]): if set, precondition the input with the tanh soft-clamp.

    Returns:
        torch.Tensor: Output tensor with the same shape as ``input`` except that the hidden
            dimension remains unchanged.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])

    output = WeightedSquaredReLUFunction.apply(input, weights, clamp_scale)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
