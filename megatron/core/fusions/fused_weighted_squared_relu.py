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
    """``tanh(ReLU(x) / clamp_scale)`` in fp32, the shared term of the clamped forward/backward."""
    return torch.tanh(F.relu(x.float()) / clamp_scale)


@jit_fuser
def weighted_clamped_squared_relu(
    x: torch.Tensor, weights: torch.Tensor, clamp_scale: float
) -> torch.Tensor:
    """Element-wise weight applied after tanh soft-clamped Squared-ReLU.

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weight tensor that will be broadcast-multiplied with the
            activation result.
        clamp_scale (float): The soft-clamp scale ``s``.

    Returns:
        torch.Tensor: The weighted activation with original ``dtype`` preserved.
    """
    out_dtype = x.dtype
    c = clamp_scale * _tanh_relu_over_scale(x, clamp_scale)
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
    c = clamp_scale * t
    act = torch.pow(c, 2)

    input_grad = (1 - torch.pow(t, 2)) * (2 * c) * g * weights

    weights_grad = act * g.float()

    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)

    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


@jit_fuser
def clamped_squared_relu(x: torch.Tensor, clamp_scale: float) -> torch.Tensor:
    """Tanh-soft-clamped squared-ReLU without token weights.

    This matches ``squared_relu(tanh_soft_clamp(x, clamp_scale))``. The clamped
    pre-activation and the square stay in FP32 and only the result is rounded back to
    the input dtype.
    """
    out_dtype = x.dtype
    clamped = clamp_scale * _tanh_relu_over_scale(x, clamp_scale)
    return torch.pow(clamped, 2).to(out_dtype)


@jit_fuser
def clamped_squared_relu_back(g: torch.Tensor, x: torch.Tensor, clamp_scale: float):
    """Backward for tanh-soft-clamped squared-ReLU, recomputed from the raw input."""
    tanh_value = _tanh_relu_over_scale(x, clamp_scale)
    clamped = clamp_scale * tanh_value
    input_grad = (1 - torch.pow(tanh_value, 2)) * (2 * clamped) * g
    return input_grad.to(x.dtype)


class WeightedSquaredReLUFunction(torch.autograd.Function):
    """Autograd wrapper around the (optionally weighted, optionally clamped) Squared-ReLU
    fused kernels.

    Only the raw input (and the weights, when given) is saved for backward. The unfused
    ``squared_relu(tanh_soft_clamp(x))`` ordering saves both ``x`` and the clamped
    intermediate, doubling the activation memory kept alive for this op.
    """

    @staticmethod
    @nvtx_decorator()
    def forward(
        ctx, input: torch.Tensor, weights: Optional[torch.Tensor], clamp_scale: Optional[float]
    ):
        """forward method for `WeightedSquaredReLUFunction`

        Args:
            ctx : context object to store intermediate tensors.
            input (torch.Tensor): input tensor.
            weights (Optional[torch.Tensor]): optional per-token weight tensor.
            clamp_scale (Optional[float]): if set, soft-clamp the input with
                ``clamp_scale * tanh(input / clamp_scale)`` before the activation.
        """
        assert weights is not None or clamp_scale is not None, (
            "weighted_squared_relu_impl needs weights and/or clamp_scale; "
            "use squared_relu directly otherwise."
        )
        ctx.clamp_scale = clamp_scale
        ctx.has_weights = weights is not None
        if weights is None:
            ctx.save_for_backward(input)
            return clamped_squared_relu(input, clamp_scale)

        ctx.save_for_backward(input, weights)
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
        if not ctx.has_weights:
            (input,) = ctx.saved_tensors
            return clamped_squared_relu_back(grad_output, input, ctx.clamp_scale), None, None

        input, weights = ctx.saved_tensors
        if ctx.clamp_scale is None:
            inp_grad, w_grad = weighted_squared_relu_back(grad_output, input, weights)
        else:
            inp_grad, w_grad = weighted_clamped_squared_relu_back(
                grad_output, input, weights, ctx.clamp_scale
            )
        return inp_grad, w_grad, None


def weighted_squared_relu_impl(
    input: torch.Tensor, weights: Optional[torch.Tensor] = None, clamp_scale: Optional[float] = None
) -> torch.Tensor:
    """Squared-ReLU fusion with optional per-token weights and optional tanh soft clamping.

    Args:
        input (torch.Tensor): Input tensor of shape ``(B, *, hidden_size)`` where ``*`` can be
            the sequence dimension.
        weights (Optional[torch.Tensor]): Optional per-token weights broadcastable to the
            output of ``squared_relu`` once ``input`` is flattened to ``(-1, hidden_size)``.
            When ``None``, the clamped squared-ReLU is applied and only ``input`` is saved for
            backward.
        clamp_scale (Optional[float]): if set, precondition the input with the tanh soft-clamp
            ``clamp_scale * tanh(input / clamp_scale)``. At least one of ``weights`` and
            ``clamp_scale`` must be given.

    Returns:
        torch.Tensor: Output tensor with the same shape as ``input`` except that the hidden
            dimension remains unchanged.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])

    output = WeightedSquaredReLUFunction.apply(input, weights, clamp_scale)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
