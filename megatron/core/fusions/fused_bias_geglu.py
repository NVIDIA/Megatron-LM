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
    """Performs GEGLU (GELU-Gated Linear Unit) activation.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of GEGLU activation: GELU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))) * y_2


@jit_fuser
def bias_geglu(bias, y):
    """Performs GEGLU activation with bias addition.

    Args:
        bias (torch.Tensor): Bias tensor to be added to the input.
        y (torch.Tensor): Input tensor to be split and gated.

    Returns:
        torch.Tensor: Result of bias addition followed by GEGLU activation.
    """
    y = y + bias
    return geglu(y)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def geglu_back(g, y):
    """Computes the gradient for the GEGLU activation.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    tanh_out = torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * y_1 * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * y_1 * y_1)) + 0.5 * (
        1 + tanh_out
    )
    return torch.cat(((g * y_2) * ff, g * (y_1 * 0.5 * (1.0 + tanh_out))), -1)


@jit_fuser
def bias_geglu_back(g, y, bias):
    """Computes the gradient for the biased GEGLU activation.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor after bias addition.
    """
    y = y + bias
    return geglu_back(g, y)


class BiasGeGLUFunction(torch.autograd.Function):
    """Custom autograd function for GEGLU activation with bias support."""

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        """Forward pass of biased GEGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply GEGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before GEGLU.

        Returns:
            torch.Tensor: Result of applying bias addition followed by GEGLU activation.
        """
        ctx.save_for_backward(input, bias)
        return bias_geglu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of biased GEGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing gradients with respect to the input and bias tensors.
        """
        input, bias = ctx.saved_tensors
        tmp = bias_geglu_back(grad_output, input, bias)
        return tmp, tmp


class GeGLUFunction(torch.autograd.Function):
    """Custom autograd function for GEGLU activation without bias."""

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        """Forward pass of GEGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply GEGLU to.

        Returns:
            torch.Tensor: Result of applying GEGLU activation.
        """
        ctx.save_for_backward(input)
        return geglu(input)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of GEGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            torch.Tensor: Gradient with respect to the input tensor.
        """
        input = ctx.saved_tensors
        tmp = geglu_back(grad_output, input[0])
        return tmp


def bias_geglu_impl(input, bias):
    """Implementation of biased GEGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the GEGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply GEGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free GEGLU variant.

    Returns:
        torch.Tensor: Result of biased GEGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
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
def weighted_quick_geglu(
    y: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
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
    """Backward helper for Quick-GEGLU.

    Args:
        g (torch.Tensor): Upstream gradient tensor.
        y (torch.Tensor): Input tensor used in the forward pass.
        linear_offset (float, optional): Linear offset used in the forward pass. Defaults to 0.0.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
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


# ---------------- Weighted Bias Quick-GEGLU helpers -----------------


@jit_fuser
def weighted_bias_quick_geglu(
    y: torch.Tensor, bias: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
    """Token-wise weighted Quick-GEGLU activation with bias.

    Args:
        y: Input tensor before bias addition.
        bias: Bias tensor broadcastable to `y`.
        weights: Weight tensor with shape `[tokens, 1]` broadcasting over feature dim.
        linear_offset: Optional linear offset for the second half before gating.

    Returns:
        Activated tensor with same dtype as `y`.
    """
    dtype = y.dtype
    res = quick_geglu(y + bias, linear_offset) * weights
    return res.to(dtype)


@jit_fuser
def weighted_bias_quick_geglu_back(g, y, bias, weights, linear_offset: float = 0.0):
    """Backward helper for weighted Quick-GEGLU with bias.

    Returns gradients w.r.t input `y`, `bias`, and `weights`.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype

    # Forward input with bias
    x = y + bias

    # Gradient w.r.t input (and thus bias) via chain rule
    input_grad = quick_geglu_back(g * weights, x, linear_offset)

    # Gradient w.r.t weights
    weights_grad = quick_geglu(x, linear_offset) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)

    # bias gradient identical to input gradient
    bias_grad = input_grad

    return input_grad.to(input_dtype), bias_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedQuickGeGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted Quick-GEGLU (no bias)."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
        linear_offset: torch.Tensor,
    ):
        """Forward pass of weighted Quick-GEGLU.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor of shape [N, 2H].
            weights (torch.Tensor): Per-token weights of shape [N, 1].
            fp8_input_store (bool): If True, stores input for backward in FP8.
            linear_offset (torch.Tensor): Scalar tensor offset added to the linear half.

        Returns:
            torch.Tensor: Output tensor of shape [N, H] after weighted Quick-GEGLU.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights, linear_offset)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_quick_geglu(input, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of weighted Quick-GEGLU.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Upstream gradient w.r.t. the output.

        Returns:
            tuple: Gradients with respect to (input, weights, fp8_input_store, linear_offset).
                The latter two gradients are None.
        """
        input, weights, linear_offset = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, wgrad = weighted_quick_geglu_back(grad_output, input, weights, linear_offset)
        return input_grad, wgrad, None, None


class WeightedBiasQuickGeGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted Quick-GEGLU with bias support."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bias: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
        linear_offset: torch.Tensor,
    ):
        """Forward pass of weighted Quick-GEGLU.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor of shape [N, 2H].
            bias (torch.Tensor): Bias tensor of shape [N, 1].
            weights (torch.Tensor): Per-token weights of shape [N, 1].
            fp8_input_store (bool): If True, stores input for backward in FP8.
            linear_offset (torch.Tensor): Scalar tensor offset added to the linear half.

        Returns:
            torch.Tensor: Output tensor of shape [N, H] after weighted Quick-GEGLU with bias.
        """
        # Optionally store the input in FP8 for memory savings.
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input

        # Save tensors for backward.
        ctx.save_for_backward(input_for_backward, bias, weights, linear_offset)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store

        # Compute activation using fused helper that includes bias and weighting.
        return weighted_bias_quick_geglu(input, bias, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of weighted Quick-GEGLU with bias.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Upstream gradient w.r.t. the output.

        Returns:
            tuple: Gradients with respect to (input, bias, weights, fp8_input_store, linear_offset).
                The latter two gradients are None.
        """
        input, bias, weights, linear_offset = ctx.saved_tensors

        # Restore original input dtype if it was stored in FP8.
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input

        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            grad_output, input, bias, weights, linear_offset
        )

        return input_grad, bias_grad, weights_grad, None, None


def weighted_bias_quick_geglu_impl(
    input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None
):
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
    if clamp_value is not None:
        x_glu, x_linear = input.chunk(2, -1)
        input = torch.cat(
            (
                x_glu.clamp(min=None, max=clamp_value),
                x_linear.clamp(min=-clamp_value, max=clamp_value),
            ),
            -1,
        )
    input = input.view(-1, ori_shape[-1])
    linear_offset = torch.tensor(linear_offset, dtype=input.dtype, device=input.device)
    if bias is not None:
        output = WeightedBiasQuickGeGLUFunction.apply(
            input, bias, weights, fp8_input_store, linear_offset
        )
    else:
        output = WeightedQuickGeGLUFunction.apply(input, weights, fp8_input_store, linear_offset)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
