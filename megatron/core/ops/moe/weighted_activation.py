# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bound mixed-precision router weighting in the forward and backward passes."""

from collections.abc import Callable

import torch
from torch.autograd.function import once_differentiable

_MAX_TEMP_ELEMENTS = 16 * 1024 * 1024


class _ChunkedRouterWeight(torch.autograd.Function):
    """Multiply router probabilities without retaining a full promoted product."""

    @staticmethod
    def forward(ctx, activations, probs, chunk_rows):
        """Apply the original promoted product to bounded row chunks."""
        ctx.save_for_backward(activations if ctx.needs_input_grad[1] else None, probs)
        ctx.activation_dtype = activations.dtype
        ctx.chunk_rows = chunk_rows
        ctx.product_dtype = torch.promote_types(activations.dtype, probs.dtype)
        output = torch.empty_like(activations)
        for start in range(0, activations.shape[0], chunk_rows):
            end = start + chunk_rows
            output[start:end].copy_(
                (activations[start:end] * probs[start:end]).to(activations.dtype)
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Return activation and router gradients without a full promoted product."""
        activations, probs = ctx.saved_tensors
        grad_activations = torch.empty_like(grad_output) if ctx.needs_input_grad[0] else None
        grad_probs = torch.empty_like(probs) if ctx.needs_input_grad[1] else None
        for start in range(0, grad_output.shape[0], ctx.chunk_rows):
            end = start + ctx.chunk_rows
            # Match the gradient of the cast after the original promoted product.
            grad = grad_output[start:end].to(ctx.product_dtype)
            if grad_activations is not None:
                grad_activations[start:end].copy_(
                    (grad * probs[start:end]).to(ctx.activation_dtype)
                )
            if grad_probs is not None:
                grad_probs[start:end].copy_(
                    (grad * activations[start:end]).sum(dim=-1, keepdim=True).to(probs.dtype)
                )
        return grad_activations, grad_probs, None


class _ChunkedWeightedActivation(torch.autograd.Function):
    """Recompute a pointwise activation one row chunk at a time during backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, activation_func, inputs, probs, chunk_rows):
        """Save inputs instead of retaining full eager activation intermediates."""
        ctx.save_for_backward(inputs, probs)
        ctx.activation_func = activation_func
        ctx.chunk_rows = chunk_rows
        output = None
        for start in range(0, inputs.shape[0], chunk_rows):
            end = start + chunk_rows
            activation = activation_func(inputs[start:end])
            if output is None:
                output = activation.new_empty((inputs.shape[0], activation.shape[1]))
            output[start:end].copy_((activation * probs[start:end]).to(activation.dtype))
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):
        """Recompute each activation chunk with its forward autocast context."""
        inputs, probs = ctx.saved_tensors
        needs_input, needs_probs = ctx.needs_input_grad[1:3]
        grad_inputs = torch.empty_like(inputs) if needs_input else None
        grad_probs = torch.empty_like(probs) if needs_probs else None
        for start in range(0, inputs.shape[0], ctx.chunk_rows):
            end = start + ctx.chunk_rows
            with torch.enable_grad():
                local_input = inputs[start:end].detach().requires_grad_(needs_input)
                local_probs = probs[start:end].detach().requires_grad_(needs_probs)
                activation = ctx.activation_func(local_input)
                weighted = (activation * local_probs).to(activation.dtype)
                variables = [value for value in (local_input, local_probs) if value.requires_grad]
                gradients = iter(torch.autograd.grad(weighted, variables, grad_output[start:end]))
            if grad_inputs is not None:
                grad_inputs[start:end].copy_(next(gradients))
            if grad_probs is not None:
                grad_probs[start:end].copy_(next(gradients))
        return None, grad_inputs, grad_probs, None


def apply_weighted_activation(
    activation_func: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Recompute large mixed-precision activations for first-order training.

    Save FC1 inputs and probabilities instead of full eager GLU intermediates.
    Use the original activation and autograd in each chunk to retain its dtype
    rounding, clamp derivatives, and router probability gradients.
    The callback must be deterministic, pointwise, and parameter-free.

    Args:
        activation_func: A deterministic, parameter-free, row-local activation.
        inputs: FC1 outputs with shape ``[tokens, features]``.
        probs: Router probabilities with shape ``[tokens, 1]``.

    Returns:
        Router-weighted activations in the activation's original dtype.

    Note:
        The chunked path supports first-order gradients only. Small inputs and
        inputs without dtype promotion keep the original autograd path.
        CUDA router gradients can differ by FP32 reduction roundoff.
    """
    if (
        inputs.numel() <= _MAX_TEMP_ELEMENTS
        or torch.promote_types(inputs.dtype, probs.dtype) == inputs.dtype
        or inputs.ndim != 2
        or probs.shape != (inputs.shape[0], 1)
    ):
        activation = activation_func(inputs)
        return apply_router_probs(activation, probs)
    chunk_rows = max(1, _MAX_TEMP_ELEMENTS // inputs.shape[1])
    return _ChunkedWeightedActivation.apply(activation_func, inputs, probs, chunk_rows)


def apply_router_probs(activations: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Keep FP32 router weights without a full-size promoted activation tensor.

    Large dropless expert batches can receive far more tokens than a balanced
    rank. Target chunks of at most 16M elements, or 64 MiB for FP32, with a
    minimum of one row per chunk. A wider row exceeds that target.
    The output, saved activations, and returned gradients retain their original
    precision. Small batches and products without dtype promotion use autograd.

    Args:
        activations: Expert activations with shape ``[tokens, features]``.
        probs: Router probabilities broadcastable to the activations.

    Returns:
        The weighted activation in its original dtype.

    Note:
        The chunked path supports first-order gradients only. Other layouts keep
        native broadcasting and autograd.
        CUDA router gradients can differ by FP32 reduction roundoff.
    """
    if (
        activations.numel() <= _MAX_TEMP_ELEMENTS
        or torch.promote_types(activations.dtype, probs.dtype) == activations.dtype
        or activations.ndim != 2
        or probs.shape != (activations.shape[0], 1)
    ):
        return (activations * probs).to(activations.dtype)
    chunk_rows = max(1, _MAX_TEMP_ELEMENTS // activations.shape[1])
    return _ChunkedRouterWeight.apply(activations, probs, chunk_rows)
