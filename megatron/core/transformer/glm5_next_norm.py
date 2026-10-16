# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Normalization primitives with the released GLM5Next rounding contract."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor, nn


class _GLM5NextRMSNormFunction(torch.autograd.Function):
    """Released forward plus an allocation-bounded, operation-ordered VJP."""

    @staticmethod
    def forward(
        ctx, hidden_states: Tensor, weight: Tensor, normalized_shape: tuple[int, ...], eps: float
    ) -> Tensor:
        """Normalize in FP32, round to the input dtype, then apply gamma."""
        input_dtype = hidden_states.dtype
        normalized_dims = tuple(range(-len(normalized_shape), 0))
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(dim=normalized_dims, keepdim=True)
        inverse_rms = torch.rsqrt(variance + eps)
        normalized = (hidden_states_fp32 * inverse_rms).to(input_dtype)
        weight_view_shape = (1,) * (hidden_states.ndim - len(normalized_shape)) + normalized_shape

        # Do not retain the full FP32 normalized activation.  The VJP only
        # needs the BF16/FP16 input and gamma plus one FP32 inverse RMS value
        # per normalized row.
        ctx.normalized_shape = normalized_shape
        ctx.normalized_dims = normalized_dims
        ctx.weight_view_shape = weight_view_shape
        ctx.save_for_backward(hidden_states, weight, inverse_rms)
        return weight.view(weight_view_shape) * normalized

    @staticmethod
    def backward(ctx, output_grad: Tensor):
        """Return gradients that repeat the forward rounding boundary."""
        hidden_states, weight, inverse_rms = ctx.saved_tensors
        hidden_states_fp32 = hidden_states.to(torch.float32)
        normalized = (hidden_states_fp32 * inverse_rms).to(hidden_states.dtype)
        weight_view = weight.view(ctx.weight_view_shape)

        weight_grad = output_grad * normalized
        broadcast_dims = tuple(range(hidden_states.ndim - len(ctx.normalized_shape)))
        if broadcast_dims:
            weight_grad = weight_grad.sum(dim=broadcast_dims)
        weight_grad = weight_grad.to(weight.dtype)

        # The forward cast to input dtype is an observable rounding boundary.
        # Its VJP likewise rounds the affine gradient to that dtype before
        # continuing through the FP32 normalization graph.
        normalized_grad = (output_grad * weight_view).to(hidden_states.dtype)
        normalized_grad_fp32 = normalized_grad.to(torch.float32)
        direct_input_grad = normalized_grad_fp32 * inverse_rms
        inverse_rms_grad = (normalized_grad_fp32 * hidden_states_fp32).sum(
            dim=ctx.normalized_dims, keepdim=True
        )
        variance_grad = inverse_rms_grad * (-0.5) * inverse_rms.pow(3)
        square_grad = variance_grad / math.prod(ctx.normalized_shape)
        variance_input_grad = square_grad * (2.0 * hidden_states_fp32)
        input_grad = (direct_input_grad + variance_input_grad).to(hidden_states.dtype)
        return input_grad, weight_grad, None, None


class GLM5NextRMSNorm(nn.Module):
    """RMSNorm that preserves GLM5Next's observable BF16 affine boundary.

    The released implementation normalizes in FP32, casts the normalized
    activation back to the input dtype, and then applies the affine weight.
    Transformer Engine instead applies the affine in FP32 and casts once at the
    end.  Those expressions are algebraically equivalent but not numerically
    equivalent in BF16, and the difference compounds across the released
    45-block checkpoint.

    The custom VJP avoids retaining a full FP32 normalized activation across
    forward/backward: it saves the input and gamma in their original dtypes and
    one FP32 inverse-RMS value per normalized row.  Both passes preserve the
    released operation order exactly and are CUDA-graph capture safe.
    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        sequence_parallel: Optional[bool] = None,
        params_dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        device: Optional[torch.device | str | int] = None,
        dtype: Optional[torch.dtype] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if zero_centered_gamma:
            raise ValueError("GLM5Next RMSNorm does not support zero-centered gamma.")
        if dtype is not None and params_dtype is not None and dtype != params_dtype:
            raise ValueError("dtype and params_dtype must agree when both are provided.")

        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(normalized_shape)
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"normalized_shape must contain positive dimensions, got {shape!r}.")

        self.normalized_shape = shape
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps!r}.")
        parameter_dtype = dtype if dtype is not None else params_dtype
        if parameter_dtype is None:
            parameter_dtype = torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(shape, device=device, dtype=parameter_dtype))
        self.sequence_parallel = bool(sequence_parallel)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        if self.weight.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the released unit-gamma initialization contract."""

        if self.weight.device.type != "meta":
            nn.init.ones_(self.weight)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize in FP32, then round before applying the affine weight."""

        if tuple(hidden_states.shape[-len(self.normalized_shape) :]) != self.normalized_shape:
            raise ValueError(
                f"Input shape {tuple(hidden_states.shape)} does not end in "
                f"normalized_shape={self.normalized_shape}."
            )
        return _GLM5NextRMSNormFunction.apply(
            hidden_states, self.weight, self.normalized_shape, self.eps
        )

    def extra_repr(self) -> str:
        """Describe the normalized shape and epsilon."""
        return f"{self.normalized_shape}, eps={self.eps}"
