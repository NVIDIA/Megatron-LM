# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Normalization primitives with the released GLM5Next rounding contract."""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor, nn


def _rmsnorm_forward(
    hidden_states: Tensor,
    weight: Tensor,
    normalized_dims: tuple[int, ...],
    weight_view_shape: tuple[int, ...],
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Normalize in FP32, round to the input dtype, then apply gamma."""
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(dim=normalized_dims, keepdim=True)
    inverse_rms = torch.rsqrt(variance + eps)
    normalized = (hidden_states_fp32 * inverse_rms).to(hidden_states.dtype)
    return weight.view(weight_view_shape) * normalized, inverse_rms


def _rmsnorm_backward(
    output_grad: Tensor,
    hidden_states: Tensor,
    weight: Tensor,
    inverse_rms: Tensor,
    normalized_dims: tuple[int, ...],
    weight_view_shape: tuple[int, ...],
    normalized_numel: int,
) -> tuple[Tensor, Tensor]:
    """Return gradients that repeat the forward rounding boundary."""
    hidden_states_fp32 = hidden_states.to(torch.float32)
    normalized = (hidden_states_fp32 * inverse_rms).to(hidden_states.dtype)
    weight_view = weight.view(weight_view_shape)

    weight_grad = output_grad * normalized
    broadcast_dims = tuple(range(hidden_states.ndim - len(normalized_dims)))
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
        dim=normalized_dims, keepdim=True
    )
    variance_grad = inverse_rms_grad * (-0.5) * inverse_rms.pow(3)
    square_grad = variance_grad / normalized_numel
    variance_input_grad = square_grad * (2.0 * hidden_states_fp32)
    input_grad = (direct_input_grad + variance_input_grad).to(hidden_states.dtype)
    return input_grad, weight_grad


@lru_cache(maxsize=1)
def _compiled_rmsnorm() -> tuple:
    """Compile both passes once, keeping every dtype cast of the eager graph.

    Inductor otherwise drops a round trip such as FP32 -> BF16 -> FP32, which is exactly the
    boundary this norm exists to keep; ``emulate_precision_casts`` restores it.
    """
    message = (
        "rmsnorm_cast_before_weight_compiled requires torch.compile support for "
        "emulate_precision_casts."
    )
    try:
        from torch._inductor import list_options
    except ImportError as error:
        raise RuntimeError(message) from error
    if "emulate_precision_casts" not in list_options():
        raise RuntimeError(message)
    options = {"emulate_precision_casts": True}
    return (
        torch.compile(_rmsnorm_forward, options=options),
        torch.compile(_rmsnorm_backward, options=options),
    )


class _GLM5NextRMSNormFunction(torch.autograd.Function):
    """Released forward plus an allocation-bounded, operation-ordered VJP."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        normalized_shape: tuple[int, ...],
        eps: float,
        compiled: bool = False,
    ) -> Tensor:
        """Normalize in FP32, round to the input dtype, then apply gamma."""
        normalized_dims = tuple(range(-len(normalized_shape), 0))
        weight_view_shape = (1,) * (hidden_states.ndim - len(normalized_shape)) + normalized_shape
        forward_fn = _compiled_rmsnorm()[0] if compiled else _rmsnorm_forward
        output, inverse_rms = forward_fn(
            hidden_states, weight, normalized_dims, weight_view_shape, eps
        )

        # Do not retain the full FP32 normalized activation.  The VJP only
        # needs the BF16/FP16 input and gamma plus one FP32 inverse RMS value
        # per normalized row.
        ctx.normalized_shape = normalized_shape
        ctx.normalized_dims = normalized_dims
        ctx.weight_view_shape = weight_view_shape
        ctx.compiled = compiled
        ctx.save_for_backward(hidden_states, weight, inverse_rms)
        return output

    @staticmethod
    def backward(ctx, output_grad: Tensor):
        """Return gradients that repeat the forward rounding boundary."""
        hidden_states, weight, inverse_rms = ctx.saved_tensors
        backward_fn = _compiled_rmsnorm()[1] if ctx.compiled else _rmsnorm_backward
        input_grad, weight_grad = backward_fn(
            output_grad,
            hidden_states,
            weight,
            inverse_rms,
            ctx.normalized_dims,
            ctx.weight_view_shape,
            math.prod(ctx.normalized_shape),
        )
        return input_grad, weight_grad, None, None, None


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

    With ``compiled=True`` both passes run as Inductor kernels instead of about
    twenty eager kernels.  Every dtype cast is kept (``emulate_precision_casts``),
    so the rounding contract is unchanged, but the mean square and the gamma
    gradient are summed in Inductor's reduction order and can differ from the
    eager result in the last bit.
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
        compiled: bool = False,
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
        self.compiled = bool(compiled)
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
            hidden_states, self.weight, self.normalized_shape, self.eps, self.compiled
        )

    def extra_repr(self) -> str:
        """Describe the normalized shape and epsilon."""
        return f"{self.normalized_shape}, eps={self.eps}"
