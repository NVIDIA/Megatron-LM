"""Explicit autograd boundary for inference-only module outputs."""

from __future__ import annotations

import torch


class _InferenceOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        *dependencies: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        del dependencies
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx, grad_output
        raise NotImplementedError(
            "Backward crossed an inference-only module boundary. "
            "Replace the module with a training-capable implementation."
        )


def inference_only(
    value: torch.Tensor,
    *dependencies: torch.Tensor,
) -> torch.Tensor:
    """Return ``value`` unchanged and reject backward through dependencies.

    Deployment kernels commonly produce tensors under ``torch.inference_mode``.
    Explicit dependencies keep this boundary connected to trainable inputs and
    BF16 master weights without changing the visible forward value.
    """

    # A tensor allocated under inference_mode suppresses autograd even for
    # subsequent operations.  Clone it after leaving that mode so the explicit
    # boundary below can own the grad_fn while preserving the exact values.
    if value.is_inference():
        value = value.clone()
    return _InferenceOnly.apply(value, *dependencies)


__all__ = ["inference_only"]
