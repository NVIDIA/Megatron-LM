# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared errors, qualification constants, and numerical helpers for MLA latent CP."""

from __future__ import annotations

import math
from typing import Final, TypeAlias

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend

CUDNN_FRONTEND_SOURCE_REV: Final[str] = "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
"""Immutable source revision used to implement and qualify the cuDNN Graph adapter."""

QualifiedBackendTuple: TypeAlias = tuple[AttnBackend, str, str, tuple[int, int]]

# This feature is fail-closed. These are the complete, exact tuples backed by the checked-in
# qualification contract; wildcards, version ranges, and runtime overrides are unsupported.
QUALIFIED_BACKEND_CONFIGS: Final[tuple[QualifiedBackendTuple, ...]] = (
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)),
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)),
    (AttnBackend.flash, "4.0.0b11", "flash-attn-4==4.0.0b11", (10, 0)),
)


class LatentCPError(RuntimeError):
    """Base error for the experimental latent-CP implementation."""


class BackendNotQualifiedError(LatentCPError):
    """Raised when the exact backend/package/device tuple is not qualified."""


class BackendPlanNotSupportedError(LatentCPError):
    """Raised before P2P when a public backend reports that a phase plan is unsupported."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MLAWithLatentCP: {message}")


def scatter_upper_phase(
    output: Tensor, lse: Tensor, back_indices: Tensor, local_tokens: int
) -> tuple[Tensor, Tensor]:
    """Functionally scatter an upper rectangular phase into full local Q rows."""

    output_full = torch.zeros(
        (local_tokens, *output.shape[1:]), dtype=torch.float32, device=output.device
    ).index_copy(0, back_indices, output.float())
    lse_full = torch.full(
        (local_tokens, lse.size(1)), -torch.inf, dtype=torch.float32, device=lse.device
    ).index_copy(0, back_indices, lse.float())
    return output_full, lse_full


def merge_attention_partials(
    output_a: Tensor, lse_a: Tensor, output_b: Tensor, lse_b: Tensor
) -> tuple[Tensor, Tensor]:
    """Stable FP32 online-softmax merge for two attention partials."""

    _require(
        output_a.dtype == output_b.dtype == torch.float32,
        "partial outputs must be FP32",
    )
    _require(lse_a.dtype == lse_b.dtype == torch.float32, "partial LSE must be FP32")
    merged_lse = torch.logaddexp(lse_a, lse_b)
    valid_a = torch.isfinite(lse_a) & torch.isfinite(merged_lse)
    valid_b = torch.isfinite(lse_b) & torch.isfinite(merged_lse)
    delta_a = torch.where(
        valid_a, lse_a - merged_lse, torch.full_like(lse_a, -torch.inf)
    )
    delta_b = torch.where(
        valid_b, lse_b - merged_lse, torch.full_like(lse_b, -torch.inf)
    )
    weight_a = torch.exp(delta_a)
    weight_b = torch.exp(delta_b)
    merged_output = output_a * weight_a.unsqueeze(-1) + output_b * weight_b.unsqueeze(
        -1
    )
    return merged_output, merged_lse


def cudnn_backward_proxy(
    partial_output: Tensor, grad_output: Tensor, grad_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Construct cuDNN's corrected BF16 o/dO inputs in FP32.

    For safe rows, dot(G_i, O_corr) encodes the missing LSE gradient in public sdpa_backward.
    Zero and tiny rows use zero correction.
    """

    partial_output = partial_output.float()
    grad_output = grad_output.float()
    grad_lse = grad_lse.float()
    norm2 = torch.sum(grad_output * grad_output, dim=-1)
    threshold = math.sqrt(torch.finfo(torch.float32).tiny)
    safe = torch.isfinite(norm2) & (norm2 >= threshold) & torch.isfinite(grad_lse)
    denominator = torch.where(safe, norm2, torch.ones_like(norm2))
    coefficient = torch.where(safe, grad_lse / denominator, torch.zeros_like(grad_lse))
    raw_correction = coefficient.unsqueeze(-1) * grad_output
    correction = torch.where(
        safe.unsqueeze(-1), raw_correction, torch.zeros_like(raw_correction)
    )
    return partial_output - correction, grad_output
