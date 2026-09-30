# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sync-free prefix objective for MTP end-to-end acceptance probabilities."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _mtp_prefix_forward_kernel(
        acceptances,
        output,
        prefix_losses,
        num_rows,
        num_depths,
        inv_num_depths,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute the prefix losses and their mean."""
        rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < num_rows
        prefix_product = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        prefix_loss_sum = tl.zeros((BLOCK_SIZE,), tl.float32)

        depth = 0
        while depth < num_depths:
            acceptance = tl.load(
                acceptances + depth * num_rows + rows, mask=row_mask, other=1.0
            ).to(tl.float32)
            prefix_product *= acceptance
            prefix_loss = 1.0 - prefix_product
            prefix_loss_sum += prefix_loss
            tl.store(prefix_losses + depth * num_rows + rows, prefix_loss, mask=row_mask)
            depth += 1

        tl.store(output + rows, prefix_loss_sum * inv_num_depths, mask=row_mask)

    @triton.jit
    def _mtp_prefix_backward_kernel(
        acceptances,
        grad_output,
        grad_acceptances,
        num_rows,
        num_depths,
        inv_num_depths,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply a zero-safe analytical gradient without dividing by acceptance."""
        rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < num_rows
        output_gradient = tl.load(grad_output + rows, mask=row_mask, other=0.0).to(tl.float32)
        prefix_before = tl.full((BLOCK_SIZE,), 1.0, tl.float32)

        depth = 0
        while depth < num_depths:
            suffix_product = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
            suffix_sum = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
            suffix_depth = depth + 1
            while suffix_depth < num_depths:
                suffix_acceptance = tl.load(
                    acceptances + suffix_depth * num_rows + rows, mask=row_mask, other=1.0
                ).to(tl.float32)
                suffix_product *= suffix_acceptance
                suffix_sum += suffix_product
                suffix_depth += 1

            gradient = -output_gradient * inv_num_depths * prefix_before * suffix_sum
            tl.store(grad_acceptances + depth * num_rows + rows, gradient, mask=row_mask)
            acceptance = tl.load(
                acceptances + depth * num_rows + rows, mask=row_mask, other=1.0
            ).to(tl.float32)
            prefix_before *= acceptance
            depth += 1


def fused_mtp_prefix_unavailable_reason(acceptances: Tensor) -> Optional[str]:
    """Return why the fused prefix objective cannot consume the input."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if not acceptances.is_cuda:
        return "acceptances are not a CUDA tensor"
    # The fused TV primitive produces FP32 acceptance probabilities for both
    # BF16 and FP32 logits. Keep this internal contract narrow instead of
    # introducing different BF16 prefix-rounding semantics that production
    # cannot exercise.
    if acceptances.dtype != torch.float32:
        return f"acceptance dtype {acceptances.dtype} is not supported"
    if acceptances.ndim == 0 or acceptances.size(0) == 0:
        return "acceptances must have a non-empty depth dimension"
    if acceptances.numel() == 0:
        return "acceptances must have at least one row"
    if not acceptances.is_contiguous():
        return "acceptances are not contiguous"
    return None


class _FusedMTPPrefixObjective(torch.autograd.Function):
    """Triton prefix objective with a zero-safe analytical backward."""

    @staticmethod
    def forward(ctx, acceptances: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the mean and per-depth rejection probabilities."""
        num_depths = acceptances.size(0)
        num_rows = acceptances.numel() // num_depths
        inv_num_depths = float(num_depths**-1)
        output = torch.empty(
            acceptances.shape[1:], dtype=acceptances.dtype, device=acceptances.device
        )
        prefix_losses = torch.empty_like(acceptances)
        _mtp_prefix_forward_kernel[(triton.cdiv(num_rows, 256),)](
            acceptances,
            output,
            prefix_losses,
            num_rows=num_rows,
            num_depths=num_depths,
            inv_num_depths=inv_num_depths,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        ctx.save_for_backward(acceptances)
        ctx.num_depths = num_depths
        ctx.num_rows = num_rows
        ctx.inv_num_depths = inv_num_depths
        ctx.mark_non_differentiable(prefix_losses)
        return output, prefix_losses

    @staticmethod
    def backward(ctx, grad_output: Tensor, _grad_prefix_losses: Tensor) -> tuple[Tensor]:
        """Return the analytical gradient for every acceptance probability."""
        (acceptances,) = ctx.saved_tensors
        grad_acceptances = torch.empty_like(acceptances)
        _mtp_prefix_backward_kernel[(triton.cdiv(ctx.num_rows, 256),)](
            acceptances,
            grad_output.contiguous(),
            grad_acceptances,
            num_rows=ctx.num_rows,
            num_depths=ctx.num_depths,
            inv_num_depths=ctx.inv_num_depths,
            BLOCK_SIZE=256,
            num_warps=4,
        )
        return (grad_acceptances,)


def _reference_mtp_prefix_objective(acceptances: Tensor) -> tuple[Tensor, Tensor]:
    """Preserve the established PyTorch implementation as an oracle and fallback."""
    prefix_losses = 1.0 - torch.cumprod(acceptances, dim=0)
    return prefix_losses.mean(dim=0), prefix_losses.detach()


def mtp_e2e_prefix_objective(acceptances: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and detached per-depth E2E-TV losses with automatic dispatch."""
    if fused_mtp_prefix_unavailable_reason(acceptances) is None:
        return _FusedMTPPrefixObjective.apply(acceptances)
    return _reference_mtp_prefix_objective(acceptances)
