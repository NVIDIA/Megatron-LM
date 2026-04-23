# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused residual-add + RMSNorm Triton kernel for inference.

Replaces the common two-op pattern::

    residual = residual + x
    output = rmsnorm(residual, weight, eps)

with a single Triton kernel that keeps the row in registers between the add
and the normalization. At small batch sizes the launch and bandwidth saved
across every layer boundary is worth a few percent end-to-end; at larger
batch sizes the effect is smaller but still strictly positive.

The kernel accumulates the variance in fp32 regardless of the input dtype,
matching the TE RMSNorm convention.

The *cross-layer* protocol that passes ``(residual, delta)`` pairs between
layers so this kernel can absorb the previous layer's residual-add lives in
:mod:`megatron.core.fusions.deferred_add`.
"""

from typing import Tuple
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = version.parse(triton.__version__) >= version.parse("2.0.0")
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _fused_add_rmsnorm_kernel(
    residual_ptr,  # [N, H] bf16/fp16/fp32, read-write: residual ← residual + x
    x_ptr,  # [N, H], read
    weight_ptr,  # [H], read
    out_ptr,  # [N, H], write
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Loads a full row into registers, adds, normalizes,
    writes back the updated residual and the normalized output."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_offset = row * n_cols

    r = tl.load(residual_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)

    r = r + x
    # Store the updated residual (in the input dtype).
    tl.store(residual_ptr + row_offset + cols, r, mask=mask)

    var = tl.sum(r * r, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = r * rstd * w

    tl.store(out_ptr + row_offset + cols, y, mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused ``residual + x`` then RMSNorm, computed in one Triton kernel.

    Returns both the updated residual and its normalized view. The kernel
    requires contiguous memory for its row-stride arithmetic; if ``residual``
    is not contiguous we make a contiguous copy and return *that* buffer as
    the updated residual. Callers must use the returned residual for any
    downstream ops -- relying on in-place mutation of the input tensor is
    not safe in the non-contiguous case.

    Args:
        x: Tensor of shape ``(..., hidden)`` to add into the residual.
        residual: Tensor of shape ``(..., hidden)``, same dtype as ``x``.
        weight: 1D RMSNorm weight of shape ``(hidden,)``.
        eps: Epsilon added inside the reciprocal square root.

    Returns:
        ``(normed, residual_out)`` where ``normed = rmsnorm(residual + x)``
        and ``residual_out = residual + x``. ``residual_out`` may alias the
        input ``residual`` when the input was already contiguous.
    """
    if not HAVE_TRITON:
        raise RuntimeError("fused_add_rmsnorm requires Triton >= 2.0.0")

    assert x.dtype == residual.dtype, (
        f"dtype mismatch: x {x.dtype} vs residual {residual.dtype}"
    )
    assert weight.ndim == 1 and weight.shape[0] == residual.shape[-1], (
        f"weight shape {weight.shape} incompatible with hidden dim {residual.shape[-1]}"
    )
    assert x.is_cuda, "fused_add_rmsnorm only supports CUDA tensors"

    # Broadcast ``x`` to match ``residual``'s shape so the kernel (which
    # expects identical shapes) can run. This mirrors the implicit
    # broadcasting done by the unfused ``residual + x`` path.
    if x.shape != residual.shape:
        x = x.expand_as(residual)

    # The kernel indexes by ``row * n_cols``, so both operands must be
    # row-major contiguous. ``.contiguous()`` is a no-op when the input
    # is already contiguous -- in that case ``residual_out`` aliases the
    # caller's buffer and the mutation is visible through their reference.
    x = x.contiguous()
    residual_out = residual.contiguous()
    weight = weight.contiguous()

    hidden = x.shape[-1]
    n_rows = x.numel() // hidden
    out = torch.empty_like(x)

    # One block per row; full row fits in registers/SMEM for hidden <= 16384.
    block_size = triton.next_power_of_2(hidden)
    # num_warps heuristic: 4 is good up to ~2k, 8 for ~4-8k, 16 beyond.
    if hidden <= 2048:
        num_warps = 4
    elif hidden <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    _fused_add_rmsnorm_kernel[(n_rows,)](
        residual_out,
        x,
        weight,
        out,
        hidden,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, residual_out
