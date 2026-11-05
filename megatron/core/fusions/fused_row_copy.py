# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Contiguous copy of a row-strided tensor: contiguous rows at a uniform stride, such as a slice
of the last dimension.

``Tensor.contiguous`` copies such a view with PyTorch's generic strided copy kernel, which does
not vectorize its accesses. This kernel copies 2D tiles instead. The values are copied unchanged
and the gradient is passed through, as for ``Tensor.contiguous``.
"""

from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _row_copy_kernel(
    x_ptr, y_ptr, n_rows, n_cols, row_stride, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    rows64 = rows[:, None].to(tl.int64)
    values = tl.load(x_ptr + rows64 * row_stride + cols[None, :], mask=mask)
    tl.store(y_ptr + rows64 * n_cols + cols[None, :], values, mask=mask)


def _row_layout(x: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """Return (rows, columns, row stride) if ``x`` is contiguous rows at a uniform stride."""
    if x.dim() < 2 or x.stride(-1) != 1 or x.numel() == 0:
        return None
    leading = [(x.size(i), x.stride(i)) for i in range(x.dim() - 1) if x.size(i) != 1]
    n_cols = x.size(-1)
    if not leading:
        return 1, n_cols, n_cols
    for (_, outer_stride), (inner_size, inner_stride) in zip(leading[:-1], leading[1:]):
        if outer_stride != inner_stride * inner_size:
            return None
    row_stride = leading[-1][1]
    if row_stride < n_cols:
        return None
    return x.numel() // n_cols, n_cols, row_stride


class _RowContiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_rows, n_cols, row_stride):
        """Copy the rows of ``x`` into a contiguous tensor."""
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        block_c = min(1024, triton.next_power_of_2(n_cols))
        block_r = max(1, 4096 // block_c)
        grid = (triton.cdiv(n_rows, block_r), triton.cdiv(n_cols, block_c))
        _row_copy_kernel[grid](
            x, y, n_rows, n_cols, row_stride, BLOCK_R=block_r, BLOCK_C=block_c, num_warps=4
        )
        return y

    @staticmethod
    def backward(ctx, grad):
        """Pass the gradient through, as the backward of ``Tensor.contiguous`` does."""
        return grad, None, None, None


def contiguous_rows(x: torch.Tensor) -> torch.Tensor:
    """Return ``x.contiguous()``, copying row-strided CUDA tensors with a 2D-tiled kernel.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: ``x`` itself if it is contiguous, otherwise a contiguous copy.
    """
    if x.is_contiguous():
        return x
    layout = _row_layout(x)
    if not HAVE_TRITON or not x.is_cuda or layout is None:
        return x.contiguous()
    return _RowContiguous.apply(x, *layout)
