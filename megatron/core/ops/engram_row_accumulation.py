# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton row accumulation for the Engram symmetric (GIN/LSA) lookup.

Ported from NVIDIA/Megatron-LM#7224 by Hongxiao Bai (hxbai), commits 610449cd5 and 906ccd59e.
The kernel is verbatim.

:func:`accumulate_rows` adds ``gradients[i]`` to ``target[rows[i]]``. It launches one program
per position of the stably sorted rows. The first position of every run of equal rows sums the
whole run in FP32 and writes that row once, so no atomics are needed and the result does not
depend on the order in which the programs run.

This module imports Triton at module import. The Engram package reaches it only through
``megatron.core.ops._backends.require``.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _engram_accumulate_rows_kernel(
    target,
    rows,
    order,
    gradients,
    N,
    D: tl.constexpr,
    TARGET_STRIDE: tl.constexpr,
    GRAD_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """One writer per sorted segment; sum duplicates in FP32 without atomics."""
    first = tl.program_id(0)
    source = tl.load(order + first)
    row = tl.load(rows + source * ROW_STRIDE)
    previous = tl.load(order + first - 1, first > 0, other=0)
    previous_row = tl.load(rows + previous * ROW_STRIDE)
    if (first == 0) | (row != previous_row):
        cols = tl.arange(0, BLOCK_D)
        values = tl.full((BLOCK_D,), 0, tl.float32)
        more = True
        start = first
        while more:
            positions = start + tl.arange(0, BLOCK_ROWS)
            indices = tl.load(order + positions, positions < N, other=0)
            ids = tl.load(rows + indices * ROW_STRIDE, positions < N, other=-1)
            valid = (positions < N) & (ids == row)
            chunk = tl.load(
                gradients + indices[:, None] * GRAD_STRIDE + cols[None, :],
                valid[:, None] & (cols[None, :] < D),
                other=0,
            ).to(tl.float32)
            values += tl.sum(chunk, axis=0)
            start += BLOCK_ROWS
            more = (start < N) & (tl.sum(valid.to(tl.int32), axis=0) == BLOCK_ROWS)
        old = tl.load(target + row * TARGET_STRIDE + cols, cols < D, other=0).to(tl.float32)
        tl.store(target + row * TARGET_STRIDE + cols, old + values, cols < D)


def accumulate_rows(target: Tensor, rows: Tensor, order: Tensor, gradients: Tensor) -> bool:
    """Add ``gradients [N, D]`` to ``target[rows]``, summing duplicate rows in FP32.

    ``order`` is the stable argsort of ``rows [N]``. Returns ``False`` without writing for
    inputs the kernel does not cover: CPU tensors, FP64, or a column stride other than one.
    The caller then runs its PyTorch path.
    """
    if not (
        target.is_cuda
        and target.dtype != torch.float64
        and gradients.dtype != torch.float64
        and target.stride(1) == gradients.stride(1) == 1
    ):
        return False
    _engram_accumulate_rows_kernel[(rows.numel(),)](
        target,
        rows,
        order,
        gradients,
        rows.numel(),
        target.shape[1],
        target.stride(0),
        gradients.stride(0),
        rows.stride(0),
        triton.next_power_of_2(target.shape[1]),
        16,
    )
    return True
