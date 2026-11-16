# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernel that zeroes the padding-token rows of a token-major activation.

Dynamic inference pads the token dimension, both on eager steps and up to a captured size on
CUDA-graph steps. Padding rows still feed the per-tensor amax of FP8/FP4 GEMMs, so they are
zeroed first. The boundary between real and padding rows is read from a fixed-address
``int32[1]`` GPU tensor, so the kernel is safe to capture in a CUDA graph: only the value
behind the pointer changes between replays. It stores zeros to the padding rows only and
never reads the activation.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


# Coverage.py cannot trace Triton device bodies after they are JIT-compiled. Eager,
# CUDA-graph replay, and determinism behavior are exercised by dedicated CUDA tests.
@triton.jit
def _zero_padding_rows_kernel(  # pragma: no cover
    x_ptr,  # [num_rows, row_numel], contiguous
    real_token_count_ptr,  # int32* [1]
    num_rows,
    row_numel,
    first_token_idx,  # token index of row 0
    BLOCK_M: tl.constexpr,  # rows per program
    BLOCK_N: tl.constexpr,  # elements per row per program
):
    """Store zeros to rows whose token index is at or past the real token count."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    real_token_count = tl.load(real_token_count_ptr)
    is_padding = (rows + first_token_idx >= real_token_count) & (rows < num_rows)
    mask = is_padding[:, None] & (cols < row_numel)[None, :]

    offsets = rows[:, None].to(tl.int64) * row_numel + cols[None, :]
    zeros = tl.zeros((BLOCK_M, BLOCK_N), dtype=x_ptr.dtype.element_ty)
    tl.store(x_ptr + offsets, zeros, mask=mask)


def zero_padding_rows(tensor: Tensor, real_token_count: Tensor, first_token_idx: int = 0) -> None:
    """Zero, in place, the rows of ``tensor`` whose token index is at or past the real count.

    Args:
        tensor: Activation whose first dimension is the padded token dimension.
        real_token_count: ``[1]`` int32 GPU tensor holding the number of real tokens. It is
            read on device, so the boundary follows the value at CUDA-graph replay.
        first_token_idx: Token index of row 0 of ``tensor``, e.g. the offset of this rank's
            sequence-parallel shard.
    """
    num_rows = tensor.shape[0]
    if tensor.numel() == 0:
        return

    if HAVE_TRITON and tensor.is_cuda and tensor.is_contiguous():
        row_numel = tensor.numel() // num_rows
        block_n = min(1024, triton.next_power_of_2(row_numel))
        block_m = 8
        grid = (triton.cdiv(num_rows, block_m), triton.cdiv(row_numel, block_n))
        _zero_padding_rows_kernel[grid](
            tensor,
            real_token_count,
            num_rows,
            row_numel,
            first_token_idx,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        return

    token_idx = torch.arange(
        first_token_idx,
        first_token_idx + num_rows,
        dtype=real_token_count.dtype,
        device=tensor.device,
    )
    is_padding = (token_idx >= real_token_count).view(num_rows, *([1] * (tensor.dim() - 1)))
    tensor.masked_fill_(is_padding, 0.0)
