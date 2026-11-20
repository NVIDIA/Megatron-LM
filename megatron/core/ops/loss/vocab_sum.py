# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-order vocabulary-shard sums for batch-invariant cross entropy."""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

from megatron.core.ops.collectives.ordered_reduce import ordered_all_reduce_shards

_ROW_SUM_BLOCK = 1024


@triton.jit
def _row_shard_sum_kernel(x_ptr, out_ptr, num_rows, row_stride, shard_len, BLOCK: tl.constexpr):
    """out[s, r] = sum of row r over shard s: BLOCK-strided fp32 lanes, then one tree sum.

    One program per (row, shard), so the order depends only on the shard length and the
    fixed BLOCK, never on the number of rows. Launched by :func:`batch_invariant_vocab_sum`
    (the cross-entropy softmax denominator).
    """
    row = tl.program_id(0).to(tl.int64)
    shard = tl.program_id(1).to(tl.int64)
    base = x_ptr + row * row_stride + shard * shard_len
    lanes = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, shard_len, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        lanes += tl.load(base + offs, mask=offs < shard_len, other=0.0).to(tl.float32)
    tl.store(out_ptr + shard * num_rows + row, tl.sum(lanes, axis=0))


def batch_invariant_vocab_sum(x: torch.Tensor, group, num_shards: int) -> torch.Tensor:
    """fp32 ``x.sum(-1)`` over a vocab split across ``group``, with batch-invariant bits.

    ``x`` [..., V/TP] holds this rank's ``num_shards`` consecutive vocab shards. Each shard is
    summed per row in a fixed order, and the shard sums of all ranks are folded in global shard
    order (:func:`ordered_all_reduce_shards`), so the result depends neither on the number of
    rows nor, for a fixed total shard count, on the TP size.

    Used by the vocab-parallel cross entropy (``cross_entropy.py``, unfused and fused) for the
    softmax denominator under batch-invariant mode.
    """
    *lead, vocab = x.shape
    assert vocab % num_shards == 0, f"local vocab {vocab} must divide into {num_shards} shards"
    x2 = x.reshape(-1, vocab)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    parts = torch.empty((num_shards, rows), dtype=torch.float32, device=x.device)
    if rows:
        _row_shard_sum_kernel[(rows, num_shards)](
            x2, parts, rows, x2.stride(0), vocab // num_shards, BLOCK=_ROW_SUM_BLOCK
        )
    return ordered_all_reduce_shards(parts, group).view(lead)
