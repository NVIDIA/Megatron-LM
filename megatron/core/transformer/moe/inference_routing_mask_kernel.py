# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernel for masking CUDA-graph padding rows of a local routing map.

Under CUDA-graph capture the local token count is padded up to a captured
graph size; those padding rows have garbage routing indices and, if left
alone, would dispatch padding tokens to real experts. This kernel zeroes
that out by writing ``-1`` into every topk slot of rows in
``[real_token_count, local_tokens)``.

The kernel reads ``real_token_count`` from a fixed-address ``int32[1]`` GPU
tensor, so it is safe to call from inside a captured graph: only the value
behind the pointer changes between replays.
"""

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
    triton.autotune = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}),
        triton.Config({"BLOCK_M": 128})
    ],
    key=["total_rows", "TOPK"],
)
@triton.jit
def _mask_routing_padding_kernel(
    routing_map_ptr,          # int64* [total_rows, topk]
    real_token_count_ptr,     # int32* [1]
    total_rows: tl.int32,
    tp_rank: tl.int32,        # SP/TP rank — local row r maps to global row r + tp_rank*total_rows
    TOPK: tl.constexpr,       # actual topk
    BLOCK_M: tl.constexpr,    # rows per program (autotuned)
    BLOCK_TOPK: tl.constexpr, # next_power_of_2(TOPK), column block
):
    """Fill `routing_map[real_token_count:, :]` with -1, BLOCK_M rows per program."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    real_count = tl.load(real_token_count_ptr).to(tl.int32)

    # real_count is in the global (pre-SP-shard) frame; rows is local to this SP rank.
    global_rows = rows + tp_rank * total_rows
    row_mask = (global_rows >= real_count) & (rows < total_rows)

    cols = tl.arange(0, BLOCK_TOPK)
    col_mask = cols < TOPK

    offs = rows[:, None].to(tl.int64) * TOPK + cols[None, :].to(tl.int64)
    mask = row_mask[:, None] & col_mask[None, :]

    neg_one = tl.full((BLOCK_M, BLOCK_TOPK), -1, dtype=tl.int64)
    tl.store(routing_map_ptr + offs, neg_one, mask=mask)


def mask_routing_padding(
    routing_map: Tensor, real_token_count_tensor: Tensor, tp_rank: int = 0
) -> None:
    """In-place fill -1 into ``routing_map[real_token_count:, :]``.

    Args:
        routing_map: ``[N, topk]`` int64 local routing map. ``N`` is the
            (possibly CUDA-graph-padded) local token count.
        real_token_count_tensor: ``[1]`` int32 GPU tensor holding the real
            (unpadded) token count for this step, in the global (pre-SP-shard)
            frame. Read inside the kernel so the mask boundary moves correctly
            across CUDA-graph replays.
        tp_rank: This rank's index in the SP/TP group. Local row ``r`` is
            row ``r + tp_rank * N`` in the global frame; the kernel uses this
            offset to compare against ``real_token_count_tensor``.
    """
    assert routing_map.is_cuda, "routing_map must be on CUDA"
    assert routing_map.dim() == 2, f"expected 2D routing_map, got {routing_map.shape}"
    assert routing_map.dtype.is_floating_point is False, "routing_map must be integer"

    total_rows, topk = routing_map.shape
    if total_rows == 0:
        return

    BLOCK_TOPK = triton.next_power_of_2(topk)
    # BLOCK_M is picked by @triton.autotune above; key=(total_rows, TOPK).
    grid = lambda META: (triton.cdiv(total_rows, META["BLOCK_M"]),)  # noqa: E731

    _mask_routing_padding_kernel[grid](
        routing_map,
        real_token_count_tensor,
        total_rows=total_rows,
        tp_rank=tp_rank,
        TOPK=topk,
        BLOCK_TOPK=BLOCK_TOPK,
    )
