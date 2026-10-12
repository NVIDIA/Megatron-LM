# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import NamedTuple, Optional
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()

# Tokens per program of both kernels and rows of the permuted buffer cleared per loop iteration.
BLOCK_TOKENS = 128
BLOCK_ROWS = 128
NUM_WARPS = 4


class LocalRoutingMetadata(NamedTuple):
    """HybridEP's routing metadata of a single-rank expert group, from the local routing map.

    The tensors named as in HybridEP's dispatch handle have its layout and values: the dense
    index of a token is the number of tokens routed somewhere before it, expert ``e``'s permuted
    rows start at the sum of the padded raw counts of the experts before it, a token's row for
    expert ``e`` is that start plus the number of earlier tokens routed to ``e``, and rows at or
    past the permuted-token budget are dropped (-1 in the maps, counts clipped, overflow flag
    set when the last expert's rows run past the budget).
    """

    tokens_per_expert: torch.Tensor
    """[num_experts] int32, after the drop against the budget."""
    padded_tokens_per_expert: torch.Tensor
    """[num_experts] int64, tokens_per_expert rounded up to the pad multiple."""
    dense_to_expert_map: torch.Tensor
    """[num_token_slots, num_experts] int32, permuted row of dense token d at expert e or -1;
    rows at or past the number of dense tokens are not written."""
    dense_chunk_layout: torch.Tensor
    """[ceil(num_token_slots / chunk_size)] int32, dense tokens before chunk c + 1; the last entry
    is the number of dense tokens."""
    sparse_to_dense_map: torch.Tensor
    """[num_token_slots, 1] int32, dense index of each token or -1 when routed nowhere."""
    rdma_to_attn_map: torch.Tensor
    """[num_token_slots, 1] bool, whether the token is routed somewhere."""
    num_dispatched_tokens: torch.Tensor
    """[1] int32, the number of dense tokens."""
    overflow_flag: torch.Tensor
    """[1] int32, 1 when tokens were dropped."""
    row_to_token: torch.Tensor
    """[num_permuted_tokens] int32, source token of each permuted row, -1 on padding rows."""
    permuted_probs: Optional[torch.Tensor]
    """[num_permuted_tokens] float32, probability of each permuted row, 0 on padding rows."""


@triton.jit
def _block_counts_kernel(
    routing_map_ptr,
    partials_ptr,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_PADDED: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """partials[b, e]: tokens of block b routed to expert e; [b, EXPERTS_PADDED]: routed anywhere"""
    block = tl.program_id(0)
    tokens = block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    experts = tl.arange(0, EXPERTS_PADDED)
    mask = (tokens < num_tokens)[:, None] & (experts < NUM_EXPERTS)[None, :]
    routed = tl.load(
        routing_map_ptr + tokens.to(tl.int64)[:, None] * NUM_EXPERTS + experts[None, :],
        mask=mask,
        other=0,
    )
    routed = (routed != 0).to(tl.int32)
    needed = (tl.sum(routed, axis=1) > 0).to(tl.int32)
    tl.store(partials_ptr + block * (EXPERTS_PADDED + 1) + experts, tl.sum(routed, axis=0))
    tl.store(partials_ptr + block * (EXPERTS_PADDED + 1) + EXPERTS_PADDED, tl.sum(needed, axis=0))


@triton.jit
def _routing_layout_kernel(
    routing_map_ptr,
    probs_ptr,
    partials_ptr,
    tokens_per_expert_ptr,
    padded_tokens_per_expert_ptr,
    dense_to_expert_map_ptr,
    dense_chunk_layout_ptr,
    sparse_to_dense_map_ptr,
    rdma_to_attn_map_ptr,
    num_dispatched_tokens_ptr,
    overflow_flag_ptr,
    over_budget_ptr,
    row_to_token_ptr,
    permuted_probs_ptr,
    num_tokens,
    num_token_slots,
    num_permuted_tokens,
    num_blocks,
    num_chunks,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_PADDED: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAD_MULTIPLE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HAS_PROBS: tl.constexpr,
    HAS_OVER_BUDGET: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    experts = tl.arange(0, EXPERTS_PADDED)
    expert_mask = experts < NUM_EXPERTS

    # Totals over all blocks and the exclusive prefix over the blocks before this one, from the
    # per-block partial counts (L2-resident).
    totals = tl.zeros([EXPERTS_PADDED], dtype=tl.int32)
    before = tl.zeros([EXPERTS_PADDED], dtype=tl.int32)
    needed_totals = tl.zeros([32], dtype=tl.int32)
    needed_before = tl.zeros([32], dtype=tl.int32)
    for start in range(0, num_blocks, 32):
        blocks = start + tl.arange(0, 32)
        block_mask = blocks < num_blocks
        tile = tl.load(
            partials_ptr + blocks[:, None] * (EXPERTS_PADDED + 1) + experts[None, :],
            mask=block_mask[:, None] & expert_mask[None, :],
            other=0,
        )
        totals += tl.sum(tile, axis=0)
        before += tl.sum(tl.where((blocks < block)[:, None], tile, 0), axis=0)
        needed = tl.load(
            partials_ptr + blocks * (EXPERTS_PADDED + 1) + EXPERTS_PADDED, mask=block_mask, other=0
        )
        needed_totals += needed
        needed_before += tl.where(blocks < block, needed, 0)
    num_dense_tokens = tl.sum(needed_totals, axis=0)
    dense_before = tl.sum(needed_before, axis=0)

    # HybridEP's expert layout: every expert starts after the padded raw counts of the experts
    # before it; rows past the budget are dropped.
    padded_raw = ((totals + (PAD_MULTIPLE - 1)) // PAD_MULTIPLE) * PAD_MULTIPLE
    starts = tl.cumsum(padded_raw, axis=0) - padded_raw
    ends_raw = starts + totals
    counts = tl.where(
        ends_raw <= num_permuted_tokens,
        totals,
        tl.where(starts < num_permuted_tokens, num_permuted_tokens - starts, 0),
    )
    counts = tl.where(expert_mask, counts, 0)
    padded = ((counts + (PAD_MULTIPLE - 1)) // PAD_MULTIPLE) * PAD_MULTIPLE
    rows_used = tl.sum(padded, axis=0)
    overflow = (
        tl.sum(tl.where(experts == NUM_EXPERTS - 1, ends_raw, 0), axis=0) > num_permuted_tokens
    )

    if block == 0:
        one = tl.arange(0, 1)
        tl.store(tokens_per_expert_ptr + experts, counts, mask=expert_mask)
        tl.store(padded_tokens_per_expert_ptr + experts, padded.to(tl.int64), mask=expert_mask)
        tl.store(
            dense_chunk_layout_ptr + (num_chunks - 1) + one,
            num_dense_tokens + tl.zeros([1], tl.int32),
        )
        tl.store(num_dispatched_tokens_ptr + one, num_dense_tokens + tl.zeros([1], tl.int32))
        overflow_flag = overflow.to(tl.int32) + tl.zeros([1], tl.int32)
        tl.store(overflow_flag_ptr + one, overflow_flag)
        if HAS_OVER_BUDGET:
            over_budget = tl.load(over_budget_ptr + one)
            tl.store(
                over_budget_ptr + one, ((over_budget != 0) | (overflow_flag != 0)).to(tl.uint8)
            )

    # This block's tokens: dense index, rank within each expert, map rows, chunk layout,
    # row_to_token and permuted probabilities.
    tokens = block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    tokens64 = tokens.to(tl.int64)
    token_mask = tokens < num_tokens
    slot_mask = tokens < num_token_slots
    routed = tl.load(
        routing_map_ptr + tokens64[:, None] * NUM_EXPERTS + experts[None, :],
        mask=token_mask[:, None] & expert_mask[None, :],
        other=0,
    )
    routed = (routed != 0).to(tl.int32)
    needed = (tl.sum(routed, axis=1) > 0).to(tl.int32)
    dense = dense_before + tl.cumsum(needed, axis=0) - needed
    rows = (starts + before)[None, :] + tl.cumsum(routed, axis=0) - routed
    kept = (routed != 0) & (rows < num_permuted_tokens)
    tl.store(
        dense_to_expert_map_ptr + dense.to(tl.int64)[:, None] * NUM_EXPERTS + experts[None, :],
        tl.where(kept, rows, -1),
        mask=(needed > 0)[:, None] & expert_mask[None, :],
    )
    tl.store(sparse_to_dense_map_ptr + tokens, tl.where(needed > 0, dense, -1), mask=slot_mask)
    tl.store(rdma_to_attn_map_ptr + tokens, needed.to(tl.uint8), mask=slot_mask)
    chunks = tokens // CHUNK_SIZE
    chunk_first = ((tokens % CHUNK_SIZE) == 0) & (chunks > 0) & (chunks < num_chunks) & slot_mask
    tl.store(dense_chunk_layout_ptr + (chunks - 1), dense, mask=chunk_first)
    rows64 = rows.to(tl.int64)
    tl.store(
        row_to_token_ptr + rows64,
        tl.broadcast_to(tokens[:, None], (BLOCK_TOKENS, EXPERTS_PADDED)),
        mask=kept,
    )
    if HAS_PROBS:
        probs = tl.load(
            probs_ptr + tokens64[:, None] * NUM_EXPERTS + experts[None, :], mask=kept, other=0.0
        )
        tl.store(permuted_probs_ptr + rows64, probs, mask=kept)

    # The padding rows of this block's slice of the permuted buffer: the alignment padding of
    # every expert and the unused tail.
    ends = starts + counts
    ends_padded = starts + padded
    for start in range(0, ROWS_PER_BLOCK, BLOCK_ROWS):
        slice_rows = block * ROWS_PER_BLOCK + start + tl.arange(0, BLOCK_ROWS)
        in_padding = (
            (slice_rows[:, None] >= ends[None, :])
            & (slice_rows[:, None] < ends_padded[None, :])
            & expert_mask[None, :]
        )
        is_padding = (tl.sum(in_padding.to(tl.int32), axis=1) > 0) | (slice_rows >= rows_used)
        mask = is_padding & (slice_rows < num_permuted_tokens)
        tl.store(row_to_token_ptr + slice_rows, tl.full([BLOCK_ROWS], -1, tl.int32), mask=mask)
        if HAS_PROBS:
            tl.store(permuted_probs_ptr + slice_rows, tl.zeros([BLOCK_ROWS], tl.float32), mask=mask)


def fused_local_routing_metadata(
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_token_slots: int,
    num_permuted_tokens: int,
    pad_multiple: int,
    chunk_size: int,
    over_budget: Optional[torch.Tensor] = None,
) -> LocalRoutingMetadata:
    """Build HybridEP's routing metadata of a single-rank expert group from its routing map.

    Two launches over the [num_tokens, num_experts] bool map: per-block counts, then the layout
    (every block re-reduces the per-block counts for its prefix). No host synchronization and
    no data-dependent shapes, so it can be captured in a CUDA graph.

    Args:
        routing_map: [num_tokens, num_experts] bool.
        probs: [num_tokens, num_experts] float32 routing probabilities, or None.
        num_token_slots: HybridEP's token-slot count (>= num_tokens, a multiple of 16); the
            slots past num_tokens are routed nowhere.
        num_permuted_tokens: the permuted-token budget, a multiple of pad_multiple.
        pad_multiple: alignment of every expert's rows.
        chunk_size: HybridEP's preprocessing chunk size (the dense_chunk_layout granularity).
        over_budget: [1] bool, OR-ed in place with the overflow flag when given.
    """
    assert HAVE_TRITON, "fused_local_routing_metadata needs Triton"
    assert routing_map.dim() == 2 and routing_map.dtype == torch.bool
    num_tokens, num_experts = routing_map.shape
    assert num_token_slots >= num_tokens and num_token_slots % 16 == 0
    assert pad_multiple > 0 and num_permuted_tokens % pad_multiple == 0 and chunk_size > 0
    if probs is not None:
        assert probs.shape == routing_map.shape and probs.dtype == torch.float32
        probs = probs.contiguous()
    if over_budget is not None:
        assert over_budget.dtype == torch.bool and over_budget.numel() == 1
    routing_map = routing_map.contiguous()
    device = routing_map.device
    experts_padded = triton.next_power_of_2(num_experts)
    num_blocks = triton.cdiv(num_token_slots, BLOCK_TOKENS)
    num_chunks = triton.cdiv(num_token_slots, chunk_size)
    rows_per_block = triton.cdiv(triton.cdiv(num_permuted_tokens, num_blocks), BLOCK_ROWS)
    rows_per_block *= BLOCK_ROWS

    partials = torch.empty((num_blocks, experts_padded + 1), dtype=torch.int32, device=device)
    tokens_per_expert = torch.empty((num_experts,), dtype=torch.int32, device=device)
    padded_tokens_per_expert = torch.empty((num_experts,), dtype=torch.int64, device=device)
    dense_to_expert_map = torch.empty(
        (num_token_slots, num_experts), dtype=torch.int32, device=device
    )
    dense_chunk_layout = torch.empty((num_chunks,), dtype=torch.int32, device=device)
    sparse_to_dense_map = torch.empty((num_token_slots, 1), dtype=torch.int32, device=device)
    rdma_to_attn_map = torch.empty((num_token_slots, 1), dtype=torch.bool, device=device)
    num_dispatched_tokens = torch.empty((1,), dtype=torch.int32, device=device)
    overflow_flag = torch.empty((1,), dtype=torch.int32, device=device)
    row_to_token = torch.empty((num_permuted_tokens,), dtype=torch.int32, device=device)
    permuted_probs = None
    if probs is not None:
        permuted_probs = torch.empty((num_permuted_tokens,), dtype=torch.float32, device=device)

    routing_map_u8 = routing_map.view(torch.uint8)
    _block_counts_kernel[(num_blocks,)](
        routing_map_u8,
        partials,
        num_tokens,
        NUM_EXPERTS=num_experts,
        EXPERTS_PADDED=experts_padded,
        BLOCK_TOKENS=BLOCK_TOKENS,
        num_warps=NUM_WARPS,
    )
    _routing_layout_kernel[(num_blocks,)](
        routing_map_u8,
        probs if probs is not None else routing_map_u8,
        partials,
        tokens_per_expert,
        padded_tokens_per_expert,
        dense_to_expert_map,
        dense_chunk_layout,
        sparse_to_dense_map,
        rdma_to_attn_map.view(torch.uint8),
        num_dispatched_tokens,
        overflow_flag,
        over_budget.view(torch.uint8) if over_budget is not None else overflow_flag,
        row_to_token,
        permuted_probs if permuted_probs is not None else row_to_token,
        num_tokens,
        num_token_slots,
        num_permuted_tokens,
        num_blocks,
        num_chunks,
        NUM_EXPERTS=num_experts,
        EXPERTS_PADDED=experts_padded,
        BLOCK_TOKENS=BLOCK_TOKENS,
        PAD_MULTIPLE=pad_multiple,
        CHUNK_SIZE=chunk_size,
        HAS_PROBS=probs is not None,
        HAS_OVER_BUDGET=over_budget is not None,
        ROWS_PER_BLOCK=rows_per_block,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=NUM_WARPS,
    )
    return LocalRoutingMetadata(
        tokens_per_expert,
        padded_tokens_per_expert,
        dense_to_expert_map,
        dense_chunk_layout,
        sparse_to_dense_map,
        rdma_to_attn_map,
        num_dispatched_tokens,
        overflow_flag,
        row_to_token,
        permuted_probs,
    )
