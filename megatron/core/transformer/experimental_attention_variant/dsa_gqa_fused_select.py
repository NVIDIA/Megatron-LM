# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exact top-k selection for DSA-over-GQA inference without a [rows, context] score buffer.

At long context, scoring every key of every row into a buffer and selecting from it costs a full
write and read of that buffer. Instead, per step:

1. Sample: each row scores ``SAMPLES_PER_RUN`` keys per run of ``SAMPLE_RUN`` keys (at fixed
   pseudo-random offsets) and takes the ``RANK``-th largest sampled score as its threshold.
2. Filter: one pass scores every key on tensor cores and appends those reaching the row's
   threshold to its candidate buffer of ``CAPACITY`` slots (``dsa_gqa_filter_kernels``).
3. Select: DeepSelect takes each row's top-k among its candidates.

Sampling and both selections read only as far as each row's keys reach, so a step costs what its
real context lengths need even when its bound (a CUDA graph's) is far longer.
4. Fallback: a row whose candidates fell short of top-k, or overflowed its buffer, takes an
   exact radix select over all its scores. One launch covers all rows; unflagged rows exit at
   once.

The result equals the top-k of the full scores (up to ties). Every launch size depends only on
the padded rows, blocks, and the step's context bound, so the selection runs inside CUDA graphs.

Why the thresholds rarely miss: with sampling rate p = 3/128, the number of samples among a
row's m highest-scoring keys is about Poisson(m p). A row comes up short only if its top k - 1
keys hold RANK samples, and overflows only if its top CAPACITY keys hold fewer than RANK samples.
For k = 2048, RANK = 96, CAPACITY = 9216: P(short) ~ 7e-10 and P(overflow) ~ 2e-20 per row
(512 and 1024 have the same orders of magnitude).
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

try:
    import deep_select

    HAVE_DEEP_SELECT = True
except ImportError:
    deep_select = None
    HAVE_DEEP_SELECT = False

from megatron.core.transformer.experimental_attention_variant import dsa_gqa_filter_kernels

# Sampling rate (``_sample_position`` hard-codes it).
SAMPLE_RUN = 128
SAMPLES_PER_RUN = 3
# Per top-k: (RANK, CAPACITY); see the module docstring.
SELECTOR = {512: (39, 4608), 1024: (60, 6144), 2048: (96, 9216)}
# The fused selection pays a fixed cost (sampling, candidate top-k) that the materialized path
# does not; it wins once contexts reach about this many keys.
FUSED_MIN_CONTEXT = 65536
# Sampling: rows per block (the score blocks of build_row_metadata) and samples per tile.
SAMPLE_ROW_BLOCK = 32
SAMPLE_BLOCK = 64
# Filter programs to aim for per SM: the decode filter is light; the prefill filter fills an SM.
DECODE_FILTER_PROGRAMS_PER_SM = 8
PREFILL_FILTER_PROGRAMS_PER_SM = 1
# Exact fallback: keys per tile.
FALLBACK_BLOCK = 128


def fused_select_supported(
    device: torch.device, index_topk: int, index_dim: int, page_size: int
) -> bool:
    """Whether this device and model can take the fused selection."""
    return (
        HAVE_TRITON
        and HAVE_DEEP_SELECT
        and dsa_gqa_filter_kernels.HAVE_CUTE_DSL
        and index_topk in SELECTOR
        and index_dim == dsa_gqa_filter_kernels.INDEX_DIM
        and page_size % dsa_gqa_filter_kernels.BLOCK_KEYS == 0
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


if HAVE_TRITON:

    @triton.jit
    def _sample_position(s):
        # Sample s is the (s % 3)-th of run s // 3: one sample in each third of the run's 128
        # keys (43 + 43 + 42), at a data-independent pseudo-random offset.
        run = s // 3
        j = s % 3
        return run * 128 + j * 43 + (run * 7919 + j * 104729 + 12345) % 42

    @triton.jit(do_not_specialize=["stride_s", "width_samples"])
    def _sample_kernel(
        query,
        key_cache,
        block_table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        samples,
        stride_q,
        stride_k_page,
        stride_k_pos,
        stride_table,
        stride_s,
        width_samples,
        max_pages,
        num_pages,
        INDEX_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        worker = tl.program_id(0)
        block = tl.program_id(1)
        workers = tl.num_programs(0)
        num_rows = tl.load(block_rows + block)
        request = tl.load(block_requests + block).to(tl.int64)
        r = tl.arange(0, ROW_BLOCK)
        row_valid = r < num_rows
        rows = (tl.load(block_starts + block) + r).to(tl.int64)
        lengths = tl.load(row_lengths + rows, mask=row_valid, other=0)
        dims = tl.arange(0, INDEX_DIM)
        q = tl.load(
            query + rows[:, None] * stride_q + dims[None, :], mask=row_valid[:, None], other=0.0
        )
        longest = tl.max(lengths, axis=0)
        # Only runs the block's rows reach (none for an empty block).
        limit = tl.minimum(width_samples, (longest + 127) // 128 * 3)
        for start in range(worker * BLOCK_S, limit, workers * BLOCK_S):
            s = start + tl.arange(0, BLOCK_S)
            positions = _sample_position(s)
            logical_page = positions // PAGE_SIZE
            key_valid = (positions < longest) & (logical_page < max_pages)
            page = tl.load(
                block_table + request * stride_table + logical_page, mask=key_valid, other=-1
            )
            key_valid = key_valid & (page >= 0) & (page < num_pages)
            page = tl.where(key_valid, page, 0).to(tl.int64)
            k = tl.load(
                key_cache
                + page[:, None] * stride_k_page
                + (positions % PAGE_SIZE)[:, None] * stride_k_pos
                + dims[None, :],
                mask=key_valid[:, None],
                other=0.0,
            )
            score = tl.dot(q, tl.trans(k))
            in_row = (positions[None, :] < lengths[:, None]) & key_valid[None, :]
            score = tl.where(in_row, score, float("-inf"))
            tl.store(
                samples + rows[:, None] * stride_s + s[None, :],
                score,
                mask=row_valid[:, None] & (s[None, :] < width_samples),
            )

    @triton.jit
    def _orderable(score):
        # FP32 -> uint32 with the same order.
        bits = score.to(tl.uint32, bitcast=True)
        return tl.where((bits >> 31) == 1, ~bits, bits | 0x80000000)

    @triton.jit
    def _row_scores(
        query,
        key_cache,
        block_table,
        request,
        row,
        positions,
        length,
        stride_q,
        stride_k_page,
        stride_k_pos,
        stride_table,
        max_pages,
        num_pages,
        INDEX_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        dims = tl.arange(0, INDEX_DIM)
        q = tl.load(query + row.to(tl.int64) * stride_q + dims).to(tl.float32)
        logical_page = positions // PAGE_SIZE
        valid = (positions < length) & (logical_page < max_pages)
        page = tl.load(
            block_table + request.to(tl.int64) * stride_table + logical_page, mask=valid, other=-1
        )
        valid = valid & (page >= 0) & (page < num_pages)
        page = tl.where(valid, page, 0).to(tl.int64)
        k = tl.load(
            key_cache
            + page[:, None] * stride_k_page
            + (positions % PAGE_SIZE)[:, None] * stride_k_pos
            + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        return tl.sum(k.to(tl.float32) * q[None, :], axis=1), valid

    @triton.jit
    def _radix_level(
        query,
        key_cache,
        block_table,
        request,
        row,
        length,
        hist_row,
        prefix,
        need,
        stride_q,
        stride_k_page,
        stride_k_pos,
        stride_table,
        max_pages,
        num_pages,
        SHIFT_HI: tl.constexpr,
        SHIFT: tl.constexpr,
        BINS: tl.constexpr,
        INDEX_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One radix level: histogram this digit of the row's keys that match the prefix so
        far, then pick the digit holding the need-th largest key."""
        bins = tl.arange(0, BINS)
        tl.store(hist_row + bins, tl.zeros((BINS,), tl.int32))
        # The program's threads share this histogram row: order the zeroing before the
        # increments, and every increment before the read below.
        tl.debug_barrier()
        for start in range(0, length, BLOCK_T):
            positions = start + tl.arange(0, BLOCK_T)
            score, valid = _row_scores(
                query,
                key_cache,
                block_table,
                request,
                row,
                positions,
                length,
                stride_q,
                stride_k_page,
                stride_k_pos,
                stride_table,
                max_pages,
                num_pages,
                INDEX_DIM,
                PAGE_SIZE,
            )
            key = _orderable(score)
            match = valid
            if SHIFT_HI < 32:
                match = match & ((key >> SHIFT_HI) == (prefix >> SHIFT_HI))
            digit = ((key >> SHIFT) & (BINS - 1)).to(tl.int32)
            tl.atomic_add(hist_row + digit, 1, mask=match)
        tl.debug_barrier()
        counts = tl.load(hist_row + bins).to(tl.int64)
        at_or_above = tl.cumsum(counts, axis=0, reverse=True)
        chosen = tl.maximum(tl.sum((at_or_above >= need).to(tl.int64), axis=0) - 1, 0)
        need = need - tl.sum(tl.where(bins > chosen, counts, 0), axis=0)
        prefix = prefix | (chosen.to(tl.uint32) << SHIFT)
        return prefix, need

    @triton.jit(do_not_specialize=["stride_hist", "stride_out"])
    def _radix_fallback_kernel(
        query,
        key_cache,
        block_table,
        row_requests,
        row_lengths,
        flags,
        hist,
        out,
        stride_q,
        stride_k_page,
        stride_k_pos,
        stride_table,
        stride_hist,
        stride_out,
        max_pages,
        num_pages,
        TOPK: tl.constexpr,
        INDEX_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Exact top-k of one flagged row by a three-level radix select over its 32-bit score
        keys (11, 11 and 10 bits); unflagged rows exit at once."""
        row = tl.program_id(0)
        if tl.load(flags + row) != 0:
            length = tl.load(row_lengths + row)
            request = tl.maximum(tl.load(row_requests + row), 0)
            hist_row = hist + row.to(tl.int64) * stride_hist
            prefix = tl.zeros((), tl.uint32)
            need = tl.full((), TOPK, tl.int64)
            prefix, need = _radix_level(
                query,
                key_cache,
                block_table,
                request,
                row,
                length,
                hist_row,
                prefix,
                need,
                stride_q,
                stride_k_page,
                stride_k_pos,
                stride_table,
                max_pages,
                num_pages,
                32,
                21,
                2048,
                INDEX_DIM,
                PAGE_SIZE,
                BLOCK_T,
            )
            prefix, need = _radix_level(
                query,
                key_cache,
                block_table,
                request,
                row,
                length,
                hist_row,
                prefix,
                need,
                stride_q,
                stride_k_page,
                stride_k_pos,
                stride_table,
                max_pages,
                num_pages,
                21,
                10,
                2048,
                INDEX_DIM,
                PAGE_SIZE,
                BLOCK_T,
            )
            prefix, need = _radix_level(
                query,
                key_cache,
                block_table,
                request,
                row,
                length,
                hist_row,
                prefix,
                need,
                stride_q,
                stride_k_page,
                stride_k_pos,
                stride_table,
                max_pages,
                num_pages,
                10,
                0,
                1024,
                INDEX_DIM,
                PAGE_SIZE,
                BLOCK_T,
            )
            # Emit the keys above the k-th key, then `need` keys equal to it, in position order.
            above_total = TOPK - need
            above_seen = tl.zeros((), tl.int64)
            equal_seen = tl.zeros((), tl.int64)
            out_row = out + row.to(tl.int64) * stride_out
            for start in range(0, length, BLOCK_T):
                positions = start + tl.arange(0, BLOCK_T)
                score, valid = _row_scores(
                    query,
                    key_cache,
                    block_table,
                    request,
                    row,
                    positions,
                    length,
                    stride_q,
                    stride_k_page,
                    stride_k_pos,
                    stride_table,
                    max_pages,
                    num_pages,
                    INDEX_DIM,
                    PAGE_SIZE,
                )
                key = _orderable(score)
                above = (valid & (key > prefix)).to(tl.int64)
                equal = (valid & (key == prefix)).to(tl.int64)
                above_slot = above_seen + tl.cumsum(above, axis=0) - 1
                equal_slot = equal_seen + tl.cumsum(equal, axis=0) - 1
                tl.store(out_row + above_slot, positions, mask=above != 0)
                tl.store(
                    out_row + above_total + equal_slot,
                    positions,
                    mask=(equal != 0) & (equal_slot < need),
                )
                above_seen += tl.sum(above, axis=0)
                equal_seen += tl.sum(equal, axis=0)

    @triton.jit(do_not_specialize=["stride_pick", "stride_out", "capacity"])
    def _assemble_kernel(
        pick,
        cand_idx,
        fallback,
        row_lengths,
        flags,
        out,
        stride_pick,
        stride_out,
        capacity,
        TOPK: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Each row's top-k: all its keys when it has at most top-k, the fallback when flagged,
        else the candidates DeepSelect picked."""
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        slots = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        in_k = slots < TOPK
        length = tl.load(row_lengths + row)
        flagged = tl.load(flags + row) != 0
        base = row.to(tl.int64)
        if length <= TOPK:
            result = tl.where(slots < length, slots, -1)
        elif flagged:
            result = tl.load(fallback + base * stride_out + slots, mask=in_k, other=-1)
        else:
            p = tl.load(pick + base * stride_pick + slots, mask=in_k, other=-1)
            usable = (p >= 0) & (p < capacity)
            result = tl.load(
                cand_idx + base * capacity + tl.where(usable, p, 0), mask=in_k & usable, other=-1
            )
        tl.store(out + base * stride_out + slots, result, mask=in_k)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@torch.no_grad()
def fused_select_topk(
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_requests: torch.Tensor,
    row_lengths: torch.Tensor,
    score_blocks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    filter_blocks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    width: int,
    index_topk: int,
    decode_only: bool,
) -> torch.Tensor:
    """Each row's exact top-k key positions, ``[rows,index_topk]`` int32, -1 padded.

    Args:
        index_query: BF16 indexer queries ``[rows,Di]``, contiguous.
        index_key_cache: Paged BF16 indexer keys ``[pages,page_size,Di]``, contiguous.
        block_table: Int32 ``[requests,max_pages]``.
        row_requests: Int32 ``[rows]`` block-table row of each row; -1 for padding.
        row_lengths: Int32 ``[rows]`` causal key count of each row; 0 for padding.
        score_blocks: ``(starts, rows, requests)`` of blocks of at most ``SAMPLE_ROW_BLOCK``
            rows (``build_row_metadata``); a decode-only step's blocks are single rows.
        filter_blocks: The same with at most ``PREFILL_ROW_BLOCK`` rows, for prefill steps.
        width: Bound on every row length (the step's context bound).
        index_topk: Keys each row selects (a key of ``SELECTOR``).
        decode_only: Every row is its request's only query token this step.
    """
    rank, capacity = SELECTOR[index_topk]
    rows = index_query.size(0)
    device = index_query.device
    sms = _num_sms(device.index)
    page_size = index_key_cache.size(1)
    common = (
        index_query.stride(0),
        index_key_cache.stride(0),
        index_key_cache.stride(1),
        block_table.stride(0),
    )

    # 1. Sample, and threshold each row at its rank-th largest sample.
    width_samples = -(-width // SAMPLE_RUN) * SAMPLES_PER_RUN
    alignment = deep_select.get_stride_requirement()[0] // torch.float32.itemsize
    sample_width = -(-max(width_samples, rank) // alignment) * alignment
    # Each row's samples cover the runs its keys reach; DeepSelect reads no further.
    samples = torch.empty((rows, sample_width), device=device)
    sample_ends = ((row_lengths + SAMPLE_RUN - 1) // SAMPLE_RUN * SAMPLES_PER_RUN).to(torch.int32)
    # A CUDA graph's padded batch has many empty blocks (they exit at once), so programs per
    # block follow the blocks the step's tokens can fill, not the padded block count.
    # (A decode row is a block of its own.)
    num_score_blocks = score_blocks[0].numel()
    filled_blocks = rows if decode_only else -(-rows // SAMPLE_ROW_BLOCK)
    workers = max(1, min(-(-width_samples // SAMPLE_BLOCK), 8 * sms // max(1, filled_blocks)))
    _sample_kernel[(workers, num_score_blocks)](
        index_query,
        index_key_cache,
        block_table,
        row_lengths,
        *score_blocks,
        samples,
        *common,
        samples.stride(0),
        width_samples,
        block_table.size(1),
        index_key_cache.size(0),
        INDEX_DIM=index_query.size(1),
        PAGE_SIZE=page_size,
        ROW_BLOCK=SAMPLE_ROW_BLOCK,
        BLOCK_S=SAMPLE_BLOCK,
        num_warps=4,
    )
    # A row with fewer than rank samples gets -inf fill values, so it keeps every key.
    values, _ = deep_select.topk(
        samples, rank, end=sample_ends, indices_type=torch.int32, return_value=True
    )
    thresholds = values.min(dim=-1).values.contiguous()

    # 2. Filter every key against its row's threshold into the candidate buffers.
    counts = torch.zeros(rows, device=device, dtype=torch.int32)
    cand_idx = torch.empty(rows, capacity, device=device, dtype=torch.int32)
    cand_val = torch.empty(rows, capacity, device=device)
    blocks = score_blocks if decode_only else filter_blocks
    per_sm = DECODE_FILTER_PROGRAMS_PER_SM if decode_only else PREFILL_FILTER_PROGRAMS_PER_SM
    row_block = 1 if decode_only else dsa_gqa_filter_kernels.PREFILL_ROW_BLOCK
    workers = max(
        1,
        min(
            -(-width // dsa_gqa_filter_kernels.BLOCK_KEYS),
            per_sm * sms // max(1, -(-rows // row_block)),
        ),
    )
    dsa_gqa_filter_kernels.filter_candidates(
        index_query,
        index_key_cache,
        block_table,
        row_lengths,
        thresholds,
        blocks,
        counts,
        cand_idx,
        cand_val,
        topk=index_topk,
        workers=workers,
        prefill=not decode_only,
    )

    # 3. Exact top-k among the candidates (unfilled slots are -inf).
    pick = deep_select.topk(
        cand_val,
        index_topk,
        end=counts.clamp(max=capacity),
        indices_type=torch.int32,
        return_value=False,
        idx_oob_fill_value=-1,
    )[1]

    # 4. Rows the threshold missed take the exact fallback; then assemble every row's result.
    flags = ((row_lengths > index_topk) & ((counts < index_topk) | (counts > capacity))).to(
        torch.int32
    )
    fallback = torch.full((rows, index_topk), -1, device=device, dtype=torch.int32)
    hist = torch.empty(rows, 2048, device=device, dtype=torch.int32)
    _radix_fallback_kernel[(rows,)](
        index_query,
        index_key_cache,
        block_table,
        row_requests,
        row_lengths,
        flags,
        hist,
        fallback,
        *common,
        hist.stride(0),
        fallback.stride(0),
        block_table.size(1),
        index_key_cache.size(0),
        TOPK=index_topk,
        INDEX_DIM=index_query.size(1),
        PAGE_SIZE=page_size,
        BLOCK_T=FALLBACK_BLOCK,
    )
    out = torch.empty(rows, index_topk, device=device, dtype=torch.int32)
    _assemble_kernel[(rows, triton.cdiv(index_topk, 1024))](
        pick,
        cand_idx,
        fallback,
        row_lengths,
        flags,
        out,
        pick.stride(0),
        out.stride(0),
        capacity,
        TOPK=index_topk,
        BLOCK_K=1024,
    )
    return out
