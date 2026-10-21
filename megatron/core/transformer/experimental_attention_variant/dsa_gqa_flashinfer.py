# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA-over-GQA inference through FlashInfer's DSA (SDSA) kernels.

The same kernels and plan as vLLM's DSA deployment, on Megatron's paged caches:

- **Decode rows** (single-token requests): dense BF16 indexer scores
  (``nvfp4_paged_mqa_logits``), exact top-k (gvr_2), then trtllm-gen token-sparse attention.
- **Prefill rows**, in 128-row tiles of one request each: a threshold from sampled scores, one
  fused sweep that keeps the keys above it (``sdsa_select``, with an exact in-launch repair), the
  exact top-k among them, then FA4 attention over each 8-token group's union of selected keys
  with a per-token bitmask (the "union" core; at every context length measured, 4K prompts to
  2K-token chunks on 1M keys, it beat attending per token).

Megatron's caches are used as they are: the ``[blocks, block_size, Di]`` indexer cache is viewed
as FlashInfer's 64-key pages (a derived block table), and the ``[blocks, block_size, 1, D]`` K/V
caches as page-size-1 slots. The per-step plan (``build_step_plan``) is built on the CPU from
the step's request lengths with sizes fixed by the padded batch, so every launch has a static
shape and the whole path can run inside a CUDA graph.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

try:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.utils import ssa_to_scalar
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.sdsa import (
        group_union_masks,
        nvfp4_paged_mqa_logits,
        sdsa_gather_sample,
        sdsa_sample_threshold,
        sdsa_select,
        token_sparse_kv_indices,
    )
    from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen, workspace_bytes
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    HAVE_FLASHINFER_DSA = True
except ImportError:
    HAVE_FLASHINFER_DSA = False

try:
    import deep_select

    HAVE_DEEP_SELECT = True
except ImportError:
    deep_select = None
    HAVE_DEEP_SELECT = False

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# FlashInfer's indexer page, and prefill query tiles.
FI_PAGE = 64
TILE_ROWS = 128
# Threshold samples: SAMPLES keys at random offsets into each run of PAGES pages; a row's
# threshold is its RANK-th largest sampled score, and it has CAPACITY candidate slots after its
# top-k tie slots. A row the threshold misses is repaired exactly inside the select launch.
SAMPLES, PAGES = 3, 2
SELECTOR = {512: (39, 4608), 1024: (60, 6144), 2048: (96, 9216)}
# trtllm-gen token-sparse attention workspace, and the most attention rows it is sized for.
CORE_WORKSPACE_BYTES = 128 << 20
MAX_CORE_ROWS = 65535


def union_group(heads: int) -> int:
    """Tokens per union-core group: the most whose query heads fit FA4's 128-row tile."""
    return 1 << ((TILE_ROWS // heads).bit_length() - 1)


def flashinfer_dsa_supported(
    device: torch.device, index_topk: int, index_dim: int, page_size: int, heads: int
) -> bool:
    """Whether this device and model can take the FlashInfer DSA path."""
    return (
        HAVE_FLASHINFER_DSA
        and HAVE_DEEP_SELECT
        and HAVE_TRITON
        and index_topk in SELECTOR
        and index_dim == 128
        and page_size % FI_PAGE == 0
        and heads <= TILE_ROWS
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


@dataclass
class StepPlan:
    """One step's FlashInfer DSA plan (device tensors) and its static launch sizes."""

    decode_rows: int  # rows [0, decode_rows) may take the decode path
    decode_requests: torch.Tensor  # [decode_rows] block-table row of each decode-path row
    decode_lengths: torch.Tensor  # [decode_rows] its keys, 0 for rows not on the decode path
    num_tiles: int  # prefill query tiles (0: no prefill path)
    tile_requests: torch.Tensor  # [tiles] block-table row of each tile
    tile_ordinals: torch.Tensor  # [tiles] the tile's prefill request ordinal
    row_ends: torch.Tensor  # [tiles * 128] causal key count of each tiled row (0: none)
    query_rows: torch.Tensor  # [tiles * 128] the step row of each tiled row (-1: none)
    request_rows: torch.Tensor  # [prefill requests] block-table row of each prefill request
    sample_ends: torch.Tensor  # [prefill requests] sampled keys of each prefill request
    batch_starts: torch.Tensor  # [tiles * 128 / group + 1] first step row of each group
    max_context: int  # bound on every row length (the graph's bound under CUDA graphs)


def plan_sizes(padded_rows: int, padded_requests: int, decode_only: bool) -> Dict[str, int]:
    """Static plan sizes for a padded batch: decode-path rows, prefill tiles and requests."""
    if decode_only:
        return {"decode_rows": padded_rows, "tiles": 0, "prefill_requests": 0}
    return {
        "decode_rows": min(padded_rows, padded_requests),
        "tiles": -(-padded_rows // TILE_ROWS) + padded_requests,
        "prefill_requests": padded_requests,
    }


def build_step_plan(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    *,
    padded_rows: int,
    padded_requests: int,
    decode_only: bool,
    heads: int,
) -> Dict[str, np.ndarray]:
    """The plan arrays (int32, CPU) of a step whose request ``i`` adds ``query_lengths[i]`` rows,
    in order, the last at key ``kv_lengths[i] - 1``; sizes depend only on the padded batch.

    The leading single-token requests whose rows are among the first ``decode_rows`` rows take
    the decode path; every other request's rows go into prefill tiles.
    """
    sizes = plan_sizes(padded_rows, padded_requests, decode_only)
    group = union_group(heads)
    q_len = np.asarray(query_lengths, dtype=np.int64)
    kv_len = np.asarray(kv_lengths, dtype=np.int64)
    first_row = np.cumsum(q_len) - q_len
    # The decode path takes the leading single-token requests (the dynamic context orders
    # decode requests first), so the prefill rows stay consecutive for the union core.
    multi = np.flatnonzero(q_len > 1)
    leading = np.arange(len(q_len)) < (multi[0] if len(multi) else len(q_len))
    on_decode = leading & (first_row < sizes["decode_rows"])
    decode_requests = np.zeros(sizes["decode_rows"], np.int32)
    decode_lengths = np.zeros(sizes["decode_rows"], np.int32)
    decode_ids = np.flatnonzero(on_decode)
    decode_requests[first_row[decode_ids]] = decode_ids
    decode_lengths[first_row[decode_ids]] = kv_len[decode_ids]
    if decode_only:
        assert on_decode.sum() == len(q_len), "A decode-only step holds single-token requests."
    tiles, prefill_requests = sizes["tiles"], sizes["prefill_requests"]
    prefill = np.flatnonzero(~on_decode)
    assert len(prefill) <= prefill_requests
    request_tiles = (q_len[prefill] + TILE_ROWS - 1) // TILE_ROWS
    assert request_tiles.sum() <= tiles
    tile_ordinal = np.repeat(np.arange(len(prefill)), request_tiles)
    tile_start = TILE_ROWS * (
        np.arange(len(tile_ordinal))
        - np.repeat(np.cumsum(request_tiles) - request_tiles, request_tiles)
    )
    request = prefill[tile_ordinal]
    count = np.minimum(TILE_ROWS, q_len[request] - tile_start)
    lane = np.arange(TILE_ROWS)
    live = lane < count[:, None]
    row_ends = np.zeros((tiles, TILE_ROWS), np.int32)
    query_rows = np.full((tiles, TILE_ROWS), -1, np.int32)
    real = len(tile_ordinal)
    row_ends[:real] = np.where(
        live, (kv_len - q_len)[request][:, None] + tile_start[:, None] + lane + 1, 0
    )
    query_rows[:real] = np.where(live, first_row[request][:, None] + tile_start[:, None] + lane, -1)
    tile_requests = np.zeros(tiles, np.int32)
    tile_requests[:real] = request
    tile_ordinals = np.zeros(tiles, np.int32)
    tile_ordinals[:real] = tile_ordinal
    request_rows = np.zeros(prefill_requests, np.int32)
    request_rows[: len(prefill)] = prefill
    sample_ends = np.zeros(prefill_requests, np.int32)
    sample_ends[: len(prefill)] = kv_len[prefill] // (PAGES * FI_PAGE) * SAMPLES
    # Groups of `group` tiled rows: each is a union-core batch over its live rows, which are
    # consecutive step rows; an empty group is an empty batch at the previous one's end.
    group_rows = query_rows.reshape(-1, group)
    group_live = (group_rows >= 0).sum(1)
    live_groups = group_live > 0
    ends = np.where(live_groups, group_rows[:, 0] + group_live, 0)
    first = int(group_rows[live_groups, 0].min()) if live_groups.any() else 0
    # An empty group starts (and ends) where the last live group before it ended.
    carried = np.maximum(np.maximum.accumulate(ends), first)
    previous = np.concatenate(([first], carried[:-1]))
    batch_starts = np.empty(len(ends) + 1, np.int32)
    batch_starts[:-1] = np.where(live_groups, group_rows[:, 0], previous)
    batch_starts[-1] = carried[-1] if len(carried) else first
    return {
        "decode_requests": decode_requests,
        "decode_lengths": decode_lengths,
        "tile_requests": tile_requests,
        "tile_ordinals": tile_ordinals,
        "row_ends": row_ends.ravel(),
        "query_rows": query_rows.ravel(),
        "request_rows": request_rows,
        "sample_ends": sample_ends,
        "batch_starts": batch_starts,
    }


if HAVE_FLASHINFER_DSA:

    @cute.jit
    def _union_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        """Bits of union columns kv_idx .. kv_idx + 127 for query q_idx of its batch, from that
        query's row of the bitmask aux_tensors[0] (FA4's tile padding rows read the batch's
        last row)."""
        q = cutlass.min(ssa_to_scalar(q_idx), seqlen_info.seqlen_q - 1)
        row = aux_tensors[0][seqlen_info.offset_q + q, None]
        words = cute.make_rmem_tensor(4, cutlass.Int32)
        cute.autovec_copy(cute.local_tile(row, (4,), (ssa_to_scalar(kv_idx) >> 7,)), words)
        return cute.recast_tensor(words, cutlass.Uint32).load()

    _union_mask_mod.__vec_size__ = 128


if HAVE_TRITON:

    @triton.jit(do_not_specialize=["rows"])
    def _candidate_scores_kernel(
        candidates, ranges, out, stride_c, stride_o, rows, BLOCK: tl.constexpr
    ):
        """out[r, j] = the score of candidate slot starts[r] + j for j < stop - start, else -inf
        (candidates hold (key, score bits) pairs; ranges is [2, rows])."""
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        start = tl.load(ranges + row)
        stop = tl.load(ranges + rows + row)
        j = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = j < stop - start
        bits = tl.load(
            candidates + row.to(tl.int64) * stride_c + (start + j) * 2 + 1, mask=valid, other=0
        )
        score = tl.where(valid, bits.to(tl.float32, bitcast=True), float("-inf"))
        tl.store(out + row.to(tl.int64) * stride_o + j, score, mask=j < stride_o)


@functools.cache
def _persistent(device_index: int, heads: int, max_tiles_hint: int):
    """Per device: trtllm-gen workspace and split-KV counters (left zeroed by every call),
    gvr_2's workspace, sdsa_select's arrival counts (zeroed once, left zeroed), and the fixed
    random sample offsets."""
    device = torch.device("cuda", device_index)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    core = torch.zeros(CORE_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    counters = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(MAX_CORE_ROWS, heads, sms),
        dtype=torch.uint8,
        device=device,
    )
    topk = torch.zeros(workspace_bytes() // 4, dtype=torch.int32, device=device)
    arrivals = torch.zeros(max_tiles_hint, dtype=torch.int32, device=device)
    generator = torch.Generator(device).manual_seed(0x5D5A)
    offsets = torch.randint(
        PAGES * FI_PAGE, (1 << 20,), generator=generator, dtype=torch.uint8, device=device
    )
    return core, counters, topk, arrivals, offsets


def fi_block_table(block_table: torch.Tensor, page_size: int, min_pages: int) -> torch.Tensor:
    """[requests, blocks] of page_size-key blocks -> FlashInfer's [requests, pages] of 64-key
    pages, padded to cover at least min_pages (entries past a row's length are never read)."""
    per = page_size // FI_PAGE
    table = (
        block_table.to(torch.int32)[:, :, None] * per
        + torch.arange(per, device=block_table.device, dtype=torch.int32)
    ).reshape(block_table.size(0), -1)
    return torch.nn.functional.pad(table, (0, max(0, min_pages - table.size(1))))


def _token_core(query, keys, requests, table, key_rows, value_rows, out, index_topk, scale, heads):
    """Each row of `query` attends to its selected key positions `keys` (-1: none) of request
    `requests[row]` with trtllm-gen token-sparse attention (page size 1)."""
    core_ws, counters, _, _, _ = _persistent(query.device.index, heads, 1)
    rows = query.size(0)
    slot_tables = torch.empty(rows, 2, index_topk, device=query.device, dtype=torch.int32)
    lengths = torch.empty(rows, device=query.device, dtype=torch.int32)
    token_sparse_kv_indices(
        keys, requests, table, block_tables=slot_tables, seq_lens=lengths, requests_per_token=1
    )
    trtllm_batch_decode_with_kv_cache(
        query,
        (key_rows, value_rows),
        core_ws,
        slot_tables,
        lengths,
        index_topk,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        out=out,
        backend="trtllm-gen",
        uses_shared_paged_kv_idx=False,
        multi_ctas_kv_counter_buffer=counters,
        sparse_top_k=index_topk,
    )
    return lengths


@torch.no_grad()
def flashinfer_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    plan: StepPlan,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Attention output ``[rows,Hq,Dv]`` of every step row through FlashInfer's DSA kernels;
    rows with no keys (padding) are zero.

    Args:
        query: BF16 ``[rows,Hq,D]``.
        key_cache, value_cache: Paged BF16 ``[blocks,block_size,1,D]``.
        index_query: BF16 indexer queries ``[rows,Di]``.
        index_key_cache: Paged BF16 indexer keys ``[blocks,block_size,Di]``.
        block_table: Int32 ``[requests,max_blocks]``.
        plan: The step's plan (``StepPlan``).
        index_topk: Keys each row selects.
        softmax_scale: Attention softmax scale.
    """
    device = query.device
    rows, heads, dim = query.shape
    value_dim = value_cache.size(-1)
    page_size = key_cache.size(1)
    max_context = plan.max_context
    key_tiles = -(-max_context // TILE_ROWS)
    width = -(-max(max_context, index_topk) // 4) * 4
    table = fi_block_table(block_table, page_size, max(-(-width // FI_PAGE), 2 * key_tiles))
    pages = index_key_cache.view(-1, FI_PAGE, index_key_cache.size(-1))
    key_rows = key_cache.view(-1, 1, 1, dim)
    value_rows = value_cache.view(-1, 1, 1, value_dim)
    index_query = index_query.contiguous()
    query = query.contiguous()
    _, _, topk_ws, arrivals, offsets = _persistent(device.index, heads, 4096)
    out = torch.zeros(rows, heads, value_dim, device=device, dtype=query.dtype)

    if plan.num_tiles:
        rank, capacity = SELECTOR[index_topk]
        tiles = plan.num_tiles
        tiled = tiles * TILE_ROWS
        assert tiles <= arrivals.numel()
        query_rows = plan.query_rows
        packed = (
            index_query[query_rows.clamp(min=0).long()]
            * (query_rows >= 0).to(index_query.dtype)[:, None]
        )
        # Threshold samples of each prefill request's keys.
        sample_cap = TILE_ROWS * -(-max_context * SAMPLES // (TILE_ROWS * FI_PAGE * PAGES))
        num_requests = plan.request_rows.numel()
        sample_keys = torch.empty(
            num_requests, sample_cap, pages.size(-1), dtype=pages.dtype, device=device
        )
        sdsa_gather_sample(
            pages,
            None,
            table.index_select(0, plan.request_rows.long()),
            offsets[:sample_cap],
            plan.sample_ends,
            sample_keys=sample_keys,
            sample_scales=None,
            samples_per_run=SAMPLES,
            pages_per_run=PAGES,
        )
        sample_table = torch.arange(
            num_requests * sample_cap // FI_PAGE, dtype=torch.int32, device=device
        ).view(num_requests, -1)
        sample_row_ends = (plan.row_ends // (PAGES * FI_PAGE) * SAMPLES).to(torch.int32)
        sample_scores = torch.empty(tiled, sample_cap, device=device)
        nvfp4_paged_mqa_logits(
            packed,
            None,
            sample_keys.view(-1, FI_PAGE, pages.size(-1)),
            None,
            sample_table,
            sample_row_ends.view(-1, 8).amax(1),
            requests=plan.tile_ordinals.repeat_interleave(TILE_ROWS // 8),
            scores=sample_scores,
        )
        sample_top = torch.empty(tiled, rank, dtype=torch.int32, device=device)
        run_varlen(
            sample_scores,
            None,
            sample_row_ends,
            sample_top,
            next_n=1,
            max_seq_len=sample_cap,
            workspace=topk_ws,
            top_k=rank,
        )
        thresholds = torch.empty(tiled, device=device)
        counters = torch.empty(tiled, 2, dtype=torch.int32, device=device)
        sdsa_sample_threshold(sample_scores, sample_top, thresholds=thresholds, counters=counters)
        # Candidates and ties of every row in one sweep, repaired exactly inside the launch.
        ranges = torch.empty(2, tiled, dtype=torch.int32, device=device)
        slots = index_topk + capacity
        candidates = torch.empty(tiled, slots, 2, dtype=torch.int32, device=device)
        sdsa_select(
            packed,
            None,
            pages,
            None,
            plan.row_ends,
            plan.tile_requests,
            table,
            thresholds=thresholds,
            counters=counters,
            arrivals=arrivals[:tiles],
            ranges=ranges,
            candidates=candidates,
            key_tiles=key_tiles,
            ties=index_topk,
        )
        # The exact top-k among each row's slots, as positions from its range start.
        score_width = -(-slots // 256) * 256
        candidate_scores = torch.empty(tiled, score_width, device=device)
        _candidate_scores_kernel[(tiled, triton.cdiv(score_width, 1024))](
            candidates,
            ranges,
            candidate_scores,
            candidates.stride(0),
            candidate_scores.stride(0),
            tiled,
            BLOCK=1024,
        )
        positions = deep_select.topk(
            candidate_scores,
            index_topk,
            end=(ranges[1] - ranges[0]).contiguous(),
            indices_type=torch.int32,
            return_value=False,
            idx_oob_fill_value=-1,
        )[1].contiguous()
        starts = ranges[0].contiguous()
        group = union_group(heads)
        groups = tiled // group
        columns = -(-min(group * index_topk, TILE_ROWS * key_tiles) // 128) * 128
        unions = torch.zeros(groups, columns, dtype=torch.int32, device=device)
        union_sizes = torch.zeros(groups, dtype=torch.int32, device=device)
        mask = torch.zeros(rows, columns // 32, dtype=torch.int32, device=device)
        group_union_masks(
            positions,
            candidates,
            starts,
            query_rows,
            torch.arange(groups, dtype=torch.int32, device=device),
            plan.tile_requests,
            table,
            union=unions,
            union_sizes=union_sizes,
            mask=mask,
            key_tiles=key_tiles,
        )
        _flash_attn_fwd(
            query,
            key_rows,
            value_rows,
            page_table=unions,
            seqused_k=union_sizes,
            softmax_scale=softmax_scale,
            out=out,
            cu_seqlens_q=plan.batch_starts,
            max_seqlen_q=group,
            mask_mod=_union_mask_mod,
            aux_tensors=[mask],
        )

    if plan.decode_rows:
        decode_rows = plan.decode_rows
        scores = torch.empty(decode_rows, width, device=device)
        nvfp4_paged_mqa_logits(
            index_query[:decode_rows],
            None,
            pages,
            None,
            table,
            plan.decode_lengths,
            requests=plan.decode_requests,
            scores=scores,
        )
        topk = torch.empty(decode_rows, index_topk, device=device, dtype=torch.int32)
        run_varlen(
            scores,
            None,
            plan.decode_lengths,
            topk,
            next_n=1,
            max_seq_len=width,
            workspace=topk_ws,
            top_k=index_topk,
        )
        decode_out = torch.empty(decode_rows, heads, value_dim, device=device, dtype=query.dtype)
        _token_core(
            query[:decode_rows],
            topk,
            plan.decode_requests,
            table,
            key_rows,
            value_rows,
            decode_out,
            index_topk,
            softmax_scale,
            heads,
        )
        on_decode = (plan.decode_lengths > 0)[:, None, None]
        out[:decode_rows] = torch.where(on_decode, decode_out, out[:decode_rows])
    return out
