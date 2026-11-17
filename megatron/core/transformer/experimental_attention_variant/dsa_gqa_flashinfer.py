# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA-over-GQA inference through FlashInfer's DSA (SDSA) kernels, on Megatron's paged caches.

Without FlashInfer's DSA kernels (``flashinfer.sdsa`` and ``flashinfer.token_sparse``) or FA4
this module still imports, and ``require_flashinfer_dsa`` raises when the path is requested.

- **Decode rows** (single-token requests): dense indexer scores (``indexer_scores``), exact top-k
  (gvr_2), then trtllm-gen token-sparse attention.
- **Prefill rows**, in 128-row tiles of one request each: a threshold from sampled scores, one
  fused sweep that keeps the keys above it (``indexer_threshold_select``, with an exact in-launch
  repair), the exact top-k among them (``top_k_from_candidates``), then FA4 attention over each
  8-token group's union of selected keys with a per-token bitmask (the "union" core; at every
  context length measured, 4K prompts to 2K-token chunks on 1M keys, it beat attending per
  token).

Megatron's caches are used as they are: the ``[blocks, block_size, Di]`` indexer cache is viewed
as FlashInfer's 64-key pages (a derived block table), and the ``[blocks, block_size, 1, D]`` K/V
caches as page-size-1 slots. The per-step plan (``build_step_plan``) is built on the CPU from
the step's request lengths with sizes fixed by the padded batch, so every launch has a static
shape and the whole path can run inside a CUDA graph.

Indexer keys are BF16 by default. With NVFP4 index keys (``dsa_indexer_key_dtype="nvfp4"``) the
cache holds, for every 64-key page, the keys' packed E2M1 values (64 bytes a key) followed by
their E4M3 scales (one per 16 values, 8 bytes a key), and ``quantize_write_index_qk_nvfp4``
quantizes a step's keys into it and its indexer queries into rows of the same format, so decode
and prefill selection both score NVFP4 queries against NVFP4 keys.
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
        gather_key_sample,
        indexer_scores,
        indexer_threshold_select,
        thresholds_from_sample,
        top_k_from_candidates,
    )
    from flashinfer.token_sparse import group_union_masks, token_sparse_kv_indices
    from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen, workspace_bytes
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    HAVE_FLASHINFER_DSA = True
except ImportError:
    HAVE_FLASHINFER_DSA = False

try:
    from flashinfer.sdsa import quantize_write_index_qk_nvfp4

    HAVE_FLASHINFER_DSA_NVFP4 = HAVE_FLASHINFER_DSA
except ImportError:
    HAVE_FLASHINFER_DSA_NVFP4 = False

# FlashInfer's indexer page, and prefill query tiles.
FI_PAGE = 64
TILE_ROWS = 128
# Threshold samples: SAMPLES keys at random offsets into each run of PAGES pages; a row's
# threshold is its RANK-th largest sampled score, and it has CAPACITY candidate slots after its
# top-k tie slots. A row the threshold misses is repaired exactly inside the select launch.
SAMPLES, PAGES = 3, 2
SELECTOR = {512: (39, 4608), 1024: (60, 6144), 2048: (96, 9216)}
# gvr_2's top-k launches one grid row per input row, and a grid's y dimension holds at most
# 65535. A prefill plan pads a tile per request, so a large request budget exceeds that; select
# in slabs of rows instead, as FlashInfer's own prefill top-k does.
TOPK_ROW_SLAB = 32768
# NVFP4 index keys of the 128-dim indexer: packed E2M1 bytes and E4M3 scales (one per 16 values)
# per key; a 64-key cache page holds its keys' bytes, then their scales.
INDEX_KEY_DTYPES = ("bf16", "nvfp4")
NVFP4_KEY_BYTES, NVFP4_SCALES = 64, 8
NVFP4_PAGE_BYTES = FI_PAGE * (NVFP4_KEY_BYTES + NVFP4_SCALES)
# NVFP4 scores take few distinct values, so keys often tie at a row's top-k boundary, and the
# selection kernels keep any of the tied keys, in an order that can vary between runs.
NVFP4_BATCH_INVARIANT_ERROR = (
    "NVFP4 DSA index keys do not support batch-invariant (deterministic) inference: NVFP4 "
    "scores often tie at the top-k boundary, and which tied key is selected can vary between "
    "runs. Use dsa_indexer_key_dtype='bf16'."
)
# trtllm-gen token-sparse attention workspace, and the most attention rows it is sized for.
CORE_WORKSPACE_BYTES = 128 << 20
MAX_CORE_ROWS = 65535


def union_group(heads: int) -> int:
    """Tokens per union-core group: the most whose query heads fit FA4's 128-row tile."""
    return 1 << ((TILE_ROWS // heads).bit_length() - 1)


def require_flashinfer_dsa(index_key_dtype: str = "bf16") -> None:
    """Raise unless the packages this path needs, with ``index_key_dtype`` index keys, are
    installed."""
    if index_key_dtype not in INDEX_KEY_DTYPES:
        raise ValueError(
            f"Unknown DSA indexer key dtype {index_key_dtype!r}; expected one of "
            f"{INDEX_KEY_DTYPES}."
        )
    missing = [
        name
        for name, present in (
            ("FlashInfer with DSA kernels (flashinfer.sdsa) and FA4", HAVE_FLASHINFER_DSA),
            (
                "FlashInfer's NVFP4 indexer kernels (quantize_write_index_qk_nvfp4)",
                HAVE_FLASHINFER_DSA_NVFP4 or index_key_dtype != "nvfp4",
            ),
        )
        if not present
    ]
    if missing:
        raise ImportError(
            "DSA-over-GQA inference with dsa_backend='flashinfer' needs " + ", ".join(missing) + "."
        )


def flashinfer_dsa_supported(
    device: torch.device, index_topk: int, index_dim: int, page_size: int, heads: int
) -> bool:
    """Whether this device and model shape can take the FlashInfer DSA path (packages aside)."""
    return (
        index_topk in SELECTOR
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
    page_table: torch.Tensor  # [requests, pages] FlashInfer's 64-key pages (fill_page_table)


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


@functools.cache
def _persistent(device_index: int, heads: int, max_tiles_hint: int):
    """Per device: trtllm-gen workspace and split-KV counters (left zeroed by every call),
    gvr_2's workspace, indexer_threshold_select's arrival counts (zeroed once, left zeroed), and
    the fixed random sample offsets."""
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


def page_table_columns(max_kv_blocks: int, page_size: int, max_context: int, index_topk: int):
    """Columns of the FlashInfer page table: the requests' 64-key pages, and at least the score
    width and twice the key tiles (``indexer_scores`` / ``indexer_threshold_select`` read it;
    entries past a row's length are never read)."""
    width = -(-max(max_context, index_topk) // 4) * 4
    key_tiles = -(-max_context // TILE_ROWS)
    return max(max_kv_blocks * (page_size // FI_PAGE), -(-width // FI_PAGE), 2 * key_tiles)


def fill_page_table(block_table: torch.Tensor, page_size: int, out: torch.Tensor) -> None:
    """Write FlashInfer's 64-key pages of [requests, blocks] page_size-key blocks into the leading
    columns of ``out`` (block b holds pages b * per .. b * per + per - 1), once per step."""
    per = page_size // FI_PAGE
    requests, blocks = block_table.shape
    pages = out[:requests, : blocks * per].view(requests, blocks, per)
    # One kernel: arange(per) + per * block.
    offsets = torch.arange(per, device=out.device, dtype=out.dtype)
    torch.add(offsets, block_table[:, :, None], alpha=per, out=pages)


def nvfp4_index_cache_shape(num_blocks: int, block_size: int) -> tuple:
    """Shape of one layer's NVFP4 index-key cache (uint8): each block's 64-key pages, every page
    its keys' packed values, then their scales."""
    if block_size % FI_PAGE:
        raise ValueError(
            f"NVFP4 DSA index keys need a KV block size that is a multiple of {FI_PAGE}, "
            f"got {block_size}."
        )
    return (num_blocks, block_size // FI_PAGE, NVFP4_PAGE_BYTES)


def nvfp4_index_pages(cache: torch.Tensor):
    """FlashInfer's views of an NVFP4 index-key cache (``nvfp4_index_cache_shape``): packed keys
    uint8 ``[pages, 64, 64]`` and their E4M3 scale bytes uint8 ``[pages, 64, 8]``."""
    pages = cache.view(-1, NVFP4_PAGE_BYTES)
    split = FI_PAGE * NVFP4_KEY_BYTES
    keys = pages[:, :split].unflatten(1, (FI_PAGE, NVFP4_KEY_BYTES))
    scales = pages[:, split:].unflatten(1, (FI_PAGE, NVFP4_SCALES))
    return keys, scales


def write_nvfp4_index_keys(
    key: torch.Tensor,
    query: torch.Tensor,
    cache: torch.Tensor,
    block_idx: torch.Tensor,
    positions: torch.Tensor,
):
    """Quantize a step's indexer keys ``[tokens, 128]`` into an NVFP4 index-key cache at
    (``block_idx[t]``, ``positions[t]``) (tokens with a negative block are skipped), and its
    indexer queries ``[rows, 128]`` into NVFP4 rows, in one launch.

    Returns:
        The packed queries, uint8 ``[rows, 64]``, and their E4M3 scale bytes, uint8 ``[rows, 8]``.
    """
    block_size = cache.size(1) * FI_PAGE
    keys, scales = nvfp4_index_pages(cache)
    block_idx = block_idx[: key.size(0)].long()
    slots = torch.where(
        block_idx >= 0, block_idx * block_size + positions[: key.size(0)].long(), -1
    )
    rows = query.size(0)
    packed_query = torch.empty(rows, NVFP4_KEY_BYTES, device=query.device, dtype=torch.uint8)
    query_scales = torch.empty(rows, NVFP4_SCALES, device=query.device, dtype=torch.uint8)
    quantize_write_index_qk_nvfp4(
        key.contiguous(),
        slots,
        query.contiguous(),
        torch.arange(rows, device=query.device, dtype=torch.int64),
        key_cache=keys,
        key_scales=scales,
        packed_query=packed_query,
        query_scales=query_scales,
    )
    return packed_query, query_scales


def _top_k_rows(scores, lengths, out, workspace, top_k):
    """Each row's ``top_k`` largest of its first ``lengths[row]`` scores (gvr_2), in slabs of
    at most ``TOPK_ROW_SLAB`` rows. Rows are independent, so slabbing doesn't change the result."""
    for start in range(0, scores.size(0), TOPK_ROW_SLAB):
        end = start + TOPK_ROW_SLAB
        run_varlen(
            scores[start:end],
            None,
            lengths[start:end],
            out[start:end],
            next_n=1,
            max_seq_len=scores.size(1),
            workspace=workspace,
            top_k=top_k,
        )


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
    plan: StepPlan,
    index_topk: int,
    softmax_scale: float,
    deterministic: bool = False,
    index_query_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention output ``[rows,Hq,Dv]`` of every step row through FlashInfer's DSA kernels;
    rows with no keys (padding) are zero.

    Args:
        query: BF16 ``[rows,Hq,D]``.
        key_cache, value_cache: Paged BF16 ``[blocks,block_size,1,D]``.
        index_query: Indexer queries: BF16 ``[rows,Di]``, or with ``index_query_scales`` NVFP4
            packed uint8 ``[rows,Di/2]`` (``write_nvfp4_index_keys``).
        index_key_cache: Paged indexer keys: BF16 ``[blocks,block_size,Di]``, or with NVFP4
            queries the uint8 NVFP4 cache (``nvfp4_index_cache_shape``).
        plan: The step's plan (``StepPlan``), whose page table maps the requests' KV blocks.
        index_topk: Keys each row selects.
        softmax_scale: Attention softmax scale.
        deterministic: Sort each decode row's selected keys so that decode replays bit-exactly
            (the prefill path already does).
        index_query_scales: E4M3 scale bytes ``[rows,Di/16]`` of NVFP4 indexer queries; None for
            BF16 queries and keys.
    """
    if deterministic and index_query_scales is not None:
        raise ValueError(NVFP4_BATCH_INVARIANT_ERROR)
    device = query.device
    rows, heads, dim = query.shape
    value_dim = value_cache.size(-1)
    max_context = plan.max_context
    key_tiles = -(-max_context // TILE_ROWS)
    width = -(-max(max_context, index_topk) // 4) * 4
    table = plan.page_table
    assert table.size(1) >= max(-(-width // FI_PAGE), 2 * key_tiles), "page table too narrow"
    nvfp4 = index_query_scales is not None
    if nvfp4:
        pages, key_scales = nvfp4_index_pages(index_key_cache)
        index_query_scales = index_query_scales.contiguous()
    else:
        pages, key_scales = index_key_cache.view(-1, FI_PAGE, index_key_cache.size(-1)), None
    key_rows = key_cache.view(-1, 1, 1, dim)
    value_rows = value_cache.view(-1, 1, 1, value_dim)
    index_query = index_query.contiguous()
    query = query.contiguous()
    _, _, topk_ws, arrivals, offsets = _persistent(device.index, heads, 4096)
    out = (
        torch.zeros(rows, heads, value_dim, device=device, dtype=query.dtype)
        if plan.num_tiles
        else None
    )

    if plan.num_tiles:
        rank, capacity = SELECTOR[index_topk]
        tiles = plan.num_tiles
        tiled = tiles * TILE_ROWS
        assert tiles <= arrivals.numel()
        query_rows = plan.query_rows
        live_rows = query_rows.clamp(min=0).long()
        packed = index_query[live_rows] * (query_rows >= 0).to(index_query.dtype)[:, None]
        packed_scales = sample_scales = None
        if nvfp4:
            # Padding rows: zero values and zero scales, as BF16 padding rows are zero.
            packed_scales = (
                index_query_scales[live_rows] * (query_rows >= 0).to(torch.uint8)[:, None]
            )
        # Threshold samples of each prefill request's keys.
        sample_cap = TILE_ROWS * -(-max_context * SAMPLES // (TILE_ROWS * FI_PAGE * PAGES))
        num_requests = plan.request_rows.numel()
        sample_keys = torch.empty(
            num_requests, sample_cap, pages.size(-1), dtype=pages.dtype, device=device
        )
        if nvfp4:
            sample_scales = torch.empty(
                num_requests, sample_cap, NVFP4_SCALES, dtype=torch.uint8, device=device
            )
        gather_key_sample(
            pages,
            table.index_select(0, plan.request_rows.long()),
            offsets[:sample_cap],
            plan.sample_ends,
            sample_keys=sample_keys,
            samples_per_run=SAMPLES,
            pages_per_run=PAGES,
            key_scales=key_scales,
            sample_scales=sample_scales,
        )
        sample_table = torch.arange(
            num_requests * sample_cap // FI_PAGE, dtype=torch.int32, device=device
        ).view(num_requests, -1)
        sample_row_ends = (plan.row_ends // (PAGES * FI_PAGE) * SAMPLES).to(torch.int32)
        sample_scores = torch.empty(tiled, sample_cap, device=device)
        indexer_scores(
            packed,
            sample_keys.view(-1, FI_PAGE, pages.size(-1)),
            sample_table,
            sample_row_ends.view(-1, 8).amax(1),
            requests=plan.tile_ordinals.repeat_interleave(TILE_ROWS // 8),
            scores=sample_scores,
            query_scales=packed_scales,
            key_scales=(
                None if sample_scales is None else sample_scales.view(-1, FI_PAGE, NVFP4_SCALES)
            ),
        )
        sample_top = torch.empty(tiled, rank, dtype=torch.int32, device=device)
        _top_k_rows(sample_scores, sample_row_ends, sample_top, topk_ws, rank)
        thresholds = torch.empty(tiled, device=device)
        counters = torch.empty(tiled, 2, dtype=torch.int32, device=device)
        thresholds_from_sample(sample_scores, sample_top, thresholds=thresholds, counters=counters)
        # Candidates and ties of every row in one sweep, repaired exactly inside the launch.
        ranges = torch.empty(2, tiled, dtype=torch.int32, device=device)
        slots = index_topk + capacity
        candidates = torch.empty(tiled, slots, 2, dtype=torch.int32, device=device)
        indexer_threshold_select(
            packed,
            pages,
            plan.row_ends,
            plan.tile_requests,
            table,
            thresholds=thresholds,
            counters=counters,
            candidates=candidates,
            max_context=max_context,
            top_k=index_topk,
            arrivals=arrivals[:tiles],
            ranges=ranges,
            query_scales=packed_scales,
            key_scales=key_scales,
        )
        # The keys of the exact top-k among each row's candidates and ties (-1 last).
        positions = torch.empty(tiled, index_topk, dtype=torch.int32, device=device)
        top_k_from_candidates(candidates, ranges, top_keys=positions)
        group = union_group(heads)
        groups = tiled // group
        columns = -(-min(group * index_topk, TILE_ROWS * key_tiles) // 128) * 128
        unions = torch.zeros(groups, columns, dtype=torch.int32, device=device)
        union_sizes = torch.zeros(groups, dtype=torch.int32, device=device)
        mask = torch.zeros(rows, columns // 32, dtype=torch.int32, device=device)
        group_union_masks(
            positions,
            query_rows,
            torch.arange(groups, dtype=torch.int32, device=device),
            plan.tile_requests.repeat_interleave(TILE_ROWS // group),
            table,
            union=unions,
            union_sizes=union_sizes,
            mask=mask,
            num_keys=max_context,
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
        indexer_scores(
            index_query[:decode_rows],
            pages,
            table,
            plan.decode_lengths,
            requests=plan.decode_requests,
            scores=scores,
            query_scales=None if index_query_scales is None else index_query_scales[:decode_rows],
            key_scales=key_scales,
        )
        topk = torch.empty(decode_rows, index_topk, device=device, dtype=torch.int32)
        _top_k_rows(scores, plan.decode_lengths, topk, topk_ws, index_topk)
        if deterministic:
            # gvr_2 returns the selected keys in an order that varies between runs, and
            # attention sums in that order; sorting (descending, so -1 padding stays last)
            # makes decode replay bit-exactly.
            topk = topk.sort(dim=-1, descending=True).values
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
        if plan.num_tiles:
            out[:decode_rows] = torch.where(on_decode, decode_out, out[:decode_rows])
        else:
            # Decode-only: every row is a decode row or padding (no keys -> zeros).
            out = decode_out.mul_(on_decode)
    return out


@torch.no_grad()
def dynamic_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    inference_context,
    index_topk: int,
    softmax_scale: float,
    index_query_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """DSA attention over paged caches for a decode, prefill, or mixed dynamic batch, through
    FlashInfer's DSA kernels and the step plan in ``inference_context.dsa_metadata``
    (``FlashInferDSAMetadata``). Takes the arguments of
    ``dsa_gqa_inference.dynamic_sparse_attention``.

    Args:
        query: Queries ``[T,1,Hq,D]`` for the step's padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,1,D]``.
        value_cache: Paged values ``[blocks,block_size,1,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``, or NVFP4 packed uint8 ``[T,Di/2]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``, or the NVFP4 cache.
        inference_context: The step's dynamic inference context.
        index_topk: Number of keys each query attends to.
        softmax_scale: Attention softmax scale.
        index_query_scales: E4M3 scale bytes ``[T,Di/16]`` of NVFP4 indexer queries (the cache
            then holds NVFP4 keys); None for BF16.

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows of padding tokens are zero.
    """
    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "DSA inference decodes one token per request; MTP speculative decoding is not "
            "supported."
        )
    num_tokens, _, num_query_heads, _ = query.shape
    if key_cache.size(2) != 1:
        raise NotImplementedError(
            f"DSA-over-GQA inference needs one local KV group, got {key_cache.size(2)}."
        )
    # NVFP4 queries pack two values a byte.
    index_dim = index_query.size(-1) * (2 if index_query_scales is not None else 1)
    if not flashinfer_dsa_supported(
        query.device, index_topk, index_dim, key_cache.size(1), num_query_heads
    ):
        raise NotImplementedError(
            "FlashInfer's DSA kernels need SM100/SM103, top-k 512/1024/2048, indexer head dim "
            "128, at most 128 query heads per rank, and a KV block size that is a multiple of 64."
        )
    output = flashinfer_sparse_attention(
        query=query[:, 0],
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query.reshape(num_tokens, -1),
        index_key_cache=index_key_cache,
        plan=inference_context.dsa_metadata.plan(),
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        deterministic=inference_context.batch_invariant_mode,
        index_query_scales=index_query_scales,
    )
    return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))
