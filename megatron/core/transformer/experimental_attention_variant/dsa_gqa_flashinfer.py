# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA-over-GQA inference through FlashInfer's DSA (SDSA) kernels, on Megatron's paged caches.

Without FlashInfer's DSA kernels (``flashinfer.sdsa`` and ``flashinfer.token_sparse``) or FA4
this module still imports, and ``require_flashinfer_dsa`` raises when the path is requested.

- **Decode rows** (single-token requests): dense BF16 indexer scores (``indexer_scores``), exact
  top-k (gvr_2), then trtllm-gen token-sparse attention.
- **Prefill rows**, in 128-row tiles of one request each: a threshold from sampled scores, one
  fused sweep that keeps the keys above it (``indexer_threshold_select``, with an exact in-launch
  repair), the exact top-k among them (``top_k_from_candidates``), then FA4 attention over each
  8-token group's union of selected keys with a per-token bitmask (the "union" core; at every
  context length measured, 4K prompts to 2K-token chunks on 1M keys, it beat attending per
  token).

Megatron's caches are used as they are: the ``[blocks, block_size, Di]`` indexer cache is viewed
as FlashInfer's 64-key pages (a derived block table), and the ``[blocks, block_size, 1, D]`` K/V
caches as page-size-1 slots. The per-step plan (``dsa_flashinfer_plan.build_step_plan``, re-exported here) is built on
the CPU from the step's request lengths with sizes fixed by the padded batch, so every launch
has a static shape and the whole path can run inside a CUDA graph.

``flashinfer_sparse_attention`` runs the step as a pipeline of named stages:

- prefill: ``_tiled_index_queries`` -> ``_sampled_thresholds`` -> ``_select_prefill_topk``
  -> ``_union_core_attention``;
- decode: ``_decode_attention``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from megatron.core.inference.contexts.attention_context.dsa_flashinfer_plan import (  # noqa: F401
    FI_PAGE,
    MAX_TOPK_ROWS,
    PAGES,
    SAMPLES,
    TILE_ROWS,
    StepPlan,
    build_step_plan,
    ceil_div,
    check_plan_rows,
    fill_page_table,
    page_table_columns,
    plan_sizes,
    union_group,
)

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

# Threshold selection per top-k: (RANK, CAPACITY). A prefill row's threshold is the RANK-th
# largest of its sampled scores (SAMPLES keys per run of PAGES pages, dsa_flashinfer_plan), and
# it has CAPACITY candidate slots after its top-k tie slots; a row the threshold misses is
# repaired exactly inside the select launch.
SELECTOR = {512: (39, 4608), 1024: (60, 6144), 2048: (96, 9216)}
# trtllm-gen token-sparse attention workspace, and the most attention rows it is sized for.
CORE_WORKSPACE_BYTES = 128 << 20
MAX_CORE_ROWS = 65535


# Score rows are padded to a multiple of this many keys (indexer_scores' vector width).
SCORE_ALIGN = 4
# The fixed random threshold-sample offsets: seed and count (a key's offset into its run of
# PAGES pages is drawn once per device and reused by every step).
SAMPLE_OFFSET_SEED = 0x5D5A
SAMPLE_OFFSET_COUNT = 1 << 20


def require_flashinfer_dsa() -> None:
    """Raise unless the packages this path needs are installed."""
    missing = [
        name
        for name, present in (
            ("FlashInfer with DSA kernels (flashinfer.sdsa) and FA4", HAVE_FLASHINFER_DSA),
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
def _arrivals(device_index: int, tiles: int) -> torch.Tensor:
    """indexer_threshold_select's arrival counts for up to ``tiles`` query tiles (zeroed once,
    left zeroed by every call), for plans beyond the default count."""
    return torch.zeros(tiles, dtype=torch.int32, device=torch.device("cuda", device_index))


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
    generator = torch.Generator(device).manual_seed(SAMPLE_OFFSET_SEED)
    offsets = torch.randint(
        PAGES * FI_PAGE,
        (SAMPLE_OFFSET_COUNT,),
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )
    return core, counters, topk, arrivals, offsets


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


@dataclass
class _CacheViews:
    """FlashInfer's views of a layer's paged caches."""

    pages: torch.Tensor  # indexer keys as 64-key pages: BF16 [pages,64,Di]
    key_rows: torch.Tensor  # K cache as page-size-1 slots [slots,1,1,D]
    value_rows: torch.Tensor  # V cache as page-size-1 slots [slots,1,1,Dv]


def _with_empty_row(t: torch.Tensor) -> torch.Tensor:
    """``t`` with one zero row appended (the row every padding entry reads)."""
    return torch.cat((t, t.new_zeros((1, *t.shape[1:]))))


def _tiled_index_queries(plan, index_query):
    """Each tiled row's indexer query (zero for padding rows)."""
    query_rows = plan.query_rows
    live_rows = query_rows.clamp(min=0).long()
    return index_query[live_rows] * (query_rows >= 0).to(index_query.dtype)[:, None]


def _sampled_thresholds(plan, caches, packed, *, rank, offsets, topk_ws):
    """Each tiled row's selection threshold, and its counters for the select launch.

    Gathers ``SAMPLES`` keys per run of ``PAGES`` pages of every prefill request
    (``gather_key_sample``), scores the live tiled rows against their request's samples
    (``indexer_scores``), and takes each row's ``rank``-th largest score (gvr_2 top-k and
    ``thresholds_from_sample``). The per-row kernels run on the live rows only (compacted, plus
    one empty row every padding tiled row reads), so their rows are bounded by the step's tokens
    rather than by its tiles.
    """
    device = packed.device
    pages = caches.pages
    sample_cap = TILE_ROWS * ceil_div(plan.max_context * SAMPLES, TILE_ROWS * FI_PAGE * PAGES)
    num_requests = plan.request_rows.numel()
    sample_keys = torch.empty(
        num_requests, sample_cap, pages.size(-1), dtype=pages.dtype, device=device
    )
    gather_key_sample(
        pages,
        plan.page_table.index_select(0, plan.request_rows.long()),
        offsets[:sample_cap],
        plan.sample_ends,
        sample_keys=sample_keys,
        samples_per_run=SAMPLES,
        pages_per_run=PAGES,
    )
    sample_table = torch.arange(
        num_requests * sample_cap // FI_PAGE, dtype=torch.int32, device=device
    ).view(num_requests, -1)
    compact = plan.compact_rows.long()
    compact_query = _with_empty_row(packed[compact])
    compact_ends = _with_empty_row(plan.compact_ends)
    sample_row_ends = (compact_ends // (PAGES * FI_PAGE) * SAMPLES).to(torch.int32)
    compact_rows = compact_ends.numel()
    sample_scores = torch.empty(compact_rows, sample_cap, device=device)
    indexer_scores(
        compact_query,
        sample_keys.view(-1, FI_PAGE, pages.size(-1)),
        sample_table,
        sample_row_ends,
        requests=_with_empty_row(plan.compact_ordinals),
        scores=sample_scores,
    )
    sample_top = torch.empty(compact_rows, rank, dtype=torch.int32, device=device)
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
    compact_thresholds = torch.empty(compact_rows, device=device)
    compact_counters = torch.empty(compact_rows, 2, dtype=torch.int32, device=device)
    thresholds_from_sample(
        sample_scores, sample_top, thresholds=compact_thresholds, counters=compact_counters
    )
    tiled_compact = plan.tiled_compact.long()
    return compact_thresholds[tiled_compact], compact_counters[tiled_compact].contiguous()


def _select_prefill_topk(
    plan, caches, packed, thresholds, counters, *, index_topk, capacity, arrivals
):
    """The keys of each tiled row's exact top-k (``[tiled, index_topk]``, -1 last).

    One fused sweep keeps every key above the row's threshold plus its ties
    (``indexer_threshold_select``, which repairs a row the threshold missed inside the launch),
    then ``top_k_from_candidates`` takes the exact top-k among them.
    """
    device = packed.device
    tiled = plan.num_tiles * TILE_ROWS
    ranges = torch.empty(2, tiled, dtype=torch.int32, device=device)
    candidates = torch.empty(tiled, index_topk + capacity, 2, dtype=torch.int32, device=device)
    indexer_threshold_select(
        packed,
        caches.pages,
        plan.row_ends,
        plan.tile_requests,
        plan.page_table,
        thresholds=thresholds,
        counters=counters,
        candidates=candidates,
        max_context=plan.max_context,
        top_k=index_topk,
        arrivals=arrivals[: plan.num_tiles],
        ranges=ranges,
    )
    positions = torch.empty(tiled, index_topk, dtype=torch.int32, device=device)
    top_k_from_candidates(candidates, ranges, top_keys=positions)
    return positions


def _union_core_attention(plan, caches, query, positions, out, *, index_topk, softmax_scale):
    """FA4 attention of the prefill rows into ``out``, in union-core groups.

    Each group of ``union_group(heads)`` consecutive tiled rows attends to the union of its
    rows' selected keys (``group_union_masks``), with a per-row bitmask (``_union_mask_mod``)
    that keeps each row on its own keys; ``plan.batch_starts`` are the groups' FA4
    ``cu_seqlens_q``.
    """
    rows, heads, _ = query.shape
    device = query.device
    group = union_group(heads)
    groups = plan.num_tiles * TILE_ROWS // group
    key_tiles = ceil_div(plan.max_context, TILE_ROWS)
    columns = ceil_div(min(group * index_topk, TILE_ROWS * key_tiles), 128) * 128
    unions = torch.zeros(groups, columns, dtype=torch.int32, device=device)
    union_sizes = torch.zeros(groups, dtype=torch.int32, device=device)
    mask = torch.zeros(rows, columns // 32, dtype=torch.int32, device=device)
    group_union_masks(
        positions,
        plan.query_rows,
        torch.arange(groups, dtype=torch.int32, device=device),
        plan.tile_requests.repeat_interleave(TILE_ROWS // group),
        plan.page_table,
        union=unions,
        union_sizes=union_sizes,
        mask=mask,
        num_keys=plan.max_context,
    )
    _flash_attn_fwd(
        query,
        caches.key_rows,
        caches.value_rows,
        page_table=unions,
        seqused_k=union_sizes,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_q=plan.batch_starts,
        max_seqlen_q=group,
        mask_mod=_union_mask_mod,
        aux_tensors=[mask],
    )


def _decode_attention(
    plan,
    caches,
    query,
    index_query,
    out,
    *,
    index_topk,
    width,
    softmax_scale,
    deterministic,
    topk_ws,
):
    """Attention of the decode rows: dense indexer scores of all their keys, exact top-k
    (gvr_2), then trtllm-gen token-sparse attention. Returns the step output: ``out`` with its
    decode rows replaced (mixed step), or the decode rows' output (decode-only step) with its
    padding rows zeroed."""
    _, heads, _ = query.shape
    device = query.device
    decode_rows = plan.decode_rows
    scores = torch.empty(decode_rows, width, device=device)
    indexer_scores(
        index_query[:decode_rows],
        caches.pages,
        plan.page_table,
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
    if deterministic:
        # gvr_2 returns the selected keys in an order that varies between runs, and
        # attention sums in that order; sorting (descending, so -1 padding stays last)
        # makes decode replay bit-exactly.
        topk = topk.sort(dim=-1, descending=True).values
    value_dim = caches.value_rows.size(-1)
    decode_out = torch.empty(decode_rows, heads, value_dim, device=device, dtype=query.dtype)
    _token_core(
        query[:decode_rows],
        topk,
        plan.decode_requests,
        plan.page_table,
        caches.key_rows,
        caches.value_rows,
        decode_out,
        index_topk,
        softmax_scale,
        heads,
    )
    on_decode = (plan.decode_lengths > 0)[:, None, None]
    if out is not None:
        out[:decode_rows] = torch.where(on_decode, decode_out, out[:decode_rows])
        return out
    # Decode-only: every row is a decode row or padding (no keys). Padding rows are zeroed with
    # a select, not a multiply, so whatever the kernel left in them (even NaN) becomes 0.
    return decode_out.masked_fill_(~on_decode, 0)


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
) -> torch.Tensor:
    """Attention output ``[rows,Hq,Dv]`` of every step row through FlashInfer's DSA kernels;
    rows with no keys (padding) are zero.

    Args:
        query: BF16 ``[rows,Hq,D]``.
        key_cache, value_cache: Paged BF16 ``[blocks,block_size,1,D]``.
        index_query: BF16 indexer queries ``[rows,Di]``.
        index_key_cache: Paged BF16 indexer keys ``[blocks,block_size,Di]``.
        plan: The step's plan (``StepPlan``), whose page table maps the requests' KV blocks.
        index_topk: Keys each row selects.
        softmax_scale: Attention softmax scale.
        deterministic: Sort each decode row's selected keys so that decode replays bit-exactly
            (the prefill path already does).
    """
    device = query.device
    rows, heads, dim = query.shape
    value_dim = value_cache.size(-1)
    width = ceil_div(max(plan.max_context, index_topk), SCORE_ALIGN) * SCORE_ALIGN
    key_tiles = ceil_div(plan.max_context, TILE_ROWS)
    assert plan.page_table.size(1) >= max(
        ceil_div(width, FI_PAGE), 2 * key_tiles
    ), "page table too narrow"
    caches = _CacheViews(
        pages=index_key_cache.view(-1, FI_PAGE, index_key_cache.size(-1)),
        key_rows=key_cache.view(-1, 1, 1, dim),
        value_rows=value_cache.view(-1, 1, 1, value_dim),
    )
    index_query = index_query.contiguous()
    query = query.contiguous()
    _, _, topk_ws, arrivals, offsets = _persistent(device.index, heads, 4096)
    out = None

    if plan.num_tiles:
        out = torch.zeros(rows, heads, value_dim, device=device, dtype=query.dtype)
        rank, capacity = SELECTOR[index_topk]
        if plan.num_tiles > arrivals.numel():
            arrivals = _arrivals(device.index, 1 << (plan.num_tiles - 1).bit_length())
        packed = _tiled_index_queries(plan, index_query)
        thresholds, counters = _sampled_thresholds(
            plan, caches, packed, rank=rank, offsets=offsets, topk_ws=topk_ws
        )
        positions = _select_prefill_topk(
            plan,
            caches,
            packed,
            thresholds,
            counters,
            index_topk=index_topk,
            capacity=capacity,
            arrivals=arrivals,
        )
        _union_core_attention(
            plan, caches, query, positions, out, index_topk=index_topk, softmax_scale=softmax_scale
        )

    if plan.decode_rows:
        out = _decode_attention(
            plan,
            caches,
            query,
            index_query,
            out,
            index_topk=index_topk,
            width=width,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
            topk_ws=topk_ws,
        )
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
) -> torch.Tensor:
    """DSA attention over paged caches for a decode, prefill, or mixed dynamic batch, through
    FlashInfer's DSA kernels and the step plan in ``inference_context.dsa_metadata``
    (``FlashInferDSAMetadata``). Takes the arguments of
    ``dsa_gqa_inference.dynamic_sparse_attention``.

    Args:
        query: Queries ``[T,1,Hq,D]`` for the step's padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,1,D]``.
        value_cache: Paged values ``[blocks,block_size,1,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``.
        inference_context: The step's dynamic inference context.
        index_topk: Number of keys each query attends to.
        softmax_scale: Attention softmax scale.

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
    if not flashinfer_dsa_supported(
        query.device, index_topk, index_query.size(-1), key_cache.size(1), num_query_heads
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
    )
    return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))
