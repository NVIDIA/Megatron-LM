# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load-balanced context-parallel DSA indexer.

Under context parallelism the DSA indexer uses a contiguous sequence split, so the causal
per-query cost ``visible_k(p) = min((p + 1) // ratio, comp_len)`` grows with position and later
CP ranks become stragglers (the layout-level max/min work ratio is ``2 * cp_size - 1``). This
module removes that imbalance without a new kernel: it splits the sequence(s) into
``2 * cp_size`` near-equal chunks and gives CP rank ``r`` chunk ``r`` (a low-position "head",
light) and chunk ``2 * cp_size - 1 - r`` (a high-position "tail", heavy), so
``work(head) + work(tail)`` is ~constant across ranks. The balanced ownership is realized by the
PER-SEQUENCE ZIGZAG: packs whose padded sequence lengths divide ``2 * cp_size`` gather each
rank's head/tail chunks of every sequence via prebuilt A2A routes and score them with two
packed fused-top-k calls against synthetic per-sequence layouts with explicit RoPE positions.
Config validation requires the fused backend; the ACTUAL pack decides per microbatch:
an eager pack the zigzag builders cannot represent routes to the contiguous reference
path (``pack_eligible_for_zigzag``), while CUDA graphs require a recorded verdict and a
static composition. ``pad_packed_seq_alignment`` remains a capacity-rounding hint, not
an eligibility contract: ``None``, ``"max"``, and integer alignments are all accepted.
The fused kernel package has one verified defect: a fused call
with more than ``FUSED_INDEXER_MAX_SAFE_ROWS`` (32768) query rows is silently corrupted
from row 32768 on unless it is the process's first fused call; the balanced
synthetic-layout calls therefore fail closed above the limit (prebuild bounds ``l_local``
at twice it -- two half-row calls), while pre-existing reference callers keep their
behavior and get a once-per-process correctness warning instead. (The former chunk-pair
folding fallback measured slower than the unbalanced baseline on unequal packs and was
removed.)

``balanced_compute_cp_indexer_topk`` is a drop-in replacement for
``csa_utils.cp_utils.compute_cp_indexer_topk``: it returns the top-k in the same contiguous
``[l_local, topk]`` layout, so the downstream index-building and sparse attention are unchanged.

Triage switch: ``MCORE_DSA_CP_BAL_DEBUG=1`` logs the eligibility decision once.

CUDA-graph contract: the default mode is scoped to static pack compositions at PP=1 and is
enforced through ``prebuild_balanced_layouts`` — under CUDA graphs it MUST be called every
microbatch (as ``pretrain_gpt.get_batch`` does); frontends that skip it lose the
composition-change detection and are protected only by the in-graph divisibility assert.
The separate opt-in ``dsa_cp_balance_indexer_graph_dynamic_packs`` mode instead validates
every padded pack and builds one fixed-capacity, two-hop equal-split A2A route in that hook.
The same fixed-shape route plan is supplied as CUDA-graph tensor inputs to every DSA layer, so a
replay can refresh metadata for a different pack without recapturing or repeating route sorts.
"""

import logging
import os

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
from megatron.core.transformer.experimental_attention_variant.dsa_fused_safety import (
    FUSED_INDEXER_MAX_SAFE_ROWS,
)
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

logger = logging.getLogger(__name__)

# Run-constant A/B switches, read once at import (see the module docstring).
_GATE_DEBUG = os.environ.get("MCORE_DSA_CP_BAL_DEBUG") == "1"

# Score-buffer sizing contract for the fused cuDNN indexer kernel. Widths are
# quantized to _KV_BOUND_QUANTUM so the set of distinct (max_seqlen_q, max_seqlen_kv)
# shapes stays tiny (the kernel JIT-compiles and reallocates its fp32 score buffer per
# shape). The ceiling encodes an EMPIRICAL kernel property: a tight score width
# narrower than the declared per-sequence KV length is only safe up to
# _KV_TIGHT_WIDTH_CEILING columns (wider tight values hit an illegal memory access; a
# width equal to the declared per-sequence length is always fine). Revisit both if the
# kernel contract changes.
_KV_BOUND_QUANTUM = 8192
_KV_TIGHT_WIDTH_CEILING = 65536


# Staging buffers for the balanced exchanges, cached per (tag, group, width, dtype).
# All split sizes are static for a fixed pack size, so ``all_to_all_single`` is
# CUDA-graph capturable, and the cached staging buffers keep the allocator pool static
# (no expandable-segments cuMem churn).
# NOTE: this process-global cache is append-only by design (entries are keyed by
# (group, size, dtype, width) and reused for the life of the run). Under dynamic CP
# each bucket's subgroup adds its own entries, so the footprint is bounded by the
# number of distinct CP buckets — small in practice, but not evicted.
_A2A_BUF: dict = {}

# Last prebuilt zigzag plan per (group, rank) — latest wins, so the slot count is
# bounded regardless of how many distinct pack capacities an eager varlen run
# produces. TE's graph-capture argument cloning strips dynamically attached
# PackedSeqParams attributes, so the per-batch layout cache can be invisible
# during capture; this module-level copy keeps the prebuilt plan (and its route
# tensors) alive and reachable there. Pack composition must be static under CUDA
# graphs, so the last plan is always the right one; every consumer still
# re-validates capacity via plan["half"] * 2 == l_local.
_LAST_PLAN: dict = {}

# Per-(group, l_local) verdict from the last prebuild or eager probe: the CURRENT
# pack is zigzag-representable (every padded sequence length divisible by 2 * cp_size).
# Both outcomes are recorded because eager execution may switch between the balanced and
# contiguous paths from one pack to the next. Capture consults this registry because
# probing is impossible while recording a graph.
_ZZ_PACK_OK: dict = {}

# Last OBSERVED (l_local, cu composition) per group, recorded on every prebuild
# call — independent of whether a plan is ever built. The static-composition
# gate compares against this, not against a plan: a reference-fallback run (e.g. pad
# alignment that never opens the zigzag gate) builds no plans, yet its captured
# graphs still bake composition-sensitive host state (mq/gkv widths derived
# from max_seqlen_q, and the gate/single-multi host branches), so a composition
# change must fail loudly there too.
_SEEN_CU: dict = {}

# A graph-dynamic route is built once per PackedSeqParams in the data-step hook,
# then flattened into ordinary tensor kwargs for TE CUDA graphs.  Keeping the
# representation private avoids making feature-specific fields part of the public
# PackedSeqParams dataclass while still giving every layer a fixed-shape input.
_GRAPH_DYNAMIC_PLAN_ATTR = "_dsa_cp_balance_graph_plan"
_GRAPH_DYNAMIC_PLAN_KWARGS = {
    "validated_cu": "dsa_cp_graph_validated_cu",
    "pos_head": "dsa_cp_graph_pos_head",
    "pos_tail": "dsa_cp_graph_pos_tail",
    "score_cu_q": "dsa_cp_graph_score_cu_q",
    "score_cu_kv": "dsa_cp_graph_score_cu_kv",
    "head_offsets": "dsa_cp_graph_head_offsets",
    "tail_offsets": "dsa_cp_graph_tail_offsets",
    "output_cu_q": "dsa_cp_graph_output_cu_q",
    "output_cu_kv": "dsa_cp_graph_output_cu_kv",
    "output_offsets": "dsa_cp_graph_output_offsets",
    "src_slot": "dsa_cp_graph_src_slot",
    "relay_perm": "dsa_cp_graph_relay_perm",
    "dst_slot": "dsa_cp_graph_dst_slot",
}


def _is_capturing() -> bool:
    """True only while a CUDA stream capture is in progress.

    CUDA-less torch builds (e.g. macOS / CPU-only wheels, where the CPU half of
    these layout helpers still runs) expose ``is_current_stream_capturing`` as a
    dummy that raises instead of returning False, so short-circuit on
    availability first.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _group_key(cp_group):
    """Stable cache key for a process group.

    ``id()`` alone can be recycled if a group is destroyed and a new one is allocated at
    the same address; c10d's ``group_name`` is unique per created group, so prefer it.
    """
    return getattr(cp_group, "group_name", None) or id(cp_group)


def _a2a_buf(tag, rows, width, dtype, dev, cp_group, persistent=True):
    if not persistent:
        # S-sized fallback buffers: allocate transiently. In eager the caching
        # allocator reuses the block; under capture the allocation comes from the
        # graph's private pool, which already guarantees a stable address across
        # replays — caching a pool-owned tensor in this process-global dict would
        # instead risk a dangling pointer if that graph is ever re-recorded.
        return torch.empty((rows, width), dtype=dtype, device=dev)
    # Grow-only capacity pool: rows is NOT part of the key. Eager varlen with an
    # integer pad alignment produces a different l_local per pack total; keying on
    # rows would leak one full buffer set per distinct value (tens of MB each),
    # never evicted. Instead one buffer per (tag, group, width, dtype) grows to
    # the largest rows seen — bounded by capacity // cp_size — and smaller packs
    # use a prefix view. Under CUDA graphs the composition is static (enforced at
    # prebuild), so the buffer never regrows and capture-baked addresses stay
    # valid.
    key = (tag, _group_key(cp_group), width, dtype)
    buf = _A2A_BUF.get(key)
    if buf is None or buf.shape[0] < rows:
        if _is_capturing():
            # First touch (or regrowth) of this tag happens INSIDE a capture: the
            # allocation is graph-pool-owned, so caching it process-globally would
            # dangle if that graph is ever re-recorded.
            # Use it transiently; the pool keeps its address stable across replays.
            return torch.empty((rows, width), dtype=dtype, device=dev)
        buf = torch.empty((rows, width), dtype=dtype, device=dev)
        _A2A_BUF[key] = buf
    return buf[:rows]


def dispatch_chunks_async(
    indexer_qr,
    weights_indexer_cp,
    cp_group,
    cp_size,
    l_local,
    layout_cache=None,
    cu_seqlens=None,
    cu_seqlens_compressed=None,
    graph_dynamic_packs=False,
    graph_dynamic_plan=None,
):
    """Issue the chunk dispatch as early as possible; returns an opaque handle.

    Called by the CSA forward right after ``indexer_qr``/``weights_indexer_cp`` are
    produced, so the all_to_all overlaps with the compressed-K/KV all-gathers already in
    flight (and the local top-k preparation) instead of sitting on the critical path
    right before the top-k. ``balanced_compute_cp_indexer_topk(dispatch_handle=...)`` waits
    on it. When qr and weights share a dtype they ride one all_to_all (single launch).
    ``cu_seqlens`` feeds the pack-eligibility consistency check
    (``_ensure_pack_zigzag_ok`` -- the CSA caller routes ineligible packs to the
    reference path before dispatching); the compute side follows the returned
    handle's kind, so both stay consistent.

    Handle ``kind`` legend — which transport carried the (qr | weights) payload:

    - ``"zzr"``: zigzag + prebuilt-route ``all_to_all_single`` — each rank exchanges
      only ~``l_local`` rows; splits/rows come from ``prebuild_balanced_layouts``
      (host ints), so the exchange is CUDA-graph capturable.
    - ``"gdr"`` / ``"gdr2"``: opt-in replay-dynamic two-hop equal-split A2A;
      metadata and invocation-owned staging buffers are captured with the graph.
    - ``"ag"``: zigzag fallback — no usable routed plan (never prebuilt, capacity
      mismatch, or plan lacks route fields): one static-shape S-row AllGather of the
      merged payload; row selection is deferred to consume time.
    - ``"ag2"``: same fallback, but qr/weights dtypes differ so the merged payload
      cannot carry both — two separate AllGathers.
    - ``None``: ``cp_size <= 1`` — nothing to balance, nothing to prefetch.
    """
    if cp_size <= 1:
        return None
    if graph_dynamic_packs:
        return _graph_dynamic_dispatch_chunks_async(
            indexer_qr,
            weights_indexer_cp,
            cu_seqlens,
            cu_seqlens_compressed,
            cp_group,
            cp_size,
            l_local,
            graph_dynamic_plan,
        )
    q_lora = indexer_qr.shape[-1]
    n_heads = weights_indexer_cp.shape[-1]
    # Detached: the dispatch feeds only the integer top-k; no gradient flows back.
    q2 = indexer_qr.detach().reshape(-1, q_lora)
    w2 = weights_indexer_cp.detach().reshape(-1, n_heads)
    _ensure_pack_zigzag_ok(cu_seqlens, cp_group, cp_size, l_local, layout_cache)
    plan = layout_cache.get(("zigzag", cp_group.rank())) if layout_cache else None
    if plan is not None and plan.get("half", 0) * 2 != l_local:
        # Plan built against different local row count (e.g. prebuild saw a
        # padded cu ending short of the physical pack): unusable here.
        plan = None
    if plan is None and _is_capturing():
        # TE graph capture clones PackedSeqParams and strips the per-microbatch
        # cache; the module-level slot is the capture-time source of truth. In
        # eager, a missing cache entry means prebuild did not vet THIS pack, so
        # do not trust a slot left over from a previous one. (CUDA graphs with
        # this flag require a static pack composition — enforced at data-prep
        # time by prebuild_balanced_layouts — so under capture the slot always
        # describes the pack being captured.)
        plan = _LAST_PLAN.get((_group_key(cp_group), cp_group.rank()))
        if plan is not None and plan.get("half", 0) * 2 != l_local:
            plan = None  # latest-slot plan built at a different capacity
    if plan is not None and "disp_send_rows" in plan and q2.dtype == w2.dtype:
        # Route-A2A dispatch (PR #5664-style prebuilt exchange): each rank sends and
        # receives only ~l_local rows instead of the S-row AllGather. Splits are host
        # ints prebuilt at data-prep time, so the exchange is CUDA-graph capturable.
        width = q_lora + n_heads
        payload = _a2a_buf("zzr_pay", l_local, width, q2.dtype, q2.device, cp_group)
        payload[:, :q_lora].copy_(q2)
        payload[:, q_lora:].copy_(w2)
        send = _a2a_buf("zzr_send", l_local, width, q2.dtype, q2.device, cp_group)
        # copy_(index_select(...)) instead of index_select(out=...): keeps the
        # cached staging buffer as the destination without imposing out= dtype /
        # autograd constraints. q2/w2 are detached; no gradient flows through
        # the dispatch.
        send.copy_(torch.index_select(payload, 0, plan["disp_send_rows"]))
        recv = _a2a_buf("zzr_recv", l_local, width, q2.dtype, q2.device, cp_group)
        work = dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=plan["disp_out_splits"],
            input_split_sizes=plan["disp_in_splits"],
            group=cp_group,
            async_op=True,
        )
        return {"kind": "zzr", "works": [work], "recv": recv, "plan": plan, "q_lora": q_lora}
    # Fallback: static-shape AllGather; row selection happens on device from
    # cu_seqlens at consume time (CUDA-graph friendly with per-iteration packs).
    if plan is not None and "disp_send_rows" in plan:
        # Routed plan exists but qr/weights dtypes differ: the merged-payload
        # exchange cannot carry both.
        _warn_allgather_fallback("prebuilt routes unavailable for mixed qr/weights dtypes")
    else:
        _warn_allgather_fallback(_NO_PLAN_REASON)
    if q2.dtype == w2.dtype:
        width = q_lora + n_heads
        payload = _a2a_buf("zz_qw", l_local, width, q2.dtype, q2.device, cp_group)
        payload[:, :q_lora].copy_(q2)
        payload[:, q_lora:].copy_(w2)
        work, g = _all_gather_rows_buf(payload, l_local, cp_group, cp_size, "zz_qw")
        return {"kind": "ag", "works": [work], "g": g, "q_lora": q_lora}
    wq, gq = _all_gather_rows_buf(q2, l_local, cp_group, cp_size, "zz_q")
    ww, gw = _all_gather_rows_buf(w2, l_local, cp_group, cp_size, "zz_w")
    return {"kind": "ag2", "works": [wq, ww], "gq": gq, "gw": gw, "q_lora": q_lora}


_FP8_AUTOCAST = None  # resolved once by _no_fp8_ctx: TE fp8_autocast, or False when TE absent


def _no_fp8_ctx():
    """FP8-disabled context for the no-grad chunk projections.

    The balanced path projects each chunk through ``linear_wq_b`` IN ADDITION to the
    loss path's grad-tracked projection. Under an FP8 delayed-scaling recipe covering
    that TE linear, letting these extra forwards run in FP8 would append to its amax
    history — the quantization-scale trajectory (and hence loss-path numerics) would
    depend on the balanced indexer. Running them in the ambient non-FP8 dtype keeps
    the loss path's amax stream identical to the reference during TRAINING forwards
    (one recording per step) and scores the top-k at bf16-or-better precision.

    Known, accepted divergences under FP8: (a) eval/no-grad forwards skip the loss
    projection entirely (csa.py), so the flag removes eval-time amax recordings the
    reference would have made — eval steps no longer perturb training scales, which
    differs from the reference trajectory by design; (b) the sparse-attention
    SELECTION uses these non-FP8 chunk scores while the indexer LOSS trains against
    the FP8-projected loss-path scores — the selection is at least as precise as the
    reference's.

    NOTE(fp8_model_init): with FP8-only parameters (--fp8-param-gather /
    fp8_model_init), running a TE linear outside fp8_autocast is TE-version
    dependent (dequantize-on-the-fly vs error). Any failure is loud, not silent;
    validate on the target TE build before enabling the flag under FP8 params.
    """
    global _FP8_AUTOCAST
    if _FP8_AUTOCAST is None:
        try:
            from transformer_engine.pytorch import fp8_autocast

            _FP8_AUTOCAST = fp8_autocast
        except ImportError:
            _FP8_AUTOCAST = False
    if _FP8_AUTOCAST is False:
        import contextlib

        return contextlib.nullcontext()
    return _FP8_AUTOCAST(enabled=False)


_AG_FALLBACK_WARNED: set = set()


def _warn_allgather_fallback(reason):
    """One-time-per-reason heads-up that the heavy AllGather fallback engaged (eager only)."""
    if reason in _AG_FALLBACK_WARNED or _is_capturing():
        return
    _AG_FALLBACK_WARNED.add(reason)
    logger.warning(
        "balanced CP indexer: %s; using the S-row AllGather dispatch/combine "
        "fallback (correct, but heavier than the routed exchange).",
        reason,
    )


_NO_PLAN_REASON = (
    "no prebuilt zigzag plan for this pack (call prebuild_balanced_layouts from the "
    "data step, see pretrain_gpt.get_batch, to enable the ~l_local-row A2A routes; "
    "if it IS called, its plans were built at a different capacity than the forward "
    "sees — check the capacity/pad settings passed to prebuild)"
)


def _all_gather_rows_buf(x, l_local, cp_group, cp_size, tag):
    """AllGather into a buffer (async): returns (work, gathered[cp_size*l_local, D]).

    The l_local-sized send stage stays cached; the S-sized receive buffer is always
    transient (see ``_a2a_buf``).
    """
    D = x.shape[-1]
    x2 = x.reshape(l_local, D)
    send = _a2a_buf(tag + "_agsend", l_local, D, x2.dtype, x.device, cp_group)
    send.copy_(x2)
    g = _a2a_buf(
        tag + "_agrecv", cp_size * l_local, D, x2.dtype, x.device, cp_group, persistent=False
    )
    work = dist.all_gather_into_tensor(g, send, group=cp_group, async_op=True)
    return work, g


def _excl_cumsum(x):
    z = torch.zeros_like(x[:1])
    return torch.cat((z, torch.cumsum(x, 0)[:-1]))


def _pack_zigzag_verdict(cu_seqlens, cp_group, cp_size, l_local, layout_cache):
    """Is the ACTUAL microbatch pack zigzag-representable? Returns a bool.

    Structural requirement of the zigzag builders (nothing to do with the kernel
    row-limit defect): every (padded) sequence length and the capacity tail must
    be divisible by ``2 * cp_size``, and the per-rank row count must be even;
    otherwise the two-half ragged chunk enumeration would emit out-of-range gather
    indices. Verdicts are cached on
    the microbatch ``layout_cache`` (written by prebuild or an earlier probe of
    the same pack, capacity-tagged) and in the module registry keyed by
    (group, l_local). Probing needs a D2H read of ``cu_seqlens``, which is
    impossible under capture — capture therefore requires a recorded verdict
    (eager warmup or prebuild) and raises without one.
    """
    if layout_cache is not None:
        cached = layout_cache.get("zz_pack_ok")
        if cached is not None and cached[0] == l_local:
            # Verdicts are only valid for the capacity they were probed at: prebuild
            # may have seen a padded cu ending short of the physical pack.
            return cached[1]
    key = (_group_key(cp_group), l_local)
    if _is_capturing():
        if key in _ZZ_PACK_OK:
            return _ZZ_PACK_OK[key]
        # Probing is impossible here and guessing would bake an unverified branch.
        raise RuntimeError(
            "balanced CP indexer: no pack-eligibility verdict is available during "
            "graph capture; run an eager warmup (or prebuild_balanced_layouts) "
            "before capturing."
        )
    if cu_seqlens is None:
        if key in _ZZ_PACK_OK:
            return _ZZ_PACK_OK[key]
        raise RuntimeError(
            "balanced CP indexer: cannot verify pack eligibility — no cu_seqlens to "
            "probe and no recorded verdict for this (group, capacity). Run "
            "prebuild_balanced_layouts at data-prep time."
        )
    S = cp_size * l_local
    nch = 2 * cp_size
    cu = cu_seqlens.reshape(-1).cpu()
    lens = cu[1:] - cu[:-1]
    total = int(cu[-1])
    ok = (
        l_local % 2 == 0
        and total <= S
        and (S - total) % nch == 0
        and bool(((lens % nch) == 0).all())
    )
    if _GATE_DEBUG and not getattr(_pack_zigzag_verdict, "_logged", False):
        _pack_zigzag_verdict._logged = True
        logger.info("[zz-gate] pack verdict=%s: S=%s l_local=%s nch=%s", ok, S, l_local, nch)
    _ZZ_PACK_OK[key] = ok
    if layout_cache is not None:
        layout_cache["zz_pack_ok"] = (l_local, ok)
    return ok


def pack_eligible_for_zigzag(packed_seq_params, cu_seqlens, cp_group, cp_size, l_local):
    """Per-pack routing decision for the CSA forward (eager: bool; capture: recorded).

    An ineligible pack takes the original contiguous ``compute_cp_indexer_topk``
    path for that microbatch. Routing costs nothing beyond the one cached D2H probe
    for frontends that never prebuild. Ensures the
    per-microbatch layout cache exists so the verdict (and layer 1's plans) are
    shared by every layer.
    """
    cache = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
    if cache is None:
        cache = {}
        packed_seq_params._dsa_cp_balance_layout_cache = cache
    return _pack_zigzag_verdict(cu_seqlens, cp_group, cp_size, l_local, cache)


def _ensure_pack_zigzag_ok(cu_seqlens, cp_group, cp_size, l_local, layout_cache):
    """Internal consistency check: callers must route ineligible packs away first.

    The CSA integration routes ineligible packs to the contiguous reference path
    (``pack_eligible_for_zigzag``) before dispatching, so reaching this raise
    means a caller bypassed that routing; proceeding would emit out-of-range
    gather indices (silent corruption), hence a real raise.
    """
    if not _pack_zigzag_verdict(cu_seqlens, cp_group, cp_size, l_local, layout_cache):
        raise ValueError(
            "balanced CP indexer: this pack is not zigzag-representable "
            f"(l_local={l_local}, capacity={cp_size * l_local}; every padded sequence "
            f"length and the capacity tail must be divisible by 2 * cp_size = "
            f"{2 * cp_size}). Callers must route such packs to the contiguous "
            "compute_cp_indexer_topk path, as the CSA integration does."
        )


def _rope_positions(q, pos_ids, cu_q, nope_dim, pos_dim, indexer, config, table_len):
    """Apply MLA RoPE at explicit sequence-relative positions (zigzag packed rows)."""
    if config.apply_rope_fusion:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace

        cos, sin = indexer.rotary_pos_emb.get_cached_cos_sin(
            table_len, dtype=q.dtype, packed_seq=True, mscale=1.0
        )
        return fused_mla_rope_inplace(
            q,
            cos,
            sin,
            nope_dim,
            pos_dim,
            cu_seqlens_q=cu_q,
            remove_interleaving=True,
            position_ids=pos_ids,
        )
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

    rpe = indexer.rotary_pos_emb(table_len, packed_seq=True)
    rpe = rpe[0] if isinstance(rpe, tuple) else rpe
    freqs = torch.index_select(rpe, 0, pos_ids)
    content, rotary = torch.split(q, [nope_dim, pos_dim], dim=-1)
    rotary = _apply_rotary_pos_emb_bshd(
        rotary,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=1.0,
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=True,
    )
    return torch.cat((content, rotary), dim=-1)


def _zigzag_plan(cu_seqlens, cu_seqlens_compressed, cp_size, l_local, r, dev, layout_cache):
    """Per-sequence zigzag plan for this microbatch (cached on the layout cache).

    Requires every (padded) sequence length — including the capacity-padding pseudo-sequence —
    to be divisible by ``2 * cp_size`` (verified by the per-pack routing gate).
    Rank ``r`` computes, for EVERY sequence, its intra-sequence chunks ``r`` (head) and
    ``2N-1-r`` (tail), so per-rank causal work is exactly constant for any pack composition.
    All indices/layouts are computed on device from ``cu_seqlens`` (no host sync); shapes are
    static, so the plan is CUDA-graph friendly (values refresh through captured kernels).
    """
    if layout_cache is not None:
        cached = layout_cache.get(("zigzag", r))
        if cached is not None and cached.get("half", 0) * 2 == l_local:
            # Wrong-capacity entries (prebuild probed a cu ending short of the
            # physical pack) must be rebuilt, not resurrected.
            return cached
    N = cp_size
    nch = 2 * N
    S = N * l_local
    cu = cu_seqlens.reshape(-1)
    dt = cu.dtype
    # Sequence starts/lengths including the capacity-padding pseudo-sequence [cu[-1], S).
    starts = torch.cat((cu[:-1], cu[-1:]))
    ends = torch.cat((cu[1:], torch.full_like(cu[:1], S)))
    lens = ends - starts
    c = torch.div(lens, nch, rounding_mode="floor")  # exact when alignment % (2*cp) == 0
    if not _is_capturing():
        if bool((lens % nch != 0).any()):
            # One eager-only D2H check per plan build (rare): the ragged enumeration
            # below floors ``lens / 2N`` and would emit out-of-range gather indices
            # for non-2N-aligned packs. The fail-fast check (_ensure_pack_zigzag_ok)
            # raises for such packs before any dispatch is issued; reaching this
            # error means a caller bypassed it.
            raise ValueError(
                "balanced CP indexer zigzag plan requires every packed sequence length "
                f"(including capacity padding) to be divisible by 2 * cp_size = {nch}."
            )
    else:
        # D2H is impossible while capturing, and the capture-time verdict may come
        # from a module registry written by a PREVIOUS eager pack (frontends that
        # never prebuild). Record the check into the graph instead: every replay
        # re-validates the refreshed cu values on device, so a pack that drifts to a
        # non-representable composition trips a device-side assert rather than
        # silently gathering out-of-range rows.
        torch._assert_async((lens % nch == 0).all())
    half = l_local // 2
    nseg = int(starts.shape[0])

    # Ragged [seg, offset] enumeration over the head (and tail) rows: static shape `half`.
    base = _excl_cumsum(c)  # offset of each sequence inside the packed half
    rows = torch.arange(half, device=dev, dtype=dt)
    seg_id = (torch.bucketize(rows, base[1:] if nseg > 1 else base[:0], right=True)).clamp_max(
        nseg - 1
    )
    off = rows - base[seg_id]
    head_idx = starts[seg_id] + r * c[seg_id] + off
    tail_idx = starts[seg_id] + (nch - 1 - r) * c[seg_id] + off
    gather_idx = torch.cat((head_idx, tail_idx)).long()
    pos_head = (r * c)[seg_id] + off  # sequence-relative positions for RoPE
    pos_tail = ((nch - 1 - r) * c)[seg_id] + off
    # Capacity-padding pseudo-sequence rows carry position 0, matching the reference
    # ``_thd_cp_position_ids`` clamp: their top-k is discarded, and unclamped positions
    # could exceed the RoPE table length when the capacity tail is long.
    real_end = cu[-1]
    pos_head = torch.where(head_idx < real_end, pos_head, torch.zeros_like(pos_head))
    pos_tail = torch.where(tail_idx < real_end, pos_tail, torch.zeros_like(pos_tail))

    # Inverse permutation: for each of my OWNED contiguous rows, its position in the
    # rank-major all-gathered zigzag output Z[[rank] * l_local rows each, [heads | tails]].
    g = torch.arange(r * l_local, (r + 1) * l_local, device=dev, dtype=dt)
    s_own = torch.bucketize(g, starts[1:], right=True).clamp_max(nseg - 1)
    p = g - starts[s_own]
    k = torch.div(p, c[s_own].clamp_min(1), rounding_mode="floor").clamp_max(nch - 1)
    q_in = p - k * c[s_own]
    rho = torch.where(k < N, k, (nch - 1) - k)
    is_head = k < N
    pos_in_block = torch.where(is_head, base[s_own] + q_in, half + base[s_own] + q_in)
    inv_idx = (rho * l_local + pos_in_block).long()

    # Synthetic packed layouts (one segment per sequence; K keeps full compressed ranges).
    cu_comp = cu_seqlens_compressed.reshape(-1)
    comp_pad = torch.cat((cu_comp, cu_comp[-1:]))  # padding pseudo-seq: zero K rows
    cu_q = torch.cat((torch.zeros_like(c[:1]), torch.cumsum(c, 0))).to(dt)
    head_layout = (cu_q, comp_pad, (r * c).to(dt))
    tail_layout = (cu_q, comp_pad, ((nch - 1 - r) * c).to(dt))

    plan = {
        "gather_idx": gather_idx,
        "inv_idx": inv_idx,
        # int32, matching the dtype the reference path (_thd_cp_position_ids)
        # feeds the same fused RoPE kernel.
        "pos_head": pos_head.int(),
        "pos_tail": pos_tail.int(),
        "head_layout": head_layout,
        "tail_layout": tail_layout,
        "half": half,
    }
    if layout_cache is not None:
        layout_cache[("zigzag", r)] = plan
    return plan


def _segmented_rank(key: torch.Tensor, tie: torch.Tensor) -> torch.Tensor:
    """Return each element's zero-based rank among equal ``key`` values.

    This is deliberately a fixed-shape tensor program. In particular it avoids
    ``nonzero``/masked selection and host reads, so data preparation can build a
    same-shaped source plan for every pack and copy it into TE's replay inputs.
    """
    count = key.numel()
    order = torch.argsort(key * (count + 1) + tie)
    sorted_key = key.index_select(0, order)
    positions = torch.arange(count, dtype=torch.long, device=key.device)
    is_start = torch.cat(
        (torch.ones((1,), dtype=torch.bool, device=key.device), sorted_key[1:] != sorted_key[:-1])
    )
    group_start = torch.cummax(
        torch.where(is_start, positions, torch.zeros_like(positions)), dim=0
    ).values
    rank_sorted = positions - group_start
    result = torch.empty_like(rank_sorted)
    result.scatter_(0, order, rank_sorted)
    return result


def _graph_dynamic_zigzag_plan(cu_seqlens, cu_seqlens_compressed, cp_size, l_local, r, dev):
    """Build zigzag scoring and fixed-capacity A2A metadata for one pack.

    This ordinary CUDA tensor program runs once in the data-step prebuild, not once
    per DSA layer.  Its fixed-shape outputs are passed to TE graphs as replay inputs.
    The two-hop route uses an equal ``all_to_all_single`` at both hops: each peer
    block has ``C = min(L, floor(L/N) + N - 1)`` rows and every rank sends
    ``R = N*C`` rows per hop.  ``C`` is sufficient for every permutation with
    source/destination row sums ``L``; unused slots are completed into a full relay
    permutation, which makes the exact reverse route fixed-shape as well.
    """
    N = cp_size
    L = l_local
    half = L // 2
    S = N * L
    nch = 2 * N
    cu = cu_seqlens.reshape(-1)
    dt = cu.dtype

    # Prebuild rejects these conditions before creating a replay source plan.
    # Keep tensor-side assertions here as a fail-closed boundary for direct
    # callers of this builder; replay separately checks the plan's validated_cu
    # snapshot against the refreshed graph input.
    seq_lens = cu[1:] - cu[:-1]
    torch._assert_async(cu[0] == 0)
    torch._assert_async((seq_lens >= 0).all())
    torch._assert_async(cu[-1] == S)
    torch._assert_async((seq_lens % nch == 0).all())

    # Per-sequence zigzag ownership for ALL destination ranks, in the scorer's
    # [all heads | all tails] row order.  Eligibility and the fixed physical
    # capacity are checked by prebuild_balanced_layouts before every replay.
    starts = torch.cat((cu[:-1], cu[-1:]))
    ends = torch.cat((cu[1:], torch.full_like(cu[:1], S)))
    chunks = torch.div(ends - starts, nch, rounding_mode="floor")
    torch._assert_async(2 * chunks.sum() == L)
    base = _excl_cumsum(chunks)
    rows = torch.arange(half, dtype=dt, device=dev)
    seg_id = torch.bucketize(rows, base[1:], right=True).clamp_max(starts.shape[0] - 1)
    off = rows - base.index_select(0, seg_id)
    starts_by_row = starts.index_select(0, seg_id)
    chunks_by_row = chunks.index_select(0, seg_id)
    ranks = torch.arange(N, dtype=dt, device=dev).view(N, 1)
    ordered_all = torch.cat(
        (
            starts_by_row.view(1, half) + ranks * chunks_by_row.view(1, half) + off,
            starts_by_row.view(1, half) + (nch - 1 - ranks) * chunks_by_row.view(1, half) + off,
        ),
        dim=1,
    ).long()

    # Build the two-hop route for the global permutation O[d, v].  ``src_slot``
    # scatters this source rank's L rows into relay blocks; ``dst_slot`` gathers
    # this destination rank's L rows after hop 2.  ``relay_perm`` is a full
    # permutation of [0, R), including padding-to-padding completion.
    E = N * L
    edge = torch.arange(E, dtype=torch.long, device=dev)
    dst = torch.div(edge, L, rounding_mode="floor")
    global_row = ordered_all.reshape(E)
    src = torch.div(global_row, L, rounding_mode="floor")
    edge_rank = _segmented_rank(src * N + dst, edge)
    relay = torch.remainder(edge_rank, N)
    src_relay_rank = _segmented_rank(src * N + relay, edge)

    C = min(L, L // N + N - 1)
    R = N * C
    relay_input = src * C + src_relay_rank
    src_send_slot = relay * C + src_relay_rank
    relay_dst_rank = _segmented_rank(relay * N + dst, relay_input)
    relay_output = dst * C + relay_dst_rank
    dst_recv_slot = relay * C + relay_dst_rank

    src_slots_all = torch.empty((E,), dtype=torch.long, device=dev)
    src_slots_all.scatter_(0, global_row, src_send_slot)
    src_slot = src_slots_all.view(N, L)[r]
    dst_slot = dst_recv_slot.view(N, L)[r]

    relay_valid = torch.full((N * R,), -1, dtype=torch.long, device=dev)
    relay_valid.scatter_(0, relay * R + relay_output, relay_input)
    relay_valid = relay_valid.view(N, R)
    valid_input = torch.zeros((N * R,), dtype=torch.bool, device=dev)
    valid_input.scatter_(0, relay * R + relay_input, True)
    valid_input = valid_input.view(N, R)
    valid_output = relay_valid >= 0
    positions = torch.arange(R, dtype=torch.long, device=dev).expand(N, R)
    # Unused relay inputs sort first.  Their order is paired one-for-one with
    # unused output positions to complete a genuine permutation.
    padding_inputs = torch.argsort(valid_input.long() * R + positions, dim=1)
    padding_output_rank = torch.cumsum(~valid_output, dim=1) - 1
    padding_fill = torch.take_along_dim(padding_inputs, padding_output_rank.clamp_min(0), dim=1)
    relay_perm = torch.where(valid_output, relay_valid, padding_fill)[r]

    # Dynamic scoring metadata for this rank.  Score widths and K slices remain
    # conservative/static in the consumer; only these fixed-shape tensor values
    # change with the pack composition.
    gather_idx = ordered_all[r]
    pos_head = r * chunks_by_row + off
    pos_tail = (nch - 1 - r) * chunks_by_row + off
    real_end = cu[-1]
    pos_head = torch.where(gather_idx[:half] < real_end, pos_head, torch.zeros_like(pos_head))
    pos_tail = torch.where(gather_idx[half:] < real_end, pos_tail, torch.zeros_like(pos_tail))
    cu_comp = cu_seqlens_compressed.reshape(-1)
    comp_pad = torch.cat((cu_comp, cu_comp[-1:]))
    cu_q = torch.cat((torch.zeros_like(chunks[:1]), torch.cumsum(chunks, 0))).to(dt)
    head_offsets = (r * chunks).to(dt)
    tail_offsets = ((nch - 1 - r) * chunks).to(dt)

    # Contiguous output layout for this CP rank.  It is composition metadata too,
    # so build it once with the route instead of recording its cumsum in every
    # DSA layer graph.
    global_start = r * L
    global_end = global_start + L
    local_starts = cu[:-1].clamp_min(global_start)
    local_ends = cu[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    zero = torch.zeros((1,), dtype=dt, device=dev)
    padding_q = (global_end - cu[-1].clamp_min(global_start)).clamp_min(0)
    output_cu_q = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    output_offsets = torch.cat((torch.where(q_lens > 0, local_starts - cu[:-1], 0), zero))
    plan = {
        # A private snapshot, rather than an alias of the frontend's cu tensor.
        # The consumer compares it with the replay cu input, catching a frontend
        # that updates cu_seqlens but accidentally reuses stale route metadata.
        "validated_cu": cu.clone(),
        "gather_idx": gather_idx,
        "pos_head": pos_head.int(),
        "pos_tail": pos_tail.int(),
        "score_cu_q": cu_q,
        "score_cu_kv": comp_pad,
        "head_offsets": head_offsets,
        "tail_offsets": tail_offsets,
        "output_cu_q": output_cu_q,
        "output_cu_kv": comp_pad,
        "output_offsets": output_offsets,
        "head_layout": (cu_q, comp_pad, head_offsets),
        "tail_layout": (cu_q, comp_pad, tail_offsets),
        "output_layout": (output_cu_q, comp_pad, output_offsets),
        "half": half,
        "src_slot": src_slot,
        "relay_perm": relay_perm,
        "dst_slot": dst_slot,
        "route_rows": R,
    }
    return plan


def _validate_graph_dynamic_capacity(capacity, cp_size):
    """Validate the fixed physical capacity shared by every graph replay."""
    if cp_size <= 1:
        raise ValueError(
            "graph-dynamic balanced CP route requires cp_size greater than one "
            f"(cp_size={cp_size})."
        )
    if capacity <= 0 or capacity % cp_size != 0:
        raise ValueError(
            "graph-dynamic balanced CP route capacity must be positive and divisible "
            f"by cp_size (capacity={capacity}, cp_size={cp_size})."
        )
    l_local = capacity // cp_size
    if l_local % 2 != 0:
        raise ValueError(
            "graph-dynamic balanced CP route requires an even per-rank capacity "
            f"(capacity={capacity}, cp_size={cp_size}, l_local={l_local})."
        )
    if l_local // 2 > FUSED_INDEXER_MAX_SAFE_ROWS:
        raise RuntimeError(
            f"graph-dynamic balanced CP indexer: per-rank pack capacity {l_local} would "
            f"issue fused calls of {l_local // 2} rows, above the verified-safe limit "
            f"of {FUSED_INDEXER_MAX_SAFE_ROWS}. Increase CP or reduce padded capacity."
        )
    return l_local


def build_graph_dynamic_plan(cu_seqlens, cp_group, capacity):
    """Build and return one graph-input route plan for the current microbatch.

    The data-step hook calls this exactly once per ``PackedSeqParams``. All DSA
    layers then consume this source plan through their TE input surfaces, avoiding
    composition-dependent Python caches as well as repeated in-graph sorting.
    """
    cu = cu_seqlens.reshape(-1)
    cp_size = cp_group.size()
    if cu.numel() < 2:
        raise ValueError("graph-dynamic balanced CP route requires at least one sequence.")
    l_local = _validate_graph_dynamic_capacity(capacity, cp_size)
    compressed_lens = torch.div(cu[1:] - cu[:-1], 4, rounding_mode="floor")
    cu_compressed = torch.cat(
        (torch.zeros_like(cu[:1]), torch.cumsum(compressed_lens, dim=0, dtype=torch.int32))
    )
    with torch.no_grad():
        return _graph_dynamic_zigzag_plan(
            cu, cu_compressed, cp_size, l_local, cp_group.rank(), cu.device
        )


def attach_graph_dynamic_plan(packed_seq_params, plan):
    """Attach an invocation-owned graph route without publishing process-global state."""
    setattr(packed_seq_params, _GRAPH_DYNAMIC_PLAN_ATTR, plan)


def get_graph_dynamic_plan(packed_seq_params):
    """Return the per-pack graph route, or ``None`` when the frontend did not prebuild it."""
    return getattr(packed_seq_params, _GRAPH_DYNAMIC_PLAN_ATTR, None)


def copy_graph_dynamic_plan_(destination, source):
    """Refresh an already-captured plan's tensor surfaces from another pack."""
    destination_plan = get_graph_dynamic_plan(destination)
    source_plan = get_graph_dynamic_plan(source)
    if destination_plan is None or source_plan is None:
        raise RuntimeError("both packs must have prebuilt graph-dynamic balanced CP routes")
    with torch.no_grad():
        for key in _GRAPH_DYNAMIC_PLAN_KWARGS:
            destination_plan[key].copy_(source_plan[key])


def add_graph_dynamic_plan_to_kwargs(packed_seq_params, kwargs, *, required=False):
    """Flatten a route plan into tensor-only TE CUDA graph kwargs."""
    plan = get_graph_dynamic_plan(packed_seq_params)
    if plan is None:
        if required:
            raise RuntimeError(
                "graph-dynamic balanced CP indexer is missing its per-pack route. "
                "The frontend must call prebuild_balanced_layouts before model forward."
            )
        return
    for plan_key, kwarg_key in _GRAPH_DYNAMIC_PLAN_KWARGS.items():
        kwargs[kwarg_key] = plan[plan_key]


def pop_graph_dynamic_plan_from_kwargs(kwargs, cp_size, l_local):
    """Reconstruct a route plan from TE CUDA graph tensor kwargs."""
    first_key = next(iter(_GRAPH_DYNAMIC_PLAN_KWARGS.values()))
    if first_key not in kwargs:
        return None
    missing = [name for name in _GRAPH_DYNAMIC_PLAN_KWARGS.values() if name not in kwargs]
    if missing:
        raise RuntimeError(
            "incomplete graph-dynamic balanced CP route kwargs: " + ", ".join(missing)
        )
    plan = {
        plan_key: kwargs.pop(kwarg_key)
        for plan_key, kwarg_key in _GRAPH_DYNAMIC_PLAN_KWARGS.items()
    }
    plan["half"] = l_local // 2
    plan["route_rows"] = cp_size * min(l_local, l_local // cp_size + cp_size - 1)
    plan["head_layout"] = (plan["score_cu_q"], plan["score_cu_kv"], plan["head_offsets"])
    plan["tail_layout"] = (plan["score_cu_q"], plan["score_cu_kv"], plan["tail_offsets"])
    plan["output_layout"] = (plan["output_cu_q"], plan["output_cu_kv"], plan["output_offsets"])
    return plan


def add_graph_dynamic_plan_static_inputs(static_inputs, cu_seqlens, cp_group, capacity):
    """Add a valid full-capacity sample route to a layer's TE capture inputs."""
    plan = build_graph_dynamic_plan(cu_seqlens, cp_group, capacity)
    for plan_key, kwarg_key in _GRAPH_DYNAMIC_PLAN_KWARGS.items():
        static_inputs[kwarg_key] = plan[plan_key]


def _graph_dynamic_dispatch_chunks_async(
    indexer_qr,
    weights_indexer_cp,
    cu_seqlens,
    cu_seqlens_compressed,
    cp_group,
    cp_size,
    l_local,
    plan,
):
    """Dispatch through the replay-dynamic two-hop fixed-capacity route."""
    if cu_seqlens is None or cu_seqlens_compressed is None or plan is None:
        raise RuntimeError(
            "graph-dynamic balanced CP indexer requires padded cu_seqlens, "
            "cu_seqlens_compressed, and a prebuilt per-pack graph route."
        )
    q_lora = indexer_qr.shape[-1]
    n_heads = weights_indexer_cp.shape[-1]
    q2 = indexer_qr.detach().reshape(l_local, q_lora)
    w2 = weights_indexer_cp.detach().reshape(l_local, n_heads)
    replay_cu = cu_seqlens.reshape(-1)
    torch._assert_async((plan["validated_cu"] == replay_cu).all())
    torch._assert_async(replay_cu[-1] == cp_size * l_local)
    R = plan["route_rows"]

    def _launch(payload):
        width = payload.shape[-1]
        # These buffers intentionally have invocation/graph-slot ownership.  An
        # allocation recorded during capture belongs to that graph's private pool;
        # publishing it through the process-global _A2A_BUF would make graph slots
        # alias and could leave dangling pointers after re-recording.
        # Only ``src_slot`` rows carry real payload.  Every remaining row stays
        # on the padding-only side of the completed relay permutation, so its
        # value is never observed by a destination row.  Avoid zero-filling the
        # potentially large fixed-capacity staging buffer on every replay.
        send1 = torch.empty((R, width), dtype=payload.dtype, device=payload.device)
        send1.index_copy_(0, plan["src_slot"], payload)
        recv1 = torch.empty_like(send1)
        work = dist.all_to_all_single(recv1, send1, group=cp_group, async_op=True)
        return work, send1, recv1

    if q2.dtype == w2.dtype:
        payload = torch.empty((l_local, q_lora + n_heads), dtype=q2.dtype, device=q2.device)
        payload[:, :q_lora].copy_(q2)
        payload[:, q_lora:].copy_(w2)
        work, send1, recv1 = _launch(payload)
        return {
            "kind": "gdr",
            "works": [work],
            "recv1": recv1,
            "send1": send1,
            "plan": plan,
            "q_lora": q_lora,
        }
    work_q, send_q, recv_q = _launch(q2)
    work_w, send_w, recv_w = _launch(w2)
    return {
        "kind": "gdr2",
        "works": [work_q, work_w],
        "recv_q": recv_q,
        "recv_w": recv_w,
        "send_q": send_q,
        "send_w": send_w,
        "plan": plan,
        "q_lora": q_lora,
    }


def balanced_compute_cp_indexer_topk(
    indexer_qr,  # [l_local, 1, q_lora]  detached indexer qr (pre-projection)
    weights_indexer_cp,  # [l_local, n_heads]    already-scaled indexer weights
    indexer,  # module: linear_wq_b, rotary_pos_emb, index_*_dim, qk_pos_emb_head_dim
    k_seq_major,  # [comp, head_dim]      all-gathered compressed K (full sequence)
    cu_seqlens,  # global (padded) cu_seqlens
    cu_seqlens_compressed,  # compressed cu_seqlens
    config,
    cp_group,
    cp_size,
    l_local,
    global_start,
    ratio,
    topk,
    softmax_scale,
    max_seqlen_q,
    use_fused=True,
    dispatch_handle=None,
    layout_cache=None,
    graph_dynamic_packs=False,
):
    """Balanced drop-in replacement for ``compute_cp_indexer_topk``.

    Returns ``(compressed_topk, layout)`` in the same contiguous ``[l_local, topk]`` layout the
    caller expects, so ``build_attention_indices`` / sparse attention are unchanged. Every
    sequence is tiled into ``2 * cp_size`` chunks; this rank scores chunk ``r`` (head) and chunk
    ``2 * cp_size - 1 - r`` (tail) of every sequence — one cheap and one expensive under
    the causal mask — via per-chunk calls that follow the reference (RoPE positions,
    causal offsets, packing, tight KV bounds), then combines the top-k back to contiguous
    order. Eligibility is decided independently for every pack. The CSA integration
    routes a pack with a sequence length not divisible by ``2 * cp_size`` to the
    contiguous reference path before dispatch; the internal check here still raises if
    a caller bypasses that routing and reaches the zigzag builders with an invalid pack.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    if graph_dynamic_packs:
        # Defensive even for direct callers: the opt-in path has no host route/layout
        # cache contract. All composition-dependent tensors come from the current
        # invocation's prebuilt source plan and are supplied as graph inputs.
        layout_cache = None

    q_lora = indexer_qr.shape[-1]
    n_heads, head_dim = indexer.index_n_heads, indexer.index_head_dim
    pos_dim = indexer.qk_pos_emb_head_dim
    nope_dim = head_dim - pos_dim
    dev = indexer_qr.device
    comp = int(k_seq_major.shape[0])
    S = cp_size * l_local
    r = cp_group.rank()
    graph_plan = None
    if graph_dynamic_packs:
        if dispatch_handle is None or dispatch_handle.get("kind") not in ("gdr", "gdr2"):
            raise RuntimeError(
                "graph-dynamic balanced CP indexer requires its prebuilt dynamic dispatch handle."
            )
        graph_plan = dispatch_handle["plan"]

    # Contiguous layout for the return value: downstream builds indices from it plus the (balanced,
    # re-contiguous) top-k, so it is identical to the reference path.
    def _layout_at(gs_, rows_):
        if graph_dynamic_packs:
            if gs_ == int(global_start) and rows_ == l_local:
                return graph_plan["output_layout"]
            # Do not call cp_utils._build_cp_indexer_layout here: that helper is
            # torch.compile'd, while this opt-in path intentionally records only
            # ordinary CUDA ops (the controlled GB200 toy exposed an Inductor
            # miscompile for the replay-dynamic builder). Degenerate subrange calls
            # still build their one-off layout here; the normal full-rank layout is
            # part of the per-pack source plan above.
            global_end = gs_ + rows_
            cu_q = cu_seqlens.reshape(-1)
            cu_comp = cu_seqlens_compressed.reshape(-1)
            zero = torch.zeros((1,), dtype=cu_q.dtype, device=cu_q.device)
            local_starts = cu_q[:-1].clamp_min(gs_)
            local_ends = cu_q[1:].clamp_max(global_end)
            q_lens = (local_ends - local_starts).clamp_min(0)
            q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
            padding_q = (global_end - cu_q[-1].clamp_min(gs_)).clamp_min(0)
            return (
                torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1))),
                torch.cat((cu_comp, cu_comp[-1:])),
                torch.cat((torch.where(q_lens > 0, local_starts - cu_q[:-1], 0), zero)),
            )
        # cu_seqlens is fixed for a microbatch, so layouts depend only on (offset, rows);
        # layout_cache (scoped to the microbatch's PackedSeqParams) reuses them across layers.
        if layout_cache is not None:
            cached = layout_cache.get((gs_, rows_))
            if cached is not None:
                return cached
        built = _cu._build_cp_indexer_layout(
            cu_seqlens.reshape(-1), cu_seqlens_compressed.reshape(-1), gs_, rows_
        )
        if layout_cache is not None:
            layout_cache[(gs_, rows_)] = built
        return built

    layout = _layout_at(int(global_start), l_local)

    @torch.no_grad()
    def _chunk_topk(qr_rows, w_rows, gs, sz):
        # Project -> per-chunk RoPE at the chunk's true global positions (the real ``cu_seqlens`` is
        # passed, so multi-sequence packs get correct per-segment positions) -> rotate -> reference
        # top-k. Delegating to ``compute_cp_indexer_topk`` reuses its multi-segment layout builder
        # and its fused/unfused split, so ``use_fused=False`` is honored. A query's top-k depends
        # only on its own position and K, so scoring a chunk matches the same rows of a full call
        # (up to GEMM reduction order of the chunked projection: exact score ties may resolve
        # differently; the output is integer indices with no gradient path).
        with _no_fp8_ctx():  # keep the loss path's FP8 amax stream reference-identical
            q, _ = indexer.linear_wq_b(qr_rows.reshape(sz, 1, q_lora))
        q = q.reshape(sz, n_heads, head_dim)
        if config.apply_rope_fusion:
            # cos/sin dtype from the projected q (post linear_wq_b), matching the reference path
            # (csa.py uses q_indexer_cp.dtype) so the two stay bit-identical if linear_wq_b's output
            # dtype ever diverges from the qr dtype (e.g. FP8). get_cached_cos_sin is cached.
            cos, sin = indexer.rotary_pos_emb.get_cached_cos_sin(
                max_seqlen_q, dtype=q.dtype, packed_seq=True, mscale=1.0
            )
            q = _cu.apply_thd_cp_local_rope_fused(q, cos, sin, nope_dim, pos_dim, cu_seqlens, gs)
        else:
            rpe = indexer.rotary_pos_emb(max_seqlen_q, packed_seq=True)
            rpe = rpe[0] if isinstance(rpe, tuple) else rpe
            q = _cu.apply_thd_cp_local_rope_unfused(
                q, rpe, nope_dim, pos_dim, cu_seqlens, gs, config
            )
        q = rotate_activation(q)
        # Tight capacity bounds for this chunk. The fused scorer materializes an fp32
        # ``(rows, max_seqlen_kv)`` buffer and masks its full width, so passing the global
        # sequence bounds would make every chunk pay full-sequence bandwidth (and thrash the
        # allocator). A chunk row at global offset ``< gs + sz`` has sequence-relative position
        # ``<= gs + sz - 1``, so its causal visible width is ``<= (gs + sz) // ratio``; segment
        # lengths inside the chunk are ``<= sz``.
        mq = max(1, min(int(max_seqlen_q), sz))
        gkv = max(1, int(max_seqlen_q) // int(ratio))
        # Balanced-feature policy for its own ordinary-layout calls: inside a
        # balanced run other fused calls have already been issued, so an
        # above-limit fused call here is the verified-corrupt pattern -- take the
        # unfused path instead (this layout is an ordinary cached
        # _build_cp_indexer_layout result, so unfused masking is valid). Legacy
        # callers of compute_cp_indexer_topk are NOT rerouted (see the policy note
        # on FUSED_INDEXER_MAX_SAFE_ROWS).
        use_fused_here = use_fused and sz <= _cu.FUSED_INDEXER_MAX_SAFE_ROWS
        # Exact causal need for this chunk, rounded UP to the shared width quantum
        # (see _KV_BOUND_QUANTUM/_KV_TIGHT_WIDTH_CEILING at module scope for the
        # kernel contract). Rounding up is always safe: the kernel's block count is
        # clamped by the true per-sequence KV length, so extra columns are never
        # written or read.
        bound = min(gkv, (gs + sz) // int(ratio))
        q_ = _KV_BOUND_QUANTUM
        bound_q = max(q_, (bound + q_ - 1) // q_ * q_)
        # Full width with the real layout beyond the tight-width ceiling: always
        # safe (width == declared per-sequence length, the reference call's
        # contract shape). The former single-full-pack K-prefix special case died
        # with the folding fallback — _chunk_topk now only serves the degenerate
        # exits, where the kernel either never runs or the pack is far below the
        # ceiling.
        k_pass = k_seq_major
        layout_pass = _layout_at(gs, sz)
        if bound_q <= _KV_TIGHT_WIDTH_CEILING:
            mkv = bound_q
        else:
            mkv = gkv
        tk, _ = _cu.compute_cp_indexer_topk(
            q,
            w_rows.reshape(sz, n_heads),
            k_pass,
            cu_seqlens,
            cu_seqlens_compressed,
            gs,
            ratio,
            topk,
            softmax_scale,
            max_seqlen_q=mq,
            use_fused=use_fused_here,
            max_seqlen_kv=mkv,
            prebuilt_layout=layout_pass,
        )
        return tk

    zero_width = int(max_seqlen_q) // int(ratio) == 0
    if cp_size <= 1 or comp == 0 or int(topk) == 0 or zero_width:
        # Nothing to balance / nothing to select: one call over this rank's own rows. An
        # already-issued async dispatch must still be completed, otherwise its NCCL work is
        # orphaned and the persistent staging buffers could be rewritten while still in flight.
        if dispatch_handle is not None:
            for work in dispatch_handle["works"]:
                work.wait()
        if zero_width:
            # Mirror the reference contract exactly: when every packed sequence is
            # shorter than the compress ratio, the reference's default score width
            # (max_seqlen_q // ratio) is zero and it returns (None, None) — keeping
            # ``use_indexer_loss`` off downstream. The chunk call below could not
            # reproduce that: it passes an explicit quantum-floored max_seqlen_kv,
            # so the kernel would run and return a dense all--1 top-k instead.
            return None, None
        tk = _chunk_topk(indexer_qr, weights_indexer_cp, int(global_start), l_local)
        # Mirror the reference contract: (None, None) when nothing was selected.
        return tk, (layout if tk is not None else None)

    # KNOWN KERNEL-PACKAGE DEFECT (verified on cudnn-frontend 1.26.0; no known-good
    # version demonstrated yet): a fused indexer call with more than 32768 query rows
    # is silently corrupted from row 32768 on unless it is the process's FIRST fused
    # call. The predecessor's shape is irrelevant (a bit-identical predecessor also
    # triggers it) and calls at or below the limit are immune to process history --
    # shape VARIATION between calls is explicitly safe below the limit (controlled
    # matrix in the WORKSPACE NOTE of the unit tests). Policy: the balanced
    # synthetic-layout calls fail closed above the limit (prebuild bounds l_local at
    # twice it), balanced-internal ordinary-layout calls take the unfused path, and
    # pre-existing reference/non-CP callers keep their behavior with a
    # once-per-process correctness warning (see FUSED_INDEXER_MAX_SAFE_ROWS).

    # "kind" legend lives on dispatch_chunks_async: zzr/ag/ag2 are the three zigzag
    # transports (routed a2a / merged AllGather / split AllGather); a None handle means
    # the caller skipped the early dispatch. Eligibility was already enforced by the
    # dispatch (or prebuild); re-check here only when no dispatch ran.
    if dispatch_handle is None:
        _ensure_pack_zigzag_ok(cu_seqlens, cp_group, cp_size, l_local, layout_cache)
    # ---- Per-sequence zigzag: exact balance for any pack composition -------------
    if graph_dynamic_packs:
        # Per-pack tensor-input plan: never consult layout_cache/_LAST_PLAN here.
        plan = graph_plan
    elif dispatch_handle is not None and dispatch_handle.get("kind") == "zzr":
        plan = dispatch_handle["plan"]
    else:
        plan = (layout_cache or {}).get(("zigzag", r))
        if plan is not None and plan.get("half", 0) * 2 != l_local:
            plan = None  # see dispatch_chunks_async: wrong-capacity plan
        if plan is None and _is_capturing():
            # See dispatch_chunks_async: the module slot is only trustworthy
            # under capture, where the static-composition contract (enforced by
            # prebuild_balanced_layouts) guarantees it describes this pack.
            plan = _LAST_PLAN.get((_group_key(cp_group), r))
            if plan is not None and plan.get("half", 0) * 2 != l_local:
                plan = None  # latest-slot plan built at a different capacity
        if plan is None:
            plan = _zigzag_plan(
                cu_seqlens, cu_seqlens_compressed, cp_size, l_local, r, dev, layout_cache
            )
    half = plan["half"]
    mq = max(1, min(int(max_seqlen_q), half))
    gkv = max(1, int(max_seqlen_q) // int(ratio))

    @torch.no_grad()
    def _packed_topk(qr_rows, w_rows, layout3, pos_ids, kv_rows, mkv):
        # Integer top-k output only: no gradient flows through the balanced
        # scoring, so skip autograd tracking for the per-chunk projection/RoPE.
        sz = qr_rows.shape[0]
        with _no_fp8_ctx():  # see _chunk_topk: keep the FP8 amax stream untouched
            q, _ = indexer.linear_wq_b(qr_rows.reshape(sz, 1, q_lora))
        q = q.reshape(sz, n_heads, head_dim)
        q = _rope_positions(
            q, pos_ids, layout3[0], nope_dim, pos_dim, indexer, config, int(max_seqlen_q)
        )
        q = rotate_activation(q)
        tk, _ = _cu.compute_cp_indexer_topk(
            q,
            w_rows.reshape(sz, n_heads),
            kv_rows,
            cu_seqlens,
            cu_seqlens_compressed,
            0,
            ratio,
            topk,
            softmax_scale,
            max_seqlen_q=mq,
            use_fused=True,
            max_seqlen_kv=mkv,
            prebuilt_layout=layout3,
            synthetic_layout=True,
        )
        return tk

    # Per-call tight compressed-K bounds (K-slice generalized per segment); the
    # capture-safe fallback plan carries no bounds and keeps the full width.
    # NOTE: this runtime fallback width (gkv = max_seqlen_q // ratio, the
    # reference call's contract shape) is narrower than prebuild's
    # _kv_bounds fallback (total // ratio): prebuild cannot trust
    # max_seqlen_q (frontends may leave it unset/tensor-valued), so it stays
    # conservative. Both exceed every row's causal need; only buffer size
    # differs.
    mkv_h = gkv if graph_dynamic_packs else int(plan.get("mkv_head", gkv))
    mkv_t = gkv if graph_dynamic_packs else int(plan.get("mkv_tail", gkv))
    k_rows_total = k_seq_major.shape[0]
    # Host-int invariants (free): a plan's K-slice end can never exceed the
    # physical K buffer of the pack it was built for, and a prebuilt plan must
    # have been built for this compress ratio; a violation means a stale or
    # foreign plan is being consumed. Real raises, not asserts: these guard
    # silent corruption and must survive ``python -O``. (mkv_* is a rounded-up
    # score-buffer CAPACITY — floored at the width quantum — not a need, so it
    # has no such bound.)
    if (
        plan.get("k_end_head", 0) > k_rows_total
        or plan.get("k_end_tail", 0) > k_rows_total
        or plan.get("_ratio", ratio) != ratio
    ):
        # Complete any in-flight dispatch first: an orphaned NCCL work could
        # see the persistent staging buffers rewritten during teardown. The
        # inputs to this check are per-rank host ints, so the raise itself
        # can be rank-divergent and peers may still block in their next
        # collective until the NCCL watchdog fires — an extra per-layer
        # collective to synchronize a should-never-fire verdict is not worth
        # it (unlike _validate_hep_order, which runs once per key).
        if dispatch_handle is not None:
            for work in dispatch_handle["works"]:
                work.wait()
        raise RuntimeError(
            "balanced CP indexer: stale or foreign zigzag plan (K-slice ends "
            f"{plan.get('k_end_head')}/{plan.get('k_end_tail')} vs {k_rows_total} "
            f"K rows, plan ratio {plan.get('_ratio')} vs {ratio})."
        )
    # NOTE: k_h / k_t must stay prefix VIEWS of the full gathered buffer. The
    # packed layouts may declare per-sequence K ranges past the slice end
    # (mkv_* is a capacity, k_end_* the true causal need); reads past the
    # slice land on valid full-buffer memory only while these are views — a
    # .contiguous() here would turn them into real out-of-bounds reads.
    k_h = (
        k_seq_major[: plan["k_end_head"]]
        if plan.get("k_end_head", k_rows_total) < k_rows_total
        else k_seq_major
    )
    k_t = (
        k_seq_major[: plan["k_end_tail"]]
        if plan.get("k_end_tail", k_rows_total) < k_rows_total
        else k_seq_major
    )

    nvtx_range_push("Bal_Dispatch")
    if dispatch_handle is not None:
        for work in dispatch_handle["works"]:
            work.wait()
        # The async collectives have consumed their send tensors.  Drop those
        # references before allocating the second hop/scoring buffers so the
        # CUDA graph pool can reuse their storage during capture and replay.
        dispatch_handle["works"].clear()
        if dispatch_handle["kind"] == "gdr":
            qlw = dispatch_handle["q_lora"]
            dispatch_handle.pop("send1", None)
            recv1 = dispatch_handle.pop("recv1")
            send2 = torch.index_select(recv1, 0, plan["relay_perm"])
            recv2 = torch.empty_like(send2)
            dist.all_to_all_single(recv2, send2, group=cp_group)
            rows = torch.index_select(recv2, 0, plan["dst_slot"])
            qr_h, w_h = rows[:half, :qlw].contiguous(), rows[:half, qlw:].contiguous()
            qr_t, w_t = rows[half:, :qlw].contiguous(), rows[half:, qlw:].contiguous()
            del recv1, send2, recv2, rows
        elif dispatch_handle["kind"] == "gdr2":
            dispatch_handle.pop("send_q", None)
            dispatch_handle.pop("send_w", None)
            first_q = dispatch_handle.pop("recv_q")
            send_q = torch.index_select(first_q, 0, plan["relay_perm"])
            recv_q = torch.empty_like(send_q)
            dist.all_to_all_single(recv_q, send_q, group=cp_group)
            qh = torch.index_select(recv_q, 0, plan["dst_slot"])
            qr_h, qr_t = qh[:half].contiguous(), qh[half:].contiguous()
            del first_q, send_q, recv_q, qh

            first_w = dispatch_handle.pop("recv_w")
            send_w = torch.index_select(first_w, 0, plan["relay_perm"])
            recv_w = torch.empty_like(send_w)
            dist.all_to_all_single(recv_w, send_w, group=cp_group)
            wh = torch.index_select(recv_w, 0, plan["dst_slot"])
            w_h, w_t = wh[:half].contiguous(), wh[half:].contiguous()
            del first_w, send_w, recv_w, wh
        elif dispatch_handle["kind"] == "zzr":
            qlw = dispatch_handle["q_lora"]
            recv = dispatch_handle["recv"]
            rows = torch.empty_like(recv)
            rows.index_copy_(0, plan["disp_recv_rows"], recv)
            qr_h, w_h = rows[:half, :qlw].contiguous(), rows[:half, qlw:].contiguous()
            qr_t, w_t = rows[half:, :qlw].contiguous(), rows[half:, qlw:].contiguous()
        elif dispatch_handle["kind"] == "ag":
            g = dispatch_handle["g"]
            qlw = dispatch_handle["q_lora"]
            rows = torch.index_select(g, 0, plan["gather_idx"])
            qr_h, w_h = rows[:half, :qlw].contiguous(), rows[:half, qlw:].contiguous()
            qr_t, w_t = rows[half:, :qlw].contiguous(), rows[half:, qlw:].contiguous()
        else:
            gq, gw = dispatch_handle["gq"], dispatch_handle["gw"]
            qh = torch.index_select(gq, 0, plan["gather_idx"])
            wh = torch.index_select(gw, 0, plan["gather_idx"])
            qr_h, qr_t = qh[:half].contiguous(), qh[half:].contiguous()
            w_h, w_t = wh[:half].contiguous(), wh[half:].contiguous()
    else:
        _warn_allgather_fallback(_NO_PLAN_REASON)
        q2 = indexer_qr.detach().reshape(l_local, q_lora)
        w2 = weights_indexer_cp.detach().reshape(l_local, n_heads)
        wq, gq = _all_gather_rows_buf(q2, l_local, cp_group, cp_size, "zz_q")
        ww, gw = _all_gather_rows_buf(w2, l_local, cp_group, cp_size, "zz_w")
        wq.wait()
        ww.wait()
        qh = torch.index_select(gq, 0, plan["gather_idx"])
        wh = torch.index_select(gw, 0, plan["gather_idx"])
        qr_h, qr_t = qh[:half].contiguous(), qh[half:].contiguous()
        w_h, w_t = wh[:half].contiguous(), wh[half:].contiguous()
    nvtx_range_pop("Bal_Dispatch")

    if graph_dynamic_packs:
        head_layout = (plan["score_cu_q"], plan["score_cu_kv"], plan["head_offsets"])
        tail_layout = (plan["score_cu_q"], plan["score_cu_kv"], plan["tail_offsets"])
    else:
        head_layout, tail_layout = plan["head_layout"], plan["tail_layout"]

    nvtx_range_push("BalancedIndexerScore")
    nvtx_range_push("Bal_Head")
    tk_head = _packed_topk(qr_h, w_h, head_layout, plan["pos_head"], k_h, mkv_h)
    del qr_h, w_h
    nvtx_range_pop("Bal_Head")
    nvtx_range_push("Bal_Tail")
    tk_tail = _packed_topk(qr_t, w_t, tail_layout, plan["pos_tail"], k_t, mkv_t)
    del qr_t, w_t
    nvtx_range_pop("Bal_Tail")
    nvtx_range_pop("BalancedIndexerScore")

    nvtx_range_push("Bal_Combine")
    tkw = tk_head.shape[-1]
    if graph_dynamic_packs:
        # Exact reverse of the two equal-split dispatch hops.
        send1 = torch.empty((plan["route_rows"], tkw), dtype=tk_head.dtype, device=tk_head.device)
        send1.index_copy_(0, plan["dst_slot"][:half], tk_head)
        send1.index_copy_(0, plan["dst_slot"][half:], tk_tail)
        del tk_head, tk_tail
        recv1 = torch.empty_like(send1)
        dist.all_to_all_single(recv1, send1, group=cp_group)
        del send1
        # relay_perm is a full permutation, including padding-to-padding
        # completion, so index_copy_ initializes every row.
        send2 = torch.empty_like(recv1)
        send2.index_copy_(0, plan["relay_perm"], recv1)
        del recv1
        recv2 = torch.empty_like(send2)
        dist.all_to_all_single(recv2, send2, group=cp_group)
        del send2
        compressed_topk = torch.index_select(recv2, 0, plan["src_slot"])
        del recv2
    else:
        ht = _a2a_buf("zz_cmb_send", l_local, tkw, tk_head.dtype, dev, cp_group)
        ht[:half].copy_(tk_head)
        ht[half:].copy_(tk_tail)
    if not graph_dynamic_packs and "cmb_send_rows" in plan:
        # Route-A2A combine: exact inverse exchange, ~l_local rows per rank.
        send = _a2a_buf("zzr_cmb_send", l_local, tkw, tk_head.dtype, dev, cp_group)
        send.copy_(torch.index_select(ht, 0, plan["cmb_send_rows"]))
        recv = _a2a_buf("zzr_cmb_recv", l_local, tkw, tk_head.dtype, dev, cp_group)
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=plan["disp_in_splits"],
            input_split_sizes=plan["disp_out_splits"],
            group=cp_group,
        )
        compressed_topk = torch.empty((l_local, tkw), dtype=tk_head.dtype, device=dev)
        compressed_topk.index_copy_(0, plan["cmb_recv_rows"], recv)
    elif not graph_dynamic_packs:
        Z = _a2a_buf("zz_cmb_recv", S, tkw, tk_head.dtype, dev, cp_group, persistent=False)
        dist.all_gather_into_tensor(Z, ht, group=cp_group)
        compressed_topk = torch.index_select(Z, 0, plan["inv_idx"])
    nvtx_range_pop("Bal_Combine")
    return compressed_topk, layout


def prebuild_balanced_layouts(
    packed_seq_params,
    cp_group=None,
    pad_alignment=None,
    capacity=None,
    graphs_enabled=False,
    graph_dynamic_packs=False,
):
    """Data-prep-time prebuild of the balanced-indexer zigzag plan and multi-seq gate.

    Mirrors ``context_parallel_layout.prebuild_thd_cp_partition_routes``: this runs
    where host syncs are free, so the forward path never has to build layout
    metadata (or run its one D2H segment-count probe) inside a CUDA graph capture.

    The zigzag ownership indices come from
    ``context_parallel_layout`` zigzag segment ownership — the shared
    canonical CP layout definition — reordered into the ``[heads | tails]`` packing
    that the fused indexer requires (its packed calls allow only one segment per
    sequence per call, so head and tail chunks go into two separate calls). The
    static-shape device-side builder in ``_zigzag_plan`` remains as the
    capture-safe fallback for callers that do not prebuild.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return
    if cp_group is None:
        cp_group = getattr(packed_seq_params, "cp_group", None)
    if cp_group is None or cp_group.size() <= 1:
        return
    from megatron.core.context_parallel_layout.utils import (
        get_packed_seq_params_cp_partition_cu_seqlens,
    )

    cu = get_packed_seq_params_cp_partition_cu_seqlens(packed_seq_params)
    if cu is None:
        return
    N, r = cp_group.size(), cp_group.rank()
    cu = cu.reshape(-1)
    if graph_dynamic_packs and cu.numel() < 2:
        raise ValueError(
            "graph-dynamic balanced CP indexer requires cu_seqlens for at least one sequence."
        )
    metadata_total = int(cu[-1].item())
    if graph_dynamic_packs:
        if capacity is None:
            raise ValueError(
                "graph-dynamic balanced CP indexer requires an explicit fixed physical "
                "capacity from the graph configuration."
            )
        total = int(capacity)
    else:
        total = metadata_total
    if not graph_dynamic_packs and capacity is not None and 0 < total < capacity:
        # Middle-PP-stage PackedSeqParams carry the raw (unpadded) cu while the
        # hidden states are padded: probe and build at the PHYSICAL pack size.
        # Under graphs the pack is padded to the full target capacity; in eager
        # integer-alignment mode it is padded only to round_up(total/N, align)*N,
        # which is generally SMALLER — using the full capacity there would tag
        # every plan at the wrong l_local and silently degrade middle stages to
        # the runtime fallback. Keep the raw sequence boundaries and model
        # [raw_total, physical) as the zero-K pseudo-sequence (the runtime
        # _zigzag_plan semantics), so the compressed-K geometry matches the K
        # buffer the forward builds from the raw cu.
        if not graphs_enabled and isinstance(pad_alignment, int) and pad_alignment > 0:
            per_rank = (total + N - 1) // N
            per_rank = -(-per_rank // pad_alignment) * pad_alignment
            capacity = min(capacity, per_rank * N)
        if total < capacity:
            total = capacity
    if graph_dynamic_packs:
        l_local = _validate_graph_dynamic_capacity(total, N)
    else:
        l_local = total // N if total > 0 else 0

    if graph_dynamic_packs:
        # The opt-in graph path validates and builds one invocation-owned tensor
        # plan here. It deliberately publishes neither a host layout cache nor a
        # module-level verdict; TE receives the source plan through fixed-shape
        # per-callable input surfaces.
        cu_list_dynamic = [int(v) for v in cu.tolist()]
        if cu_list_dynamic[0] != 0:
            raise ValueError(
                "graph-dynamic balanced CP indexer requires cu_seqlens to start at zero."
            )
        if any(e < s for s, e in zip(cu_list_dynamic[:-1], cu_list_dynamic[1:])):
            raise ValueError("graph-dynamic balanced CP indexer requires monotonic cu_seqlens.")
        if metadata_total != total:
            raise ValueError(
                "graph-dynamic balanced CP indexer requires padded cu_seqlens to end at "
                f"the fixed physical capacity ({total}), got {metadata_total}."
            )
        seq_lens_dynamic = [e - s for s, e in zip(cu_list_dynamic[:-1], cu_list_dynamic[1:])] + [
            total - cu_list_dynamic[-1]
        ]
        nch = 2 * N
        if any(sl % nch for sl in seq_lens_dynamic):
            raise ValueError(
                "graph-dynamic balanced CP indexer cannot fall back during replay: every "
                "padded sequence length and the capacity tail must be divisible by "
                f"2 * cp_size = {nch} (l_local={l_local}, total={total})."
            )
        # Build exactly once per microbatch.  The resulting fixed-shape tensors
        # become replay inputs to every captured DSA layer; no route builder or
        # composition-dependent host cache lives inside an attention graph.
        attach_graph_dynamic_plan(packed_seq_params, build_graph_dynamic_plan(cu, cp_group, total))
        return

    gkey = (_group_key(cp_group), l_local)

    def _stash_verdict(ok):
        # Per-pack OK verdict on the microbatch's own cache (consulted first by
        # ``_ensure_pack_zigzag_ok``, so a prebuilt forward never re-probes) plus the
        # module registry (the capture-time fallback, keyed by (group, l_local)).
        cache_ = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
        if cache_ is None:
            cache_ = {}
            packed_seq_params._dsa_cp_balance_layout_cache = cache_
        cache_["zz_pack_ok"] = (l_local, ok)
        _ZZ_PACK_OK[gkey] = ok

    # Static-composition observation: recorded UNCONDITIONALLY on every prebuild
    # call, before every early return or invariant raise, so no pack — including
    # ones that never build a plan, and all-empty packs — can slip past the
    # contract unnoticed; a mixed flow whose eager warmup ran without
    # graphs still contributes observations. The raise itself only applies when
    # CUDA graphs are enabled (eager varlen is ordinary operation).
    seen_key = _group_key(cp_group)
    seen_cu = (l_local, [int(v) for v in cu.tolist()])
    seen = _SEEN_CU.get(seen_key)
    if graphs_enabled and seen is not None and seen != seen_cu:
        raise RuntimeError(
            "dsa_cp_balance_indexer with CUDA graphs requires a static pack "
            "composition, but cu_seqlens changed between microbatches (note that "
            "validation/eval microbatches count too). Disable the flag or CUDA "
            "graphs for varying-composition (varlen) data; varlen-with-graphs "
            "support lands in a follow-up PR."
        )
    _SEEN_CU[seen_key] = seen_cu

    if total <= 0:
        # All-empty pack: nothing to score; the forward exits before any fused call.
        return
    if total % N != 0:
        raise ValueError(
            "balanced CP indexer: physical pack capacity must be divisible by "
            f"cp_size (total={total}, cp_size={N})."
        )

    cu_list = seen_cu[1]
    # Same sequence enumeration as _zigzag_plan: real segments plus the
    # capacity-padding pseudo-sequence [cu[-1], total) (empty for full packs).
    seq_lens_list = [e - s for s, e in zip(cu_list[:-1], cu_list[1:])] + [total - cu_list[-1]]
    if l_local % 2 != 0 or any(sl % (2 * N) for sl in seq_lens_list):
        # Not zigzag-representable: record the verdict so the forward routes this
        # microbatch to the contiguous reference path. The configured alignment is
        # deliberately not part of this decision: only the physical capacity and
        # actual sequence boundaries determine whether the zigzag layout is legal.
        _stash_verdict(False)
        return

    prev = _LAST_PLAN.get((_group_key(cp_group), r))
    if prev is not None and prev.get("half", 0) * 2 != l_local:
        # Latest-slot plan was built at a different capacity (e.g. a raw middle-stage
        # cu probed with a different capacity hint): not reusable for idempotency.
        prev = None

    # The unified zigzag path also serves single-full-sequence packs (the per-sequence
    # zigzag of one pack-spanning sequence is the plain 2N-fold of the whole pack), so the
    # plan and its A2A routes are built for every eligible pack composition.

    # Idempotent prebuild: for static pack compositions (every CUDA-graph run, and
    # any fixed-length workload) the plan content is identical each microbatch.
    # Reuse the previous plan object instead of re-allocating its tensors — this
    # keeps steady-state allocations at zero (no expandable-segments cuMem churn)
    # and keeps capture-baked pointers valid.
    if prev is not None and prev.get("_cu_list") == cu_list:
        cache = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
        if cache is None:
            cache = {}
            packed_seq_params._dsa_cp_balance_layout_cache = cache
        cache[("zigzag", r)] = prev
        _stash_verdict(True)
        return

    # The balanced row-limit applies only when a plan will actually be built: an
    # ineligible pack above the limit routes to the reference path. That legacy
    # path preserves its fused behavior and emits the shared backend warning, so
    # this balanced-only prebuild must not reject it here.
    if l_local // 2 > FUSED_INDEXER_MAX_SAFE_ROWS:
        # The balanced path scores two packed fused calls of l_local // 2 rows each;
        # above this bound the fused kernel package silently corrupts rows >= 32768
        # (see FUSED_INDEXER_MAX_SAFE_ROWS in dsa_fused_safety.py). Fail at data-prep time
        # with the remedy rather than at the first forward.
        raise RuntimeError(
            f"balanced CP indexer: per-rank pack capacity {l_local} would issue fused "
            f"indexer calls of {l_local // 2} rows, above the verified-safe limit of "
            f"{FUSED_INDEXER_MAX_SAFE_ROWS} for the current fused kernel package. "
            "Increase the CP degree or reduce the pack capacity, or run the indexer "
            "unfused."
        )

    dev, dt = cu.device, cu.dtype
    half = l_local // 2
    c_list = [sl // (2 * N) for sl in seq_lens_list]
    assert 2 * sum(c_list) == l_local, (c_list, l_local)

    # Canonical zigzag ownership (chunk ``rho`` + chunk ``2N-1-rho`` of every
    # sequence, incl. the capacity-padding pseudo-sequence [cu[-1], total)),
    # enumerated directly in the [heads | tails] packed-call order. Mirrors the
    # segment math in ``context_parallel_layout.routes``; vectorized like
    # ``_zigzag_plan`` so a changed pack costs tensor ops, not O(S * N) Python.
    # TODO(cp-layout): reuse a public per-rank ownership helper once
    # ``context_parallel_layout`` exports one again.
    starts64 = torch.tensor(cu_list[:-1] + [cu_list[-1]], device=dev, dtype=torch.long)
    c64 = torch.tensor(c_list, device=dev, dtype=torch.long)
    nseg = starts64.numel()
    base = torch.cat((torch.zeros(1, device=dev, dtype=torch.long), torch.cumsum(c64, 0)[:-1]))
    rows_h = torch.arange(half, device=dev, dtype=torch.long)
    seg_id = torch.bucketize(rows_h, base[1:] if nseg > 1 else base[:0], right=True).clamp_max(
        nseg - 1
    )
    off = rows_h - base[seg_id]

    def _ordered_rows(rho):
        head = starts64[seg_id] + rho * c64[seg_id] + off
        tail = starts64[seg_id] + (2 * N - 1 - rho) * c64[seg_id] + off
        return torch.cat((head, tail))

    gather_idx = _ordered_rows(r)

    # Sequence-relative positions for RoPE; pseudo-sequence rows carry position 0
    # (see the matching clamp in _zigzag_plan).
    pos = torch.cat(((r * c64)[seg_id] + off, ((2 * N - 1 - r) * c64)[seg_id] + off))
    pos = torch.where(gather_idx < cu_list[-1], pos, torch.zeros_like(pos))
    # int32: same dtype the reference path feeds the fused RoPE kernel.
    pos_head, pos_tail = pos[:half].int(), pos[half:].int()

    # Inverse permutation: my contiguous rows' positions in the rank-major
    # [heads | tails] all-gather concat — derived from the canonical indices.
    pos_global = torch.empty(total, device=dev, dtype=torch.long)
    ar = torch.arange(l_local, device=dev, dtype=torch.long)
    for rho in range(N):
        pos_global[_ordered_rows(rho)] = rho * l_local + ar
    mine = torch.arange(r * l_local, (r + 1) * l_local, device=dev, dtype=torch.long)
    inv_idx = pos_global.index_select(0, mine)

    # Packed call layouts. ratio == 4 is the only compress ratio that instantiates
    # an indexer (mirrors CSAAttention.__init__), so its compressed cu is fixed here.
    ratio = 4
    c_t = torch.tensor(c_list, device=dev, dtype=torch.int64)
    cu_q = torch.cat((torch.zeros(1, device=dev, dtype=torch.int64), torch.cumsum(c_t, 0))).to(dt)
    comp_lens = torch.div(cu[1:] - cu[:-1], ratio, rounding_mode="floor")
    cu_comp = torch.cat(
        (torch.zeros_like(cu[:1]), torch.cumsum(comp_lens, dim=0, dtype=torch.int32))
    ).reshape(-1)
    comp_pad = torch.cat((cu_comp, cu_comp[-1:]))
    nch = 2 * N

    # ---- route-A2A metadata (PR #5664-style precomputed exchange) --------------------
    # Dispatch: contiguous rows -> my [heads | tails] order. Sender convention: rows are
    # sent to each destination ordered by the DESTINATION's [heads | tails] order, so the
    # receiver's per-source blocks interleave back via one index_copy_.
    ordered_all = [_ordered_rows(rho) for rho in range(N)]
    src_of = torch.div(ordered_all[r], l_local, rounding_mode="floor")
    disp_out_splits = torch.bincount(src_of, minlength=N).tolist()
    disp_recv_rows = torch.argsort(src_of, stable=True)
    send_parts, disp_in_splits = [], []
    for dst in range(N):
        need = ordered_all[dst]
        m = torch.div(need, l_local, rounding_mode="floor") == r
        rows = need[m] - r * l_local
        send_parts.append(rows)
        disp_in_splits.append(int(rows.numel()))
    disp_send_rows = torch.cat(send_parts).long()
    # Combine (inverse): my computed [heads | tails] rows -> their contiguous owners.
    # Send grouped by owner in my-order (stable) == the dispatch recv permutation; the
    # receiver's arrival order is (computer rank, computer-local position) == pos_global.
    cmb_send_rows = disp_recv_rows
    cmb_recv_rows = torch.argsort(inv_idx, stable=True)

    # ---- per-call tight compressed-K bounds (K-slice generalized per segment) --------
    gkv = max(1, total // ratio)
    comp_lens_list = [int(v) for v in comp_lens.tolist()] + [0]
    cu_comp_list = [int(v) for v in cu_comp.tolist()]

    def _kv_bounds(chunk_idx):
        spans, ends = [1], [1]
        for i, (ci, cl) in enumerate(zip(c_list, comp_lens_list)):
            if ci == 0 or cl == 0:
                continue
            span = min(cl, -(-((chunk_idx + 1) * ci) // ratio))
            spans.append(span)
            ends.append(cu_comp_list[i] + span)
        span = max(spans)
        q_ = _KV_BOUND_QUANTUM
        bound = max(q_, ((span + q_ - 1) // q_) * q_)
        # Tight widths past the ceiling are unsafe (see the module-scope kernel
        # contract note); fall back to the full compressed width.
        # max(1, ...): a degenerate pack whose every sequence is shorter than the
        # compress ratio has zero compressed rows; k_end == 0 would slice an empty K.
        # (The consumer's ``comp == 0`` early-exit fires first for truly empty
        # buffers, so the clamp is pure defense.)
        if bound > _KV_TIGHT_WIDTH_CEILING or bound >= gkv:
            return gkv, max(1, cu_comp_list[-1])
        return bound, max(1, min(max(ends), cu_comp_list[-1]))

    mkv_head, k_end_head = _kv_bounds(r)
    mkv_tail, k_end_tail = _kv_bounds(nch - 1 - r)

    plan = {
        "gather_idx": gather_idx.long(),
        "inv_idx": inv_idx.long(),
        "pos_head": pos_head,
        "pos_tail": pos_tail,
        "head_layout": (cu_q, comp_pad, (r * c_t).to(dt)),
        "tail_layout": (cu_q, comp_pad, ((nch - 1 - r) * c_t).to(dt)),
        "half": half,
        "disp_send_rows": disp_send_rows,
        "disp_in_splits": disp_in_splits,
        "disp_out_splits": disp_out_splits,
        "disp_recv_rows": disp_recv_rows.long(),
        "cmb_send_rows": cmb_send_rows.long(),
        "cmb_recv_rows": cmb_recv_rows.long(),
        "mkv_head": mkv_head,
        "k_end_head": k_end_head,
        "mkv_tail": mkv_tail,
        "k_end_tail": k_end_tail,
        "_ratio": ratio,
        "_cu_list": cu_list,
    }
    # Composition changes were observed (raise / varies-pin) before the early
    # returns above; here we only publish the fresh plan object. The old object is
    # never mutated in place.
    cache = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
    if cache is None:
        cache = {}
        packed_seq_params._dsa_cp_balance_layout_cache = cache
    cache[("zigzag", r)] = plan
    _LAST_PLAN[(_group_key(cp_group), r)] = plan
    _stash_verdict(True)
