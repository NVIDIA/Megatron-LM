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
Eligibility is a RUN-LEVEL INVARIANT: config validation requires the fused backend and a
pack-tail alignment divisible by ``2 * cp_size``, and a pack that violates per-sequence or
tail divisibility RAISES (at prebuild when prebuilt, else at dispatch) instead of falling
back — the fused kernel package silently corrupts fused calls whose row shape differs from
earlier calls in the process, so per-microbatch balanced/reference switching is forbidden.
Fused calls issued by this module additionally pin the per-process row count
(``_pin_fused_call_rows``), so a varying pack capacity raises as well. (The former
chunk-pair folding fallback measured slower than the unbalanced baseline on unequal packs
and was removed.)

``balanced_compute_cp_indexer_topk`` is a drop-in replacement for
``csa_utils.cp_utils.compute_cp_indexer_topk``: it returns the top-k in the same contiguous
``[l_local, topk]`` layout, so the downstream index-building and sparse attention are unchanged.

Triage switch: ``MCORE_DSA_CP_BAL_DEBUG=1`` logs the eligibility decision once.

CUDA-graph contract: graph support is scoped to static pack compositions at PP=1 and is
enforced through ``prebuild_balanced_layouts`` — under CUDA graphs it MUST be called every
microbatch (as ``pretrain_gpt.get_batch`` does); frontends that skip it lose the
composition-change detection and are protected only by the in-graph divisibility assert.
"""

import logging
import os

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
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

# Per-(group, l_local) OK verdict from the last prebuild or eager probe: the CURRENT
# pack is zigzag-representable (every padded sequence length divisible by 2 * cp_size).
# Only successes are recorded — a violating pack raises (eligibility is a run-level
# invariant). Capture consults this registry because probing is impossible while
# recording a graph.
_ZZ_PACK_OK: dict = {}

# Last OBSERVED (l_local, cu composition) per group, recorded on every prebuild
# call — independent of whether a plan is ever built. The static-composition
# gate compares against this, not against a plan: a reference-fallback run (e.g. pad
# alignment that never opens the zigzag gate) builds no plans, yet its captured
# graphs still bake composition-sensitive host state (mq/gkv widths derived
# from max_seqlen_q, and the gate/single-multi host branches), so a composition
# change must fail loudly there too.
_SEEN_CU: dict = {}

# Fail-closed guard for the fused kernel package's cross-call hazard (see the
# WORKSPACE NOTE in test_cp_balanced_indexer_layout.py): a fused call preceded by
# fused calls of OTHER row counts in the same process can be silently corrupted,
# and the falsified priming experiment shows no warmup scheme protects it. Every
# fused indexer call issued by this module therefore pins the process to a single
# row count; a call that would change it raises instead of risking wrong top-k
# indices. Scope: calls issued by this module only — it cannot see fused indexer
# calls made elsewhere in the process. Under CUDA graphs the composition is static
# (enforced at prebuild), so the pin never trips there.
_FUSED_CALL_ROWS: dict = {}


def _pin_fused_call_rows(rows: int) -> None:
    """Pin the per-process fused-call row count; raise on a transition."""
    rows = int(rows)
    prev = _FUSED_CALL_ROWS.get("rows")
    if prev is None:
        _FUSED_CALL_ROWS["rows"] = rows
        return
    if prev != rows:
        raise RuntimeError(
            f"balanced CP indexer: fused indexer call with {rows} rows after this "
            f"process pinned {prev} rows. The fused kernel package carries cross-call "
            "state that silently corrupts calls whose shape differs from earlier calls "
            "(see the WORKSPACE NOTE in test_cp_balanced_indexer_layout.py), so this "
            "fails closed. Keep one fused-call shape per run: use a fixed pack "
            "capacity and make every (padded) sequence length divisible by "
            "2 * cp_size so all packs stay on the zigzag path, or run the indexer "
            "unfused (attention_backend=unfused / dsa_kernel_backend='none'), or "
            "disable dsa_cp_balance_indexer."
        )


# Sentinel default for prebuild_balanced_layouts(pad_alignment=...): "the caller did
# not supply the config's pad alignment" (build routes; the forward gate decides).
# Passing None explicitly means "the config HAS no packed-seq alignment", for which
# the zigzag gate can never open and route construction is skipped.
_PAD_UNSPECIFIED = object()


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
    config=None,
    layout_cache=None,
    cu_seqlens=None,
):
    """Issue the chunk dispatch as early as possible; returns an opaque handle.

    Called by the CSA forward right after ``indexer_qr``/``weights_indexer_cp`` are
    produced, so the all_to_all overlaps with the compressed-K/KV all-gathers already in
    flight (and the local top-k preparation) instead of sitting on the critical path
    right before the top-k. ``balanced_compute_cp_indexer_topk(dispatch_handle=...)`` waits
    on it. When qr and weights share a dtype they ride one all_to_all (single launch).
    ``cu_seqlens`` feeds the fail-fast pack-eligibility check
    (``_ensure_pack_zigzag_ok``); the compute side follows the returned handle's kind,
    so both stay consistent.

    Handle ``kind`` legend — which transport carried the (qr | weights) payload:

    - ``"zzr"``: zigzag + prebuilt-route ``all_to_all_single`` — each rank exchanges
      only ~``l_local`` rows; splits/rows come from ``prebuild_balanced_layouts``
      (host ints), so the exchange is CUDA-graph capturable.
    - ``"ag"``: zigzag fallback — no usable routed plan (never prebuilt, capacity
      mismatch, or plan lacks route fields): one static-shape S-row AllGather of the
      merged payload; row selection is deferred to consume time.
    - ``"ag2"``: same fallback, but qr/weights dtypes differ so the merged payload
      cannot carry both — two separate AllGathers.
    - ``None``: ``cp_size <= 1`` — nothing to balance, nothing to prefetch.
    """
    if cp_size <= 1:
        return None
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


def _ensure_pack_zigzag_ok(cu_seqlens, cp_group, cp_size, l_local, layout_cache):
    """Fail fast unless the ACTUAL microbatch pack is zigzag-representable.

    Eligibility is a run-level invariant (config validation requires the fused backend
    and a compatible ``pad_packed_seq_alignment``); this check enforces the per-pack
    half of the contract — every (padded) sequence length, the capacity tail, and the
    per-rank row count must divide ``2 * cp_size`` — and RAISES on violation instead of
    falling back, because the fused kernel package silently corrupts fused calls whose
    row shape differs from earlier calls. Success verdicts are cached on the microbatch
    ``layout_cache`` (written by prebuild or an earlier probe of the same pack) and in
    the module registry keyed by (group, l_local); probing is impossible under capture,
    so capture requires a recorded verdict (eager warmup or prebuild).
    """
    if layout_cache is not None:
        cached = layout_cache.get("zz_pack_ok")
        if cached is not None and cached[0] == l_local and cached[1]:
            # Verdicts are only valid for the capacity they were probed at: prebuild
            # may have seen a padded cu ending short of the physical pack.
            return
    key = (_group_key(cp_group), l_local)
    if _is_capturing():
        if _ZZ_PACK_OK.get(key):
            return
        # Probing is impossible here and guessing would bake an unverified branch.
        raise RuntimeError(
            "balanced CP indexer: no pack-eligibility verdict is available during "
            "graph capture; run an eager warmup (or prebuild_balanced_layouts) "
            "before capturing."
        )
    if cu_seqlens is None:
        if _ZZ_PACK_OK.get(key):
            return
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
    if not ok:
        raise ValueError(
            "balanced CP indexer: this pack is not zigzag-representable "
            f"(l_local={l_local}, total={total}, capacity={S}; every padded sequence "
            f"length and the capacity tail must be divisible by 2 * cp_size = {nch}). "
            "Balanced eligibility is a run-level invariant — fix the dataset padding / "
            "pad_packed_seq_alignment so every pack conforms, or disable "
            "dsa_cp_balance_indexer; there is no per-microbatch fallback because the "
            "fused kernel package silently corrupts shape-alternating calls."
        )
    if _GATE_DEBUG and not getattr(_ensure_pack_zigzag_ok, "_logged", False):
        _ensure_pack_zigzag_ok._logged = True
        logger.info("[zz-gate] pack eligible: S=%s l_local=%s nch=%s", S, l_local, nch)
    _ZZ_PACK_OK[key] = True
    if layout_cache is not None:
        layout_cache["zz_pack_ok"] = (l_local, True)


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
    to be divisible by ``2 * cp_size`` (guaranteed by ``pad_packed_seq_alignment`` >= 2*cp).
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
    multi_seq=True,
):
    """Balanced drop-in replacement for ``compute_cp_indexer_topk``.

    Returns ``(compressed_topk, layout)`` in the same contiguous ``[l_local, topk]`` layout the
    caller expects, so ``build_attention_indices`` / sparse attention are unchanged. Every
    sequence is tiled into ``2 * cp_size`` chunks; this rank scores chunk ``r`` (head) and chunk
    ``2 * cp_size - 1 - r`` (tail) of every sequence — one cheap and one expensive under
    the causal mask — via per-chunk calls that follow the reference (RoPE positions,
    causal offsets, packing, tight KV bounds), then combines the top-k back to contiguous
    order. Eligibility is a run-level invariant: a pack the zigzag builders cannot
    represent (a sequence length not divisible by ``2 * cp_size``) RAISES — at prebuild
    when prebuilt, else at the dispatch/compute check here — instead of falling back,
    because the fused kernel package silently corrupts fused calls whose row shape
    differs from earlier calls in the process.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    q_lora = indexer_qr.shape[-1]
    n_heads, head_dim = indexer.index_n_heads, indexer.index_head_dim
    pos_dim = indexer.qk_pos_emb_head_dim
    nope_dim = head_dim - pos_dim
    dev = indexer_qr.device
    comp = int(k_seq_major.shape[0])
    S = cp_size * l_local
    r = cp_group.rank()

    # Contiguous layout for the return value: downstream builds indices from it plus the (balanced,
    # re-contiguous) top-k, so it is identical to the reference path.
    def _layout_at(gs_, rows_):
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

    # One full-pack sequence (the extreme long-context case): every chunk's rows belong to
    # that sequence, so a chunk can score against a sliced K prefix with an exact synthetic
    # layout (see _chunk_topk).
    single_full_seq = not multi_seq

    def _kslice_layout(gs_, rows_, kb_):
        key = ("kslice", gs_, rows_, kb_)
        if layout_cache is not None:
            cached = layout_cache.get(key)
            if cached is not None:
                return cached
        dt = cu_seqlens.dtype
        built = (
            torch.tensor([0, rows_, rows_], dtype=dt, device=dev),
            torch.tensor([0, kb_, kb_], dtype=dt, device=dev),
            torch.tensor([gs_, 0], dtype=dt, device=dev),
        )
        if layout_cache is not None:
            layout_cache[key] = built
        return built

    @torch.no_grad()
    def _chunk_topk(qr_rows, w_rows, gs, sz):
        # Project -> per-chunk RoPE at the chunk's true global positions (the real ``cu_seqlens`` is
        # passed, so multi-sequence packs get correct per-segment positions) -> rotate -> reference
        # top-k. Delegating to ``compute_cp_indexer_topk`` reuses its multi-segment layout builder
        # and its fused/unfused split, so ``use_fused=False`` is honored. A query's top-k depends
        # only on its own position and K, so scoring a chunk matches the same rows of a full call
        # (up to GEMM reduction order of the chunked projection: exact score ties may resolve
        # differently; the output is integer indices with no gradient path).
        if use_fused:
            _pin_fused_call_rows(sz)
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
        # Exact causal need for this chunk, rounded UP to the shared width quantum
        # (see _KV_BOUND_QUANTUM/_KV_TIGHT_WIDTH_CEILING at module scope for the
        # kernel contract). Rounding up is always safe: the kernel's block count is
        # clamped by the true per-sequence KV length, so extra columns are never
        # written or read.
        bound = min(gkv, (gs + sz) // int(ratio))
        q_ = _KV_BOUND_QUANTUM
        bound_q = max(q_, (bound + q_ - 1) // q_ * q_)
        # Beyond the tight-width ceiling, a single-full-pack sequence instead scores
        # against a sliced K prefix with a synthetic [0, bound] layout, so width ==
        # declared length (the reference call's contract shape) at any size;
        # multi-sequence packs fall back to the full width. Fused-only: the unfused
        # scorer recomputes masking from (cu_seqlens, gs) and treats the layout as
        # metadata, and compute_cp_indexer_topk's contract declares synthetic
        # layouts unsupported there — it takes the full width instead.
        k_pass = k_seq_major
        layout_pass = _layout_at(gs, sz)
        graphs_enabled = getattr(config, "cuda_graph_impl", "none") != "none"
        if bound_q <= _KV_TIGHT_WIDTH_CEILING:
            mkv = bound_q
        elif single_full_seq and use_fused and not graphs_enabled and not _is_capturing():
            mkv = min(bound_q, comp)
            k_pass = k_seq_major[:mkv]
            layout_pass = _kslice_layout(gs, sz, mkv)
        else:
            # Multi-sequence packs — and ANY graphs-enabled run: the K-slice
            # layouts live in the per-psp cache, which TE's capture-time cloning
            # strips, and rebuilding them is an H2D copy that is illegal while
            # capturing. The branch keys on the config switch (not just
            # is_current_stream_capturing) so the eager warmup takes the SAME
            # kernel shape the capture will record — the fused scorer
            # JIT-compiles and allocates per (max_seqlen_q, max_seqlen_kv)
            # shape, and a shape first seen inside stream capture would do that
            # host-side work while capturing. Full width with the real layout
            # is always safe (width == declared per-sequence length, the
            # reference call's contract shape); the graph just carries a larger
            # fp32 score buffer.
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
            use_fused=use_fused,
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

    # KNOWN KERNEL-PACKAGE HAZARD (no user-side mitigation; to be fixed by the
    # kernel owners): the fused indexer package carries cross-call state that can
    # corrupt a fused call whose shape differs from the calls that preceded it in
    # the process (see the WORKSPACE NOTE in the unit tests, which records the
    # measurements — including that a discarded same-shape priming call does NOT
    # protect a subsequent call, so no warmup scheme here can help). Runs whose
    # fused-call shapes are constant — every balanced call below scores
    # ~l_local/2 rows per call, repeated identically across layers and
    # microbatches for a fixed capacity, i.e. the CUDA-graph scope and every
    # benchmarked configuration — are unaffected (end-to-end loss parity vs the
    # reference). Regimes that would ALTERNATE fused-call shapes within one process
    # are rejected by design until the kernel-side fix: eligibility is a run-level
    # invariant (config validation plus the fail-fast pack check), and the
    # per-process row pin (_pin_fused_call_rows) rejects a varying pack capacity.

    # "kind" legend lives on dispatch_chunks_async: zzr/ag/ag2 are the three zigzag
    # transports (routed a2a / merged AllGather / split AllGather); a None handle means
    # the caller skipped the early dispatch. Eligibility was already enforced by the
    # dispatch (or prebuild); re-check here only when no dispatch ran.
    if dispatch_handle is None:
        _ensure_pack_zigzag_ok(cu_seqlens, cp_group, cp_size, l_local, layout_cache)
    # ---- Per-sequence zigzag: exact balance for any pack composition -------------
    if dispatch_handle is not None and dispatch_handle.get("kind") == "zzr":
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
        _pin_fused_call_rows(sz)
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
    mkv_h = int(plan.get("mkv_head", gkv))
    mkv_t = int(plan.get("mkv_tail", gkv))
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
        if dispatch_handle["kind"] == "zzr":
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

    nvtx_range_push("BalancedIndexerScore")
    nvtx_range_push("Bal_Head")
    tk_head = _packed_topk(qr_h, w_h, plan["head_layout"], plan["pos_head"], k_h, mkv_h)
    nvtx_range_pop("Bal_Head")
    nvtx_range_push("Bal_Tail")
    tk_tail = _packed_topk(qr_t, w_t, plan["tail_layout"], plan["pos_tail"], k_t, mkv_t)
    nvtx_range_pop("Bal_Tail")
    nvtx_range_pop("BalancedIndexerScore")

    nvtx_range_push("Bal_Combine")
    tkw = tk_head.shape[-1]
    ht = _a2a_buf("zz_cmb_send", l_local, tkw, tk_head.dtype, dev, cp_group)
    ht[:half].copy_(tk_head)
    ht[half:].copy_(tk_tail)
    if "cmb_send_rows" in plan:
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
    else:
        Z = _a2a_buf("zz_cmb_recv", S, tkw, tk_head.dtype, dev, cp_group, persistent=False)
        dist.all_gather_into_tensor(Z, ht, group=cp_group)
        compressed_topk = torch.index_select(Z, 0, plan["inv_idx"])
    nvtx_range_pop("Bal_Combine")
    return compressed_topk, layout


def prebuild_balanced_layouts(
    packed_seq_params,
    cp_group=None,
    pad_alignment=_PAD_UNSPECIFIED,
    capacity=None,
    graphs_enabled=False,
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
    total = int(cu[-1].item())
    if capacity is not None and 0 < total < capacity:
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
    l_local = total // N if total > 0 else 0
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
    if total % N != 0 or l_local % 2 != 0:
        raise ValueError(
            "balanced CP indexer: pack capacity violates the run-level invariant "
            f"(total={total}, cp_size={N}, l_local={l_local}): the physical pack must "
            "split into an even per-rank row count. Check pad_packed_seq_alignment "
            "(config validation requires an integer multiple of 2 * cp_size)."
        )

    # Multi-seq gate: one D2H probe here instead of in the first forward, using the
    # forward probe's exact predicate (csa.py): "multi" means NOT a single sequence
    # that fills the whole physical pack. For padded-cu frontends cu[-1] == total,
    # so this reduces to the segment count; in the raw-cu ``capacity=`` branch the
    # second conjunct matters — a single raw sequence with a capacity tail is NOT a
    # full-pack sequence, exactly as the forward probe would decide.
    seg_lens = cu[1:] - cu[:-1]
    nseg_real = int((seg_lens > 0).sum().item())
    multi_seq = not (nseg_real == 1 and int(cu[-1].item()) == total)
    # Capacity-tagged like the pack verdict: prebuild may have seen a padded cu
    # ending short of the physical pack, in which case the forward must re-probe.
    packed_seq_params._dsa_cp_multi_seq = (l_local, multi_seq)

    cu_list = seen_cu[1]
    prev = _LAST_PLAN.get((_group_key(cp_group), r))
    if prev is not None and prev.get("half", 0) * 2 != l_local:
        # Latest-slot plan was built at a different capacity (e.g. a raw middle-stage
        # cu probed with a different capacity hint): not reusable for idempotency.
        prev = None

    if pad_alignment is not _PAD_UNSPECIFIED and not (
        isinstance(pad_alignment, int) and pad_alignment % (2 * N) == 0
    ):
        # Config validation already requires this when dsa_cp_balance_indexer is
        # enabled; a mismatch here means the caller wired a different value.
        raise ValueError(
            "balanced CP indexer: prebuild received pad_alignment="
            f"{pad_alignment!r}, which is not an integer multiple of 2 * cp_size "
            f"= {2 * N}. Balanced eligibility is a run-level invariant; pass the "
            "config's pad_packed_seq_alignment."
        )
    # The unified zigzag path also serves single-full-sequence packs (the per-sequence
    # zigzag of one pack-spanning sequence is the plain 2N-fold of the whole pack), so the
    # plan and its A2A routes are built for every pack composition.

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
    # Same sequence enumeration as _zigzag_plan: real segments plus the
    # capacity-padding pseudo-sequence [cu[-1], total) (empty for full packs).
    seq_lens_list = [e - s for s, e in zip(cu_list[:-1], cu_list[1:])] + [total - cu_list[-1]]
    if any(sl % (2 * N) for sl in seq_lens_list):
        # Fail fast at data-prep time — the best failure point. Balanced
        # eligibility is a run-level invariant; there is no per-microbatch
        # fallback because the fused kernel package silently corrupts
        # shape-alternating calls.
        raise ValueError(
            "balanced CP indexer: this pack is not zigzag-representable — every "
            "(padded) sequence length and the capacity tail must be divisible by "
            f"2 * cp_size = {2 * N} (sequence lengths incl. tail: {seq_lens_list}). "
            "Fix the dataset padding / pad_packed_seq_alignment so every pack "
            "conforms, or disable dsa_cp_balance_indexer."
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
