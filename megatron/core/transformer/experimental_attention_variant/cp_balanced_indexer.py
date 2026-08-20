# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load-balanced context-parallel DSA indexer.

Under context parallelism the DSA indexer uses a contiguous sequence split, so the causal
per-query cost ``visible_k(p) = min((p + 1) // ratio, comp_len)`` grows with position and later
CP ranks become stragglers (the layout-level max/min work ratio is ``2 * cp_size - 1``). This
module removes that imbalance without a new kernel: it splits the global sequence into
``2 * cp_size`` near-equal chunks and gives CP rank ``r`` chunk ``r`` (a low-position "head",
light) and chunk ``2 * cp_size - 1 - r`` (a high-position "tail", heavy), so
``work(head) + work(tail)`` is ~constant across ranks. Each chunk is scored by a per-chunk call to
``compute_cp_indexer_topk`` at the chunk's true global offset, so multi-sequence packs, the
``use_fused`` flag, and non-even per-rank lengths are all handled exactly as in the reference path.

``balanced_compute_cp_indexer_topk`` is a drop-in replacement for
``csa_utils.cp_utils.compute_cp_indexer_topk``: it returns the top-k in the same contiguous
``[l_local, topk]`` layout, so the downstream index-building and sparse attention are unchanged.
"""

import logging
import warnings

import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

logger = logging.getLogger(__name__)


def _all_gather_rows(x, l_local, cp_group, cp_size):
    """AllGather x[l_local, D] across the CP group -> [cp_size*l_local, D] (global-row order)."""
    D = x.shape[-1]
    x2 = x.reshape(l_local, D).contiguous()
    g = torch.empty((cp_size * l_local, D), dtype=x2.dtype, device=x2.device)
    dist.all_gather_into_tensor(g, x2, group=cp_group)
    return g


# Static all-to-all metadata and staging buffers, cached per (group, cp_size, l_local[, dtype,
# width]). The chunk map is a fixed permutation: chunk ``c``'s rows are OWNED by rank ``c // 2``
# (contiguous layout) and COMPUTED by rank ``proc(c) = c if c < cp_size else 2*cp_size - 1 - c``.
# Dispatch moves each owned half-chunk to its computing rank; combine moves each computed top-k
# chunk back to its owner. Payload is ~l_local rows per rank per direction (vs. cp_size*l_local
# for AllGather, or an S-row zero-padded reduce_scatter for the combine). All split sizes are
# static for a fixed pack size, so ``all_to_all_single`` is CUDA-graph capturable, and the cached
# staging buffers keep the allocator pool static (no expandable-segments cuMem churn).
_A2A_META: dict = {}
_A2A_BUF: dict = {}

# Last prebuilt zigzag plan per (group, rank, l_local). TE's graph-capture argument
# cloning strips dynamically attached PackedSeqParams attributes, so the per-batch
# layout cache can be invisible during capture; this module-level copy keeps the
# prebuilt plan (and its route tensors) alive and reachable there. Pack composition
# must be static under CUDA graphs, so the last plan is always the right one.
_LAST_PLAN: dict = {}


def _group_key(cp_group):
    """Stable cache key for a process group.

    ``id()`` alone can be recycled if a group is destroyed and a new one is allocated at
    the same address; c10d's ``group_name`` is unique per created group, so prefer it.
    """
    return getattr(cp_group, "group_name", None) or id(cp_group)


def _a2a_meta(cp_group, cp_size, l_local):
    key = (_group_key(cp_group), cp_size, l_local)
    meta = _A2A_META.get(key)
    if meta is not None:
        return meta
    N, nch = cp_size, 2 * cp_size
    S = N * l_local
    r = cp_group.rank()
    bounds = [(k * S) // nch for k in range(nch + 1)]
    size = [bounds[k + 1] - bounds[k] for k in range(nch)]

    def proc(c):  # computing rank of chunk c (head for c < N, tail otherwise)
        return c if c < N else nch - 1 - c

    def own(c):  # owning rank of chunk c in the contiguous layout
        return c // 2

    head_c, tail_c = r, nch - 1 - r

    # Convention on both sides: rows for one peer are ordered by ascending chunk id, and
    # all_to_all_single orders the buffers by peer rank. head_c < tail_c and
    # own(head_c) <= own(tail_c) always hold, so [head | tail] is already peer-ordered on the
    # receive side of the dispatch and on the send side of the combine. The owner's local pair
    # (2r, 2r+1) needs a swap iff proc(2r) > proc(2r+1) (true for 2r >= N).
    d_in = [0] * N
    d_in[proc(2 * r)] += size[2 * r]
    d_in[proc(2 * r + 1)] += size[2 * r + 1]
    d_out = [0] * N
    d_out[own(head_c)] += size[head_c]
    d_out[own(tail_c)] += size[tail_c]
    c_in = [0] * N
    c_in[own(head_c)] += size[head_c]
    c_in[own(tail_c)] += size[tail_c]
    c_out = [0] * N
    c_out[proc(2 * r)] += size[2 * r]
    c_out[proc(2 * r + 1)] += size[2 * r + 1]
    swap_pair = proc(2 * r) > proc(2 * r + 1)

    meta = {
        "d_in": d_in,
        "d_out": d_out,
        "c_in": c_in,
        "c_out": c_out,
        "swap_pair": swap_pair,
        "s0": size[2 * r],
        "s1": size[2 * r + 1],
        "sh": size[head_c],
        "st": size[tail_c],
    }
    _A2A_META[key] = meta
    return meta


def _a2a_buf(tag, rows, width, dtype, dev, cp_group):
    key = (tag, _group_key(cp_group), rows, width, dtype)
    buf = _A2A_BUF.get(key)
    if buf is None:
        buf = torch.empty((rows, width), dtype=dtype, device=dev)
        _A2A_BUF[key] = buf
    return buf


def _alltoall_dispatch(x, meta, cp_group, tag):
    """Move my owned half-chunks to their computing ranks; return (head_rows, tail_rows).

    ``x`` is ``[l_local, D]`` (leading dims collapsed). The returned tensors are views into a
    cached receive buffer; they are consumed by the chunk top-k within the same layer, before the
    next dispatch reuses the buffer (stream-ordered).
    """
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    l_local = x2.shape[0]
    s0 = meta["s0"]
    send = _a2a_buf(tag + "_send", l_local, D, x2.dtype, x.device, cp_group)
    if meta["swap_pair"]:
        send[: l_local - s0].copy_(x2[s0:])
        send[l_local - s0 :].copy_(x2[:s0])
    else:
        send.copy_(x2)
    recv = _a2a_buf(tag + "_recv", meta["sh"] + meta["st"], D, x2.dtype, x.device, cp_group)
    dist.all_to_all_single(
        recv, send, output_split_sizes=meta["d_out"], input_split_sizes=meta["d_in"], group=cp_group
    )
    return recv[: meta["sh"]], recv[meta["sh"] :]


def _alltoall_dispatch_async(x, meta, cp_group, tag):
    """Async variant of :func:`_alltoall_dispatch`: returns ``(work, recv)``.

    The caller runs unrelated compute (compressor, K gather) between issue and wait, so
    the NCCL transfer overlaps instead of sitting on the critical path.
    """
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    l_local = x2.shape[0]
    s0 = meta["s0"]
    send = _a2a_buf(tag + "_send", l_local, D, x2.dtype, x.device, cp_group)
    if meta["swap_pair"]:
        send[: l_local - s0].copy_(x2[s0:])
        send[l_local - s0 :].copy_(x2[:s0])
    else:
        send.copy_(x2)
    recv = _a2a_buf(tag + "_recv", meta["sh"] + meta["st"], D, x2.dtype, x.device, cp_group)
    work = dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=meta["d_out"],
        input_split_sizes=meta["d_in"],
        group=cp_group,
        async_op=True,
    )
    return work, recv


def dispatch_chunks_async(
    indexer_qr,
    weights_indexer_cp,
    cp_group,
    cp_size,
    l_local,
    max_seqlen_q=None,
    config=None,
    use_fused=True,
    multi_seq=False,
    layout_cache=None,
):
    """Issue the chunk dispatch as early as possible; returns an opaque handle.

    Called by the CSA forward right after ``indexer_qr``/``weights_indexer_cp`` are
    produced, so the all_to_all overlaps with the compressor and compressed-K gather that
    run before the top-k. ``balanced_compute_cp_indexer_topk(dispatch_handle=...)`` waits
    on it. When qr and weights share a dtype they ride one all_to_all (single launch).
    """
    if cp_size <= 1:
        return None
    q_lora = indexer_qr.shape[-1]
    n_heads = weights_indexer_cp.shape[-1]
    q2 = indexer_qr.reshape(-1, q_lora)
    w2 = weights_indexer_cp.reshape(-1, n_heads)
    if config is not None and _use_zigzag(multi_seq, cp_size, l_local, use_fused, config):
        plan = layout_cache.get(("zigzag", cp_group.rank())) if layout_cache else None
        if plan is None:
            plan = _LAST_PLAN.get((_group_key(cp_group), cp_group.rank(), l_local))
        if plan is not None and "disp_send_rows" in plan and q2.dtype == w2.dtype:
            # Route-A2A dispatch (PR #5664-style prebuilt exchange): each rank sends and
            # receives only ~l_local rows instead of the S-row AllGather. Splits are host
            # ints prebuilt at data-prep time, so the exchange is CUDA-graph capturable.
            width = q_lora + n_heads
            payload = _a2a_buf("zzr_pay", l_local, width, q2.dtype, q2.device, cp_group)
            payload[:, :q_lora].copy_(q2)
            payload[:, q_lora:].copy_(w2)
            send = _a2a_buf("zzr_send", l_local, width, q2.dtype, q2.device, cp_group)
            # copy_(index_select(...)) instead of index_select(out=...): the staging
            # buffer picks up requires_grad from the copy_ of q2/w2, and out= variants
            # reject autograd-tracked inputs. Gradients never flow through the dispatch.
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
    meta = _a2a_meta(cp_group, cp_size, l_local)
    if q2.dtype != w2.dtype:
        wq, rq = _alltoall_dispatch_async(q2, meta, cp_group, "qr")
        ww, rw = _alltoall_dispatch_async(w2, meta, cp_group, "w")
        return {"kind": "a2a", "meta": meta, "works": [wq, ww], "recvs": [rq, rw], "q_lora": q_lora}
    rows = q2.shape[0]
    width = q_lora + n_heads
    dev = q2.device
    send = _a2a_buf("qw_send", rows, width, q2.dtype, dev, cp_group)
    s0 = meta["s0"]
    if meta["swap_pair"]:
        send[: rows - s0, :q_lora].copy_(q2[s0:])
        send[: rows - s0, q_lora:].copy_(w2[s0:])
        send[rows - s0 :, :q_lora].copy_(q2[:s0])
        send[rows - s0 :, q_lora:].copy_(w2[:s0])
    else:
        send[:, :q_lora].copy_(q2)
        send[:, q_lora:].copy_(w2)
    recv = _a2a_buf("qw_recv", meta["sh"] + meta["st"], width, q2.dtype, dev, cp_group)
    work = dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=meta["d_out"],
        input_split_sizes=meta["d_in"],
        group=cp_group,
        async_op=True,
    )
    return {"kind": "a2a", "meta": meta, "works": [work], "recvs": [recv], "q_lora": q_lora}


def _dispatch_wait(handle, meta):
    """Complete an async dispatch; return (qr_head, w_head, qr_tail, w_tail)."""
    for work in handle["works"]:
        work.wait()
    sh = meta["sh"]
    q_lora = handle["q_lora"]
    if len(handle["recvs"]) == 2:
        rq, rw = handle["recvs"]
        return rq[:sh], rw[:sh], rq[sh:], rw[sh:]
    recv = handle["recvs"][0]
    # Column slices of the merged buffer are strided; contiguify (tens of MB, D2D) so the
    # projection GEMM and the fused scorer see plain dense tensors.
    return (
        recv[:sh, :q_lora].contiguous(),
        recv[:sh, q_lora:].contiguous(),
        recv[sh:, :q_lora].contiguous(),
        recv[sh:, q_lora:].contiguous(),
    )


def _alltoall_combine(tk_head, tk_tail, meta, cp_group):
    """Move computed top-k chunks back to their owners; return contiguous ``[l_local, tkw]``.

    The output is a fresh (allocator-pooled, fixed-size) tensor rather than a cached buffer so no
    caller-held reference can alias the next layer's combine.
    """
    tkw = tk_head.shape[-1]
    dev = tk_head.device
    sh, st = meta["sh"], meta["st"]
    s0, s1 = meta["s0"], meta["s1"]
    send = _a2a_buf("cmb_send", sh + st, tkw, tk_head.dtype, dev, cp_group)
    send[:sh].copy_(tk_head)
    send[sh:].copy_(tk_tail)
    recv = _a2a_buf("cmb_recv", s0 + s1, tkw, tk_head.dtype, dev, cp_group)
    dist.all_to_all_single(
        recv, send, output_split_sizes=meta["c_out"], input_split_sizes=meta["c_in"], group=cp_group
    )
    out = torch.empty((s0 + s1, tkw), dtype=tk_head.dtype, device=dev)
    if meta["swap_pair"]:
        out[:s0].copy_(recv[s1:])
        out[s0:].copy_(recv[:s1])
    else:
        out.copy_(recv)
    return out


def _all_gather_rows_buf(x, l_local, cp_group, cp_size, tag):
    """AllGather into a persistent buffer (async): returns (work, gathered[cp_size*l_local, D])."""
    D = x.shape[-1]
    x2 = x.reshape(l_local, D)
    send = _a2a_buf(tag + "_agsend", l_local, D, x2.dtype, x.device, cp_group)
    send.copy_(x2)
    g = _a2a_buf(tag + "_agrecv", cp_size * l_local, D, x2.dtype, x.device, cp_group)
    work = dist.all_gather_into_tensor(g, send, group=cp_group, async_op=True)
    return work, g


def _excl_cumsum(x):
    z = torch.zeros_like(x[:1])
    return torch.cat((z, torch.cumsum(x, 0)[:-1]))


def _use_zigzag(multi_seq, cp_size, l_local, use_fused, config):
    """Per-sequence zigzag applies to multi-sequence packs on the fused path.

    Requires the packed-sequence alignment to be a multiple of ``2 * cp_size`` so every
    (padded) sequence splits into equal chunks. Single full-pack sequences keep the static
    all_to_all global folding (cheapest). ``MCORE_DSA_CP_BAL_PACK_SCOPE=1`` forces the
    global-folding path (A/B testing).
    """
    import os

    verdict = None
    if os.environ.get("MCORE_DSA_CP_BAL_PACK_SCOPE") == "1":
        verdict = False
    elif not use_fused:
        verdict = False
    if verdict is None:
        # Unified path: single-full-pack sequences are the nseg==1 special case of the
        # per-sequence zigzag (identical chunks to the global folding), served by the same
        # plan/route machinery with per-call tight compressed-K bounds (K-slice general).
        pad = getattr(config, "pad_packed_seq_alignment", None)
        verdict = isinstance(pad, int) and pad % (2 * cp_size) == 0 and (l_local % 2 == 0)
    if os.environ.get("MCORE_DSA_CP_BAL_DEBUG") == "1" and not getattr(
        _use_zigzag, "_logged", False
    ):
        _use_zigzag._logged = True
        logger.info(
            "[zz-gate] verdict=%s multi_seq=%s S=%s use_fused=%s pad=%r env=%r",
            verdict,
            multi_seq,
            cp_size * l_local,
            use_fused,
            getattr(config, "pad_packed_seq_alignment", None),
            os.environ.get("MCORE_DSA_CP_BAL_PACK_SCOPE"),
        )
    return verdict


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
        if cached is not None:
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
        "pos_head": pos_head.long(),
        "pos_tail": pos_tail.long(),
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
    dispatch='alltoall',
    dispatch_handle=None,
    layout_cache=None,
    multi_seq=False,
):
    """Balanced drop-in replacement for ``compute_cp_indexer_topk``.

    Returns ``(compressed_topk, layout)`` in the same contiguous ``[l_local, topk]`` layout the
    caller expects, so ``build_attention_indices`` / sparse attention are unchanged. The global
    sequence is tiled into ``2 * cp_size`` near-equal chunks; this rank dispatches its head chunk
    (``r``) and tail chunk (``2 * cp_size - 1 - r``), scores each with a per-chunk call to
    ``compute_cp_indexer_topk`` at the chunk's global offset (so RoPE positions, causal offsets,
    multi-sequence packing, and the ``use_fused`` flag all follow the reference), then combines the
    top-k back to contiguous order. ``dispatch`` selects the redistribute backend ('alltoall' or
    'hybridep'; 'hybridep' requires an even ``l_local`` and otherwise uses 'alltoall').
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils as _cu

    q_lora = indexer_qr.shape[-1]
    n_heads, head_dim = indexer.index_n_heads, indexer.index_head_dim
    pos_dim = indexer.qk_pos_emb_head_dim
    nope_dim = head_dim - pos_dim
    dev = indexer_qr.device
    comp = int(k_seq_major.shape[0])
    S = cp_size * l_local
    nch = 2 * cp_size
    r = cp_group.rank()
    head_c, tail_c = r, nch - 1 - r

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

    def _chunk_topk(qr_rows, w_rows, gs, sz):
        # Project -> per-chunk RoPE at the chunk's true global positions (the real ``cu_seqlens`` is
        # passed, so multi-sequence packs get correct per-segment positions) -> rotate -> reference
        # top-k. Delegating to ``compute_cp_indexer_topk`` reuses its multi-segment layout builder
        # and its fused/unfused split, so ``use_fused=False`` is honored. A query's top-k depends
        # only on its own position and K, so scoring a chunk equals the same rows of a full call.
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
        mq = min(int(max_seqlen_q), sz)
        gkv = max(1, int(max_seqlen_q) // int(ratio))
        # Exact causal need for this chunk, rounded UP to a multiple of 8192. Rounding up is
        # always safe (the kernel's block count is clamped by the true per-sequence KV length,
        # so extra columns are never written or read), and it keeps the set of distinct widths
        # tiny: the fused kernel JIT-compiles per (max_seqlen_q, max_seqlen_kv) pair and
        # reallocates the score buffer per shape, so unquantized widths would trigger
        # recompilation whenever varlen packs change max_seqlen_q between iterations.
        bound = min(gkv, (gs + sz) // int(ratio))
        bound_q = max(8192, (bound + 8191) // 8192 * 8192)
        # Empirical kernel guard: score widths narrower than the declared per-sequence KV
        # length are only safe up to 65536 columns on the cuDNN indexer kernel (wider tight
        # values hit an illegal memory access; a width equal to the declared per-sequence
        # length is always fine). Beyond 64K, a single-full-pack sequence instead scores
        # against a sliced K prefix with a synthetic [0, bound] layout, so width == declared
        # length (the reference call's contract shape) at any size; multi-sequence packs
        # fall back to the full width.
        k_pass = k_seq_major
        layout_pass = _layout_at(gs, sz)
        if bound_q <= 65536:
            mkv = bound_q
        elif single_full_seq:
            mkv = min(bound_q, comp)
            k_pass = k_seq_major[:mkv]
            layout_pass = _kslice_layout(gs, sz, mkv)
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
            use_fused=use_fused,
            max_seqlen_kv=mkv,
            prebuilt_layout=layout_pass,
        )
        return tk

    if cp_size <= 1 or comp == 0 or int(topk) == 0:
        # Nothing to balance / nothing to select: one call over this rank's own rows. An
        # already-issued async dispatch must still be completed, otherwise its NCCL work is
        # orphaned and the persistent staging buffers could be rewritten while still in flight.
        if dispatch_handle is not None:
            for work in dispatch_handle["works"]:
                work.wait()
        return _chunk_topk(indexer_qr, weights_indexer_cp, int(global_start), l_local), layout

    zz = (
        dispatch_handle.get("kind") in ("ag", "ag2", "zzr")
        if dispatch_handle is not None
        else (
            dispatch != 'hybridep' and _use_zigzag(multi_seq, cp_size, l_local, use_fused, config)
        )
    )
    if zz:
        # ---- Per-sequence zigzag: exact balance for any pack composition -------------
        if dispatch_handle is not None and dispatch_handle.get("kind") == "zzr":
            plan = dispatch_handle["plan"]
        else:
            plan = (layout_cache or {}).get(("zigzag", r)) or _LAST_PLAN.get(
                (_group_key(cp_group), r, l_local)
            )
            if plan is None:
                plan = _zigzag_plan(
                    cu_seqlens, cu_seqlens_compressed, cp_size, l_local, r, dev, layout_cache
                )
        half = plan["half"]
        mq = max(1, min(int(max_seqlen_q), half))
        gkv = max(1, int(max_seqlen_q) // int(ratio))

        def _packed_topk(qr_rows, w_rows, layout3, pos_ids, kv_rows, mkv):
            sz = qr_rows.shape[0]
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
        mkv_h = int(plan.get("mkv_head", gkv))
        mkv_t = int(plan.get("mkv_tail", gkv))
        k_rows_total = k_seq_major.shape[0]
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
            q2 = indexer_qr.reshape(l_local, q_lora)
            w2 = weights_indexer_cp.reshape(l_local, n_heads)
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
            Z = _a2a_buf("zz_cmb_recv", S, tkw, tk_head.dtype, dev, cp_group)
            dist.all_gather_into_tensor(Z, ht, group=cp_group)
            compressed_topk = torch.index_select(Z, 0, plan["inv_idx"])
        nvtx_range_pop("Bal_Combine")
        return compressed_topk, layout

    # Tile [0, S) into ``nch`` near-equal chunks: even l_local -> bounds[k] = k*(l_local//2); odd
    # l_local -> chunk sizes differ by at most one row, so no row is ever dropped.
    bounds = [(k * S) // nch for k in range(nch + 1)]
    hs, he = bounds[head_c], bounds[head_c + 1]
    ts, te = bounds[tail_c], bounds[tail_c + 1]

    # Dispatch: bring this rank's head-chunk and tail-chunk rows of qr and weights here.
    meta = _a2a_meta(cp_group, cp_size, l_local)
    nvtx_range_push("Bal_Dispatch")
    if dispatch_handle is not None:
        # Transfer was issued early by dispatch_chunks_async and overlapped with the
        # compressor / compressed-K gather; just complete it here.
        qr_head, w_head, qr_tail, w_tail = _dispatch_wait(dispatch_handle, meta)
    elif dispatch == 'hybridep' and l_local % 2 == 0:
        # DeepEP all-to-all (moves only 2 chunks per rank). Requires equal chunks (even l_local).
        qr_head, w_head, qr_tail, w_tail = _hybridep_dispatch_chunks(
            indexer_qr,
            weights_indexer_cp,
            cp_group,
            cp_size,
            l_local,
            l_local // 2,
            r,
            q_lora,
            n_heads,
            dev,
        )
    else:
        # NCCL all_to_all_single over the fixed chunk permutation: ~l_local rows per rank instead
        # of an S-row AllGather, static splits (CUDA-graph capturable), cached staging buffers.
        qr_head, qr_tail = _alltoall_dispatch(indexer_qr, meta, cp_group, "qr")
        w_head, w_tail = _alltoall_dispatch(weights_indexer_cp, meta, cp_group, "w")
    nvtx_range_pop("Bal_Dispatch")

    nvtx_range_push("BalancedIndexerScore")
    nvtx_range_push("Bal_Head")
    tk_head = _chunk_topk(qr_head, w_head, hs, he - hs)
    nvtx_range_pop("Bal_Head")
    nvtx_range_push("Bal_Tail")
    tk_tail = _chunk_topk(qr_tail, w_tail, ts, te - ts)
    nvtx_range_pop("Bal_Tail")
    nvtx_range_pop("BalancedIndexerScore")

    # Combine: return each computed chunk to the rank that owns its contiguous rows. The chunk map
    # is a fixed permutation (each rank sends 2 chunks and receives its own 2), so this is an exact
    # all_to_all of ~l_local rows — no S-row zero-padded buffer and no SUM reduction.
    nvtx_range_push("Bal_Combine")
    compressed_topk = _alltoall_combine(tk_head, tk_tail, meta, cp_group)
    nvtx_range_pop("Bal_Combine")
    return compressed_topk, layout


# Cache one HybridEPBuffer per (cp_group, cp_size, l_local, width). The buffer's __init__ does
# cudaMalloc + NCCL/RDMA setup, which is not CUDA-graph capturable; creating it per call would break
# both partial (attention module) and full-iteration capture. It is created once on the first eager
# (warmup) call and reused, so the capture step only sees the capturable dispatch kernels.
_HEP_BUF: dict = {}


def _get_hep_buffer(cp_group, cp_size, l_local, width):
    key = (_group_key(cp_group), cp_size, l_local, width)
    buf = _HEP_BUF.get(key)
    if buf is None:
        from deep_ep import HybridEPBuffer

        buf = HybridEPBuffer(
            group=cp_group,
            hidden_dim=width,
            max_num_of_tokens_per_rank=l_local,
            num_local_experts=1,
            use_fp8=False,
        )
        _HEP_BUF[key] = buf
    return buf


# Keys for which the DeepEP dispatch ordering has been cross-checked against AllGather.
_HEP_VALIDATED: set = set()


def _validate_hep_order(qr, qr_head, qr_tail, cp_group, cp_size, l_local, C, r):
    """One-time cross-check that the ``disp[0:C]``=head / ``disp[C:2C]``=tail split is correct.

    The split relies on DeepEP landing dispatched rows grouped by ascending source rank (head owner
    < tail owner) in original within-chunk order. That ordering is not a documented contract, so on
    the first call per (group, cp_size, l_local) verify the dispatched head/tail chunks bit-match
    the AllGather-selected chunks, and raise loudly if DeepEP ever orders tokens differently (which
    would otherwise silently corrupt the top-k). One extra AllGather on the first eager call only.
    """
    key = (_group_key(cp_group), cp_size, l_local)
    if key in _HEP_VALIDATED:
        return
    gq = _all_gather_rows(qr, l_local, cp_group, cp_size)
    head_c, tail_c = r, 2 * cp_size - 1 - r
    ok = torch.equal(qr_head, gq[head_c * C : (head_c + 1) * C]) and torch.equal(
        qr_tail, gq[tail_c * C : (tail_c + 1) * C]
    )
    if not ok:
        raise RuntimeError(
            "hybridEP dispatch order does not match the expected [head | tail] layout; "
            "re-run with --dsa-cp-balance-dispatch alltoall."
        )
    _HEP_VALIDATED.add(key)


def _hybridep_dispatch_chunks(qr, weights, cp_group, cp_size, l_local, C, r, q_lora, n_heads, dev):
    """Move this rank's head chunk (index r) and tail chunk (index 2 * cp_size - 1 - r) rows here
    via a DeepEP hybridEP all-to-all instead of AllGather. Returns (qr_head, w_head, qr_tail,
    w_tail), each ``[C, *]``. Requires equal chunks (even ``l_local``, ``C = l_local // 2``) and the
    ``deep_ep`` package.

    With ``num_local_experts=1`` the dispatch is a pure token permutation across CP ranks: each
    local row's destination is the processor rank of the chunk it belongs to
    (``processor(c) = c if c < cp_size else 2 * cp_size - 1 - c``). The head owner (``r // 2``) is
    always less than the tail owner (``(2 * cp_size - 1 - r) // 2``), so the dispatched rows land as
    ``[head chunk (C rows) | tail chunk (C rows)]`` in original within-chunk order.

    The dispatch uses a fixed ``num_permuted_tokens=l_local`` (bijective permutation, no D2H sync)
    with ``non_blocking=True``; combined with the cached buffer this is CUDA-graph capturable.
    """
    N = cp_size
    local_chunk = torch.arange(l_local, device=dev) // C  # 0 for rows [0, C), 1 for [C, 2C)
    global_chunk = 2 * r + local_chunk  # owner chunk id in [0, 2 * cp_size)
    dest_rank = torch.where(global_chunk < N, global_chunk, 2 * N - 1 - global_chunk).long()
    routing_map = torch.zeros((l_local, N), dtype=torch.bool, device=dev)
    routing_map.scatter_(1, dest_rank.view(-1, 1), True)  # one True per row = destination CP rank
    probs = routing_map.to(torch.float32)  # HybridEP requires fp32 probs
    payload = torch.cat(
        [qr.reshape(l_local, q_lora), weights.reshape(l_local, n_heads)], dim=-1
    )  # move qr and weights under one permutation
    width = q_lora + n_heads
    buf = _get_hep_buffer(cp_group, cp_size, l_local, width)
    disp, _p, _s, _t, _h = buf.dispatch_with_permute(
        hidden=payload,
        routing_map=routing_map,
        probs=probs,
        scaling_factor=None,
        num_of_experts_per_rank=1,
        pad_multiple=None,
        num_permuted_tokens=l_local,
        non_blocking=True,
    )
    qr_head, w_head = disp[0:C, :q_lora], disp[0:C, q_lora:]
    qr_tail, w_tail = disp[C : 2 * C, :q_lora], disp[C : 2 * C, q_lora:]
    _validate_hep_order(qr, qr_head, qr_tail, cp_group, cp_size, l_local, C, r)
    return qr_head, w_head, qr_tail, w_tail


def _thd_zigzag_rank_indices(cu_list, cp_size, cp_rank, device):
    """Global THD token indices owned by ``cp_rank`` under the zigzag CP layout.

    Mirrors the canonical segment math in
    ``megatron.core.context_parallel_layout.routes`` (rank ``r`` owns chunk ``r``
    and chunk ``2 * cp_size - r - 1`` of every packed sequence), returned in
    rank-local storage order. That module only exposes the segments through a
    private helper today, so the expansion is kept here.

    TODO(cp-layout): drop this in favour of a public per-rank ownership helper
    once ``context_parallel_layout`` exports one again.
    """
    rows = []
    for seq_start, seq_end in zip(cu_list[:-1], cu_list[1:]):
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        if chunk_len == 0:
            continue
        head_start = seq_start + cp_rank * chunk_len
        tail_start = seq_start + (2 * cp_size - cp_rank - 1) * chunk_len
        rows.extend(range(head_start, head_start + chunk_len))
        rows.extend(range(tail_start, tail_start + chunk_len))
    return torch.tensor(rows, device=device, dtype=torch.long)


def prebuild_balanced_layouts(packed_seq_params, cp_group=None):
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
    if total <= 0 or total % N != 0:
        return
    l_local = total // N
    if l_local % 2 != 0:
        return

    # Multi-seq gate: one D2H probe here instead of in the first forward.
    cu_real = packed_seq_params.cu_seqlens_q if packed_seq_params.cu_seqlens_q is not None else cu
    seg_lens = cu_real[1:] - cu_real[:-1]
    nseg_real = int((seg_lens > 0).sum().item())
    total_real = int(cu_real[-1].item())
    packed_seq_params._dsa_cp_multi_seq = not (nseg_real == 1 and total_real == total)
    # The unified zigzag path also serves single-full-sequence packs (per-sequence zigzag
    # of one pack-spanning sequence IS the global folding), so the plan and its A2A routes
    # are built for every pack composition.

    cu_list = [int(v) for v in cu.tolist()]
    # Idempotent prebuild: for static pack compositions (every CUDA-graph run, and
    # any fixed-length workload) the plan content is identical each microbatch.
    # Reuse the previous plan object instead of re-allocating its tensors — this
    # keeps steady-state allocations at zero (no expandable-segments cuMem churn)
    # and keeps capture-baked pointers valid.
    prev = _LAST_PLAN.get((_group_key(cp_group), r, l_local))
    if prev is not None and prev.get("_cu_list") == cu_list:
        cache = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
        if cache is None:
            cache = {}
            packed_seq_params._dsa_cp_balance_layout_cache = cache
        cache[("zigzag", r)] = prev
        return
    # Same sequence enumeration as _zigzag_plan: real segments plus the
    # capacity-padding pseudo-sequence [cu[-1], total) (empty for full packs).
    seq_lens_list = [e - s for s, e in zip(cu_list[:-1], cu_list[1:])] + [total - cu_list[-1]]
    if any(sl % (2 * N) for sl in seq_lens_list):
        # Runtime gate falls back to folding for non-2N-aligned packs.
        return

    dev, dt = cu.device, cu.dtype
    half = l_local // 2
    # Canonical per-rank zigzag ownership (per-seq interleaved [head seg, tail seg]).
    per_rank = [_thd_zigzag_rank_indices(cu_list, N, rho, dev) for rho in range(N)]

    # Reorder the canonical per-seq-interleaved local order into [heads | tails].
    c_list = [sl // (2 * N) for sl in seq_lens_list]
    head_pos, tail_pos = [], []
    off = 0
    for c in c_list:
        head_pos.extend(range(off, off + c))
        tail_pos.extend(range(off + c, off + 2 * c))
        off += 2 * c
    assert off == l_local and len(head_pos) == half, (off, l_local, len(head_pos), half)
    perm = torch.tensor(head_pos + tail_pos, device=dev, dtype=torch.long)

    gather_idx = per_rank[r].index_select(0, perm)

    # Sequence-relative positions for RoPE (relative to real segment starts; rows in
    # the capacity pseudo-sequence use cu[-1] as their start, matching _zigzag_plan).
    starts_t = torch.tensor(cu_list[:-1] + [cu_list[-1]], device=dev, dtype=torch.long)
    bounds_t = torch.tensor(cu_list[1:] + [total], device=dev, dtype=torch.long)
    seq_of = torch.bucketize(gather_idx, bounds_t[:-1], right=True).clamp_max(starts_t.numel() - 1)
    pos = gather_idx - starts_t[seq_of]
    pos_head, pos_tail = pos[:half].long(), pos[half:].long()

    # Inverse permutation: my contiguous rows' positions in the rank-major
    # [heads | tails] all-gather concat — derived from the canonical indices.
    pos_global = torch.empty(total, device=dev, dtype=torch.long)
    ar = torch.arange(l_local, device=dev, dtype=torch.long)
    for rho in range(N):
        pos_global[per_rank[rho].index_select(0, perm)] = rho * l_local + ar
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
    ordered_all = [per_rank[rho].index_select(0, perm) for rho in range(N)]
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
        bound = max(8192, ((span + 8191) // 8192) * 8192)
        if bound > 65536 or bound >= gkv:
            return gkv, cu_comp_list[-1]
        return bound, min(max(ends), cu_comp_list[-1])

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
        "_cu_list": cu_list,
    }
    if prev is not None:
        # A CUDA graph may have baked pointers to the previous plan's tensors.
        # When only the values changed (same shapes; e.g. varlen data under a
        # fixed pad_packed_seq_alignment), refresh the previous tensors in
        # place so captured graphs keep reading correct data. Host-side split
        # lists are baked into captured collectives and cannot be refreshed:
        # if they change, any existing capture needs re-recording.
        same_shapes = all(
            torch.is_tensor(prev.get(k)) == torch.is_tensor(v)
            and (not torch.is_tensor(v) or prev[k].shape == v.shape)
            for k, v in plan.items()
            if k not in ("_cu_list", "head_layout", "tail_layout", "half")
        )
        if same_shapes and (
            prev["disp_in_splits"] == plan["disp_in_splits"]
            and prev["disp_out_splits"] == plan["disp_out_splits"]
        ):
            for k, v in plan.items():
                if torch.is_tensor(v):
                    prev[k].copy_(v)
                else:
                    prev[k] = v
            plan = prev
        else:
            warnings.warn(
                "prebuild_balanced_layouts: plan shapes or A2A splits changed for a "
                "previously built layout; any CUDA graph captured with the old plan "
                "must be re-recorded.",
                stacklevel=2,
            )
    cache = getattr(packed_seq_params, "_dsa_cp_balance_layout_cache", None)
    if cache is None:
        cache = {}
        packed_seq_params._dsa_cp_balance_layout_cache = cache
    cache[("zigzag", r)] = plan
    _LAST_PLAN[(_group_key(cp_group), r, l_local)] = plan
