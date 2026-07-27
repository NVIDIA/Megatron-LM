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
``csa_cp_utils.compute_cp_indexer_topk``: it returns the top-k in the same contiguous
``[l_local, topk]`` layout, so the downstream index-building and sparse attention are unchanged.
"""
import torch
import torch.distributed as dist

from megatron.core.transformer.experimental_attention_variant.dsa import rotate_activation
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


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


def dispatch_chunks_async(indexer_qr, weights_indexer_cp, cp_group, cp_size, l_local):
    """Issue the chunk dispatch as early as possible; returns an opaque handle.

    Called by the CSA forward right after ``indexer_qr``/``weights_indexer_cp`` are
    produced, so the all_to_all overlaps with the compressor and compressed-K gather that
    run before the top-k. ``balanced_compute_cp_indexer_topk(dispatch_handle=...)`` waits
    on it. When qr and weights share a dtype they ride one all_to_all (single launch).
    """
    if cp_size <= 1:
        return None
    meta = _a2a_meta(cp_group, cp_size, l_local)
    q_lora = indexer_qr.shape[-1]
    n_heads = weights_indexer_cp.shape[-1]
    q2 = indexer_qr.reshape(-1, q_lora)
    w2 = weights_indexer_cp.reshape(-1, n_heads)
    if q2.dtype != w2.dtype:
        wq, rq = _alltoall_dispatch_async(q2, meta, cp_group, "qr")
        ww, rw = _alltoall_dispatch_async(w2, meta, cp_group, "w")
        return {"meta": meta, "works": [wq, ww], "recvs": [rq, rw], "q_lora": q_lora}
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
    return {"meta": meta, "works": [work], "recvs": [recv], "q_lora": q_lora}


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
    from megatron.core.transformer.experimental_attention_variant import csa_cp_utils as _cu

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
    single_full_seq = int(max_seqlen_q) >= S

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
        # only on its own position and K, so scoring a chunk matches the same rows of a full call
        # (up to GEMM reduction order of the chunked projection: exact score ties may resolve
        # differently; the output is integer indices with no gradient path).
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
    <= tail owner; equal for the middle rank of an odd cp_size) in original within-chunk order. That ordering is not a documented contract, so on
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
    never greater than the tail owner (``(2 * cp_size - 1 - r) // 2``; equal for the middle rank
    of an odd ``cp_size``, whose rows arrive in send order), so the dispatched rows land as
    ``[head chunk (C rows) | tail chunk (C rows)]`` in original within-chunk order.

    The dispatch uses a fixed ``num_permuted_tokens=l_local`` (bijective permutation, no D2H sync)
    with ``non_blocking=True``; combined with the cached buffer this is CUDA-graph capturable.
    """
    N = cp_size
    # Integer top-k only: no gradient flows through the dispatch (mirrors the A2A
    # paths); DeepEP ops need not see autograd-tracked inputs.
    qr = qr.detach()
    weights = weights.detach()
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
