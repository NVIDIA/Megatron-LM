# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA selection helpers: host-sync-free CP layout and fast exact top-k.

* :func:`qsa_cp_layout_static` -- the allgather-CP query positions and the gathered-key
  reorder for packed THD inputs, computed with device ops at shape-only sizes (no
  boolean-mask indexing, no ``.item()``), so :meth:`QSAIndexer._select_cp` can be
  captured into a CUDA graph. Values equal
  :func:`dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder` under the
  contract ``cu_seqlens[-1] == s_local * cp_size`` (the pack fills the buffer).
"""

import functools

import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_cp_local_positions,
)


def _per_document_positions(cu: torch.Tensor, s_local: int, cp_size: int, cp_rank: int):
    """Local-row -> global-token map for per-document zigzag sharding, sync-free.

    Every document of length ``l`` contributes a front segment
    ``[start + rank * h, + h)`` and a back segment ``[end - (rank + 1) * h, + h)`` with
    ``h = (l // cp) // 2``; local rows are their concatenation in document order. Empty
    segments are skipped by the ``searchsorted`` (no compaction needed).
    """
    device = cu.device
    starts, ends = cu[:-1], cu[1:]
    h = ((ends - starts) // cp_size) // 2
    seg_starts = torch.stack((starts + cp_rank * h, ends - (cp_rank + 1) * h), dim=1).reshape(-1)
    seg_lens = torch.stack((h, h), dim=1).reshape(-1)
    cum = seg_lens.cumsum(0)
    i = torch.arange(s_local, device=device, dtype=torch.int64)
    seg = torch.searchsorted(cum, i, right=True).clamp(max=seg_lens.numel() - 1)
    return seg_starts[seg] + i - (cum[seg] - seg_lens[seg])


def qsa_cp_local_positions_static(
    cu_seqlens: torch.Tensor, s_local: int, cp_size: int, cp_rank: int, layout: str
) -> torch.Tensor:
    """Global token position of each of this rank's ``s_local`` packed rows."""
    cu = cu_seqlens.to(dtype=torch.int64)
    if layout == "per_sequence":
        return build_zigzag_cp_local_positions(s_local * cp_size, cp_size, cp_rank, cu.device)
    if layout != "per_document":
        raise ValueError(f"unknown cp packing layout {layout!r}")
    return _per_document_positions(cu, s_local, cp_size, cp_rank)


def qsa_cp_layout_static(
    cu_seqlens: torch.Tensor, s_local: int, cp_size: int, cp_rank: int, layout: str
):
    """(q_positions [s_local], key_reorder [s_local * cp_size]) without host syncs."""
    q_pos = qsa_cp_local_positions_static(cu_seqlens, s_local, cp_size, cp_rank, layout)
    gathered = torch.cat(
        [
            (
                q_pos
                if r == cp_rank
                else qsa_cp_local_positions_static(cu_seqlens, s_local, cp_size, r, layout)
            )
            for r in range(cp_size)
        ]
    )
    # Positions are a permutation of [0, S) under the fill contract: unique keys, so the
    # argsort is deterministic.
    return q_pos, torch.argsort(gathered)


# ---------------------------------------------------------------------------
# Indexer top-k (K4): scores + exact (score desc, block id asc) top-k without the
# chunked int64 ``cat`` + ``torch.topk``.
# ---------------------------------------------------------------------------
#
# Stage 1 (_qsa_select_score_kernel): the opt2 scoring arithmetic (per head an fp32 IEEE
# dot, relu, sequential head sum, * 1/sqrt(d)) written as fp32 scores at row-relative
# columns ``c = j - m_lo``; tiles no row of the tile can see are skipped (causal skip).
# Stage 2 selects per row, exactly as the int64-key torch.topk does: the k largest
# (score bits, -id) keys among the visible window.
#   * "triton": an in-register 4 x 8-bit radix select of the k-th score (tl.histogram),
#   * "cudnn":  cuDNN FE ``DSA.indexer_top_k_wrapper`` (radix, arbitrary ties) only to
#               find the k-th score value;
#   then the same compaction kernel takes every key above the threshold plus the
#   smallest-id ties at the threshold (the unique-key tie-break), and sorts the k keys
#   descending. The resulting keys are bit-identical to the opt2 path's ``run_keys``.

import triton
import triton.language as tl

ID_BITS = 21
_MAX_ID_CODE = (1 << ID_BITS) - 1


@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in [4, 8]], key=['BM', 'BN', 'DI'])
@triton.jit
def _qsa_select_score_kernel(
    q,  # [s, NH, DI]
    bk,  # [n_blocks, DI]
    out,  # [s, ld] fp32 scores at row-relative columns
    m_t,  # [s] exclusive upper visible block id
    m_lo,  # [s] first visible block id (read only when HAS_LO)
    s,
    nb,
    ld,
    scale,
    NH: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_LO: tl.constexpr,
):
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_d = tl.arange(0, DI)
    m_m = o_m < s
    mt = tl.load(m_t + o_m, mask=m_m, other=0)
    if HAS_LO:
        lo = tl.load(m_lo + o_m, mask=m_m, other=0)
    else:
        lo = tl.zeros_like(mt)
    # causal skip: no row of this tile sees any block of this column tile
    if i_n * BN >= tl.max(mt):
        return
    if HAS_LO:
        if (i_n + 1) * BN <= tl.min(tl.where(m_m, lo, nb)):
            return
    m_n = o_n < nb
    b_k = tl.load(bk + o_n[None, :] * DI + o_d[:, None], mask=m_n[None, :], other=0.0).to(
        tl.float32
    )
    b_acc = tl.zeros([BM, BN], dtype=tl.float32)
    for h in tl.static_range(NH):
        b_q = tl.load(
            q + (o_m[:, None] * NH + h) * DI + o_d[None, :], mask=m_m[:, None], other=0.0
        ).to(tl.float32)
        # same arithmetic as the opt2 qsa_indexer_keys_kernel (bit-identical scores)
        b_acc += tl.maximum(tl.dot(b_q, b_k, input_precision="ieee"), 0.0)
    score = b_acc * scale
    vis = (o_n[None, :] < mt[:, None]) & (o_n[None, :] >= lo[:, None]) & m_m[:, None]
    col = o_n[None, :] - lo[:, None]
    tl.store(out + o_m[:, None].to(tl.int64) * ld + col, score, mask=vis)


@triton.jit
def _qsa_select_topk_kernel(
    scores,  # [rows, ld] fp32 (row-relative columns)
    thr,  # [rows] int32 k-th score bits (read only when GIVEN_THR)
    keys_out,  # [rows, KP] int64
    m_t,
    m_lo,
    k,
    ld,
    max_id_code,
    TILE: tl.constexpr,
    KP: tl.constexpr,
    IDB: tl.constexpr,
    HAS_LO: tl.constexpr,
    GIVEN_THR: tl.constexpr,
):
    """One row: exact top-k keys over the visible window, streamed in TILE columns."""
    row = tl.program_id(0)
    mt = tl.load(m_t + row).to(tl.int32)
    if HAS_LO:
        lo = tl.load(m_lo + row).to(tl.int32)
    else:
        lo = mt * 0
    n_vis = mt - lo
    srow = scores + row.to(tl.int64) * ld
    ct = tl.arange(0, TILE)
    zero = mt * 0
    if GIVEN_THR:
        T = tl.load(thr + row)
        T = tl.where(n_vis > k, T, -1)
        gt = zero
        n_eq = zero
        for c0 in range(0, n_vis, TILE):
            c = c0 + ct
            valid = c < n_vis
            x = tl.load(srow + c, mask=valid, other=0.0)
            bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
            gt += tl.sum((bits > T).to(tl.int32), axis=0)
            n_eq += tl.sum(((bits == T) & valid).to(tl.int32), axis=0)
        kk = k - gt
    else:
        T = zero - 1
        kk = k + zero
        n_eq = zero
        if n_vis > k:
            prefix = zero
            bins = tl.arange(0, 256)
            for p in tl.static_range(4):
                shift = 24 - 8 * p
                hist = tl.zeros([256], dtype=tl.int32)
                for c0 in range(0, n_vis, TILE):
                    c = c0 + ct
                    valid = c < n_vis
                    x = tl.load(srow + c, mask=valid, other=0.0)
                    bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
                    if p == 0:
                        m = valid
                    else:
                        m = valid & ((bits >> (shift + 8)) == (prefix >> (shift + 8)))
                    hist += tl.histogram((bits >> shift) & 255, 256, mask=m)
                cge = tl.cumsum(hist, 0, reverse=True)
                dstar = tl.max(tl.where(cge >= kk, bins, -1), axis=0)
                kk = kk - tl.sum(tl.where(bins > dstar, hist, 0), axis=0)
                prefix = prefix | (dstar << shift)
                if p == 3:
                    n_eq = tl.sum(tl.where(bins == dstar, hist, 0), axis=0)
            T = prefix
    # Every key above T, plus the smallest-id ties at T up to k (the unique-key
    # tie-break), compacted with a per-tile scan, then sorted descending.
    base = keys_out + row.to(tl.int64) * KP
    run_eq = zero
    run_pos = zero
    partial_ties = n_eq > kk
    for c0 in range(0, n_vis, TILE):
        c = c0 + ct
        valid = c < n_vis
        x = tl.load(srow + c, mask=valid, other=0.0)
        # relu scores are >= +0: the int32 bit pattern is order-isomorphic to the value
        bits = tl.where(valid, x.to(tl.int32, bitcast=True), -1)
        if partial_ties:
            eq = (bits == T) & valid
            eq_i = eq.to(tl.int32)
            rank_eq = run_eq + tl.cumsum(eq_i, 0) - eq_i
            sel = valid & ((bits > T) | (eq & (rank_eq < kk)))
            run_eq += tl.sum(eq_i, axis=0)
        else:
            # every tie at T is taken (the common case): no rank scan needed
            sel = valid & (bits >= T)
        sel_i = sel.to(tl.int32)
        pos = run_pos + tl.cumsum(sel_i, 0) - sel_i
        key = ((bits.to(tl.int64) + 1) << IDB) | (max_id_code - (lo + c)).to(tl.int64)
        tl.store(base + pos, key, mask=sel)
        run_pos += tl.sum(sel_i, axis=0)
    o_k = tl.arange(0, KP)
    tl.store(base + o_k, tl.zeros([KP], dtype=tl.int64), mask=o_k >= run_pos)
    tl.debug_barrier()
    kv = tl.load(base + o_k)
    kv = tl.sort(kv, descending=True)
    tl.debug_barrier()
    tl.store(base + o_k, kv)


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


_TOPK_TILE = 2048  # columns per streamed step of the selection kernel
_CHUNK_ELEMS = 1 << 27  # score buffer cap: 128M fp32 = 512 MiB
_CUDNN_MAX_ROWS = 32768  # FE: larger fused top-k calls are unsafe (see notes/02)


def _row_chunk(s: int, nb: int, impl: str) -> int:
    rc = max(64, min(s, _CHUNK_ELEMS // max(nb, 1)))
    if impl == "cudnn":
        rc = min(rc, _CUDNN_MAX_ROWS)
    return rc


_AUTO_CUDNN_MIN_BLOCKS = 4096  # GB200 H2H: Triton radix wins at 1K blocks, cuDNN at 8K/16K


@functools.lru_cache(maxsize=None)
def _has_cudnn_topk() -> bool:
    try:
        from cudnn import DSA  # noqa: F401

        return hasattr(DSA, "indexer_top_k_wrapper")
    except Exception:  # pragma: no cover - depends on the container
        return False


def _auto_impl(nb: int) -> str:
    """Shape-only dispatch (never data): the measured faster exact top-k per width."""
    if nb >= _AUTO_CUDNN_MIN_BLOCKS and _has_cudnn_topk():
        return "cudnn"
    return "triton"


def _cudnn_kth_bits(buf, n_vis, k):
    """k-th largest score bits per row via cuDNN FE radix top-k (rows with n_vis <= k
    are ignored by the caller). ``buf``/``n_vis`` have a fixed row count, so the FE
    wrapper cache (keyed on n_rows / num_cols) never recompiles in steady state."""
    from cudnn import DSA  # lazy: the Triton path works without cudnn

    out = DSA.indexer_top_k_wrapper(buf, n_vis, k, 1, True)
    vals = out["values"].float()
    vals = torch.where(out["indices"] >= 0, vals, float("inf"))
    return vals.amin(dim=-1).view(torch.int32).contiguous()


def qsa_select_topk_keys(
    q: torch.Tensor,  # [s, NH, DI] (one batch row)
    block_keys: torch.Tensor,  # [n_blocks, DI]
    m_t: torch.Tensor,  # [s]
    k: int,
    m_lo=None,  # [s] or None
    impl: str = "auto",
) -> torch.Tensor:
    """Top-k int64 keys ``[s, k]`` (descending), bit-identical to the opt2 chunked
    ``torch.topk`` merge (kept as the test oracle in ``qsa_reference.py``). ``impl``:
    ``"triton"`` radix select, ``"cudnn"`` FE k-th value, or ``"auto"`` (by static width)."""
    assert impl in ("triton", "cudnn", "auto"), impl
    s, NH, DI = q.shape
    nb = block_keys.shape[0]
    if impl == "auto":
        impl = _auto_impl(nb)
    assert nb < (1 << ID_BITS)
    device = q.device
    KP = _pow2(max(k, 1))
    NBP = _pow2(max(nb, 1))
    keys = torch.empty(s, KP, dtype=torch.int64, device=device)
    if s == 0:
        return keys[:, :k]
    rc = _row_chunk(s, nb, impl)
    buf = torch.empty(rc, nb, dtype=torch.float32, device=device)
    has_lo = m_lo is not None
    m_t = m_t.contiguous()
    m_lo_t = m_lo.contiguous() if has_lo else m_t
    if impl == "cudnn":
        nvis_all = (m_t - m_lo_t if has_lo else m_t).to(torch.int32)
        nvis_buf = torch.zeros(rc, dtype=torch.int32, device=device)
    TILE = min(_TOPK_TILE, NBP)
    nw = 8 if TILE >= 2048 else 4
    for r0 in range(0, s, rc):
        r1 = min(r0 + rc, s)
        n = r1 - r0
        grid = (triton.cdiv(n, 64), triton.cdiv(nb, 64))
        _qsa_select_score_kernel[grid](
            q[r0:r1],
            block_keys,
            buf,
            m_t[r0:r1],
            m_lo_t[r0:r1],
            n,
            nb,
            nb,
            DI**-0.5,
            NH=NH,
            DI=DI,
            BM=64,
            BN=64,
            HAS_LO=has_lo,
        )
        if impl == "cudnn":
            nvis_buf[:n].copy_(nvis_all[r0:r1])
            if n < rc:
                nvis_buf[n:].zero_()
            thr = _cudnn_kth_bits(buf, nvis_buf, k)
        else:
            thr = m_t  # unused
        _qsa_select_topk_kernel[(n,)](
            buf,
            thr,
            keys[r0:r1],
            m_t[r0:r1],
            m_lo_t[r0:r1],
            k,
            nb,
            _MAX_ID_CODE,
            TILE=TILE,
            KP=KP,
            IDB=ID_BITS,
            HAS_LO=has_lo,
            GIVEN_THR=impl == "cudnn",
            num_warps=nw,
        )
    return keys[:, :k]


# ---------------------------------------------------------------------------
# Sparse-attention metadata (K3): query-tile supersets and the KV-tile query CSR
# without sorts. Both CSRs are "row-major nonzeros of a dense bitmask":
#   * supersets: M[b * n_tiles, num_blocks] (uint8, or int32 when BQ > 8), bit q set iff
#     query q of the tile selected the block -> entries = nonzero blocks of each tile row
#     in ascending id;
#   * KV-tile CSR: P[b * NT, s] int32, bit r set iff query t selected block r of the KV
#     tile -> entries = nonzero queries of each tile row in ascending row.
# One scatter kernel fills the masks (per query each block appears once, so the uint8
# read-modify-write is conflict free; P is an integer sum of distinct powers of two,
# exact in any order), a count + cumsum gives the offsets, and a compaction kernel writes
# the entries. The outputs are bit-identical to the sort-based builders in triton_qsa.py.
# ---------------------------------------------------------------------------

_META_CH = 1024  # columns per compaction step


@triton.jit
def _qsa_meta_scatter_kernel(
    block_indices,  # [b, s, S] int64
    block_counts,  # [b, s]
    M,  # [b * n_tiles, ldm] uint8 (DO_SS)
    ss_counts,  # [b * n_tiles] int64 (DO_SS)
    P,  # [b * NT, s] int32 (DO_KV)
    s,
    S,
    n_tiles,
    NT,
    ldm,
    ncols,
    BQ: tl.constexpr,
    SP: tl.constexpr,
    KB: tl.constexpr,
    CH: tl.constexpr,
    DO_SS: tl.constexpr,
    DO_KV: tl.constexpr,
):
    pid = tl.program_id(0)
    bi = pid // n_tiles
    tile = pid % n_tiles
    j = tl.arange(0, SP)
    mrow = M + pid.to(tl.int64) * ldm
    for qq in tl.static_range(BQ):
        t = tile * BQ + qq
        rv = t < s
        row = bi.to(tl.int64) * s + t
        cnt = tl.load(block_counts + row, mask=rv, other=0)
        m = (j < cnt) & (j < S) & rv
        ids = tl.load(block_indices + row * S + j, mask=m, other=0).to(tl.int32)
        if DO_SS:
            v = tl.load(mrow + ids, mask=m, other=0)
            tl.store(mrow + ids, (v | (1 << qq)).to(M.dtype.element_ty), mask=m)
            tl.debug_barrier()
        if DO_KV:
            kt = (bi * NT + ids // KB).to(tl.int64)
            tl.atomic_add(P + kt * s + t, (1 << (ids % KB)).to(tl.int32), mask=m, sem="relaxed")
    if DO_SS:
        acc = tl.zeros([CH], dtype=tl.int32)
        for c0 in range(0, ncols, CH):
            c = c0 + tl.arange(0, CH)
            mv = tl.load(mrow + c, mask=c < ncols, other=0)
            acc += (mv != 0).to(tl.int32)
        tl.store(ss_counts + pid, tl.sum(acc, axis=0).to(tl.int64))


@triton.jit
def _qsa_row_count_kernel(X, counts, ncols, ld, CH: tl.constexpr):
    r = tl.program_id(0)
    base = X + r.to(tl.int64) * ld
    acc = tl.zeros([CH], dtype=tl.int32)
    for c0 in range(0, ncols, CH):
        c = c0 + tl.arange(0, CH)
        v = tl.load(base + c, mask=c < ncols, other=0)
        acc += (v != 0).to(tl.int32)
    tl.store(counts + r, tl.sum(acc, axis=0).to(tl.int64))


@triton.jit
def _qsa_ss_compact_kernel(
    M,
    tile_offsets,
    block_bases,
    block_ends,
    entry_base,
    entry_end,
    membership,
    ncols,
    ldm,
    BQ: tl.constexpr,
    CH: tl.constexpr,
):
    r = tl.program_id(0)
    base = M + r.to(tl.int64) * ldm
    off = tl.load(tile_offsets + r)
    run = off * 0
    for c0 in range(0, ncols, CH):
        c = c0 + tl.arange(0, CH)
        v = tl.load(base + c, mask=c < ncols, other=0)
        nz = v != 0
        nz_i = nz.to(tl.int32)
        pos = off + run + (tl.cumsum(nz_i, 0) - nz_i).to(tl.int64)
        tl.store(entry_base + pos, tl.load(block_bases + c, mask=nz, other=0).to(tl.int32), mask=nz)
        tl.store(entry_end + pos, tl.load(block_ends + c, mask=nz, other=0).to(tl.int32), mask=nz)
        for qq in tl.static_range(BQ):
            tl.store(membership + pos * BQ + qq, ((v >> qq) & 1).to(tl.uint8), mask=nz)
        run += tl.sum(nz_i, axis=0).to(tl.int64)


@triton.jit
def _qsa_kv_compact_kernel(P, tile_offsets, entry_query, entry_bits, s, CH: tl.constexpr):
    r = tl.program_id(0)
    base = P + r.to(tl.int64) * s
    off = tl.load(tile_offsets + r)
    run = off * 0
    for c0 in range(0, s, CH):
        c = c0 + tl.arange(0, CH)
        v = tl.load(base + c, mask=c < s, other=0)
        nz = v != 0
        nz_i = nz.to(tl.int32)
        pos = off + run + (tl.cumsum(nz_i, 0) - nz_i).to(tl.int64)
        tl.store(entry_query + pos, c, mask=nz)
        tl.store(entry_bits + pos, v, mask=nz)
        run += tl.sum(nz_i, axis=0).to(tl.int64)


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(counts.numel() + 1, dtype=torch.long, device=counts.device)
    out[1:] = counts.cumsum(0)
    return out


def qsa_build_supersets_fused(
    block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
):
    """Fixed-shape supersets (same buffers and values as ``build_qsa_supersets_static``):
    returns (tile_offsets, entry_base, entry_end, membership, n_tiles, e_cap)."""
    b, s, S = block_indices.shape
    device = block_indices.device
    BQ = tile_queries
    n_tiles = (s + BQ - 1) // BQ
    n_flat = b * n_tiles
    u_cap = max(1, min(BQ * S, num_blocks))
    e_cap = n_flat * u_cap
    ldm = max(num_blocks, 1)
    # one bit per query of the tile (BQ <= 8 -> uint8, else int32)
    assert BQ <= 30, "superset query bitmask is an int32"
    m_dtype = torch.uint8 if BQ <= 8 else torch.int32
    M = torch.zeros(n_flat, ldm, dtype=m_dtype, device=device)
    counts = torch.empty(n_flat, dtype=torch.long, device=device)
    bi = block_indices.contiguous()
    bc = block_counts.contiguous()
    if n_flat > 0:
        _qsa_meta_scatter_kernel[(n_flat,)](
            bi,
            bc,
            M,
            counts,
            counts,
            s,
            S,
            n_tiles,
            1,
            ldm,
            num_blocks,
            BQ=BQ,
            SP=_pow2(S),
            KB=1,
            CH=_META_CH,
            DO_SS=True,
            DO_KV=False,
            num_warps=4,
        )
    tile_offsets = _offsets(counts)
    entry_base = torch.zeros(e_cap, dtype=torch.int32, device=device)
    entry_end = torch.zeros(e_cap, dtype=torch.int32, device=device)
    membership = torch.zeros(e_cap, BQ, dtype=torch.uint8, device=device)
    if n_flat > 0:
        _qsa_ss_compact_kernel[(n_flat,)](
            M,
            tile_offsets,
            block_bases.contiguous(),
            block_ends.contiguous(),
            entry_base,
            entry_end,
            membership,
            num_blocks,
            ldm,
            BQ=BQ,
            CH=_META_CH,
            num_warps=4,
        )
    return tile_offsets, entry_base, entry_end, membership, n_tiles, e_cap


def qsa_build_kv_tile_csr_fused(block_indices, block_counts, num_blocks, tile_blocks):
    """KV-tile CSR (same buffers and values as the sort-based ``build_qsa_kv_tile_csr``):
    returns (tile_offsets, entry_query, entry_bits, n_tiles)."""
    assert 1 <= tile_blocks <= 30, "the block bitmask is an int32"
    b, s, S = block_indices.shape
    device = block_indices.device
    NT = max(1, (num_blocks + tile_blocks - 1) // tile_blocks)
    n = b * s * S
    P = torch.zeros(b * NT, max(s, 1), dtype=torch.int32, device=device)
    BQ = 4
    n_qt = (s + BQ - 1) // BQ
    if b * n_qt > 0:
        _qsa_meta_scatter_kernel[(b * n_qt,)](
            block_indices.contiguous(),
            block_counts.contiguous(),
            P,
            P,
            P,
            s,
            S,
            n_qt,
            NT,
            1,
            1,
            BQ=BQ,
            SP=_pow2(S),
            KB=tile_blocks,
            CH=_META_CH,
            DO_SS=False,
            DO_KV=True,
            num_warps=4,
        )
    counts = torch.empty(b * NT, dtype=torch.long, device=device)
    _qsa_row_count_kernel[(b * NT,)](P, counts, s, max(s, 1), CH=_META_CH, num_warps=4)
    tile_offsets = _offsets(counts)
    entry_query = torch.zeros(n, dtype=torch.int32, device=device)
    entry_bits = torch.zeros(n, dtype=torch.int32, device=device)
    _qsa_kv_compact_kernel[(b * NT,)](
        P, tile_offsets, entry_query, entry_bits, s, CH=_META_CH, num_warps=4
    )
    return tile_offsets, entry_query, entry_bits, NT
