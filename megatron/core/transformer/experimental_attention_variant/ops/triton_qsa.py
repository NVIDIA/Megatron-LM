# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Self-contained Triton kernels for the QSA sparse-GQA attention path.

Design (SGLang-prefill style, replacing the earlier fla-NSA adaptation):

1. **Index-score kernel** (:func:`qsa_indexer_keys`): fused block scoring
   (``relu(q @ k).sum(heads) / sqrt(d)`` in fp32), causal-visibility masking and
   int64 sort-key encoding in one kernel — no ``[s, chunk, heads]`` intermediate is
   ever materialized. The keys feed the deterministic int64 ``torch.topk`` merge
   (valid keys are unique integers, so selection is deterministic by construction).
2. **Query-tile shared-superset gather**: the sparse attention kernels process
   ``BQ`` consecutive queries per program against the UNION of their selected
   blocks (plus per-query membership bits), so each gathered K/V tile is reused
   across the whole query tile instead of being re-fetched per query — the
   gather-traffic roofline that made the per-query kernels lose to dense flash.
   The union / membership / CSR structures are precomputed on the host
   (:func:`build_qsa_supersets`).
3. **Backward**: ``dq`` mirrors the forward per query tile; ``dk``/``dv`` run one
   program per 4-token KV block (padded to a 16-row dot tile) iterating the
   query tiles whose union contains that block — sequential accumulation, no
   atomics, so gradients are deterministic.

All varlen (packed THD) handling is folded into per-block token ``base``/``end``
arrays and the membership bits: the kernels never see ``cu_seqlens``. Selection
blocks may be partial (the trailing ragged block of a document) or fully in the
future (empty token range) — both are handled by the ``base``/``end`` masks,
which is how the query's own block reproduces the unconditional ragged tail.

The GQA group size ``G = HQ // H`` may be any value >= 1 (rows are padded to the
next power of two in-tile); no host-side head padding is required.
"""

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

BS = 4  # tokens per selection block (qsa_indexer_compress_ratio)
CB = 16  # union entries gathered per attention-kernel iteration (CB * BS = 64 tokens)
RS = 16  # padded KV rows per dkv program (>= 16 for tl.dot; BS rows are valid)
ID_BITS = 21  # block-id field width in the int64 sort keys (supports 2M blocks)


# ---------------------------------------------------------------------------
# 1. Index-score kernel: fused scoring + visibility + int64 key encoding
# ---------------------------------------------------------------------------


@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in [4, 8]], key=['BM', 'BN', 'DI'])
@triton.jit
def qsa_indexer_keys_kernel(
    q,  # [s, NH, DI] indexer queries (post norm/RoPE)
    bk,  # [n_blocks, DI] pooled block keys
    keys,  # [s, cb] int64 output
    m_t,  # [s] number of visible complete blocks per query
    j0,  # first block id of this chunk
    cb,  # chunk width (number of blocks scored)
    s,
    scale,  # 1 / sqrt(DI)
    max_id_code,  # (1 << IDB) - 1
    NH: tl.constexpr,
    DI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IDB: tl.constexpr,
):
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_d = tl.arange(0, DI)
    m_m = o_m < s
    m_n = o_n < cb

    # [DI, BN] block-key tile for this chunk
    b_k = tl.load(bk + (j0 + o_n)[None, :] * DI + o_d[:, None], mask=m_n[None, :], other=0.0).to(
        tl.float32
    )

    b_acc = tl.zeros([BM, BN], dtype=tl.float32)
    for h in tl.static_range(NH):
        b_q = tl.load(
            q + (o_m[:, None] * NH + h) * DI + o_d[None, :], mask=m_m[:, None], other=0.0
        ).to(tl.float32)
        # fp32 IEEE dot: keeps scores reproducible and tie behavior stable
        b_acc += tl.maximum(tl.dot(b_q, b_k, input_precision="ieee"), 0.0)

    score = b_acc * scale
    # relu scores are >= 0, so the fp32 bit pattern is order-isomorphic to the value
    bits = score.to(tl.int32, bitcast=True).to(tl.int64)
    key = ((bits + 1) << IDB) | (max_id_code - (j0 + o_n))[None, :]
    visible = (j0 + o_n)[None, :] < tl.load(m_t + o_m, mask=m_m, other=0)[:, None]
    key = tl.where(visible & m_n[None, :] & m_m[:, None], key, 0)
    tl.store(keys + o_m[:, None] * cb + o_n[None, :], key, mask=m_m[:, None] & m_n[None, :])


def qsa_indexer_keys(
    q: torch.Tensor,  # [s, NH, DI]
    block_keys: torch.Tensor,  # [n_blocks, DI]
    m_t: torch.Tensor,  # [s]
    j0: int,
    cb: int,
    keys_out: torch.Tensor,  # [s, cb] int64 (reused buffer)
) -> torch.Tensor:
    """Fill ``keys_out`` with int64 (score, block_id) sort keys for blocks [j0, j0+cb)."""
    s, NH, DI = q.shape
    BM, BN = 64, 64
    grid = (triton.cdiv(s, BM), triton.cdiv(cb, BN))
    qsa_indexer_keys_kernel[grid](
        q=q,
        bk=block_keys,
        keys=keys_out,
        m_t=m_t,
        j0=j0,
        cb=cb,
        s=s,
        scale=DI**-0.5,
        max_id_code=(1 << ID_BITS) - 1,
        NH=NH,
        DI=DI,
        BM=BM,
        BN=BN,
        IDB=ID_BITS,
    )
    return keys_out


# ---------------------------------------------------------------------------
# 2. Host-side superset precompute
# ---------------------------------------------------------------------------


class QSASupersets(NamedTuple):
    """Query-tile shared-superset structures (all device tensors).

    For every tile of ``BQ`` consecutive queries, the UNION of the tile's selected
    blocks is stored as a CSR of "entries"; each entry carries the block's token
    ``base``/``end`` and a ``[BQ]`` membership byte-vector saying which queries of
    the tile actually selected it. The inverse CSR groups entries by
    (batch, block) for the deterministic dkv backward.
    """

    tile_offsets: torch.Tensor  # [b * n_tiles + 1] int64 — CSR into the entry arrays
    entry_base: torch.Tensor  # [E] int32 — first token of the entry's block
    entry_end: torch.Tensor  # [E] int32 — one past the last valid token
    membership: torch.Tensor  # [E, BQ] uint8
    n_tiles: int
    num_entries: int


def build_qsa_supersets(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Build per-query-tile block unions, membership bits and the dkv CSR."""
    b, s, S = block_indices.shape
    device = block_indices.device
    BQ = tile_queries
    n_tiles = (s + BQ - 1) // BQ
    s_pad = n_tiles * BQ

    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    ids = torch.where(
        slot_valid, block_indices.long(), torch.full_like(block_indices.long(), num_blocks)
    )
    if s_pad != s:
        ids = torch.cat(
            [ids, torch.full((b, s_pad - s, S), num_blocks, dtype=ids.dtype, device=device)], 1
        )
        slot_valid = torch.cat(
            [slot_valid, torch.zeros(b, s_pad - s, S, dtype=torch.bool, device=device)], 1
        )
    n_flat = b * n_tiles
    tiles = ids.view(n_flat, BQ * S)
    sorted_ids, _ = torch.sort(tiles, dim=-1)
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    uniq = first & (sorted_ids < num_blocks)
    counts_per_tile = uniq.sum(-1)  # [n_flat]
    tile_offsets = torch.zeros(n_flat + 1, dtype=torch.long, device=device)
    tile_offsets[1:] = counts_per_tile.cumsum(0)
    num_entries = int(tile_offsets[-1])  # single host sync (allocation sizes)
    u_max = int(counts_per_tile.max()) if num_entries > 0 else 1

    entry_block = sorted_ids[uniq]  # row-major flatten => ascending within each tile
    # padded per-tile unions for one batched searchsorted
    padded = torch.full((n_flat, u_max), num_blocks, dtype=sorted_ids.dtype, device=device)
    rank = uniq.long().cumsum(-1) - 1
    row_idx = torch.arange(n_flat, device=device).unsqueeze(-1).expand_as(sorted_ids)
    padded[row_idx[uniq], rank[uniq]] = entry_block

    # membership: global entry index of every valid (query, slot)
    upos = torch.searchsorted(padded, tiles)  # [n_flat, BQ * S]
    gpos = tile_offsets[:-1].unsqueeze(-1) + upos
    valid_flat = slot_valid.view(n_flat, BQ * S)
    q_local = (torch.arange(BQ * S, device=device) // S).unsqueeze(0).expand(n_flat, -1)
    membership = torch.zeros(max(num_entries, 1), BQ, dtype=torch.uint8, device=device)
    membership[gpos[valid_flat], q_local[valid_flat]] = 1

    entry_base = block_bases[entry_block].to(torch.int32)
    entry_end = block_ends[entry_block].to(torch.int32)

    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=num_entries,
    )


def build_qsa_query_csr(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
):
    """Per-(batch, block) CSR of the query positions that selected it (dkv backward).

    Queries are stably ordered within each block, so the dkv accumulation order is
    deterministic.
    """
    b, s, S = block_indices.shape
    device = block_indices.device
    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    batch_ix = torch.arange(b, device=device).view(b, 1, 1).expand(b, s, S)[slot_valid]
    t_ix = torch.arange(s, device=device).view(1, s, 1).expand(b, s, S)[slot_valid]
    blk = block_indices.long()[slot_valid]
    key = batch_ix * num_blocks + blk
    perm = torch.argsort(key, stable=True)  # stable => ascending t within each block
    qsel_ids = t_ix[perm].to(torch.int32)
    counts = torch.bincount(key, minlength=b * num_blocks)
    qsel_offsets = torch.zeros(b * num_blocks + 1, dtype=torch.long, device=device)
    qsel_offsets[1:] = counts.cumsum(0)
    return qsel_ids, qsel_offsets


# ---------------------------------------------------------------------------
# 3. Sparse GQA forward
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=st) for w in [4, 8] for st in [1, 2, 3]],
    key=['BK', 'BV', 'GP', 'BQ'],
)
@triton.jit
def qsa_fwd_kernel(
    q,  # [B, TQ, HQ, K]
    k,  # [B, TK, H, K]
    v,  # [B, TK, H, V]
    o,  # [B, TQ, HQ, V]
    lse,  # [B, TQ, HQ] fp32
    q_pos,  # [TQ] global token position of each query row (arange when TQ == TK)
    tile_offsets,
    entry_base,
    entry_end,
    membership,
    scale,
    TQ,
    TK,
    n_tiles,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    CBZ: tl.constexpr,
):
    i_tile = tl.program_id(0).to(tl.int64)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2).to(tl.int64)
    i_b = i_tile // n_tiles
    tile = i_tile % n_tiles

    R: tl.constexpr = BQ * GP
    CBT: tl.constexpr = CBZ * BSZ
    o_r = tl.arange(0, R)
    q_loc = o_r // GP
    g = o_r % GP
    o_t = tile * BQ + q_loc  # query ROW within the (possibly CP-local) sample
    m_row = (o_t < TQ) & (g < G)
    # global position of each query row (for causal masking against K tokens)
    b_pos = tl.load(q_pos + o_t, mask=o_t < TQ, other=0).to(tl.int64)
    head = i_h * G + g
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    p_q = q + ((i_b * TQ + o_t) * HQ + head)[:, None] * K + o_d[None, :]
    b_q = tl.load(p_q, mask=m_row[:, None] & m_d[None, :], other=0.0)
    # log2-domain online softmax: fold scale * log2(e) into q so exp/log become
    # exp2/log2 (the LSE is stored in the log2 domain; only these kernels read it).
    b_q = (b_q * (scale * 1.4426950408889634)).to(b_q.dtype)

    e0 = tl.load(tile_offsets + i_tile)
    e1 = tl.load(tile_offsets + i_tile + 1)

    b_o = tl.zeros([R, BV], dtype=tl.float32)
    b_m = tl.full([R], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([R], dtype=tl.float32)
    for es in range(e0, e1, CBZ):
        o_e = es + tl.arange(0, CBZ)
        m_e = o_e < e1
        base = tl.load(entry_base + o_e, mask=m_e, other=0).to(tl.int64)
        tend = tl.load(entry_end + o_e, mask=m_e, other=0).to(tl.int64)
        offs2 = base[:, None] + tl.arange(0, BSZ)[None, :]
        val2 = (offs2 < tend[:, None]) & m_e[:, None]
        offs = tl.reshape(offs2, [CBT])
        m_tok = tl.reshape(val2, [CBT])

        p_k = k + ((i_b * TK + offs) * H + i_h)[None, :] * K + o_d[:, None]
        b_k = tl.load(p_k, mask=m_tok[None, :] & m_d[:, None], other=0.0)
        p_v = v + ((i_b * TK + offs) * H + i_h)[:, None] * V + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tok[:, None] & m_v[None, :], other=0.0)

        # [R, CBT] scores
        b_s = tl.dot(b_q, b_k)
        # membership: [BQ, CB] -> broadcast to [R, CBT]
        mem = tl.load(
            membership + o_e[None, :] * BQ + tl.arange(0, BQ)[:, None], mask=m_e[None, :], other=0
        )
        m_mem = (
            tl.reshape(
                tl.broadcast_to(tl.reshape(mem, [BQ, 1, CBZ, 1]), [BQ, GP, CBZ, BSZ]), [R, CBT]
            )
            != 0
        )
        causal = offs[None, :] <= b_pos[:, None]
        m_all = m_tok[None, :] & m_mem & causal & m_row[:, None]
        b_s = tl.where(m_all, b_s, float('-inf'))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.where(b_m == float('-inf'), 0.0, tl.exp2(b_m - b_m_new))
        b_p = tl.where(b_s == float('-inf'), 0.0, tl.exp2(b_s - b_m_new[:, None]))
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = b_m_new

    b_o = b_o / tl.maximum(b_acc, 1e-10)[:, None]
    b_lse = tl.where(b_m == float('-inf'), 0.0, b_m + tl.log2(tl.maximum(b_acc, 1e-10)))
    p_o = o + ((i_b * TQ + o_t) * HQ + head)[:, None] * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_row[:, None] & m_v[None, :])
    if i_v == 0:
        p_lse = lse + (i_b * TQ + o_t) * HQ + head
        tl.store(p_lse, b_lse, mask=m_row)


# ---------------------------------------------------------------------------
# 4. Backward: dq (per query tile) and dkv (per KV block, no atomics)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=st) for w in [4, 8] for st in [1, 2]],
    key=['BK', 'BV', 'GP', 'BQ'],
)
@triton.jit
def qsa_bwd_dq_kernel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    q_pos,
    tile_offsets,
    entry_base,
    entry_end,
    membership,
    scale,
    TQ,
    TK,
    n_tiles,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    BQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    CBZ: tl.constexpr,
):
    i_tile = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = i_tile // n_tiles
    tile = i_tile % n_tiles

    R: tl.constexpr = BQ * GP
    CBT: tl.constexpr = CBZ * BSZ
    o_r = tl.arange(0, R)
    q_loc = o_r // GP
    g = o_r % GP
    o_t = tile * BQ + q_loc
    m_row = (o_t < TQ) & (g < G)
    b_pos = tl.load(q_pos + o_t, mask=o_t < TQ, other=0).to(tl.int64)
    head = i_h * G + g
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    rowptr = (i_b * TQ + o_t) * HQ + head
    b_q = tl.load(
        q + rowptr[:, None] * K + o_d[None, :], mask=m_row[:, None] & m_d[None, :], other=0.0
    )
    # scores in the log2 domain, matching the forward's log2-domain LSE
    b_q = (b_q * (scale * 1.4426950408889634)).to(b_q.dtype)
    b_do = tl.load(
        do + rowptr[:, None] * V + o_v[None, :], mask=m_row[:, None] & m_v[None, :], other=0.0
    )
    b_lse = tl.load(lse + rowptr, mask=m_row, other=0.0)
    b_delta = tl.load(delta + rowptr, mask=m_row, other=0.0)

    e0 = tl.load(tile_offsets + i_tile)
    e1 = tl.load(tile_offsets + i_tile + 1)

    b_dq = tl.zeros([R, BK], dtype=tl.float32)
    for es in range(e0, e1, CBZ):
        o_e = es + tl.arange(0, CBZ)
        m_e = o_e < e1
        base = tl.load(entry_base + o_e, mask=m_e, other=0).to(tl.int64)
        tend = tl.load(entry_end + o_e, mask=m_e, other=0).to(tl.int64)
        offs2 = base[:, None] + tl.arange(0, BSZ)[None, :]
        val2 = (offs2 < tend[:, None]) & m_e[:, None]
        offs = tl.reshape(offs2, [CBT])
        m_tok = tl.reshape(val2, [CBT])

        b_k = tl.load(
            k + ((i_b * TK + offs) * H + i_h)[None, :] * K + o_d[:, None],
            mask=m_tok[None, :] & m_d[:, None],
            other=0.0,
        )
        b_v = tl.load(
            v + ((i_b * TK + offs) * H + i_h)[None, :] * V + o_v[:, None],
            mask=m_tok[None, :] & m_v[:, None],
            other=0.0,
        )

        b_s = tl.dot(b_q, b_k)
        mem = tl.load(
            membership + o_e[None, :] * BQ + tl.arange(0, BQ)[:, None], mask=m_e[None, :], other=0
        )
        m_mem = (
            tl.reshape(
                tl.broadcast_to(tl.reshape(mem, [BQ, 1, CBZ, 1]), [BQ, GP, CBZ, BSZ]), [R, CBT]
            )
            != 0
        )
        causal = offs[None, :] <= b_pos[:, None]
        m_all = m_tok[None, :] & m_mem & causal & m_row[:, None]

        b_p = tl.where(m_all, tl.exp2(b_s - b_lse[:, None]), 0.0)
        # [R, BV] @ [BV, CBT] -> [R, CBT]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(
        dq + rowptr[:, None] * K + o_d[None, :],
        b_dq.to(dq.dtype.element_ty),
        mask=m_row[:, None] & m_d[None, :],
    )


def _prune_dkv_bq(configs, nargs, **kwargs):
    # The dkv gather stacks BQD queries x GP group heads into a [BQD*GP]-wide tile.
    # tl.dot needs the tile at least 16 wide, and past ~1024 columns the Triton
    # compiler fails outright (which autotune cannot catch), so keep 16 <= BQD*GP
    # <= 256. At least one config always survives (BQD=16 covers GP <= 16; BQD=1
    # covers GP >= 16).
    gp = {**(nargs or {}), **kwargs}.get('GP', 1)
    kept = [c for c in configs if 16 <= c.kwargs['BQD'] * gp <= 256]
    return kept or [c for c in configs if c.kwargs['BQD'] == 1]


@triton.autotune(
    configs=[
        triton.Config({'BQD': bqd}, num_warps=w, num_stages=st)
        for bqd in [1, 2, 4, 8, 16]
        for w in [4, 8]
        for st in [1, 2]
    ],
    key=['BK', 'BV', 'GP'],
    prune_configs_by={'early_config_prune': _prune_dkv_bq},
)
@triton.jit
def qsa_bwd_dkv_kernel(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    q_pos,  # [TQ] global position of each query row
    qsel_ids,  # [Nsel] int32 — query ROWS grouped by (batch, block)
    qsel_offsets,  # [b * NB + 1] int64
    block_bases,
    block_ends,
    scale,
    TQ,
    TK,
    NB,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BSZ: tl.constexpr,
    RSZ: tl.constexpr,
    BQD: tl.constexpr,
):
    i_bj = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = i_bj // NB
    j = i_bj % NB

    CQ: tl.constexpr = BQD * GP  # query-column tile width
    base = tl.load(block_bases + j).to(tl.int64)
    tend = tl.load(block_ends + j).to(tl.int64)
    o_rs = tl.arange(0, RSZ)
    toks = base + o_rs
    m_tok = (o_rs < BSZ) & (toks < tend)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_d = o_d < K
    m_v = o_v < V

    kv_ptr = (i_b * TK + toks) * H + i_h
    b_k = tl.load(
        k + kv_ptr[:, None] * K + o_d[None, :], mask=m_tok[:, None] & m_d[None, :], other=0.0
    )
    b_v = tl.load(
        v + kv_ptr[:, None] * V + o_v[None, :], mask=m_tok[:, None] & m_v[None, :], other=0.0
    )

    e0 = tl.load(qsel_offsets + i_bj)
    e1 = tl.load(qsel_offsets + i_bj + 1)

    b_dk = tl.zeros([RSZ, BK], dtype=tl.float32)
    b_dv = tl.zeros([RSZ, BV], dtype=tl.float32)
    o_c = tl.arange(0, CQ)
    q_slot = o_c // GP
    g = o_c % GP
    head = i_h * G + g
    for es in range(e0, e1, BQD):
        o_e = es + q_slot
        m_e = o_e < e1
        o_t = tl.load(qsel_ids + o_e, mask=m_e, other=0).to(tl.int64)
        o_pt = tl.load(q_pos + o_t, mask=m_e, other=0).to(tl.int64)
        m_col = m_e & (g < G)

        rowptr = (i_b * TQ + o_t) * HQ + head
        b_q = tl.load(
            q + rowptr[:, None] * K + o_d[None, :], mask=m_col[:, None] & m_d[None, :], other=0.0
        )
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(
            do + rowptr[:, None] * V + o_v[None, :], mask=m_col[:, None] & m_v[None, :], other=0.0
        )
        b_lse = tl.load(lse + rowptr, mask=m_col, other=0.0)
        b_delta = tl.load(delta + rowptr, mask=m_col, other=0.0)

        # [RSZ, BK] @ [BK, CQ] -> [RSZ, CQ]
        b_s = tl.dot(b_k, tl.trans(b_q))
        m_all = m_tok[:, None] & m_col[None, :] & (o_pt[None, :] >= toks[:, None])
        # b_q here keeps the plain `scale` factor (it doubles as the dk weight
        # below), so convert to the log2 domain at the exponent instead.
        b_p = tl.where(m_all, tl.exp2(b_s * 1.4426950408889634 - b_lse[None, :]), 0.0)

        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[None, :])
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(
        dk + kv_ptr[:, None] * K + o_d[None, :],
        b_dk.to(dk.dtype.element_ty),
        mask=m_tok[:, None] & m_d[None, :],
    )
    tl.store(
        dv + kv_ptr[:, None] * V + o_v[None, :],
        b_dv.to(dv.dtype.element_ty),
        mask=m_tok[:, None] & m_v[None, :],
    )


# ---------------------------------------------------------------------------
# 5. Host wrappers and autograd
# ---------------------------------------------------------------------------


def _pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def qsa_choose_tile_queries(num_q_heads: int, num_kv_heads: int) -> int:
    """Queries per tile so the row dim BQ * next_pow2(G) is a >=16 dot tile."""
    gp = _pow2(num_q_heads // num_kv_heads)
    return max(4, 16 // gp)


def _launch_dims(K: int, V: int, G: int):
    BK = max(16, _pow2(K))
    BV = max(16, _pow2(V))
    assert BK <= 256 and BV <= 256, "QSA kernels support head dims up to 256"
    GP = _pow2(G)
    return BK, BV, GP


class QSASparseAttnFunction(torch.autograd.Function):
    """Autograd over the query-tile shared-superset sparse GQA kernels."""

    @staticmethod
    def forward(
        ctx, q, k, v, q_pos, supersets, query_csr, block_bases, block_ends, scale, tile_queries
    ):
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        G = HQ // H
        BK, BV, GP = _launch_dims(K, V, G)
        NV = triton.cdiv(V, BV)
        assert NV == 1

        o = torch.empty(B, TQ, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, TQ, HQ, dtype=torch.float32, device=q.device)
        n_tiles_total = B * supersets.n_tiles
        qsa_fwd_kernel[(n_tiles_total, NV, H)](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            q_pos=q_pos,
            tile_offsets=supersets.tile_offsets,
            entry_base=supersets.entry_base,
            entry_end=supersets.entry_end,
            membership=supersets.membership,
            scale=scale,
            TQ=TQ,
            TK=TK,
            n_tiles=supersets.n_tiles,
            H=H,
            HQ=HQ,
            G=G,
            GP=GP,
            BQ=tile_queries,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BSZ=BS,
            CBZ=CB,
        )
        ctx.save_for_backward(q, k, v, o, lse, q_pos, block_bases, block_ends)
        ctx.supersets = supersets
        ctx.query_csr = query_csr
        ctx.scale = scale
        ctx.tile_queries = tile_queries
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, q_pos, block_bases, block_ends = ctx.saved_tensors
        ss: QSASupersets = ctx.supersets
        B, TQ, HQ, K = q.shape
        TK, H, V = k.shape[1], k.shape[2], v.shape[-1]
        G = HQ // H
        BK, BV, GP = _launch_dims(K, V, G)
        BQ = ctx.tile_queries
        NB = block_bases.shape[0]

        do = do.contiguous()
        # delta = rowwise sum(o * do) over the value dim, fp32
        delta = (o.float() * do.float()).sum(-1)

        dq = torch.empty_like(q)
        n_tiles_total = B * ss.n_tiles
        qsa_bwd_dq_kernel[(n_tiles_total, H)](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dq=dq,
            q_pos=q_pos,
            tile_offsets=ss.tile_offsets,
            entry_base=ss.entry_base,
            entry_end=ss.entry_end,
            membership=ss.membership,
            scale=ctx.scale,
            TQ=TQ,
            TK=TK,
            n_tiles=ss.n_tiles,
            H=H,
            HQ=HQ,
            G=G,
            GP=GP,
            BQ=BQ,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BSZ=BS,
            CBZ=CB,
        )

        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        qsel_ids, qsel_offsets = ctx.query_csr
        qsa_bwd_dkv_kernel[(B * NB, H)](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dk=dk,
            dv=dv,
            q_pos=q_pos,
            qsel_ids=qsel_ids,
            qsel_offsets=qsel_offsets,
            block_bases=block_bases,
            block_ends=block_ends,
            scale=ctx.scale,
            TQ=TQ,
            TK=TK,
            NB=NB,
            H=H,
            HQ=HQ,
            G=G,
            GP=GP,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BSZ=BS,
            RSZ=RS,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def qsa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    block_bases: torch.Tensor,
    block_ends: torch.Tensor,
    scale: Optional[float] = None,
    tile_queries: Optional[int] = None,
    q_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse GQA attention over per-query selected 4-token blocks.

    Args:
        q: ``[B, TQ, HQ, K]`` (no head padding required; any ``HQ/H >= 1``).
        k/v: ``[B, TK, H, K/V]``. ``TK`` may exceed ``TQ`` under allgather context
            parallelism (local queries against the globally gathered K/V).
        block_indices: ``[B, TQ, S]`` GLOBAL block ids in the ``block_bases`` space
            (document-local ids must be offset by the caller under packed THD).
            Slots beyond ``block_counts`` are ignored.
        block_counts: ``[B, TQ]`` valid slots per query.
        block_bases / block_ends: ``[num_blocks]`` first token and one-past-last
            valid token of every block (encodes ragged document tails and
            fully-future own blocks; the kernels never see cu_seqlens).
        scale: attention scale, default ``K ** -0.5``.
        tile_queries: queries per shared-superset tile (default derived from G).
        q_positions: ``[TQ]`` global token position of each query row for causal
            masking. Defaults to ``arange(TQ)``; REQUIRED when ``TQ != TK``.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    HQ, H = q.shape[2], k.shape[2]
    if tile_queries is None:
        tile_queries = qsa_choose_tile_queries(HQ, H)
    if q_positions is None:
        assert (
            q.shape[1] == k.shape[1]
        ), "q_positions is required when the query rows are a CP shard (TQ != TK)."
        q_positions = torch.arange(q.shape[1], device=q.device, dtype=torch.int64)
    num_blocks = block_bases.shape[0]
    supersets = build_qsa_supersets(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    query_csr = build_qsa_query_csr(block_indices, block_counts, num_blocks) if needs_grad else None
    return QSASparseAttnFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q_positions.contiguous(),
        supersets,
        query_csr,
        block_bases.contiguous(),
        block_ends.contiguous(),
        scale,
        tile_queries,
    )
