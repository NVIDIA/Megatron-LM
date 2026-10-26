# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sparse-attention metadata of the QSA sparse-GQA attention.

The attention kernels (CuTe DSL, :mod:`cute_qsa_attn`) never see ``cu_seqlens``: every query's
selection is a list of GLOBAL block ids into per-block token ``[base, end)`` ranges, which
encode packed (THD) documents, ragged document tails and fully-future own blocks. This module
turns that selection into the two structures the kernels walk:

* **Query-tile supersets** (forward, dq): for every tile of ``BQ`` consecutive queries, the
  UNION of the tile's selected blocks as a CSR of entries (block token ``base``/``end``) plus a
  ``[BQ]`` membership vector per entry, so each gathered K/V row is reused by the whole query
  tile instead of being re-fetched per query.
* **KV-tile query CSR** (dk/dv): for every tile of ``tile_blocks`` consecutive blocks, the
  queries that selected any of its blocks, with a per-entry block bitmask.

On CUDA the builders are the sort-free Triton kernels of :mod:`triton_qsa_select`; the
sort-based ``*_torch`` builders are the CPU path and the bit-exact reference. The ``*_static``
variants work at shape-only upper bounds without host synchronization (CUDA-graph capturable).
"""

from typing import NamedTuple

import torch

BS = 4  # tokens per selection block (qsa_indexer_compress_ratio)


# ---------------------------------------------------------------------------
# Query-tile supersets (forward, dq) and the KV-tile query CSR (dk/dv)
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


def build_qsa_supersets_torch(
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


def build_qsa_supersets_static_torch(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Fixed-shape, host-sync-free :func:`build_qsa_supersets` (CUDA-graph capturable).

    The CSR is identical to the dynamic builder's -- same ``tile_offsets``, and the same
    entries in the same order at ``[0, num_entries)`` -- but every buffer is sized by a
    shape-only upper bound instead of the data-dependent entry count: each tile owns at
    most ``u_cap = min(BQ * S, num_blocks)`` unique blocks, so ``E_cap = n_tiles * u_cap``.
    Compaction uses device-computed scatter positions (invalid writes go to a dump slot
    that is sliced off) instead of boolean-mask indexing. Entries past a tile's count are
    never read by the kernels, which loop over ``[tile_offsets[i], tile_offsets[i + 1])``.
    """
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
    width = BQ * S
    u_cap = max(1, min(width, num_blocks))
    e_cap = n_flat * u_cap
    tiles = ids.view(n_flat, width)
    sorted_ids, _ = torch.sort(tiles, dim=-1)
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    uniq = first & (sorted_ids < num_blocks)
    counts_per_tile = uniq.sum(-1)  # [n_flat]
    tile_offsets = torch.zeros(n_flat + 1, dtype=torch.long, device=device)
    tile_offsets[1:] = counts_per_tile.cumsum(0)

    rank = uniq.long().cumsum(-1) - 1
    # Per-tile padded unions (row width u_cap) for the batched searchsorted; column u_cap
    # is a dump slot for the non-unique positions.
    padded = torch.full((n_flat, u_cap + 1), num_blocks, dtype=sorted_ids.dtype, device=device)
    padded.scatter_(1, torch.where(uniq, rank, u_cap), sorted_ids)
    padded = padded[:, :u_cap].contiguous()
    # Global CSR entries: tile i's unique blocks land at tile_offsets[i] + rank (exactly the
    # row-major compaction of the dynamic builder); position e_cap is the dump slot.
    entry_pos = torch.where(uniq, tile_offsets[:-1].unsqueeze(-1) + rank, e_cap)
    entry_block = torch.full((e_cap + 1,), num_blocks, dtype=sorted_ids.dtype, device=device)
    entry_block.scatter_(0, entry_pos.view(-1), sorted_ids.view(-1))
    entry_block = entry_block[:e_cap]

    upos = torch.searchsorted(padded, tiles)  # [n_flat, BQ * S]
    gpos = tile_offsets[:-1].unsqueeze(-1) + upos
    valid_flat = slot_valid.view(n_flat, width)
    q_local = (torch.arange(width, device=device) // S).unsqueeze(0).expand(n_flat, -1)
    membership = torch.zeros(e_cap + 1, BQ, dtype=torch.uint8, device=device)
    membership.index_put_(
        (torch.where(valid_flat, gpos, e_cap), q_local),
        torch.ones((), dtype=torch.uint8, device=device),
    )
    membership = membership[:e_cap]

    # The sentinel id (num_blocks) of the unused trailing entries maps to an empty range.
    pad = torch.zeros(1, dtype=block_bases.dtype, device=device)
    entry_base = torch.cat([block_bases, pad])[entry_block].to(torch.int32)
    entry_end = torch.cat([block_ends, pad])[entry_block].to(torch.int32)

    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=e_cap,
    )


class QSAKVTileCSR(NamedTuple):
    """Per-(batch, KV tile) CSR of the queries that selected any block of the tile.

    A KV tile is ``tile_blocks`` consecutive selection blocks of the ``block_bases``
    space. Entries of one tile are ordered by ascending query row, so the dkv
    accumulation order is deterministic and independent of the slot order.
    """

    tile_offsets: torch.Tensor  # [b * n_tiles + 1] int64 — CSR into the entry arrays
    entry_query: torch.Tensor  # [E] int32 — query ROW of the entry
    entry_bits: torch.Tensor  # [E] int32 — bit r set iff the query selected block r of the tile
    tile_blocks: int
    n_tiles: int


def build_qsa_kv_tile_csr_torch(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
    tile_blocks: int,
) -> QSAKVTileCSR:
    """Build the KV-tile query CSR for the dk/dv kernel (:class:`QSADkvSm100`).

    Host-sync free and fixed-shape (CUDA-graph capturable): the entry arrays are sized
    by the shape-only bound ``b * s * S``; the ``n`` unique (tile, query) entries occupy
    ``[0, n)`` and the kernel only reads ``[tile_offsets[i], tile_offsets[i + 1])``.
    A query selects a block at most once, so the per-entry bitmask is the integer sum of
    distinct powers of two (an exact, order-independent reduction).
    """
    assert 1 <= tile_blocks <= 30, "the block bitmask is an int32"
    b, s, S = block_indices.shape
    device = block_indices.device
    n_tiles = max(1, (num_blocks + tile_blocks - 1) // tile_blocks)
    slot_valid = torch.arange(S, device=device) < block_counts.unsqueeze(-1)
    blk = torch.where(slot_valid, block_indices.long(), 0)
    batch_ix = torch.arange(b, device=device).view(b, 1, 1)
    t_ix = torch.arange(s, device=device).view(1, s, 1)
    sentinel = b * n_tiles * s
    key = torch.where(
        slot_valid, (batch_ix * n_tiles + blk // tile_blocks) * s + t_ix, sentinel
    ).reshape(-1)
    bits = torch.where(
        slot_valid, torch.bitwise_left_shift(torch.ones_like(blk), blk % tile_blocks), 0
    )
    n = key.numel()
    sorted_key, perm = torch.sort(key)
    sorted_bits = bits.reshape(-1)[perm]
    valid = sorted_key < sentinel
    first = valid.clone()
    first[1:] &= sorted_key[1:] != sorted_key[:-1]
    cum_first = torch.zeros(n + 1, dtype=torch.long, device=device)
    cum_first[1:] = first.long().cumsum(0)
    entry_ix = cum_first[1:] - 1  # entry of every sorted element (valid ones)
    entry_query = torch.zeros(n + 1, dtype=torch.int32, device=device)
    entry_query.scatter_(0, torch.where(first, entry_ix, n), (sorted_key % s).to(torch.int32))
    entry_bits = torch.zeros(n + 1, dtype=torch.long, device=device)
    entry_bits.scatter_add_(0, torch.where(valid, entry_ix, n), sorted_bits)
    bounds = torch.arange(b * n_tiles + 1, device=device, dtype=sorted_key.dtype) * s
    tile_offsets = cum_first[torch.searchsorted(sorted_key, bounds)]
    return QSAKVTileCSR(
        tile_offsets=tile_offsets,
        entry_query=entry_query[:n],
        entry_bits=entry_bits[:n].to(torch.int32),
        tile_blocks=tile_blocks,
        n_tiles=n_tiles,
    )


# Sort-free (K3) builders: dense bitmask + count + compaction Triton kernels in
# ops/triton_qsa_select.py, bit-identical to the sort-based ``*_torch`` builders above
# (which remain the CPU path and the reference).


def _use_fused_meta(t: torch.Tensor) -> bool:
    return t.is_cuda


def build_qsa_supersets_static(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Fixed-shape, host-sync-free supersets (CUDA-graph capturable); see
    :func:`build_qsa_supersets_static_torch` for the contract."""
    if not _use_fused_meta(block_indices):
        return build_qsa_supersets_static_torch(
            block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
        )
    from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
        qsa_build_supersets_fused,
    )

    tile_offsets, entry_base, entry_end, membership, n_tiles, e_cap = qsa_build_supersets_fused(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    return QSASupersets(
        tile_offsets=tile_offsets,
        entry_base=entry_base,
        entry_end=entry_end,
        membership=membership,
        n_tiles=n_tiles,
        num_entries=e_cap,
    )


def build_qsa_supersets(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    block_bases: torch.Tensor,  # [num_blocks] token base per block
    block_ends: torch.Tensor,  # [num_blocks]
    num_blocks: int,
    tile_queries: int,
) -> QSASupersets:
    """Per-query-tile block unions, membership bits (exact-size buffers; one host sync)."""
    if not _use_fused_meta(block_indices):
        return build_qsa_supersets_torch(
            block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
        )
    ss = build_qsa_supersets_static(
        block_indices, block_counts, block_bases, block_ends, num_blocks, tile_queries
    )
    num_entries = int(ss.tile_offsets[-1])  # single host sync (allocation sizes)
    return QSASupersets(
        tile_offsets=ss.tile_offsets,
        entry_base=ss.entry_base[:num_entries],
        entry_end=ss.entry_end[:num_entries],
        membership=ss.membership[: max(num_entries, 1)],
        n_tiles=ss.n_tiles,
        num_entries=num_entries,
    )


def build_qsa_kv_tile_csr(
    block_indices: torch.Tensor,  # [b, s, S] GLOBAL block ids
    block_counts: torch.Tensor,  # [b, s]
    num_blocks: int,
    tile_blocks: int,
) -> "QSAKVTileCSR":
    """KV-tile query CSR for the dk/dv kernel (host-sync free, fixed
    shape); see :func:`build_qsa_kv_tile_csr_torch` for the contract."""
    if not _use_fused_meta(block_indices):
        return build_qsa_kv_tile_csr_torch(block_indices, block_counts, num_blocks, tile_blocks)
    from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
        qsa_build_kv_tile_csr_fused,
    )

    tile_offsets, entry_query, entry_bits, n_tiles = qsa_build_kv_tile_csr_fused(
        block_indices, block_counts, num_blocks, tile_blocks
    )
    return QSAKVTileCSR(
        tile_offsets=tile_offsets,
        entry_query=entry_query,
        entry_bits=entry_bits,
        tile_blocks=tile_blocks,
        n_tiles=n_tiles,
    )
