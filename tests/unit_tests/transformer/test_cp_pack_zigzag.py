# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage for pack-level zigzag context parallelism of the DSv4 attention stack.

``dsa_cp_pack_zigzag`` makes CP rank ``r`` hold pack chunks ``r`` and ``2*cp_size-1-r`` (two
contiguous segments of ``l_local/2`` rows) and reuses every contiguous CSA CP mechanism per
segment. These tests need neither CUDA nor a process group:

* the contiguous <-> pack-zigzag redistribution plan (``_a2a_plan``) and the point-to-point
  views that realise it (``_p2p_views``), with every rank of the CP group simulated in-process;
* the ``PackZigzagGeometry`` invariants, the chunk ownership helper and the physical
  ``kv_full_thd`` layout implied by the index bases;
* the two-block compressor row map (``remap_virtual_seq_to_rank_row`` over a host replica of
  the rank-row kernel) against a brute-force replay of the compaction kernel's group order;
* the ``dsa_cp_pack_zigzag`` configuration contract.

Device parity of the full attention path belongs in
``experimental_attention_variant/test_dsv4_hybrid_attention_cp.py``.
"""

import bisect
import math
import random

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_pack_zigzag as zz

_CP_SIZES = [2, 3, 4, 8, 16]
_CHUNK_ROWS = (1, 3)  # rows per half (c); the plan does not depend on the row width

# =============================================================================
# Layout redistribution: contiguous <-> pack-zigzag
# =============================================================================


def _contiguous_rows(cp, c, rank):
    """Pack rows of contiguous rank ``rank`` (chunks 2r and 2r+1)."""
    return list(range(rank * 2 * c, (rank + 1) * 2 * c))


def _zigzag_rows(cp, c, rank):
    """Pack rows of pack-zigzag rank ``rank`` (chunks r and 2*cp-1-r)."""
    tail = 2 * cp - 1 - rank
    return list(range(rank * c, (rank + 1) * c)) + list(range(tail * c, (tail + 1) * c))


def _plan_transfers(cp, c, to_zigzag, rows):
    """Apply every rank's plan with ``all_to_all_single`` semantics (the copying fallback of
    ``_redistribute_rows``: send buffer ordered by destination, receive buffer by source, half
    swaps as the plan says).

    ``rows[r]`` is rank r's buffer as a list of 2*c pack-row ids. Returns the resulting buffers
    and ``transfers[(src, dst)]``: the rows ``src`` hands to ``dst``, in send order.
    """
    plans = [zz._a2a_plan(cp, r, c, to_zigzag) for r in range(cp)]
    transfers = {}
    recv = [[None] * (2 * c) for _ in range(cp)]
    for src, (in_splits, _, swap_send, _) in enumerate(plans):
        send = rows[src][c:] + rows[src][:c] if swap_send else list(rows[src])
        offset = 0
        for dst, count in enumerate(in_splits):
            if count:
                transfers[(src, dst)] = send[offset : offset + count]
                recv_offset = sum(plans[dst][1][:src])
                recv[dst][recv_offset : recv_offset + count] = send[offset : offset + count]
                offset += count
        assert offset == 2 * c, (cp, src, in_splits)
    out = [(r[c:] + r[:c]) if p[3] else r for r, p in zip(recv, plans)]
    assert all(None not in o for o in out), (cp, c, to_zigzag)
    return out, transfers


def _fallback_ranks(cp):
    """Ranks whose plan is not "two halves to two peers": for odd CP the middle rank and the
    last rank exchange both halves with each other, so ``_p2p_views`` returns None for them."""
    return set() if cp % 2 == 0 else {(cp - 1) // 2, cp - 1}


@pytest.mark.internal
@pytest.mark.parametrize("cp", _CP_SIZES)
def test_a2a_plan_round_trip(cp):
    """Plans move whole halves, are pairwise consistent (what src sends to dst is what dst
    expects from src) and contiguous -> pack-zigzag -> contiguous recovers every rank's rows."""
    for c in _CHUNK_ROWS:
        for to_zigzag in (True, False):
            plans = [zz._a2a_plan(cp, r, c, to_zigzag) for r in range(cp)]
            for r, (in_splits, out_splits, _, _) in enumerate(plans):
                assert sum(in_splits) == 2 * c and sum(out_splits) == 2 * c
                assert all(n % c == 0 for n in in_splits + out_splits)
                for peer in range(cp):
                    assert in_splits[peer] == plans[peer][1][r], (cp, c, to_zigzag, r, peer)
        contiguous = [_contiguous_rows(cp, c, r) for r in range(cp)]
        zigzag, _ = _plan_transfers(cp, c, True, contiguous)
        assert zigzag == [_zigzag_rows(cp, c, r) for r in range(cp)]
        back, _ = _plan_transfers(cp, c, False, zigzag)
        assert back == contiguous


@pytest.mark.internal
@pytest.mark.parametrize("cp", _CP_SIZES)
def test_p2p_views_realise_the_plan(cp):
    """The zero-copy send/recv views move exactly the rows of the plan, write the receive
    buffer in place, and the exchanged buffers equal the geometry in both directions."""
    for c in _CHUNK_ROWS:
        rows = [_contiguous_rows(cp, c, r) for r in range(cp)]
        for to_zigzag, expected in ((True, _zigzag_rows), (False, _contiguous_rows)):
            ref_out, transfers = _plan_transfers(cp, c, to_zigzag, rows)
            flats = [torch.tensor(r_, dtype=torch.int64).view(-1, 1).repeat(1, 3) for r_ in rows]
            outs = [torch.full_like(f, -1) for f in flats]
            views = [
                zz._p2p_views(flats[r], outs[r], c, r, zz._a2a_plan(cp, r, c, to_zigzag))
                for r in range(cp)
            ]
            assert {r for r, v in enumerate(views) if v is None} == _fallback_ranks(cp)
            for r, v in enumerate(views):
                if v is None:
                    continue
                sends, recvs, self_pair = v
                assert len(sends) + (self_pair is not None) == 2, (cp, c, r)
                assert len(recvs) + (self_pair is not None) == 2, (cp, c, r)
                for view, peer in sends:
                    assert peer != r and view.shape[0] == c
                    assert view[:, 0].tolist() == transfers[(r, peer)], (cp, c, r, peer)
                for view, src in recvs:
                    assert src != r and view.shape[0] == c
                    # batch_isend_irecv delivery: the matching send of ``src`` lands here
                    payload = torch.tensor(transfers[(src, r)], dtype=torch.int64)
                    view.copy_(payload.view(-1, 1).expand(-1, 3))
                if self_pair is not None:
                    recv_view, send_view = self_pair
                    assert send_view[:, 0].tolist() == transfers[(r, r)]
                    recv_view.copy_(send_view)
                assert outs[r][:, 0].tolist() == ref_out[r] == expected(cp, c, r), (cp, c, r)
                assert torch.equal(outs[r][:, 0], outs[r][:, 2])  # whole rows moved
            rows = ref_out


@pytest.mark.internal
@pytest.mark.parametrize(
    "cp",
    [
        pytest.param(
            cp,
            marks=pytest.mark.xfail(
                cp % 2 == 1,
                strict=True,
                reason=(
                    "odd CP: ranks (cp-1)//2 and cp-1 exchange both halves with one peer, "
                    "_p2p_views returns None for them and _redistribute_rows takes the "
                    "all_to_all_single fallback while the other ranks post P2P ops on the "
                    "same group (mismatched collectives). Gate the P2P path on a group-wide "
                    "predicate (cp_size % 2 == 0) or let _p2p_views emit two ops per peer, "
                    "then drop this xfail."
                ),
            ),
        )
        for cp in _CP_SIZES
    ],
)
def test_redistribute_path_choice_is_uniform_across_ranks(cp):
    """Every rank of the group must pick the same code path (P2P views or the copying
    all_to_all_single fallback): the two are different collectives on one communicator."""
    c = 3
    for to_zigzag in (True, False):
        uses_fallback = [
            zz._p2p_views(
                torch.zeros(2 * c, 1),
                torch.zeros(2 * c, 1),
                c,
                r,
                zz._a2a_plan(cp, r, c, to_zigzag),
            )
            is None
            for r in range(cp)
        ]
        assert len(set(uses_fallback)) == 1, (cp, to_zigzag, uses_fallback)


# =============================================================================
# Geometry
# =============================================================================


@pytest.mark.parametrize("cp", _CP_SIZES)
def test_geometry_segments_partition_the_pack(cp):
    """Per rank: two consecutive segments at the virtual ranks' starts, halos one window before
    each segment, index bases matching the segment order; over the group: the segments
    partition the pack and every chunk has exactly one owner."""
    l_local, d_w = 4224, 128
    c = l_local // 2
    all_rows = []
    for r in range(cp):
        g = zz.PackZigzagGeometry(cp_size=cp, cp_rank=r, l_local=l_local, d_window=d_w)
        assert g.chunk == c
        assert g.virtual_ranks == (r, 2 * cp - 1 - r)
        assert all(zz.virtual_owner_of_chunk(v, cp) == r for v in g.virtual_ranks)
        assert g.seg_global_starts == tuple(v * c for v in g.virtual_ranks)
        assert g.seg_local_offsets == (0, c)
        assert g.block_index_base(0) == 0 and g.block_index_base(1) == d_w + c
        assert g.compressed_base == 2 * (d_w + c)
        rows = g.global_rows("cpu")
        assert rows.dtype == torch.int32 and rows.shape == (l_local,)
        halo = g.halo_global_rows("cpu")
        assert halo.dtype == torch.int32 and halo.shape == (2 * d_w,)
        for b, (start, offset) in enumerate(zip(g.seg_global_starts, g.seg_local_offsets)):
            assert torch.equal(
                rows[offset : offset + c], torch.arange(start, start + c, dtype=torch.int32)
            ), (cp, r, b)
            assert torch.equal(
                halo[b * d_w : (b + 1) * d_w], torch.arange(start - d_w, start, dtype=torch.int32)
            ), (cp, r, b)
        all_rows.append(rows)
    assert torch.equal(
        torch.sort(torch.cat(all_rows)).values, torch.arange(cp * l_local, dtype=torch.int32)
    )
    owners = [zz.virtual_owner_of_chunk(k, cp) for k in range(2 * cp)]
    assert owners == list(range(cp)) + list(range(cp - 1, -1, -1))


def test_geometry_pinned_example():
    """Numbers of the production-like case cp=4, rank=1, 4224 rows/rank, window 128."""
    g = zz.PackZigzagGeometry(cp_size=4, cp_rank=1, l_local=4224, d_window=128)
    assert g.chunk == 2112
    assert g.virtual_ranks == (1, 6)
    assert g.seg_global_starts == (2112, 12672)
    rows = g.global_rows("cpu")
    assert (int(rows[0]), int(rows[2111]), int(rows[2112]), int(rows[-1])) == (
        2112,
        4223,
        12672,
        14783,
    )
    halo = g.halo_global_rows("cpu")
    assert int(halo[0]) == 2112 - 128 and int(halo[128]) == 12672 - 128
    assert g.block_index_base(1) == 128 + 2112 and g.compressed_base == 2 * (128 + 2112)
    # rank 0's first halo window precedes the pack: negative rows (position helpers map to 0)
    g0 = zz.PackZigzagGeometry(cp_size=4, cp_rank=0, l_local=4224, d_window=128)
    assert int(g0.halo_global_rows("cpu")[0]) == -128


def test_geometry_rejects_odd_local_rows():
    with pytest.raises(ValueError, match="even per-rank row count"):
        zz.PackZigzagGeometry(cp_size=2, cp_rank=0, l_local=4225, d_window=128)


def test_geometry_requires_half_chunk_to_cover_the_window():
    with pytest.raises(RuntimeError, match="cover the boundary window"):
        zz.PackZigzagGeometry(cp_size=2, cp_rank=0, l_local=254, d_window=128)
    # the boundary itself (l_local/2 == d_window) is allowed
    assert zz.PackZigzagGeometry(cp_size=2, cp_rank=0, l_local=256, d_window=128).chunk == 128


def test_kv_full_layout_matches_index_bases():
    """``cat_kv_full_thd_pack_zigzag`` must place block b's (halo, local) pair at
    ``block_index_base(b)`` and the gathered compressed rows at ``compressed_base``."""
    cp, l_local, d_w = 4, 64, 8
    g = zz.PackZigzagGeometry(cp_size=cp, cp_rank=2, l_local=l_local, d_window=d_w)
    c = g.chunk
    halo = torch.arange(2 * d_w).view(-1, 1) + 1000
    local = torch.arange(l_local).view(-1, 1)
    compressed = torch.arange(24).view(-1, 1) + 5000
    kv_full = zz.cat_kv_full_thd_pack_zigzag(halo, local, compressed, g)
    assert kv_full.shape[0] == g.compressed_base + compressed.shape[0]
    for b in range(2):
        base = g.block_index_base(b)
        assert torch.equal(kv_full[base : base + d_w], halo[b * d_w : (b + 1) * d_w]), b
        assert torch.equal(kv_full[base + d_w : base + d_w + c], local[b * c : (b + 1) * c]), b
    assert torch.equal(kv_full[g.compressed_base :], compressed)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_boundary_rows_take_the_window_tails(ratio):
    cp, l_local = 2, 512
    d_w = 128
    d_comp = zz.compressor_d_comp(ratio)
    assert d_comp == (8 if ratio == 4 else ratio)
    g = zz.PackZigzagGeometry(cp_size=cp, cp_rank=1, l_local=l_local, d_window=d_w)
    halo = torch.arange(2 * d_w).view(-1, 1)
    rows = zz.compressor_boundary_rows(halo, g, d_comp)
    assert torch.equal(rows, torch.cat((halo[d_w - d_comp : d_w], halo[2 * d_w - d_comp :])))
    with pytest.raises(RuntimeError, match="expects 2\\*d_window"):
        zz.compressor_boundary_rows(halo[:-1], g, d_comp)
    with pytest.raises(RuntimeError, match="exceeds the boundary window"):
        zz.compressor_boundary_rows(halo, g, d_w + 1)


# =============================================================================
# Compressor: two-block (rank-major) compressed row map
# =============================================================================


def _block_groups(cu, seg_start, c, ratio, d_comp):
    """Replay the compaction kernel's group order for one contiguous block of ``c`` rows
    (same enumeration as ``_compressed_groups`` in test_csa_thd_layout_kernels)."""
    groups = []
    seg_end = seg_start + c
    for seq, (seq_start, seq_end) in enumerate(zip(cu[:-1], cu[1:])):
        local_end = min(seq_end, seg_end)
        if max(seq_start, seg_start) >= local_end:
            continue
        first_numer = max(0, seg_start - d_comp - seq_start)
        first = (first_numer + ratio - 1) // ratio
        stop = min((local_end - seq_start) // ratio, (seq_end - seq_start) // ratio)
        groups.extend((seq, gid) for gid in range(first, max(first, stop)))
    return groups


def _virtual_seq_to_rank_row(cu, cuc, l_virtual, n_virtual, ratio, d_comp, c_cap):
    """Host replica of ``thd_layout_kernels._compressor_rank_row_kernel`` for a contiguous
    world of ``n_virtual`` ranks with ``l_virtual`` rows each (the ``2*cp_size`` virtual world
    in which the pack-zigzag per-block layouts are built)."""
    n_seq = len(cu) - 1
    rows = []
    for logical in range((l_virtual * n_virtual) // ratio):
        if logical >= cuc[-1]:
            rows.append(-1)
            continue
        seq = 0
        while logical >= cuc[seq + 1]:
            seq += 1
        gid = logical - cuc[seq]
        last = cu[seq] + (gid + 1) * ratio - 1
        owner = min(n_virtual - 1, max(0, last // l_virtual))
        rank_start = owner * l_virtual
        first_seq = 0
        while first_seq + 1 < n_seq and rank_start >= cu[first_seq + 1]:
            first_seq += 1
        first_gid = (max(0, rank_start - d_comp - cu[first_seq]) + ratio - 1) // ratio
        rows.append(owner * c_cap + logical - (cuc[first_seq] + first_gid))
    return torch.tensor(rows, dtype=torch.int32)


def _random_pack(rng, total, ratio):
    """Padded THD cu_seqlens over ``total`` rows (1..4 sequences, tail merged into the last
    sequence) and the sequence-major compressed prefixes."""
    n_seq = rng.randint(1, 4)
    cuts = sorted(rng.sample(range(1, total), n_seq - 1)) if n_seq > 1 else []
    cu = [0] + cuts + [total]
    cuc = [0]
    for seq_start, seq_end in zip(cu[:-1], cu[1:]):
        cuc.append(cuc[-1] + (seq_end - seq_start) // ratio)
    return cu, cuc


@pytest.mark.parametrize("ratio", [4, 128])
def test_compact_group_capacity_matches_contiguous_formula(ratio):
    """Must equal the capacity ``thd_layout_kernels.build_cp_compressor_layout`` derives for a
    block of the same size (``build_cp_compressor_layout_pack_zigzag`` raises otherwise)."""
    d_comp = zz.compressor_d_comp(ratio)
    alignment = 32 // math.gcd(32, ratio)
    for rows in (128, 256, 384, 1024, 2112, 4224):
        expected = max(1, (rows + d_comp) // ratio)
        expected = ((expected + alignment - 1) // alignment) * alignment
        assert zz.compact_group_capacity(rows, ratio) == expected, (ratio, rows)
        assert zz.compact_group_capacity(rows, ratio) * ratio >= rows + d_comp - ratio + 1


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cp", _CP_SIZES)
def test_two_block_row_map_matches_brute_force(cp, ratio):
    """``remap_virtual_seq_to_rank_row`` over the virtual-world map must send every compressed
    group to ``owner_rank * 2*c_cap + block * c_cap + slot`` where ``slot`` is the group's
    position in the owning block's compaction, and no block may exceed ``c_cap`` groups."""
    rng = random.Random(1000 * cp + ratio)
    d_comp = zz.compressor_d_comp(ratio)
    for _trial in range(10):
        c = rng.choice([256, 384, 512, 1024]) if ratio == 4 else 128 * rng.randint(2, 10)
        l_local = 2 * c
        cu, cuc = _random_pack(rng, l_local * cp, ratio)
        c_cap = zz.compact_group_capacity(c, ratio)
        virtual = _virtual_seq_to_rank_row(cu, cuc, c, 2 * cp, ratio, d_comp, c_cap)
        got = zz.remap_virtual_seq_to_rank_row(virtual, cp, c_cap)
        assert got.dtype == torch.int32 and got.shape == ((l_local * cp) // ratio,)

        reference = {}
        for r in range(cp):
            g = zz.PackZigzagGeometry(
                cp_size=cp, cp_rank=r, l_local=l_local, d_window=max(128, d_comp)
            )
            for b, seg_start in enumerate(g.seg_global_starts):
                groups = _block_groups(cu, seg_start, c, ratio, d_comp)
                assert len(groups) <= c_cap, (cp, ratio, c, cu, r, b, len(groups), c_cap)
                for slot, (seq, gid) in enumerate(groups):
                    last = cu[seq] + (gid + 1) * ratio - 1
                    if seg_start <= last < seg_start + c:  # owned by this block
                        reference[(seq, gid)] = r * (2 * c_cap) + b * c_cap + slot

        n_logical = cuc[-1]
        assert len(reference) == n_logical
        for logical in range(n_logical):
            seq = bisect.bisect_right(cuc, logical) - 1
            gid = logical - cuc[seq]
            assert int(got[logical]) == reference[(seq, gid)], (cp, ratio, c, cu, logical)
        assert bool((got[n_logical:] == -1).all())
        physical = got[:n_logical].tolist()
        assert len(set(physical)) == n_logical and 0 <= min(physical)
        assert max(physical) < cp * 2 * c_cap


@pytest.mark.parametrize("cp", [2, 3, 4])
def test_position_ids_from_rows_match_contiguous_segments(cp):
    """``thd_cp_position_ids_from_rows`` on a rank's zigzag rows equals the contiguous helper
    applied per segment; halo rows before the pack start map to position 0."""
    cp_utils = pytest.importorskip(
        "megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils"
    )
    c, d_w = 256, 128
    total = 2 * c * cp
    cu = torch.tensor([0, total // 3, (2 * total) // 3 + 5, total], dtype=torch.int32)
    for r in range(cp):
        g = zz.PackZigzagGeometry(cp_size=cp, cp_rank=r, l_local=2 * c, d_window=d_w)
        got = cp_utils.thd_cp_position_ids_from_rows(cu, g.global_rows("cpu"))
        ref = torch.cat([cp_utils.thd_cp_position_ids(cu, s, c) for s in g.seg_global_starts])
        assert torch.equal(got, ref), (cp, r)
        halo = cp_utils.thd_cp_position_ids_from_rows(cu, g.halo_global_rows("cpu"))
        ref_halo = torch.cat(
            [cp_utils.thd_cp_position_ids(cu, s - d_w, d_w) for s in g.seg_global_starts]
        )
        assert torch.equal(halo, ref_halo), (cp, r)
        if r == 0:
            assert bool((halo[:d_w] == 0).all())


# =============================================================================
# Configuration contract
# =============================================================================


def _make_pack_zigzag_config(monkeypatch, fused=True, **overrides):
    """Minimal dsv4_hybrid CP config (the fixture dev's DSv4 tests share) with pack-zigzag
    enabled; backend probes are isolated the way test_cp_balanced_indexer_layout does it
    (``fused`` is what the DSA backend probe reports)."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: fused)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
    kwargs = dict(
        context_parallel_size=4,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=4096,
        csa_compress_ratios=[0, 4, 128, 4],
        csa_dense_mode=False,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="none",
        dsa_cp_pack_zigzag=True,
    )
    kwargs.update(overrides)
    return _make_config(**kwargs)


@pytest.mark.internal
def test_pack_zigzag_config_accepts_minimal_dsv4_cp_setup(monkeypatch):
    config = _make_pack_zigzag_config(monkeypatch)
    assert config.dsa_cp_pack_zigzag
    assert config.cp_partition_mode == "contiguous" and not config.dsa_cp_balance_indexer


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,match",
    [
        # dev rejects zigzag data splits for dsv4_hybrid before the pack-zigzag block runs;
        # both messages name the required mode.
        ({"cp_partition_mode": "zigzag"}, "cp_partition_mode='contiguous'"),
        ({"dsa_cp_balance_indexer": True}, "disable dsa_cp_balance_indexer"),
        ({"context_parallel_size": 1}, "requires context_parallel_size > 1"),
        (
            {
                "context_parallel_size": 1,
                "dynamic_context_parallel": True,
                "sequence_packing_scheduler": "default_dynamic_cp",
                "calculate_per_token_loss": True,
            },
            "does not support dynamic context parallelism",
        ),
        ({"cuda_graph_impl": "full_iteration"}, "full-iteration CUDA graphs"),
        ({"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss only"),
    ],
    ids=["zigzag_split", "balanced_indexer", "cp1", "dynamic_cp", "full_iteration", "dense_loss"],
)
def test_pack_zigzag_config_rejects_unsupported_combinations(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_pack_zigzag_config(monkeypatch, **overrides)


@pytest.mark.internal
def test_pack_zigzag_config_requires_fused_dsa_backend(monkeypatch):
    with pytest.raises(ValueError, match="requires the fused DSA kernel backend"):
        _make_pack_zigzag_config(monkeypatch, fused=False)
