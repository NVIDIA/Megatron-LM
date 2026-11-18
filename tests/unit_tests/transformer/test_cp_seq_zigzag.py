# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Coverage for per-sequence zigzag context parallelism of the DSv4 attention stack
(``attention_cp_layout='zigzag'``, ``csa_utils/cp_seq_zigzag.py``).

CPU (no process group):

* the host reference of the layout against TE's ``thd_get_partitioned_indices`` formula and
  the device geometry tables against that reference;
* the halo exchange plan: every rank's outgoing tails are exactly the halo rows its neighbours'
  blocks need, simulated for every rank of the group;
* the block compressor maps against a brute-force replay of the ownership rule (coverage of
  every group exactly once, injective inverse, consistency with the rank-major row map);
* the block indexer layout (per-block K copies) and the static compaction capacity;
* the per-sequence padding floor in the data pipeline and the configuration contract.

GPU (one device, CuTeDSL): the block-table layout kernels against the host reference, and the
block-mode attention index kernel in all three index modes against full-sequence semantics.
Multi-rank parity of the full attention path lives in
``experimental_attention_variant/test_dsv4_hybrid_attention_cp.py``.
"""

import queue
import random
import threading
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_seq_zigzag as sz

_CP_SIZES = [2, 3, 4, 8]


class _FakeGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _te_partition(cu, cp_size, cp_rank):
    """Python replica of TE's ``thd_partition_src_index`` (local row -> pack row)."""
    cu_s = [c // cp_size for c in cu]
    total = cu[-1] // cp_size
    out = []
    for token_id in range(total):
        seq_id = max(s for s in range(len(cu) - 1) if token_id >= cu_s[s])
        seq_len = cu_s[seq_id + 1] - cu_s[seq_id]
        index = token_id - cu_s[seq_id]
        offset = cp_rank if index < seq_len // 2 else (cp_size - 1) * 2 - cp_rank
        out.append(index + cu_s[seq_id] * cp_size + seq_len // 2 * offset)
    return out


def _random_cu(rng, cp_size, n_seq, unit, max_units=12, allow_empty=True, min_units=1):
    lengths = [2 * cp_size * unit * rng.randint(min_units, max_units) for _ in range(n_seq)]
    if allow_empty and n_seq > 1 and rng.random() < 0.5:
        # a zero-length entry in a leading, middle or trailing position (trailing ones are the
        # static cu_seqlens padding of thd_max_packed_sequences)
        lengths[rng.randrange(n_seq)] = 0
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return cu


def _zigzag_blocks(cu, cp, rank):
    """(sequence, chunk, rows, local offset, pack start) of a rank's blocks, from the definition
    of the per-sequence zigzag split only (no helper of the module under test)."""
    blocks, local_offset = [], 0
    for i in range(len(cu) - 1):
        c = (cu[i + 1] - cu[i]) // (2 * cp)
        for k in (rank, 2 * cp - 1 - rank):
            blocks.append((i, k, c, local_offset, cu[i] + k * c))
            local_offset += c
    return blocks


class _QueueDist:
    """Stand-in for the torch.distributed point-to-point calls of the halo exchange: one queue
    per (sender, receiver) pair, so the exchange can run for every rank of a group on CPU (one
    thread per rank)."""

    isend = "isend"
    irecv = "irecv"

    def __init__(self, cp_size):
        self.queues = {(s, d): queue.Queue() for s in range(cp_size) for d in range(cp_size)}

    @staticmethod
    def get_global_rank(group, rank):
        return rank

    @staticmethod
    def P2POp(op, tensor, peer, group):
        return op, tensor, peer, group.rank()

    def batch_isend_irecv(self, ops):
        for op, tensor, peer, me in ops:
            if op == "isend":
                self.queues[(me, peer)].put(tensor.detach().clone())
        for op, tensor, peer, me in ops:
            if op == "irecv":
                with torch.no_grad():
                    tensor.copy_(self.queues[(peer, me)].get(timeout=60))
        return []


def _run_on_every_rank(cp_size, fn):
    errors = []

    def run(rank):
        try:
            fn(rank)
        except BaseException as exc:  # pylint: disable=broad-except
            errors.append((rank, exc))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(cp_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert not any(thread.is_alive() for thread in threads), "a rank is still waiting"
    assert not errors, errors


# =============================================================================
# Layout: host reference and device tables
# =============================================================================


def test_validate_geometry_device_assert_message_fits():
    cu = torch.tensor([0, 64, 192], dtype=torch.int32)
    geom = sz.make_geometry(_FakeGroup(4, 0), cu, 48, 8)
    assert len(sz._layout_assert_message(geom)) < 255  # torch._assert_async rejects longer text
    sz.validate_geometry(
        geom, host_check=False
    )  # valid geometry: the (synchronous CPU) assert passes


def test_geometry_memoizes_row_tables():
    cu = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    geom = sz.make_geometry(_FakeGroup(4, 1), cu, 48, 8)
    rows = geom.global_rows()
    assert rows is geom.global_rows() and geom.halo_global_rows() is geom.halo_global_rows()
    assert geom.row_blocks() is geom.row_blocks()
    rows64 = geom.global_rows(dtype=torch.int64)
    assert rows64 is not rows and rows64.dtype == torch.int64
    assert rows64.tolist() == rows.tolist() == sz.ref_local_to_global(cu.tolist(), 4, 1)


def test_prepare_compressor_input_fallback_gathers_zero_padding():
    torch.manual_seed(0)
    boundary = torch.randn(6, 1, 3)
    local = torch.randn(10, 1, 3, requires_grad=True)
    idx = torch.tensor([3, -1, 0, 15, 7, -1, 9, 7], dtype=torch.int32)
    out = sz.prepare_cp_compressor_input_seq_zigzag(
        local, boundary, SimpleNamespace(compact_to_source=idx)
    )
    source = torch.cat((boundary, local.detach()))
    expected = torch.stack([source[i] if i >= 0 else torch.zeros(1, 3) for i in idx.tolist()])
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    counts = [sum(1 for i in idx.tolist() if i == 6 + j) for j in range(10)]
    torch.testing.assert_close(local.grad[:, 0, 0], torch.tensor(counts, dtype=torch.float32))


@pytest.mark.parametrize("cp", _CP_SIZES)
def test_reference_layout_matches_te_partition(cp):
    rng = random.Random(cp)
    for _ in range(20):
        cu = _random_cu(rng, cp, rng.randint(1, 4), unit=1)
        for rank in range(cp):
            assert sz.ref_local_to_global(cu, cp, rank) == _te_partition(cu, cp, rank)


@pytest.mark.parametrize("cp", _CP_SIZES)
def test_geometry_tables_match_reference(cp):
    rng = random.Random(100 + cp)
    for _ in range(10):
        cu = _random_cu(rng, cp, rng.randint(1, 4), unit=3)
        l_local = cu[-1] // cp
        cu_t = torch.tensor(cu, dtype=torch.int32)
        for rank in range(cp):
            geom = sz.make_geometry(_FakeGroup(cp, rank), cu_t, l_local, d_window=2)
            blocks = sz.ref_blocks(cu, cp, rank)
            assert geom.n_blocks == len(blocks) == 2 * (len(cu) - 1)
            assert geom.block_rows.tolist() == [b[2] for b in blocks]
            assert geom.block_local_offset.tolist() == [b[3] for b in blocks]
            assert geom.block_global_start.tolist() == [b[4] for b in blocks]
            assert geom.block_seq.tolist() == [b[0] for b in blocks]
            assert geom.block_chunk.tolist() == [b[1] for b in blocks]
            assert geom.global_rows().tolist() == sz.ref_local_to_global(cu, cp, rank)
            halo = geom.halo_global_rows().tolist()
            expect = [g0 - 2 + m for (_, _, _, _, g0) in blocks for m in range(2)]
            assert halo == expect
            assert geom.compressed_base == geom.n_blocks * 2 + l_local
            assert bool(geom.validity())


def test_geometry_validity_flags_short_sequences():
    cp = 2
    cu = torch.tensor([0, 8, 16], dtype=torch.int32)  # half-chunk 2 rows
    geom = sz.make_geometry(_FakeGroup(cp, 0), cu, 8, d_window=2)
    assert bool(geom.validity())
    geom = sz.make_geometry(_FakeGroup(cp, 0), cu, 8, d_window=3)
    assert not bool(geom.validity())
    with pytest.raises(RuntimeError, match="half-chunk"):
        sz.validate_geometry(geom, host_check=True)
    # the device-side form of the same check (synchronous on CPU)
    with pytest.raises(RuntimeError):
        sz.validate_geometry(geom, host_check=False)
    # empty sequences are exempt
    cu = torch.tensor([0, 16, 16], dtype=torch.int32)
    geom = sz.make_geometry(_FakeGroup(cp, 1), cu, 8, d_window=4)
    assert bool(geom.validity())


@pytest.mark.parametrize(
    "cu,l_local",
    [
        ([0, 10, 16], 8),  # 10 and 6 rows: not multiples of 2 * cp_size
        ([0, 2, 18], 9),  # a 2-row sequence is not an empty one
        ([0, 8, 16], 12),  # the pack covers 16 rows, the rank group holds 24
        ([0, 8, 16], 4),  # ... or 8
        ([8, 16, 24], 12),  # the right lengths and row count, but the pack does not start at 0
    ],
    ids=["indivisible", "below_one_chunk", "rows_missing", "rows_extra", "nonzero_start"],
)
def test_geometry_validity_flags_inconsistent_packs(cu, l_local):
    # Neither the divisibility of every length by 2 * cp_size (TE's zigzag split) nor the row
    # count can be recovered from the block tables, which would silently describe another pack.
    geom = sz.make_geometry(_FakeGroup(2, 0), torch.tensor(cu, dtype=torch.int32), l_local, 1)
    assert not bool(geom.validity())
    with pytest.raises(RuntimeError, match="multiple of 2\\*cp_size"):
        sz.validate_geometry(geom, host_check=True)
    with pytest.raises(RuntimeError):
        sz.validate_geometry(geom, host_check=False)


# =============================================================================
# Halo exchange plan
# =============================================================================


@pytest.mark.parametrize("cp", _CP_SIZES)
def test_tail_indices_are_the_neighbours_halo_rows(cp):
    """Rank r's block-(i,0) tails are rank r+1's block-(i,0) halo; its block-(i,1) tails are
    rank r-1's block-(i,1) halo; on rank cp-1 the block-(i,0) tails are its own block-(i,1)
    halo. Rank 0's block (i,0) halo precedes the sequence (zeros)."""
    rng = random.Random(7 + cp)
    d_w = 3
    for _ in range(8):
        cu = _random_cu(rng, cp, rng.randint(1, 4), unit=d_w)
        l_local = cu[-1] // cp
        cu_t = torch.tensor(cu, dtype=torch.int32)
        geoms = [sz.make_geometry(_FakeGroup(cp, r), cu_t, l_local, d_w) for r in range(cp)]
        n_seq = len(cu) - 1
        for r in range(cp):
            g = geoms[r]
            rows = g.global_rows().tolist()
            idx0, idx1 = g.tail_indices(0, d_w).tolist(), g.tail_indices(1, d_w).tolist()
            assert all(0 <= j < l_local for j in idx0 + idx1)  # clamped for empty sequences
            halo = lambda rr: geoms[rr].halo_global_rows().view(2 * n_seq, d_w).tolist()
            for i in range(n_seq):
                if cu[i + 1] == cu[i]:
                    continue  # empty sequence: its slots are never read
                tail0 = [rows[j] for j in idx0[i * d_w : (i + 1) * d_w]]
                tail1 = [rows[j] for j in idx1[i * d_w : (i + 1) * d_w]]
                if r + 1 < cp:
                    assert tail0 == halo(r + 1)[2 * i]
                else:
                    assert tail0 == halo(r)[2 * i + 1]
                if r > 0:
                    assert tail1 == halo(r - 1)[2 * i + 1]
                else:
                    # chunk 0 of the sequence: the halo lies before the sequence start
                    assert all(x < cu[i] for x in halo(0)[2 * i])


def test_halo_exchange_drops_the_gradient_of_empty_sequence_slots():
    # Single-rank group: block (i, 1) reads the tail of block (i, 0) (the self-served halo of
    # rank cp-1), no messages. The tail indices of the empty middle sequence alias rows 56..63
    # of the first one; a gradient in its slots must not be added there.
    group = _FakeGroup(1, 0)
    d_w = 8
    cu = torch.tensor([0, 64, 64, 160], dtype=torch.int32)
    geom = sz.make_geometry(group, cu, 160, d_w)
    assert bool(geom.validity())
    torch.manual_seed(0)
    hidden = torch.randn(160, 1, 4, requires_grad=True)
    boundary = sz.exchange_cp_boundary_hidden_seq_zigzag(hidden, geom, group)
    slots = boundary.view(geom.n_blocks, d_w, 1, 4)
    assert torch.count_nonzero(slots[0::2]) == 0  # chunk 0 of every sequence: nothing precedes
    torch.testing.assert_close(slots[1], hidden.detach()[24:32])
    torch.testing.assert_close(slots[5], hidden.detach()[104:112])
    slot_grad = torch.randn_like(boundary)  # non-zero everywhere, empty slots included
    boundary.backward(slot_grad)
    expected = torch.zeros_like(hidden)
    expected[24:32] = slot_grad.view_as(slots)[1]
    expected[104:112] = slot_grad.view_as(slots)[5]
    torch.testing.assert_close(hidden.grad, expected)


@pytest.mark.parametrize(
    "half_chunks",
    [[8, 12, 16], [8, 0, 12], [0, 8, 12], [8, 12, 0], [8, 0, 0, 12, 0]],
    ids=["no_empty", "middle_empty", "leading_empty", "trailing_empty", "consecutive_empties"],
)
@pytest.mark.parametrize("cp", [2, 3, 4])
def test_halo_exchange_across_the_whole_group(monkeypatch, cp, half_chunks):
    """The real exchange, forward and backward, for every rank of the group at once (queues in
    place of the point-to-point calls), against the layout definition: every slot of a non-empty
    block holds the d_window pack rows before the block (zeros for chunk 0), and the slot
    gradients return to exactly those rows. Both message paths run (tails received from rank
    r+1 and from rank r-1), with a non-zero gradient in the slots of the empty sequences, whose
    clamped tail indices alias rows of other sequences."""
    d_w = 4
    cu = [0]
    for c in half_chunks:
        cu.append(cu[-1] + 2 * cp * c)
    total, l_local, n_seq = cu[-1], cu[-1] // cp, len(half_chunks)
    monkeypatch.setattr(sz, "dist", _QueueDist(cp))
    torch.manual_seed(1234 + cp)
    pack = torch.randn(total, 1, 3)
    cu_t = torch.tensor(cu, dtype=torch.int32)
    groups = [_FakeGroup(cp, rank) for rank in range(cp)]
    geoms = [sz.make_geometry(groups[rank], cu_t, l_local, d_w) for rank in range(cp)]
    blocks = [_zigzag_blocks(cu, cp, rank) for rank in range(cp)]
    rows = [[g0 + j for (_, _, c, _, g0) in blocks[rank] for j in range(c)] for rank in range(cp)]
    hidden = [pack[rows[rank]].clone().requires_grad_(True) for rank in range(cp)]
    boundary = [None] * cp

    def forward(rank):
        assert bool(geoms[rank].validity())
        boundary[rank] = sz.exchange_cp_boundary_hidden_seq_zigzag(
            hidden[rank], geoms[rank], groups[rank]
        )

    _run_on_every_rank(cp, forward)
    for rank in range(cp):
        slots = boundary[rank].detach().view(2 * n_seq, d_w, 1, 3)
        for b, (_, k, c, _, g0) in enumerate(blocks[rank]):
            if c:
                expected = torch.zeros(d_w, 1, 3) if k == 0 else pack[g0 - d_w : g0]
                assert torch.equal(slots[b], expected), (rank, b)

    slot_grad = [torch.randn(2 * n_seq * d_w, 1, 3) for _ in range(cp)]
    expected = torch.zeros_like(pack)
    for rank in range(cp):
        grads = slot_grad[rank].view(2 * n_seq, d_w, 1, 3)
        for b, (_, k, c, _, g0) in enumerate(blocks[rank]):
            if c and k:  # not an unused slot, not the zero halo of chunk 0
                expected[g0 - d_w : g0] += grads[b]

    _run_on_every_rank(cp, lambda rank: boundary[rank].backward(slot_grad[rank]))
    for rank in range(cp):
        torch.testing.assert_close(hidden[rank].grad, expected[rows[rank]], msg=f"rank {rank}")


# =============================================================================
# Compressor maps
# =============================================================================


def _brute_force_ownership(cu, cp, ratio):
    """{(seq, group): (rank, half)} by the rule 'the chunk holding the group's last token'."""
    owner = {}
    for i in range(len(cu) - 1):
        length = cu[i + 1] - cu[i]
        c = length // (2 * cp)
        for g in range(length // ratio):
            k = ((g + 1) * ratio - 1) // c
            owner[(i, g)] = (sz.owner_rank_of_chunk(k, cp), int(k >= cp), k)
    return owner


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cp", [2, 3, 4])
def test_compressor_maps_match_brute_force(cp, ratio):
    rng = random.Random(11 * cp + ratio)
    d_comp = sz.compressor_d_comp(ratio)
    for case in range(8):
        legal = case < 4
        if legal:
            # the accepted regime: half-chunk c >= d_comp (the per-sequence floor), with c not
            # a multiple of the ratio (14 / 21 rows at ratio 4, 129 / 172 / 215 at ratio 128) so
            # that groups straddle chunk boundaries
            unit, lo, hi = (7, 2, 3) if ratio == 4 else (43, 3, 5)
        else:
            # below the floor the maps must still partition the groups consistently
            unit, lo, hi = (max(1, d_comp // (2 * cp)), 1, 6) if ratio == 4 else (8, 1, 6)
        cu = _random_cu(rng, cp, rng.randint(1, 3), unit=unit, max_units=hi, min_units=lo)
        l_local = cu[-1] // cp
        n_seq = len(cu) - 1
        c_cap = sz.compact_group_capacity_blocks(l_local, n_seq, ratio)
        owners = _brute_force_ownership(cu, cp, ratio)
        seen = {}
        for rank in range(cp):
            c2s, s2c, gids, pids, lcu, lcuc, gcuc = sz.ref_compressor_layout(
                cu, cp, rank, ratio, c_cap
            )
            blocks = sz.ref_blocks(cu, cp, rank)
            owned = sum(
                sz.ref_owned_groups(k, c, ratio)[1] - sz.ref_owned_groups(k, c, ratio)[0]
                for (_, k, c, _, _) in blocks
            )
            assert owned <= c_cap, "static capacity violated"
            # forward map: every compact token of block b is the right global row
            for b, (i, k, c, loff, g0) in enumerate(blocks):
                first, stop = sz.ref_owned_groups(k, c, ratio)
                assert lcuc[b + 1] - lcuc[b] == stop - first
                for q in range(lcuc[b], lcuc[b + 1]):
                    g = gids[q]
                    assert g == first + (q - lcuc[b]) and pids[q] == g * ratio
                    for tau in range(ratio):
                        src = c2s[q * ratio + tau]
                        G = cu[i] + g * ratio + tau
                        if src < 2 * n_seq * d_comp:
                            assert src // d_comp == b and G == g0 - d_comp + src % d_comp
                            assert cu[i] <= G < g0  # a row of the same sequence before the chunk
                            if legal:
                                assert G >= g0 - c  # ... inside the preceding chunk's tail
                        else:
                            j = src - 2 * n_seq * d_comp
                            assert j - loff == G - g0 and 0 <= j - loff < c
                        assert s2c[src] == q * ratio + tau
                    # canonical groups are recorded once per (seq, group)
                    if owners[(i, g)][:2] == (rank, int(k >= cp)):
                        assert (i, g) not in seen
                        seen[(i, g)] = (rank, b, q)
            # inverse map is injective and only points at mapped compact rows
            fwd = {t: src for t, src in enumerate(c2s) if src >= 0}
            inv = {src: t for src, t in enumerate(s2c) if t >= 0}
            assert inv == {src: t for t, src in fwd.items()}
            # padding groups
            assert all(g == -1 for g in gids[lcuc[-1] :]) and all(p == 0 for p in pids[lcuc[-1] :])
        assert set(seen) == set(owners), "every group is owned by exactly one (rank, block)"
        # rank-major row map: physical slot = owner * c_cap + owner-local compact group
        s2r = sz.ref_seq_to_rank_row(cu, cp, ratio, c_cap)
        gcuc = [0]
        for i in range(n_seq):
            gcuc.append(gcuc[-1] + (cu[i + 1] - cu[i]) // ratio)
        assert len(s2r) == l_local * cp // ratio
        for lr, phys in enumerate(s2r):
            if lr >= gcuc[-1]:
                assert phys == -1
                continue
            i = max(s for s in range(n_seq) if lr >= gcuc[s])
            g = lr - gcuc[i]
            rank, b, q = seen[(i, g)]
            assert phys == rank * c_cap + q, (lr, i, g, phys, rank, q)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compact_capacity_bounds_owned_groups(ratio):
    rng = random.Random(ratio)
    for cp in (2, 4, 16):
        for _ in range(20):
            n_seq = rng.randint(1, 4)
            # half-chunks on both sides of d_comp (8 rows at ratio 4, 128 rows at ratio 128)
            cu = _random_cu(
                rng, cp, n_seq, unit=1 if ratio == 4 else 16, max_units=64 // ratio + 12
            )
            l_local = cu[-1] // cp
            c_cap = sz.compact_group_capacity_blocks(l_local, n_seq, ratio)
            for rank in range(cp):
                owned = sum(
                    sz.ref_owned_groups(k, c, ratio)[1] - sz.ref_owned_groups(k, c, ratio)[0]
                    for (_, k, c, _, _) in sz.ref_blocks(cu, cp, rank)
                )
                assert owned <= c_cap
    # the contiguous single-block formula is the n = 1/2 special case up to alignment slack
    assert sz.compact_group_capacity_blocks(13568, 2, 128) >= (13568 + 128) // 128


# =============================================================================
# Indexer layout
# =============================================================================


@pytest.mark.parametrize("cp", [2, 4])
def test_indexer_layout_per_block_k_copies(cp):
    ratio = 4
    rng = random.Random(cp)
    for _ in range(6):
        cu = _random_cu(rng, cp, rng.randint(1, 4), unit=1, max_units=11)  # odd half-chunks too
        l_local = cu[-1] // cp
        n_seq = len(cu) - 1
        cu_t = torch.tensor(cu, dtype=torch.int32)
        cuc = [0]
        for i in range(n_seq):
            cuc.append(cuc[-1] + (cu[i + 1] - cu[i]) // ratio)
        seq_major_rows = l_local * cp // ratio
        for rank in range(cp):
            geom = sz.make_geometry(_FakeGroup(cp, rank), cu_t, l_local, d_window=2)
            (cu_q, cu_k, offs), src_map, total_k = sz.build_cp_indexer_layout_seq_zigzag(
                torch.tensor(cuc, dtype=torch.int32), geom, ratio, seq_major_rows
            )
            blocks = sz.ref_blocks(cu, cp, rank)
            assert cu_q.tolist() == [b[3] for b in blocks] + [l_local, l_local]
            assert offs.tolist() == [k * c for (_, k, c, _, _) in blocks] + [0]
            assert cu_k[-1].item() == total_k == src_map.numel()
            for b, (i, k, c, _, _) in enumerate(blocks):
                visible = min(cuc[i + 1] - cuc[i], ((k + 1) * c) // ratio)
                rows = src_map[cu_k[b] : cu_k[b + 1]].tolist()
                assert len(rows) == visible + 1
                assert rows[:visible] == list(range(cuc[i], cuc[i] + visible))
                assert rows[visible] == seq_major_rows  # guard row -> appended zero row
                # compact wrapper contract on the layout tensors themselves: the block's causal
                # offset plus its query rows fit its K copy (guard row included) times the ratio
                q_rows, k_rows = int(cu_q[b + 1] - cu_q[b]), int(cu_k[b + 1] - cu_k[b])
                assert q_rows == c and int(offs[b]) + q_rows <= k_rows * ratio
            # the K copies and their guard rows fit the static capacity with room to spare
            assert int(cu_k[-2]) < total_k
            tail = src_map[cu_k[-2] :].tolist()
            assert all(r == seq_major_rows for r in tail)


# =============================================================================
# Data pipeline floor
# =============================================================================


def test_packed_sequence_length_floor():
    from megatron.core.datasets.data_schedule_utils import get_packed_sequence_length

    assert get_packed_sequence_length(5, 4, True) == 8
    assert get_packed_sequence_length(5, 4, True, min_len=64) == 64
    assert get_packed_sequence_length(100, 4, True, min_len=64) == 104
    assert get_packed_sequence_length(0, 4, True, min_len=64) == 0
    assert get_packed_sequence_length(5, 1, True, min_len=64) == 5
    lengths = torch.tensor([5, 100, 0], dtype=torch.int32)
    assert get_packed_sequence_length(lengths, 4, True, min_len=64).tolist() == [64, 104, 0]


def _pad_config(**overrides):
    cfg = dict(
        pad_packed_seq_alignment=None,
        max_seqlen_per_dp_cp_rank=None,
        cuda_graph_impl='none',
        thd_max_packed_sequences=None,
        thd_tail_padding_policy=None,
        dsv4_zigzag_min_padded_seq_len=0,
        sequence_parallel=False,
        fp8=None,
        dsa_indexer_precision='bf16',
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _pad_batch(cu, cu_padded):
    total = cu_padded[-1]
    return {
        'cu_seqlens': torch.tensor(cu, dtype=torch.int32),
        'cu_seqlens_padded': torch.tensor(cu_padded, dtype=torch.int32),
        'tokens': torch.arange(total, dtype=torch.int64),
        'labels': torch.arange(total, dtype=torch.int64),
        'loss_mask': torch.ones(total),
        'position_ids': torch.arange(total, dtype=torch.int64),
        'padding_mask': torch.zeros(total, dtype=torch.bool),
    }


def test_pad_packed_batch_applies_floor_and_short_tail_rule():
    from megatron.core.datasets.data_schedule_utils import pad_packed_batch_before_cp_slice

    cp = 2
    # floor 32 rows: a 10-row sequence pads to 32 (not 12); the 86-row tail stays a dummy seq
    batch = _pad_batch([0, 10, 50], [0, 10, 50])
    config = _pad_config(
        pad_packed_seq_alignment='max',
        max_seqlen_per_dp_cp_rank=64,
        dsv4_zigzag_min_padded_seq_len=32,
    )
    pad_packed_batch_before_cp_slice(batch, config, cp, 1)
    assert batch['cu_seqlens_padded'].tolist() == [0, 32, 72, 128]
    assert batch['cu_seqlens'].tolist() == [0, 10, 50, 106]
    assert batch['tokens'][32:72].tolist() == list(range(10, 50))
    # a tail shorter than the floor extends the final sequence instead of becoming a dummy one
    # ([10, 90] -> [32, 92] = 124 rows, tail 4 < 32)
    batch = _pad_batch([0, 10, 100], [0, 10, 100])
    pad_packed_batch_before_cp_slice(batch, config, cp, 1)
    assert batch['cu_seqlens_padded'].tolist() == [0, 32, 128]
    assert batch['cu_seqlens'].tolist() == [0, 10, 100]
    # an empty final sequence stays empty: the last non-empty one absorbs the short tail
    # (floor 16, capacity 2 * 24 rows: [40, 0] leaves an 8-row tail)
    short = _pad_config(
        pad_packed_seq_alignment='max',
        max_seqlen_per_dp_cp_rank=24,
        dsv4_zigzag_min_padded_seq_len=16,
    )
    for policy in (None, 'extend_last'):
        batch = _pad_batch([0, 40, 40], [0, 40, 40])
        short.thd_tail_padding_policy = policy
        pad_packed_batch_before_cp_slice(batch, short, cp, 1)
        assert batch['cu_seqlens_padded'].tolist() == [0, 48, 48]
        assert batch['cu_seqlens'].tolist() == [0, 40, 40]
        geom = sz.make_geometry(_FakeGroup(cp, 0), batch['cu_seqlens_padded'], 24, 4)
        assert bool(geom.validity())
    # without the floor the behaviour is unchanged
    batch = _pad_batch([0, 10, 50], [0, 10, 50])
    pad_packed_batch_before_cp_slice(
        batch, _pad_config(pad_packed_seq_alignment='max', max_seqlen_per_dp_cp_rank=64), cp, 1
    )
    assert batch['cu_seqlens_padded'].tolist() == [0, 12, 52, 128]
    batch = _pad_batch([0, 40, 40], [0, 40, 40])
    pad_packed_batch_before_cp_slice(
        batch,
        _pad_config(
            pad_packed_seq_alignment='max',
            max_seqlen_per_dp_cp_rank=24,
            thd_tail_padding_policy='extend_last',
        ),
        cp,
        1,
    )
    assert batch['cu_seqlens_padded'].tolist() == [0, 40, 48]
    # ... and a pack that needs no tail does not consult the tail policy at all
    batch = _pad_batch([0, 24, 48], [0, 24, 48])
    pad_packed_batch_before_cp_slice(
        batch,
        _pad_config(
            pad_packed_seq_alignment='max',
            max_seqlen_per_dp_cp_rank=24,
            thd_tail_padding_policy='not-a-policy',
        ),
        cp,
        1,
    )
    assert batch['cu_seqlens_padded'].tolist() == [0, 24, 48]


# =============================================================================
# Configuration contract
# =============================================================================


_OMIT = object()  # override value that removes a default keyword of _make_seq_zigzag_config


def _make_seq_zigzag_config(monkeypatch, fused=True, **overrides):
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    if fused is not None:
        monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: fused)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
    kwargs = dict(
        context_parallel_size=4,
        cp_partition_mode="zigzag",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=4096,
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=128,
        csa_dense_mode=False,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="none",
    )
    kwargs.update(overrides)
    return _make_config(**{k: v for k, v in kwargs.items() if v is not _OMIT})


@pytest.mark.internal
def test_seq_zigzag_config_accepts_dsv4_zigzag_cp(monkeypatch):
    config = _make_seq_zigzag_config(monkeypatch)
    assert config.attention_cp_layout == "zigzag" and config.cp_partition_mode == "zigzag"
    assert config.dsv4_zigzag_min_padded_seq_len == 2 * 4 * 128
    contiguous = _make_seq_zigzag_config(monkeypatch, cp_partition_mode="contiguous")
    assert contiguous.dsv4_zigzag_min_padded_seq_len == 0
    cp1 = _make_seq_zigzag_config(
        monkeypatch,
        context_parallel_size=1,
        sequence_packing_scheduler=None,
        max_seqlen_per_dp_cp_rank=None,
    )
    assert cp1.dsv4_zigzag_min_padded_seq_len == 0
    # zigzag is the default layout: omitting the layout flags selects it
    default = _make_seq_zigzag_config(monkeypatch, cp_partition_mode=_OMIT)
    assert default.attention_cp_layout == "zigzag"
    assert default.dsv4_zigzag_min_padded_seq_len == 2 * 4 * 128
    # an odd per-rank limit is fine while packs are not padded to it (eager, integer alignment)
    odd = _make_seq_zigzag_config(
        monkeypatch, max_seqlen_per_dp_cp_rank=4095, pad_packed_seq_alignment=64
    )
    assert odd.max_seqlen_per_dp_cp_rank == 4095
    # the fused-indexer row limit only binds when a ratio-4 layer builds an indexer
    for no_indexer in ({"csa_dense_mode": True}, {"csa_compress_ratios": [0, 128, 128, 0]}):
        config = _make_seq_zigzag_config(monkeypatch, max_seqlen_per_dp_cp_rank=65536, **no_indexer)
        assert config.max_seqlen_per_dp_cp_rank == 65536


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dsa_cp_balance_indexer": True}, "disable the flag"),
        ({"cuda_graph_impl": "full_iteration"}, "full-iteration CUDA graphs"),
        (
            {"cuda_graph_impl": "none", "cuda_graph_modules": "full_iteration"},
            "full-iteration CUDA graphs",
        ),
        (
            {"cuda_graph_impl": "none", "cuda_graph_scope": "full_iteration"},
            "full-iteration CUDA graphs",
        ),
        (
            {"max_seqlen_per_dp_cp_rank": 4095, "pad_packed_seq_alignment": "max"},
            "needs an even max_seqlen_per_dp_cp_rank",
        ),
        (
            {
                "max_seqlen_per_dp_cp_rank": 4095,
                "pad_packed_seq_alignment": 64,
                "cuda_graph_impl": "local",
                "thd_max_packed_sequences": 8,
            },
            "needs an even max_seqlen_per_dp_cp_rank",
        ),
        ({"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss only"),
        ({"max_seqlen_per_dp_cp_rank": 65536}, "verified-safe limit"),
        ({"max_seqlen_per_dp_cp_rank": 128}, "above the pack capacity"),
        ({"csa_window_size": 0, "csa_compress_ratios": [0, 0, 0, 0]}, "needs a boundary window"),
        # Only the dp_balanced packing path applies the per-sequence floor: the other THD
        # frontends (no scheduler: SFT / inter-document masking / RL packing) and the
        # packing-aware scheduler in static-CP mode are rejected.
        ({"sequence_packing_scheduler": None}, "requires sequence_packing_scheduler='dp_balanced'"),
        (
            {"sequence_packing_scheduler": "default_dynamic_cp"},
            "requires sequence_packing_scheduler='dp_balanced'",
        ),
        (
            {
                "context_parallel_size": 1,
                "dynamic_context_parallel": True,
                "sequence_packing_scheduler": "default_dynamic_cp",
                "calculate_per_token_loss": True,
            },
            "DSv4 Hybrid with context parallelism requires cp_partition_mode='contiguous'",
        ),
    ],
    ids=[
        "balanced_indexer",
        "full_iteration",
        "full_iteration_deprecated_modules",
        "full_iteration_deprecated_scope",
        "odd_fixed_capacity",
        "odd_capacity_under_cuda_graphs",
        "dense_loss",
        "fused_row_limit",
        "floor_above_capacity",
        "no_boundary_window",
        "no_scheduler",
        "default_dynamic_cp_static",
        "dynamic_cp",
    ],
)
def test_seq_zigzag_config_rejects_unsupported_combinations(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_seq_zigzag_config(monkeypatch, **overrides)


@pytest.mark.internal
def test_explicit_row_rope_positions_and_fused_helper(monkeypatch):
    """Under this layout the RoPE helpers get one pack row per local (and halo) row. Rows of
    an empty sequence's neighbours, before the pack start (rank 0's chunk-0 halo slots) and
    beyond its end map to position 0; the fused helper hands the kernel these positions and
    no interval start."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    cu = torch.tensor([0, 6, 6, 16], dtype=torch.int32)
    rows = torch.tensor([14, 15, 0, 1, 5, 6, 7, -3, -1, 16, 17], dtype=torch.int32)
    expected = torch.tensor([8, 9, 0, 1, 5, 0, 1, 0, 0, 0, 0], dtype=torch.int32)
    assert torch.equal(cp_utils.thd_cp_position_ids_from_rows(cu, rows), expected)
    interval = torch.arange(2, 13, dtype=torch.int32)
    assert torch.equal(
        cp_utils.thd_cp_position_ids_from_rows(cu, interval),
        cp_utils.thd_cp_position_ids(cu, 2, 11),
    )

    calls = []

    def fake_fused(rope_input, *_args, **kwargs):
        calls.append(kwargs)
        return rope_input

    monkeypatch.setattr(cp_utils, "fused_mla_rope_inplace", fake_fused)
    x = torch.randn(rows.numel(), 2, 8)
    output = cp_utils.apply_thd_cp_local_rope_fused(
        x, torch.empty(0), torch.empty(0), 4, 4, cu, 3, global_rows=rows
    )
    assert output is x and len(calls) == 1
    assert calls[0]["cu_seqlens_q"] is cu
    assert "thd_global_start" not in calls[0]
    assert torch.equal(calls[0]["position_ids"], expected)


@pytest.mark.internal
@pytest.mark.parametrize("inverse", [False, True])
def test_unfused_rope_with_explicit_rows_matches_the_interval_form(inverse):
    """The unfused helper with explicit rows equals the interval form applied to the whole
    pack, restricted to those rows (rows in zigzag order: head and tail chunks)."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    torch.manual_seed(31)
    cu = torch.tensor([0, 3, 9, 9, 14], dtype=torch.int32)
    freqs = torch.randn(8, 1, 1, 4)
    config = SimpleNamespace(
        apply_rope_fusion=False, rotary_interleaved=False, multi_latent_attention=False
    )
    pack = torch.randn(14, 2, 8)
    whole = cp_utils.apply_thd_cp_local_rope_unfused(pack, freqs, 4, 4, cu, 0, config, inverse)
    rows = torch.tensor([0, 2, 3, 4, 8, 9, 13, 11], dtype=torch.int32)
    local = cp_utils.apply_thd_cp_local_rope_unfused(
        pack.index_select(0, rows.long()), freqs, 4, 4, cu, 5, config, inverse, global_rows=rows
    )
    torch.testing.assert_close(local, whole.index_select(0, rows.long()))


@pytest.mark.internal
def test_compressor_boundary_rows_take_the_tail_of_every_halo_slot():
    """The compressors read the ``d_comp`` rows before each block: the tail of its halo slot."""
    geom = SimpleNamespace(d_window=4, n_blocks=3, halo_rows=12)
    boundary = torch.arange(24, dtype=torch.float32).view(12, 1, 2)
    tails = sz.compressor_boundary_rows(boundary, geom, 3)
    assert torch.equal(tails, torch.cat([boundary[b * 4 + 1 : (b + 1) * 4] for b in range(3)]))
    assert torch.equal(sz.compressor_boundary_rows(boundary, geom, 4), boundary)
    assert sz.compressor_boundary_rows(boundary, geom, 0).shape == (0, 1, 2)
    with pytest.raises(RuntimeError, match="expects 12 rows"):
        sz.compressor_boundary_rows(boundary[:8], geom, 3)
    with pytest.raises(RuntimeError, match="exceeds the boundary window"):
        sz.compressor_boundary_rows(boundary, geom, 5)


@pytest.mark.internal
def test_csa_takes_the_zigzag_split_only_together_with_its_geometry():
    """CompressedSparseAttention runs a zigzag-tagged pack only with the geometry that
    DSv4HybridAttention builds for it, and a geometry only next to the zigzag tag; without
    either guard zigzag rows would go through the contiguous arithmetic."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant import csa

    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    reached = []
    entry = SimpleNamespace(
        n_local_heads=2,
        pg_collection=SimpleNamespace(cp=group),
        _forward_thd_cp=lambda *args, **kwargs: reached.append(kwargs) or torch.zeros(8, 1, 2, 4),
    )

    def forward(tag, **kwargs):
        packed = PackedSeqParams(qkv_format="thd", cp_partition_mode=tag)
        return csa.CompressedSparseAttention.forward(
            entry, torch.zeros(8, 2, 4), None, None, None, packed_seq_params=packed, **kwargs
        )

    with pytest.raises(ValueError, match="requires cp_partition_mode='contiguous'"):
        forward("zigzag")
    assert not reached
    forward("contiguous")
    geometry = object()
    forward("zigzag", seq_zigzag_geom=geometry)
    assert [call["seq_zigzag_geom"] for call in reached] == [None, geometry]

    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    geom = sz.make_geometry(group, cu, 16, 4)
    core = SimpleNamespace(pg_collection=SimpleNamespace(cp=group), use_fused_kernels=False)

    def forward_thd_cp(tag):
        packed = PackedSeqParams(
            qkv_format="thd",
            cp_partition_mode=tag,
            cu_seqlens_q=cu,
            cu_seqlens_q_padded=cu,
            max_seqlen_q=16,
        )
        return csa.CompressedSparseAttention._forward_thd_cp(
            core,
            torch.zeros(16, 2, 4),
            torch.zeros(16, 1, 4),
            torch.zeros(16, 1, 4),
            None,
            None,
            None,
            packed,
            seq_zigzag_geom=geom,
        )

    with pytest.raises(RuntimeError, match="geometry does not match this CSA call"):
        forward_thd_cp("contiguous")
    # With the matching tag the call gets past the geometry check (this stub has no fused kernels).
    with pytest.raises(NotImplementedError, match="requires the fused CSA kernels"):
        forward_thd_cp("zigzag")


@pytest.mark.internal
def test_layer_ratio_must_fit_the_padding_floor(monkeypatch):
    """The padding floor of the data pipeline comes from ``csa_compress_ratios``; a layer built
    with a ratio that the list does not contain and that needs a larger boundary window (an H
    layer of a hybrid pattern with a narrow window) fails at construction, not on device."""
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _build_cpu_attention_for_ratio_resolution,
    )

    sz.validate_layer_padding_floor(1024, 4, 64, 128)
    sz.validate_layer_padding_floor(512, 4, 64, 4)
    sz.validate_layer_padding_floor(512, 4, 64, 0)
    with pytest.raises(ValueError, match="List every compress ratio"):
        sz.validate_layer_padding_floor(512, 4, 64, 128)

    overrides = dict(
        csa_window_size=64, csa_compress_ratios=[4, 4, 4, 4], perform_initialization=False
    )
    zigzag = _make_seq_zigzag_config(monkeypatch, **overrides)
    assert zigzag.dsv4_zigzag_min_padded_seq_len == 2 * 4 * 64
    attention = _build_cpu_attention_for_ratio_resolution(monkeypatch, zigzag, explicit_ratio=4)
    assert attention._dsv4_compress_ratio == 4
    with pytest.raises(ValueError, match="List every compress ratio"):
        _build_cpu_attention_for_ratio_resolution(monkeypatch, zigzag, explicit_ratio=128)
    # No floor outside the zigzag layout: the same layer builds.
    contiguous = _make_seq_zigzag_config(monkeypatch, cp_partition_mode="contiguous", **overrides)
    assert contiguous.dsv4_zigzag_min_padded_seq_len == 0
    attention = _build_cpu_attention_for_ratio_resolution(
        monkeypatch, contiguous, explicit_ratio=128
    )
    assert attention._dsv4_compress_ratio == 128


@pytest.mark.internal
def test_hybrid_layer_pattern_is_limited_to_dsv4_attention_mlp_and_moe():
    """Only the DSv4 attention layers implement the layout; MLP and MoE layers act on single
    tokens. Every other layer type (Mamba, GDN, KDA, standard / DSA / MLA attention) is
    rejected, in the main pattern and in the MTP pattern."""
    sz.validate_hybrid_layer_pattern("WCH-E|WE-CH", "CE")
    sz.validate_hybrid_layer_pattern("W-C-H-", None)
    sz.validate_hybrid_layer_pattern(None, None)
    for main, mtp in (
        ("WM-E", None),
        ("CG", None),
        ("HK", None),
        ("W*-", None),
        ("WD", None),
        ("W+", None),
        ("WCH-E", "M-"),
    ):
        with pytest.raises(ValueError, match="supports hybrid layer patterns made of DSv4"):
            sz.validate_hybrid_layer_pattern(main, mtp)


@pytest.mark.internal
def test_hybrid_model_checks_the_layer_pattern_under_dsv4_zigzag(monkeypatch):
    """HybridModel applies the pattern rule exactly when the model is configured for the
    per-sequence zigzag layout (non-zero padding floor); the rule runs right after the pattern
    is parsed, before any layer is built."""
    from megatron.core.models.hybrid import hybrid_model

    seen = []
    monkeypatch.setattr(
        sz, "validate_hybrid_layer_pattern", lambda main, mtp: seen.append((main, mtp))
    )

    class _Stop(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise _Stop

    # The first statement after the pattern is parsed: stop there, no layer is built.
    monkeypatch.setattr(hybrid_model, "_get_hash_moe_layer_threshold", stop)
    # LanguageModule pins process-wide Transformer Engine backend variables; not under test.
    monkeypatch.setattr(
        "megatron.core.models.common.language_module.language_module.set_attention_backend",
        lambda _config: None,
    )
    process_groups = SimpleNamespace(cp=None, tp=object(), pp=None, embd=None)

    def build(config, pattern):
        with pytest.raises(_Stop):
            hybrid_model.HybridModel(
                config=config,
                hybrid_stack_spec=None,
                vocab_size=128,
                max_sequence_length=4096,
                hybrid_layer_pattern=pattern,
                pg_collection=process_groups,
            )

    zigzag = _make_seq_zigzag_config(monkeypatch)
    assert zigzag.dsv4_zigzag_min_padded_seq_len > 0
    build(zigzag, "WM-E/CE")
    assert seen == [("WM-E", "CE")]
    contiguous = _make_seq_zigzag_config(monkeypatch, cp_partition_mode="contiguous")
    assert contiguous.dsv4_zigzag_min_padded_seq_len == 0
    build(contiguous, "WM-E/CE")
    assert seen == [("WM-E", "CE")]


@pytest.mark.internal
def test_seq_zigzag_config_requires_fused_dsa_backend(monkeypatch):
    with pytest.raises(ValueError, match="requires the fused DSA kernel backend"):
        _make_seq_zigzag_config(monkeypatch, fused=False)


@pytest.mark.internal
def test_seq_zigzag_config_warns_about_a_large_halo_slot_region(monkeypatch):
    from megatron.core.transformer import transformer_config as transformer_config_module

    messages = []
    monkeypatch.setattr(
        transformer_config_module,
        "log_single_rank",
        lambda logger, level, msg, *args, **kwargs: messages.append(str(msg)),
    )
    monkeypatch.setattr(transformer_config_module, "_DSV4_ZIGZAG_HALO_WARNED", set())
    # 2 * 64 * 128 = 16384 halo slot rows per rank against 4096 local rows: flagged, still legal
    config = _make_seq_zigzag_config(monkeypatch, thd_max_packed_sequences=64)
    assert config.thd_max_packed_sequences == 64
    assert sum("halo rows per rank" in m for m in messages) == 1
    # the training entry points rebuild the config for every microbatch: reported once
    _make_seq_zigzag_config(monkeypatch, thd_max_packed_sequences=64)
    assert sum("halo rows per rank" in m for m in messages) == 1
    messages.clear()
    _make_seq_zigzag_config(monkeypatch, thd_max_packed_sequences=4)  # 1024 rows: a quarter
    assert not any("halo rows per rank" in m for m in messages)


@pytest.mark.internal
def test_thd_cuda_graph_capture_bound_counts_the_per_sequence_floor():
    # The dp_balanced capture plan bounds the packs of a global batch by assuming every sample
    # has the longest length; with the floor no sequence occupies less than the floor.
    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

    def bound(floor):
        helper = SimpleNamespace(
            dp_group=SimpleNamespace(size=lambda: 1),
            dp_cp_group=SimpleNamespace(size=lambda: 16),
            micro_batch_size=1,
            seq_length=32768,
            thd_sequence_length_upper_bound=1000,  # every sample is shorter than the floor
            config=SimpleNamespace(
                sequence_packing_scheduler="dp_balanced",
                max_seqlen_per_dp_cp_rank=2048,
                thd_max_packed_sequences=None,
                virtual_pipeline_model_parallel_size=None,
                dsv4_zigzag_min_padded_seq_len=floor,
            ),
            _get_dp_balanced_thd_max_num_microbatches=(
                TECudaGraphHelper._get_dp_balanced_thd_max_num_microbatches
            ),
        )
        return TECudaGraphHelper._get_thd_varlen_max_num_microbatches(helper, 64, None)

    # capacity 2048 * 16 = 32768 rows: 32 samples of 1000 rows per pack, 8 at the 4096-row floor
    assert bound(0) == (2, "thd_varlen_upper_bound")
    assert bound(4096) == (8, "thd_varlen_upper_bound")
    # ... which is what the scheduler really produces for that global batch
    from megatron.core.datasets.data_schedule import DpBalancedScheduler

    scheduler = DpBalancedScheduler(2048, 16, 1, None, pad_sequences=True, min_seq_len=4096)
    packs = scheduler.get_groups_and_subsamples(list(enumerate([1000] * 64)))
    assert bound(0)[0] < len(packs) <= bound(4096)[0]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cudnn DSA backend needs CUDA")
def test_seq_zigzag_config_sees_the_folded_legacy_fusion_flag(monkeypatch):
    # apply_dsa_kernel_fusion is folded into dsa_kernel_backend late in __post_init__; the
    # zigzag validation must judge the folded backend (real predicate, no monkeypatch).
    config = _make_seq_zigzag_config(monkeypatch, fused=None, apply_dsa_kernel_fusion=True)
    assert config.dsa_kernel_backend == "cudnn"
    with pytest.raises(ValueError, match="requires the fused DSA kernel backend"):
        _make_seq_zigzag_config(monkeypatch, fused=None, apply_dsa_kernel_fusion=False)


# =============================================================================
# GPU: block-table layout kernels against the host reference
# =============================================================================


def _cute_available():
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        thd_layout_kernels,
    )

    return torch.cuda.is_available() and thd_layout_kernels._CUTE_AVAILABLE


@pytest.mark.skipif(not _cute_available(), reason="CUDA + CuTeDSL required")
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("cp", [2, 4])
def test_block_compressor_layout_kernel_matches_reference(cp, ratio):
    rng = random.Random(3 * cp + ratio)
    d_comp = sz.compressor_d_comp(ratio)
    for _ in range(4):
        cu = _random_cu(
            rng,
            cp,
            rng.randint(1, 3),
            unit=(d_comp if ratio == 4 else 2 * d_comp) // (2 * cp) + 1,
            max_units=5,
        )
        l_local = cu[-1] // cp
        n_seq = len(cu) - 1
        c_cap = sz.compact_group_capacity_blocks(l_local, n_seq, ratio)
        cu_t = torch.tensor(cu, dtype=torch.int32, device='cuda')
        s2r_ref = sz.ref_seq_to_rank_row(cu, cp, ratio, c_cap)
        for rank in range(cp):
            geom = sz.make_geometry(_FakeGroup(cp, rank), cu_t, l_local, d_window=max(d_comp, 2))
            layout = sz.build_cp_compressor_layout_seq_zigzag(cu_t, geom, ratio, c_cap)
            torch.cuda.synchronize()
            c2s, s2c, gids, pids, lcu, lcuc, gcuc = sz.ref_compressor_layout(
                cu, cp, rank, ratio, c_cap
            )
            assert layout.compact_to_source.tolist() == c2s
            assert layout.source_to_compact.tolist() == s2c
            assert layout.group_ids.tolist() == gids
            assert layout.position_ids.tolist() == pids
            assert layout.local_cu_seqlens.tolist() == lcu
            assert layout.local_cu_seqlens_compressed.tolist() == lcuc
            assert layout.cu_seqlens_compressed.tolist() == gcuc
            assert layout.seq_to_rank_row.tolist() == s2r_ref
            assert layout.boundary_rows == 2 * n_seq * d_comp and layout.global_start == -1
            # the fallback compaction gathers through the maps
            hidden = torch.randn(l_local, 1, 8, device='cuda')
            boundary = torch.randn(2 * n_seq * d_comp, 1, 8, device='cuda')
            compact = sz.prepare_cp_compressor_input_seq_zigzag(hidden, boundary, layout)
            source = torch.cat((boundary, hidden))
            for t, src in enumerate(c2s):
                if src >= 0:
                    assert torch.equal(compact[t], source[src])
                else:
                    assert torch.count_nonzero(compact[t]) == 0


def _ref_attention_indices(
    cu, cp, rank, d_w, window, ratio, comp_topk, s2r, compressed_base, compressed_rows
):
    """Python reference of the block-mode index kernel (index_mode 0 / 1)."""
    blocks = sz.ref_blocks(cu, cp, rank)
    n_blocks = len(blocks)
    cuc = [0]
    for i in range(len(cu) - 1):
        cuc.append(cuc[-1] + (cu[i + 1] - cu[i]) // ratio if ratio > 1 else cuc[-1])
    rows_out, lens_out = [], []
    for b, (i, k, c, loff, g0) in enumerate(blocks):
        for j in range(loff, loff + c):
            G = g0 + (j - loff)
            start = max(cu[i], G - window + 1)
            idx = []
            for pos in range(start, G + 1):
                if pos < g0:
                    idx.append(b * d_w + pos - (g0 - d_w))
                else:
                    idx.append(n_blocks * d_w + loff + pos - g0)
            if ratio > 1:
                if comp_topk is None:
                    count = min((G - cu[i] + 1) // ratio, cuc[i + 1] - cuc[i])
                    ids = range(count)
                else:
                    ids = [x for x in comp_topk[j] if 0 <= x < cuc[i + 1] - cuc[i]]
                for cid in ids:
                    phys = s2r[cuc[i] + cid]
                    if 0 <= phys < compressed_rows:
                        idx.append(compressed_base + phys)
            rows_out.append(idx)
            lens_out.append(len(idx))
    return rows_out, lens_out


@pytest.mark.skipif(not _cute_available(), reason="CUDA + CuTeDSL required")
@pytest.mark.parametrize("mode", ["window_only", "all_compressed", "topk"])
@pytest.mark.parametrize("cp", [2, 4])
def test_block_attention_indices_kernel_matches_reference(cp, mode):
    rng = random.Random(cp + len(mode))
    gen = torch.Generator().manual_seed(cp + len(mode))
    ratio = 1 if mode == "window_only" else 4
    window = 6
    d_w = 8
    for _ in range(3):
        cu = _random_cu(rng, cp, rng.randint(1, 3), unit=d_w // (2 * cp) + 1, max_units=4)
        l_local = cu[-1] // cp
        n_seq = len(cu) - 1
        cu_t = torch.tensor(cu, dtype=torch.int32, device='cuda')
        for rank in range(cp):
            geom = sz.make_geometry(_FakeGroup(cp, rank), cu_t, l_local, d_w)
            if ratio > 1:
                c_cap = sz.compact_group_capacity_blocks(l_local, n_seq, ratio)
                layout = sz.build_cp_compressor_layout_seq_zigzag(cu_t, geom, ratio, c_cap)
                s2r = layout.seq_to_rank_row
                cuc = layout.cu_seqlens_compressed
                compressed_rows = c_cap * cp
            else:
                s2r = cuc = None
                compressed_rows = None
            if mode == "topk":
                width = 3
                comp_topk = torch.randint(
                    -1, 6, (l_local, width), dtype=torch.int32, generator=gen
                ).cuda()
            else:
                comp_topk = None
                width = (cu[-1] // ratio) if ratio > 1 else 0
            out = sz.build_attention_indices_seq_zigzag(
                cu_t,
                geom,
                window,
                ratio,
                width,
                comp_topk,
                cu_seqlens_compressed=cuc,
                seq_to_rank_row=s2r,
                compressed_rows=compressed_rows,
                output_alignment=1,
            )
            topk_idxs, topk_length, _, _ = out
            torch.cuda.synchronize()
            ref_rows, ref_lens = _ref_attention_indices(
                cu,
                cp,
                rank,
                d_w,
                window,
                ratio,
                None if comp_topk is None else comp_topk.tolist(),
                None if s2r is None else s2r.tolist(),
                geom.compressed_base,
                compressed_rows or 0,
            )
            assert topk_length.tolist() == ref_lens
            got = topk_idxs.tolist()
            for j, (row, n) in enumerate(zip(ref_rows, ref_lens)):
                assert got[j][:n] == row, (rank, j)
                if mode != "topk":
                    assert all(v == -1 for v in got[j][n:]) or n == len(got[j])


@pytest.mark.skipif(not _cute_available(), reason="CUDA + CuTeDSL required")
@pytest.mark.parametrize("cp", [2, 3, 4])
def test_block_attention_indices_kernel_matches_full_sequence_semantics(cp):
    """Block-mode index kernel in its three index modes, checked through the identity of every
    physical row of ``kv_full_thd = [halo slots | local rows | rank-major compressed rows]``
    rather than through the kernel's own offset arithmetic.

    A halo row is a pack token (the row ``d_window - m`` before its block), a zero row (chunk 0)
    or unused (empty sequence); a local row is its pack token; a compressed row is group ``g`` of
    sequence ``i`` held by its owner chunk. Every query row must list the previous ``window``
    tokens of its own sequence and the visible (mode 1), selected (mode 0) or selected-first
    (mode 2, the training path of the ratio-4 layers, with the indexer's physical rows and the
    query padding mask) compressed groups of that sequence.
    """
    rng = random.Random(4242 + cp)
    gen = torch.Generator().manual_seed(4242 + cp)
    try:
        import transformer_engine_torch as tex
    except ImportError:
        tex = None
    topk_width = 5
    # (ratio, d_window, window): window == d_window (the production shape) and smaller windows
    for ratio, d_w, window in [(0, 8, 8), (0, 16, 3), (4, 8, 6), (4, 8, 8), (16, 16, 16)]:
        shapes = [
            [d_w],  # one sequence, half-chunk exactly d_window
            [d_w, d_w + 1, 2 * d_w + 3],
            [d_w + 2, d_w, 0, 0],  # trailing empty slots
            [d_w + 3, 0, d_w, 0, 2 * d_w],  # middle empty slots
            [0, d_w + 1, d_w + 4],  # leading empty slot
            [rng.choice((0, d_w, d_w + rng.randint(1, 9))) for _ in range(4)] + [d_w],
        ]
        for half_chunks in shapes:
            lens = [2 * cp * c for c in half_chunks]
            real = [rng.randint(1, length) if length else 0 for length in lens]
            align = rng.choice((1, 8))
            tag = f"cp={cp} ratio={ratio} d_w={d_w} window={window} lens={lens} real={real}"
            cu, cu_real = [0], [0]
            for length, logical in zip(lens, real):
                cu.append(cu[-1] + length)
                cu_real.append(cu_real[-1] + logical)
            n_seq, total = len(lens), cu[-1]
            l_local, n_blocks = total // cp, 2 * n_seq
            groups = [(length // ratio if ratio > 1 else 0) for length in lens]
            cu_t = torch.tensor(cu, dtype=torch.int32, device='cuda')
            cu_real_t = torch.tensor(cu_real, dtype=torch.int32, device='cuda')
            geoms = [sz.make_geometry(_FakeGroup(cp, r), cu_t, l_local, d_w) for r in range(cp)]
            compressed, compressed_rows, s2r_t, cuc_t = [], None, None, None
            if ratio > 1:
                c_cap = sz.compact_group_capacity_blocks(l_local, n_seq, ratio)
                compressed_rows = cp * c_cap
                for r in range(cp):
                    layout = sz.build_cp_compressor_layout_seq_zigzag(cu_t, geoms[r], ratio, c_cap)
                    if r == 0:
                        s2r_t, cuc_t = layout.seq_to_rank_row, layout.cu_seqlens_compressed
                    assert layout.seq_to_rank_row.tolist() == s2r_t.tolist(), tag
                    group_ids = layout.group_ids.tolist()
                    local_cuc = layout.local_cu_seqlens_compressed.tolist()
                    blocks = _zigzag_blocks(cu, cp, r)
                    for slot in range(c_cap):
                        block = [
                            b for b in range(n_blocks) if local_cuc[b] <= slot < local_cuc[b + 1]
                        ]
                        if not block or group_ids[slot] < 0:
                            compressed.append(("pad",))
                            continue
                        i, k, c, _, _ = blocks[block[0]]
                        g = group_ids[slot]
                        owner = min(((g + 1) * ratio - 1) // c, 2 * cp - 1)
                        compressed.append(("group", i, g, owner == k))
            width_all = (max(lens) // ratio) if ratio > 1 else 0
            for r in range(cp):
                geom, blocks = geoms[r], _zigzag_blocks(cu, cp, r)
                pack_rows = [g0 + j for (_, _, c, _, g0) in blocks for j in range(c)]
                assert geom.global_rows().tolist() == pack_rows, tag
                if tex is not None:
                    assert tex.thd_get_partitioned_indices(cu_t, total, cp, r).tolist() == pack_rows
                halo = []
                for i, k, c, _, g0 in blocks:
                    for m in range(d_w):
                        halo.append(
                            ("unused",)
                            if c == 0
                            else ("zero",) if k == 0 else ("token", g0 - d_w + m)
                        )
                rows = halo + [("token", g) for g in pack_rows] + compressed
                base = n_blocks * d_w + l_local
                assert geom.compressed_base == base and geom.halo_rows == n_blocks * d_w

                def identity(index, rows=rows):
                    return rows[index] if 0 <= index < len(rows) else ("out_of_range", index)

                def window_of(j, pack_rows=pack_rows):
                    row = pack_rows[j]
                    i = max(s for s in range(n_seq) if cu[s] <= row and cu[s + 1] > cu[s])
                    return (
                        i,
                        row - cu[i],
                        [("token", x) for x in range(max(cu[i], row - window + 1), row + 1)],
                    )

                kwargs = dict(
                    cu_seqlens_compressed=cuc_t,
                    seq_to_rank_row=s2r_t,
                    compressed_rows=compressed_rows,
                    output_alignment=align,
                )
                # index mode 1: the window, then every visible compressed group
                idx, length, _, mask = sz.build_attention_indices_seq_zigzag(
                    cu_t, geom, window, ratio, width_all, None, **kwargs
                )
                assert mask is None
                assert idx.shape[1] == (window + width_all + align - 1) // align * align
                idx, length = idx.tolist(), length.tolist()
                for j in range(l_local):
                    i, pos, expected = window_of(j)
                    visible = min((pos + 1) // ratio, groups[i]) if ratio > 1 else 0
                    expected = expected + [("group", i, g, True) for g in range(visible)]
                    assert [identity(v) for v in idx[j][: length[j]]] == expected, (tag, r, j)
                    assert all(v == -1 for v in idx[j][length[j] :]), (tag, r, j)
                if ratio <= 1:
                    continue
                topk = torch.randint(
                    -2, max(groups) + 3, (l_local, topk_width), dtype=torch.int32, generator=gen
                ).cuda()
                topk_ids = topk.tolist()
                # index mode 0 (inference): the window, then the selected compressed groups
                idx, length, _, _ = sz.build_attention_indices_seq_zigzag(
                    cu_t, geom, window, ratio, topk_width, topk, **kwargs
                )
                idx, length = idx.tolist(), length.tolist()
                for j in range(l_local):
                    i, _, expected = window_of(j)
                    selected = [x for x in topk_ids[j] if 0 <= x < groups[i]]
                    expected = expected + [("group", i, g, True) for g in selected]
                    assert [identity(v) for v in idx[j][: length[j]]] == expected, (tag, r, j)
                    assert all(v == -1 for v in idx[j][length[j] :]), (tag, r, j)
                # index mode 2 (training with the sparse indexer loss): selected groups in their
                # top-k columns, then the window; physical indexer rows and the q padding mask
                for with_mask in (True, False):
                    idx, length, physical, mask = sz.build_attention_indices_seq_zigzag(
                        cu_t,
                        geom,
                        window,
                        ratio,
                        topk_width,
                        topk,
                        for_indexer_loss=True,
                        cu_seqlens_unpadded=cu_real_t if with_mask else None,
                        **kwargs,
                    )
                    assert length is None and (mask is not None) == with_mask
                    idx, physical = idx.tolist(), physical.tolist()
                    mask = mask.tolist() if with_mask else None
                    for j in range(l_local):
                        i, pos, expected = window_of(j)
                        for col in range(topk_width):
                            group = topk_ids[j][col]
                            if 0 <= group < groups[i]:
                                assert identity(idx[j][col]) == ("group", i, group, True), (
                                    tag,
                                    r,
                                    j,
                                )
                                assert physical[j][col] == idx[j][col] - base, (tag, r, j)
                            else:
                                assert idx[j][col] == -1 and physical[j][col] == -1, (tag, r, j)
                        window_cols = idx[j][topk_width : topk_width + len(expected)]
                        assert [identity(v) for v in window_cols] == expected, (tag, r, j)
                        assert all(v == -1 for v in idx[j][topk_width + len(expected) :]), (
                            tag,
                            r,
                            j,
                        )
                        if with_mask:
                            assert bool(mask[j]) == (pos >= real[i]), (tag, r, j)
