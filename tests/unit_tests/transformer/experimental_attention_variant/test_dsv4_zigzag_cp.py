# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSv4 hybrid THD context parallelism on the zigzag partition (``dsv4_cp_zigzag``).

Under zigzag CP, rank ``r`` holds chunks ``r`` (segment A) and ``2 * CP - 1 - r`` (segment B)
of a single packed sequence. The CPU tests check the geometry (segments, positions, halos),
the merged compressor layout and its zigzag compressed-row map, the two-segment attention
indices, the halo exchange (gloo) and the config guards, and run a pure-PyTorch emulation of
one H-layer core (RoPE, window, compressor, all-gather, sparse attention, inverse RoPE) with
the CP split emulated on CPU: zigzag and contiguous outputs and gradients must match a
full-sequence reference row by row. The CuTe kernels are replaced on CPU by the host oracles
in ``csa_utils/packed_layout.py``.

The CUDA tests (skipped without a GPU) compare the CuTe kernels with the oracles on one GPU
and, under ``torchrun --nproc-per-node 2|4``, run ``DSv4HybridSelfAttention`` with zigzag CP
against the contiguous partition and the CP1 reference.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.csa import (
    _pool_compressor_values,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    packed_layout,
    thd_layout_kernels,
)

_ROPE_CONFIG = SimpleNamespace(
    apply_rope_fusion=False, rotary_interleaved=False, multi_latent_attention=False
)


# =============================================================================
# Shared helpers
# =============================================================================


def _cu(total, device="cpu"):
    return torch.tensor([0, total], dtype=torch.int32, device=device)


def _zigzag_rows(cp_rank, cp_size, segment_rows, device="cpu"):
    """Global rows held by a zigzag CP rank, in local order [A | B]."""
    _, (start_a, start_b) = cp_utils.thd_cp_zigzag_segments(2 * segment_rows, cp_rank, cp_size)
    return torch.cat(
        (
            torch.arange(start_a, start_a + segment_rows, device=device),
            torch.arange(start_b, start_b + segment_rows, device=device),
        )
    )


def _halo_rows(starts, d_window, device="cpu"):
    """Global rows of the left windows of the given segment starts (negative: no row)."""
    return torch.cat([torch.arange(start - d_window, start, device=device) for start in starts])


def _take_rows(values, rows):
    """values[rows] with zero rows for negative row ids; differentiable."""
    padded = torch.cat((values.new_zeros((1,) + tuple(values.shape[1:])), values))
    return padded.index_select(0, torch.where(rows >= 0, rows + 1, 0).long())


@pytest.fixture
def cpu_layout_kernels(monkeypatch):
    """Run the CuTe layout launches through their host oracles on CPU."""
    monkeypatch.setattr(thd_layout_kernels, "_require_cute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        thd_layout_kernels, "_compressor_row_maps", packed_layout.compressor_row_maps_reference
    )
    monkeypatch.setattr(
        thd_layout_kernels, "_compressor_rank_rows", packed_layout.compressor_rank_rows_reference
    )


def _expected_keys(position, window_size, ratio):
    """Logical keys of one query of a single sequence: window rows, then compressed blocks."""
    keys = [("row", row) for row in range(max(0, position - window_size + 1), position + 1)]
    if ratio > 1:
        keys += [("block", block) for block in range((position + 1) // ratio)]
    return keys


def _compressed_block_owner_map(seq_to_rank_row):
    physical_to_block = {}
    for block, physical in enumerate(seq_to_rank_row.tolist()):
        if physical >= 0:
            assert physical not in physical_to_block, "two blocks map to one physical row"
            physical_to_block[physical] = block
    return physical_to_block


# =============================================================================
# Geometry
# =============================================================================


@pytest.mark.parametrize("cp_size", [2, 4, 8, 64])
@pytest.mark.parametrize("segment_rows", [128, 256])
def test_zigzag_segments_partition_rows_like_mcore_zigzag(cp_size, segment_rows):
    """Every row is held once; the order matches MCore's per-sequence zigzag partition."""
    total = 2 * cp_size * segment_rows
    seen = torch.zeros(total, dtype=torch.int64)
    for cp_rank in range(cp_size):
        rows = _zigzag_rows(cp_rank, cp_size, segment_rows)
        seen[rows] += 1
        spans = dsa_layout._host_packed_cp_spans([0, total], cp_size, cp_rank)
        assert spans == [
            (cp_rank * segment_rows, segment_rows),
            ((2 * cp_size - 1 - cp_rank) * segment_rows, segment_rows),
        ]
        mcore_rows = dsa_layout.build_packed_allgather_cp_local_positions_from_host(
            [0, total], cp_size, cp_rank, torch.device("cpu")
        )
        assert torch.equal(mcore_rows.long(), rows)
    assert torch.equal(seen, torch.ones_like(seen))


def test_zigzag_segments_reject_odd_rows():
    with pytest.raises(ValueError, match="even number of local rows"):
        cp_utils.thd_cp_zigzag_segments(255, 0, 2)


@pytest.mark.parametrize("cp_size", [2, 4, 64])
@pytest.mark.parametrize("compress_ratio,window", [(128, 128), (0, 128), (8, 12)])
def test_zigzag_layout_positions(cpu_layout_kernels, cp_size, compress_ratio, window):
    """q positions cover [A | B]; kv positions cover [halo_A | halo_B | A | B]."""
    d_window = cp_utils.dsv4_cp_window_rows(compress_ratio, window)
    segment_rows = 128 if compress_ratio != 8 else 16
    local_rows = 2 * segment_rows
    total = cp_size * local_rows
    cu = _cu(total)
    for cp_rank in range(cp_size):
        layout = cp_utils.build_dsv4_zigzag_cp_layout(
            cu, cp_rank, cp_size, local_rows, compress_ratio, window
        )
        rows = _zigzag_rows(cp_rank, cp_size, segment_rows)
        assert layout.segment_rows == segment_rows
        assert layout.d_window == d_window
        assert layout.segment_starts == (int(rows[0]), int(rows[segment_rows]))
        assert torch.equal(layout.q_position_ids.long(), rows)
        halo = _halo_rows(layout.segment_starts, d_window)
        expected_kv = torch.cat((halo.clamp_min(0), rows))
        assert torch.equal(layout.kv_position_ids.long(), expected_kv)
        assert (layout.compressor_layout is None) == (compress_ratio <= 1)


def test_zigzag_layout_is_cached_per_microbatch(cpu_layout_kernels):
    """Layers of one microbatch share one layout; graphs and new microbatches rebuild it."""
    group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    eager = SimpleNamespace(cuda_graph_impl="none")
    packed = SimpleNamespace()
    cu = _cu(4 * 512)
    first = cp_utils.get_dsv4_zigzag_cp_layout(packed, cu, group, 512, 128, 128, eager)
    again = cp_utils.get_dsv4_zigzag_cp_layout(packed, cu, group, 512, 128, 128, eager)
    window_only = cp_utils.get_dsv4_zigzag_cp_layout(packed, cu, group, 512, 0, 128, eager)
    assert again is first
    assert window_only is not first and window_only.compressor_layout is None
    assert first.q_position_ids._do_not_offload
    other_microbatch = cp_utils.get_dsv4_zigzag_cp_layout(
        SimpleNamespace(), cu, group, 512, 128, 128, eager
    )
    assert other_microbatch is not first
    graphs = SimpleNamespace(cuda_graph_impl="transformer_engine")
    graph_packed = SimpleNamespace()
    graph_layout = cp_utils.get_dsv4_zigzag_cp_layout(
        graph_packed, cu, group, 512, 128, 128, graphs
    )
    assert not hasattr(graph_packed, "_dsv4_zigzag_cp_layout_cache")
    assert torch.equal(graph_layout.q_position_ids, first.q_position_ids)


@pytest.mark.parametrize(
    "cu_values,local_rows,ratio,error,match",
    [
        ([0, 256, 512], 256, 128, NotImplementedError, "one sequence per pack"),
        ([0, 512], 255, 128, ValueError, "two equal chunks"),
        ([0, 512], 128, 128, ValueError, "two equal chunks"),
        ([0, 768], 384, 128, ValueError, "whole compression blocks"),
        ([0, 1024], 512, 4, NotImplementedError, "ratio-4"),
    ],
    ids=["multi-sequence", "odd-rows", "chunk-below-window", "misaligned-chunk", "ratio-4"],
)
def test_zigzag_layout_fails_closed(cu_values, local_rows, ratio, error, match):
    """Unsupported packs raise before any kernel launch (no CuTe on CPU)."""
    cu = torch.tensor(cu_values, dtype=torch.int32)
    with pytest.raises(error, match=match):
        cp_utils.build_dsv4_zigzag_cp_layout(cu, 0, 2, local_rows, ratio, 128)


def test_zigzag_layout_rejects_a_pack_shorter_than_the_cp_rows(cpu_layout_kernels):
    with pytest.raises(RuntimeError, match="local_rows \\* cp_size"):
        cp_utils.build_dsv4_zigzag_cp_layout(_cu(1000), 0, 2, 512, 128, 128)


# =============================================================================
# Compressor layout and compressed-row map
# =============================================================================


@pytest.mark.parametrize(
    "lengths,ratio,cp_size,global_start_rank",
    [((1024,), 128, 4, 0), ((1024,), 128, 4, 2), ((300, 724), 128, 4, 1), ((60, 30, 102), 8, 3, 1)],
)
def test_row_map_oracles_match_compaction_oracle(lengths, ratio, cp_size, global_start_rank):
    """The new row-map oracles agree with the existing torch compaction oracle."""
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    local_rows = sum(lengths) // cp_size
    global_start = global_start_rank * local_rows
    d_comp, c_cap = thd_layout_kernels._cp_compressor_capacity(local_rows, ratio)
    maps = packed_layout.compressor_row_maps_reference(
        cu, global_start, local_rows, ratio, d_comp, c_cap
    )
    rank_rows = packed_layout.compressor_rank_rows_reference(
        cu, maps[6], local_rows, cp_size, ratio, d_comp, c_cap
    )
    hidden = torch.arange(1, local_rows + 1, dtype=torch.float64).unsqueeze(1)
    boundary = torch.arange(-d_comp, 0, dtype=torch.float64).unsqueeze(1)
    gathered, group_ids, positions, comp_cu, row_map = packed_layout._compact_compressor_input(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap, cp_size
    )
    assert torch.equal(maps[2], group_ids.int())
    assert torch.equal(maps[3], positions.int())
    assert torch.equal(maps[6], comp_cu.int())
    assert torch.equal(rank_rows, row_map.int())
    source = torch.cat((boundary, hidden)).squeeze(1)
    from_map = torch.where(maps[0] >= 0, source[maps[0].clamp_min(0).long()], 0)
    assert torch.equal(from_map, gathered.squeeze(1))


def _check_inverse_maps(compact_to_source, source_to_compact):
    for compact, source in enumerate(compact_to_source.tolist()):
        if source >= 0:
            assert int(source_to_compact[source]) == compact
    for source, compact in enumerate(source_to_compact.tolist()):
        if compact >= 0:
            assert int(compact_to_source[compact]) == source


@pytest.mark.parametrize(
    "cp_size,segment_rows,ratio", [(2, 128, 128), (4, 256, 128), (8, 128, 128), (64, 128, 128)]
)
def test_zigzag_compressor_layout_owns_every_block_once(
    cpu_layout_kernels, cp_size, segment_rows, ratio
):
    """Each compression block is pooled from its own rows in exactly one owner slot."""
    total = 2 * cp_size * segment_rows
    cu = _cu(total)
    d_comp, c_cap = thd_layout_kernels._cp_compressor_capacity(segment_rows, ratio)
    # Logical rows of the all-gathered buffer: (block id or -1, first and last source row).
    gathered_blocks, gathered_sources = [], []
    for cp_rank in range(cp_size):
        _, starts = cp_utils.thd_cp_zigzag_segments(2 * segment_rows, cp_rank, cp_size)
        layout = thd_layout_kernels.build_cp_zigzag_compressor_layout(
            cu, starts, segment_rows, cp_size, ratio
        )
        assert layout.boundary_rows == 2 * d_comp and layout.boundary_segments == 2
        assert layout.global_start == starts[0]
        assert layout.group_ids.shape == (2 * c_cap,)
        assert torch.equal(layout.local_cu_seqlens, torch.tensor([0, 1, 2]) * c_cap * ratio)
        assert torch.equal(layout.local_cu_seqlens_compressed, torch.tensor([0, 1, 2]) * c_cap)
        assert torch.equal(layout.cu_seqlens_compressed, torch.tensor([0, total // ratio]))
        _check_inverse_maps(layout.compact_to_source, layout.source_to_compact)
        # Global row of each merged source row [halo_A[-dc:] | halo_B[-dc:] | A | B].
        source_rows = torch.cat(
            (_halo_rows(starts, d_comp), _zigzag_rows(cp_rank, cp_size, segment_rows))
        )
        for slot in range(2 * c_cap):
            group = int(layout.group_ids[slot])
            sources = layout.compact_to_source[slot * ratio : (slot + 1) * ratio]
            if group < 0:
                assert bool((sources < 0).all())
                assert int(layout.position_ids[slot]) == 0
                gathered_blocks.append(-1)
                gathered_sources.append(None)
                continue
            rows = source_rows[sources.long()]
            assert torch.equal(rows, torch.arange(group * ratio, (group + 1) * ratio))
            assert int(layout.position_ids[slot]) == group * ratio
            gathered_blocks.append(group)
            gathered_sources.append(sources)
        seq_to_rank_row = layout.seq_to_rank_row
    assert seq_to_rank_row.shape == (total // ratio,)
    for block in range(total // ratio):
        physical = int(seq_to_rank_row[block])
        assert gathered_blocks[physical] == block
        # The owner slot pools only local rows (never a halo duplicate).
        assert int(gathered_sources[physical].min()) >= 2 * d_comp


def test_zigzag_rank_rows_remap_matches_brute_force():
    cp_size, c_cap = 4, 5
    virtual = torch.arange(2 * cp_size * c_cap, dtype=torch.int32)
    virtual = torch.cat((virtual, torch.tensor([-1], dtype=torch.int32)))
    remapped = thd_layout_kernels.cp_zigzag_rank_rows(virtual, cp_size, c_cap)
    for index, value in enumerate(virtual.tolist()):
        if value < 0:
            assert int(remapped[index]) == -1
            continue
        chunk, slot = divmod(value, c_cap)
        rank = chunk if chunk < cp_size else 2 * cp_size - 1 - chunk
        segment = int(chunk >= cp_size)
        assert int(remapped[index]) == rank * 2 * c_cap + segment * c_cap + slot
    assert remapped.dtype == torch.int32


def test_zigzag_compressor_layout_rejects_non_zigzag_starts(cpu_layout_kernels):
    with pytest.raises(ValueError, match="not the zigzag chunks"):
        thd_layout_kernels.build_cp_zigzag_compressor_layout(_cu(1024), (0, 256), 128, 4, 128)


# =============================================================================
# Attention indices and causal masks
# =============================================================================


def _decode_zigzag_indices(indices, starts, segment_rows, d_window, physical_to_block):
    """Decode each row's valid physical indices to logical keys, in index order."""
    compressed_base = 2 * d_window + 2 * segment_rows
    decoded = []
    for row in indices.tolist():
        keys = []
        for index in row:
            if index < 0:
                continue
            if index < 2 * d_window:
                segment, offset = divmod(index, d_window)
                keys.append(("row", starts[segment] - d_window + offset))
            elif index < compressed_base:
                segment, offset = divmod(index - 2 * d_window, segment_rows)
                keys.append(("row", starts[segment] + offset))
            else:
                keys.append(("block", physical_to_block[index - compressed_base]))
        decoded.append(keys)
    return decoded


@pytest.mark.parametrize(
    "cp_size,segment_rows,ratio,window",
    [
        (2, 128, 128, 128),
        (4, 256, 128, 128),
        (8, 128, 128, 128),
        (64, 128, 128, 128),
        (4, 16, 8, 12),
        (2, 32, 0, 16),
    ],
)
@pytest.mark.parametrize("output_alignment", [1, 64])
def test_zigzag_attention_indices_match_contiguous_keys(
    cpu_layout_kernels, cp_size, segment_rows, ratio, window, output_alignment
):
    """Every zigzag row sees the same keys, in the same order, as the causal reference."""
    total = 2 * cp_size * segment_rows
    cu = _cu(total)
    d_window = cp_utils.dsv4_cp_window_rows(ratio, window)
    compressed_width = total // ratio if ratio > 1 else 0
    for cp_rank in range(cp_size):
        layout = cp_utils.build_dsv4_zigzag_cp_layout(
            cu, cp_rank, cp_size, 2 * segment_rows, ratio, window
        )
        starts = layout.segment_starts
        compressor = layout.compressor_layout
        kwargs = {}
        physical_to_block = {}
        if compressor is not None:
            kwargs = dict(
                cu_seqlens_compressed=compressor.cu_seqlens_compressed,
                seq_to_rank_row=compressor.seq_to_rank_row,
                compressed_rows=cp_size * compressor.group_ids.shape[0],
            )
            physical_to_block = _compressed_block_owner_map(compressor.seq_to_rank_row)
        indices, lengths, _, _ = packed_layout._build_attention_indices(
            cu,
            starts[0],
            2 * segment_rows,
            d_window,
            window,
            ratio,
            compressed_width,
            output_alignment=output_alignment,
            segment_rows=segment_rows,
            second_global_start=starts[1],
            **kwargs,
        )
        assert indices.shape[1] % output_alignment == 0
        decoded = _decode_zigzag_indices(indices, starts, segment_rows, d_window, physical_to_block)
        rows = _zigzag_rows(cp_rank, cp_size, segment_rows).tolist()
        for local, position in enumerate(rows):
            expected = _expected_keys(position, window, ratio)
            assert decoded[local] == expected, f"rank {cp_rank} row {position}"
            assert int(lengths[local]) == len(expected)
            # Causality: no key past the query, no block that is not complete.
            for kind, key in decoded[local]:
                assert key <= position if kind == "row" else ratio * (key + 1) <= position + 1


def test_contiguous_attention_indices_unchanged_by_segment_support():
    """The oracle's one-segment path is unchanged (defaults reproduce the old formula)."""
    cu = torch.tensor([0, 100, 256], dtype=torch.int32)
    seq_to_rank_row = torch.arange(32, dtype=torch.int32).flip(0)
    args = (cu, 64, 64, 8, 6, 8, 12)
    kwargs = dict(
        cu_seqlens_compressed=torch.tensor([0, 12, 32], dtype=torch.int32),
        seq_to_rank_row=seq_to_rank_row,
        compressed_rows=32,
    )
    indices, lengths, _, _ = packed_layout._build_attention_indices(*args, **kwargs)
    rows = torch.arange(64, 128)
    window = torch.maximum(rows - 5, torch.where(rows >= 100, 100, 0))[:, None] + torch.arange(6)
    expected_window = torch.where(window <= rows[:, None], window - 64 + 8, -1)
    assert torch.equal(indices[:, :6], expected_window.int())
    assert int(indices[:, 6:].max()) < 8 + 64 + 32


# =============================================================================
# Halo exchange (gloo, CPU)
# =============================================================================


def _zigzag_exchange_worker(rank, world_size, store_path, d_window, segment_rows, width):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    try:
        group = dist.group.WORLD
        total = 2 * world_size * segment_rows
        generator = torch.Generator().manual_seed(1234)
        full = torch.randn(total, width, generator=generator, dtype=torch.float64)
        halo_grads = torch.randn(
            world_size, 2 * d_window, width, generator=generator, dtype=torch.float64
        )
        rows = _zigzag_rows(rank, world_size, segment_rows)
        local = full[rows].clone().unsqueeze(1).requires_grad_(True)
        halo = cp_utils.exchange_cp_zigzag_boundary_hidden(local, 0, d_window, group)
        assert halo.shape == (2 * d_window, 1, width)
        starts = (int(rows[0]), int(rows[segment_rows]))
        expected = _take_rows(full, _halo_rows(starts, d_window))
        assert torch.equal(halo.squeeze(1), expected), f"rank {rank} halo mismatch"
        halo.squeeze(1).backward(halo_grads[rank])

        expected_grad = torch.zeros_like(full)
        for peer in range(world_size):
            peer_rows = _zigzag_rows(peer, world_size, segment_rows)
            peer_starts = (int(peer_rows[0]), int(peer_rows[segment_rows]))
            for index, row in enumerate(_halo_rows(peer_starts, d_window).tolist()):
                if row >= 0:
                    expected_grad[row] += halo_grads[peer, index]
        assert torch.equal(local.grad.squeeze(1), expected_grad[rows]), f"rank {rank} grad"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("d_window,segment_rows", [(4, 4), (3, 8)])
def test_zigzag_boundary_exchange_gloo(tmp_path, world_size, d_window, segment_rows):
    """Forward returns [halo_A | halo_B]; backward returns each halo gradient to its owner."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo is not available")
    mp.spawn(
        _zigzag_exchange_worker,
        args=(world_size, str(tmp_path / "store"), d_window, segment_rows, 5),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# RoPE helpers
# =============================================================================


@pytest.mark.parametrize("inverse", [False, True])
def test_unfused_rope_position_ids_match_global_start(inverse):
    """Explicit positions reproduce the consecutive-interval path, forward and backward."""
    torch.manual_seed(5)
    cu = torch.tensor([0, 9, 30], dtype=torch.int32)
    freqs = torch.randn(32, 1, 1, 4, dtype=torch.float64)
    x = torch.randn(7, 2, 8, dtype=torch.float64)
    expected_x = x.clone().requires_grad_(True)
    expected = cp_utils.apply_thd_cp_local_rope_unfused(
        expected_x, freqs, 4, 4, cu, 5, _ROPE_CONFIG, inverse=inverse
    )
    positions = cp_utils.thd_cp_position_ids(cu, 5, 7)
    actual_x = x.clone().requires_grad_(True)
    actual = cp_utils.apply_thd_cp_local_rope_unfused(
        actual_x, freqs, 4, 4, cu, None, _ROPE_CONFIG, inverse=inverse, position_ids=positions
    )
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_x.grad, expected_x.grad)


def test_rope_helpers_require_one_row_source(monkeypatch):
    x = torch.zeros(4, 1, 8)
    cu = _cu(8)
    with pytest.raises(ValueError, match="exactly one of global_start"):
        cp_utils.apply_thd_cp_local_rope_unfused(
            x, torch.zeros(8, 1, 1, 4), 4, 4, cu, 0, _ROPE_CONFIG, position_ids=torch.arange(4)
        )
    with pytest.raises(ValueError, match="exactly one of global_start"):
        cp_utils.apply_thd_cp_local_rope_fused(x, None, None, 4, 4, cu, None)
    with pytest.raises(ValueError, match="must have shape"):
        cp_utils.apply_thd_cp_local_rope_fused(
            x, None, None, 4, 4, cu, None, position_ids=torch.arange(3)
        )

    calls = []

    def fake_fused(rope_input, *_args, **kwargs):
        calls.append(kwargs)
        return rope_input

    monkeypatch.setattr(cp_utils, "fused_mla_rope_inplace", fake_fused)
    positions = torch.arange(4, dtype=torch.int32)
    cp_utils.apply_thd_cp_local_rope_fused(x, None, None, 4, 4, cu, None, position_ids=positions)
    assert calls[-1]["position_ids"] is positions and "thd_global_start" not in calls[-1]


# =============================================================================
# H-layer core emulation: contiguous vs zigzag vs full-sequence reference
# =============================================================================


class _CoreWeights(torch.nn.Module):
    """Weights of a toy H-layer core (MQA, latent KV, compressor, sink)."""

    def __init__(self, hidden, heads, nope_dim, pos_dim, ratio):
        super().__init__()
        head_dim = nope_dim + pos_dim
        self.heads, self.nope_dim, self.pos_dim = heads, nope_dim, pos_dim
        self.wq = torch.nn.Parameter(torch.randn(hidden, heads * head_dim, dtype=torch.float64))
        self.wkv = torch.nn.Parameter(torch.randn(hidden, head_dim, dtype=torch.float64))
        self.wc_kv = torch.nn.Parameter(torch.randn(hidden, head_dim, dtype=torch.float64))
        self.wc_gate = torch.nn.Parameter(torch.randn(hidden, head_dim, dtype=torch.float64))
        self.ape = torch.nn.Parameter(torch.randn(max(ratio, 1), head_dim, dtype=torch.float64))
        self.sink = torch.nn.Parameter(torch.randn(heads, dtype=torch.float64))
        for parameter in (self.wq, self.wkv, self.wc_kv, self.wc_gate):
            parameter.data.mul_(hidden**-0.5)


def _rope(x, freqs, cu, model, inverse=False, global_start=None, position_ids=None):
    return cp_utils.apply_thd_cp_local_rope_unfused(
        x,
        freqs,
        model.nope_dim,
        model.pos_dim,
        cu,
        global_start,
        _ROPE_CONFIG,
        inverse=inverse,
        position_ids=position_ids,
    )


def _pool(model, source, compact_to_source, groups, ratio):
    """Compressor: project, gather groups by the compact map, gated pooling."""
    kv = _take_rows(source @ model.wc_kv, compact_to_source).view(groups, ratio, 1, -1)
    score = _take_rows(source @ model.wc_gate, compact_to_source).view(groups, ratio, 1, -1)
    score = score + model.ape.view(1, ratio, 1, -1)
    return _pool_compressor_values(kv, score, source.dtype).squeeze(1)


def _attend(model, q, kv_full, indices):
    output = unfused_compressed_sparse_attn(
        q, kv_full, model.sink, indices, softmax_scale=q.shape[-1] ** -0.5
    )
    return output.view(q.shape)


def _reference_core(model, x, freqs, window, ratio):
    """Full-sequence reference: per-row window and complete compressed blocks."""
    total = x.shape[0]
    cu = _cu(total)
    q = _rope((x @ model.wq).view(total, model.heads, -1), freqs, cu, model, global_start=0)
    kv = _rope(x @ model.wkv, freqs, cu, model, global_start=0)
    blocks = total // ratio if ratio > 1 else 0
    kv_full = kv
    if blocks:
        compressed = _pool(model, x, torch.arange(blocks * ratio), blocks, ratio)
        positions = torch.arange(blocks, dtype=torch.int32) * ratio
        compressed = _rope(compressed, freqs, cu, model, position_ids=positions)
        kv_full = torch.cat((kv, compressed))
    indices = torch.full((total, window + blocks), -1, dtype=torch.int32)
    for position in range(total):
        keys = _expected_keys(position, window, ratio)
        indices[position, : len(keys)] = torch.tensor(
            [key if kind == "row" else total + key for kind, key in keys], dtype=torch.int32
        )
    output = _attend(model, q, kv_full, indices)
    return _rope(output, freqs, cu, model, inverse=True, global_start=0)


def _emulated_cp_core(model, x, freqs, window, ratio, cp_size, zigzag):
    """Run the CP data flow of one H-layer core rank by rank, as CSA's THD CP path does.

    Halos are the rows the boundary exchange delivers; the compressed blocks of every rank
    are concatenated rank-major, as the CP all-gather does.
    """
    total = x.shape[0]
    local_rows = total // cp_size
    cu = _cu(total)
    d_window = cp_utils.dsv4_cp_window_rows(ratio, window)
    ranks = []
    for cp_rank in range(cp_size):
        if zigzag:
            layout = cp_utils.build_dsv4_zigzag_cp_layout(
                cu, cp_rank, cp_size, local_rows, ratio, window
            )
            starts = layout.segment_starts
            rows = _zigzag_rows(cp_rank, cp_size, layout.segment_rows)
            compressor = layout.compressor_layout
            rope_q = dict(position_ids=layout.q_position_ids)
            rope_kv = dict(position_ids=layout.kv_position_ids)
            index_kwargs = dict(segment_rows=layout.segment_rows, second_global_start=starts[1])
        else:
            starts = (cp_rank * local_rows,)
            rows = torch.arange(starts[0], starts[0] + local_rows)
            compressor = None
            if ratio > 1:
                compressor = thd_layout_kernels.build_cp_compressor_layout(
                    cu, starts[0], local_rows, cp_size, ratio
                )
            rope_q = dict(global_start=starts[0])
            rope_kv = dict(global_start=starts[0] - d_window)
            index_kwargs = {}
        halo = _take_rows(x, _halo_rows(starts, d_window))
        local = x.index_select(0, rows)
        compressed = None
        if compressor is not None:
            d_comp = compressor.boundary_rows // len(starts)
            compressor_halo = torch.cat(
                [halo[(s + 1) * d_window - d_comp : (s + 1) * d_window] for s in range(len(starts))]
            )
            groups = compressor.group_ids.shape[0]
            compressed = _pool(
                model,
                torch.cat((compressor_halo, local)),
                compressor.compact_to_source,
                groups,
                ratio,
            )
            compressed = _rope(compressed, freqs, cu, model, position_ids=compressor.position_ids)
        ranks.append(
            (starts, rows, halo, local, compressor, compressed, rope_q, rope_kv, index_kwargs)
        )

    gathered = None
    if ratio > 1:
        gathered = torch.cat([rank[5] for rank in ranks])
    outputs, global_rows = [], []
    for starts, rows, halo, local, compressor, _, rope_q, rope_kv, index_kwargs in ranks:
        q = _rope((local @ model.wq).view(local_rows, model.heads, -1), freqs, cu, model, **rope_q)
        kv = _rope(torch.cat((halo, local)) @ model.wkv, freqs, cu, model, **rope_kv)
        kv_full = kv if gathered is None else torch.cat((kv, gathered))
        compressed_kwargs = {}
        if compressor is not None:
            compressed_kwargs = dict(
                cu_seqlens_compressed=compressor.cu_seqlens_compressed,
                seq_to_rank_row=compressor.seq_to_rank_row,
                compressed_rows=gathered.shape[0],
            )
        indices, _, _, _ = packed_layout._build_attention_indices(
            cu,
            starts[0],
            local_rows,
            d_window,
            window,
            ratio,
            total // ratio if ratio > 1 else 0,
            **compressed_kwargs,
            **index_kwargs,
        )
        output = _attend(model, q, kv_full, indices)
        output = _rope(output, freqs, cu, model, inverse=True, **rope_q)
        outputs.append(output)
        global_rows.append(rows)
    output = torch.cat(outputs)
    order = torch.argsort(torch.cat(global_rows))
    return output.index_select(0, order)


@pytest.mark.parametrize(
    "cp_size,segment_rows,ratio,window",
    [(2, 16, 8, 6), (4, 16, 8, 12), (2, 32, 16, 16), (4, 8, 0, 8), (8, 16, 16, 16)],
    ids=["cp2-r8-w6", "cp4-r8-w12", "cp2-r16-w16", "cp4-window-only", "cp8-r16-w16"],
)
def test_emulated_core_zigzag_matches_contiguous_and_reference(
    cpu_layout_kernels, cp_size, segment_rows, ratio, window
):
    """Per-row outputs and all gradients agree across zigzag, contiguous and full sequence."""
    torch.manual_seed(1000 + cp_size * 31 + ratio)
    total = 2 * cp_size * segment_rows
    hidden, heads, nope_dim, pos_dim = 16, 2, 6, 4
    model = _CoreWeights(hidden, heads, nope_dim, pos_dim, ratio)
    positions = torch.arange(total, dtype=torch.float64)
    inv_freq = 1.0 / (100.0 ** (torch.arange(0, pos_dim, 2, dtype=torch.float64) / pos_dim))
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1).view(total, 1, 1, pos_dim)
    x = torch.randn(total, hidden, dtype=torch.float64)
    grad_output = torch.randn(total, heads, nope_dim + pos_dim, dtype=torch.float64)

    results = {}
    for name, run in (
        ("reference", lambda inputs: _reference_core(model, inputs, freqs, window, ratio)),
        (
            "contiguous",
            lambda inputs: _emulated_cp_core(model, inputs, freqs, window, ratio, cp_size, False),
        ),
        (
            "zigzag",
            lambda inputs: _emulated_cp_core(model, inputs, freqs, window, ratio, cp_size, True),
        ),
    ):
        model.zero_grad(set_to_none=True)
        inputs = x.clone().requires_grad_(True)
        output = run(inputs)
        output.backward(grad_output)
        grads = {
            param_name: param.grad.clone()
            for param_name, param in model.named_parameters()
            if param.grad is not None
        }
        results[name] = (output.detach(), inputs.grad, grads)

    reference_output, reference_dx, reference_grads = results["reference"]
    for name in ("contiguous", "zigzag"):
        output, dx, grads = results[name]
        # The attention core computes in FP32 internally; inputs per row are identical.
        torch.testing.assert_close(output, reference_output, rtol=1e-6, atol=1e-6, msg=name)
        torch.testing.assert_close(dx, reference_dx, rtol=1e-5, atol=1e-5, msg=name)
        assert grads.keys() == reference_grads.keys()
        for parameter, grad in grads.items():
            torch.testing.assert_close(
                grad, reference_grads[parameter], rtol=1e-5, atol=1e-5, msg=f"{name}:{parameter}"
            )
    zigzag_output, contiguous_output = results["zigzag"][0], results["contiguous"][0]
    difference = (zigzag_output - contiguous_output).abs().max().item()
    equal_rows = (zigzag_output == contiguous_output).flatten(1).all(dim=1).double().mean().item()
    print(f"zigzag vs contiguous: max |diff| {difference:.3e}, bitwise-equal rows {equal_rows:.4f}")
    # Host BLAS may block GEMMs of different row counts differently, so CPU rows need not be
    # bitwise equal; they must agree to the FP32 attention core's rounding.
    torch.testing.assert_close(zigzag_output, contiguous_output, rtol=1e-6, atol=1e-6)


# =============================================================================
# Config guards
# =============================================================================


def _dsv4_config(**overrides):
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    kwargs = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        output_projection_groups=8,
        output_projection_lora_rank=64,
        rope_type="rope",
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[0, 128, 0, 128],
        csa_window_size=128,
        csa_dense_mode=True,
        dsa_kernel_backend="none",
        context_parallel_size=4,
        max_seqlen_per_dp_cp_rank=1024,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(cp_partition_mode="contiguous"), None),
        (dict(cp_partition_mode="zigzag"), "requires cp_partition_mode='contiguous'"),
        (dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True), None),
        (dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, thd_max_packed_sequences=1), None),
        (dict(cp_partition_mode="contiguous", dsv4_cp_zigzag=True), "to be 'zigzag'"),
        (
            dict(
                cp_partition_mode="zigzag", dsv4_cp_zigzag=True, csa_compress_ratios=[0, 4, 128, 4]
            ),
            "compress ratio 4",
        ),
        (
            dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, dsa_cp_balance_indexer=True),
            "dsa_cp_balance_indexer",
        ),
        (
            dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, delay_wgrad_compute=True),
            "delay_wgrad_compute",
        ),
        (dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, thd_max_packed_sequences=2), None),
        (
            dict(
                cp_partition_mode="zigzag",
                dsv4_cp_zigzag=True,
                sequence_packing_scheduler="dp_balanced",
            ),
            None,
        ),
        (
            dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, cuda_graph_impl="local"),
            "CUDA graphs",
        ),
        (dict(cp_partition_mode="zigzag", dsv4_cp_zigzag=True, context_parallel_size=1), None),
    ],
    ids=[
        "default-contiguous",
        "zigzag-without-flag",
        "zigzag-with-flag",
        "zigzag-one-sequence",
        "flag-with-contiguous",
        "ratio-4",
        "balanced-indexer",
        "delay-wgrad",
        "multi-sequence-packs",
        "packing-scheduler",
        "cuda-graphs",
        "cp1-no-op",
    ],
)
def test_dsv4_cp_zigzag_config_guards(overrides, error):
    context = pytest.raises(ValueError, match=error) if error else nullcontext()
    with context:
        config = _dsv4_config(**overrides)
    if error is None:
        assert config.dsv4_cp_zigzag == overrides.get("dsv4_cp_zigzag", False)
        if config.dsv4_cp_zigzag:
            assert config.attention_cp_layout == config.linear_cp_layout == "zigzag"


def test_dsv4_cp_zigzag_flag_requires_dsv4():
    from megatron.core.transformer.transformer_config import TransformerConfig

    with pytest.raises(ValueError, match="experimental_attention_variant='dsv4_hybrid'"):
        TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, dsv4_cp_zigzag=True)


def test_dsv4_attention_rejects_mismatched_partition_metadata():
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as dsv4,
    )

    zigzag_packed = SimpleNamespace(cp_partition_mode="zigzag")
    contiguous_packed = SimpleNamespace(cp_partition_mode="contiguous")
    on, off = SimpleNamespace(dsv4_cp_zigzag=True), SimpleNamespace(dsv4_cp_zigzag=False)
    assert dsv4._uses_dsv4_zigzag_cp(on, zigzag_packed)
    assert not dsv4._uses_dsv4_zigzag_cp(off, contiguous_packed)
    for config, packed in ((off, zigzag_packed), (on, contiguous_packed)):
        with pytest.raises(ValueError, match="dsv4_cp_zigzag"):
            dsv4._uses_dsv4_zigzag_cp(config, packed)


def test_validate_packed_inputs_accepts_opted_in_zigzag():
    packed = SimpleNamespace(
        qkv_format="thd", local_cp_size=None, cp_group=None, max_seqlen_q=None, max_seqlen_kv=None
    )
    group = SimpleNamespace(size=lambda: 2)
    base = dict(dsa_kernel_backend="cudnn", deterministic_mode=False)
    with pytest.raises(ValueError, match="attention_cp_layout='contiguous'"):
        cp_utils.validate_packed_inputs(
            packed, SimpleNamespace(attention_cp_layout="zigzag", **base), group
        )
    # The opted-in zigzag layout passes the layout check and reaches the next one.
    with pytest.raises(ValueError, match="host-known maximum"):
        cp_utils.validate_packed_inputs(
            packed,
            SimpleNamespace(attention_cp_layout="zigzag", dsv4_cp_zigzag=True, **base),
            group,
        )


# =============================================================================
# CUDA: CuTe kernels against the oracles (1 GPU)
# =============================================================================


def _require_cute_cuda():
    if not torch.cuda.is_available() or not thd_layout_kernels._CUTE_AVAILABLE:
        pytest.skip("CuTe DSv4 THD layout kernels require CUDA and CuTeDSL.")


def _valid_first(indices):
    """Move each row's valid entries to the front, keeping their order."""
    order = torch.argsort((indices < 0).int(), dim=1, stable=True)
    return indices.gather(1, order)


@pytest.mark.parametrize(
    "cp_size,segment_rows,ratio,window",
    [
        (2, 128, 128, 128),
        (4, 512, 128, 128),
        (8, 256, 128, 128),
        (64, 128, 128, 128),
        (4, 128, 0, 128),
    ],
)
@pytest.mark.parametrize("output_alignment", [1, 64])
def test_cuda_zigzag_layout_and_indices_match_oracles(
    cp_size, segment_rows, ratio, window, output_alignment
):
    """The CuTe row maps, rank-row map and two-segment indices equal the host oracles."""
    _require_cute_cuda()
    total = 2 * cp_size * segment_rows
    cu = _cu(total, "cuda")
    d_window = cp_utils.dsv4_cp_window_rows(ratio, window)
    for cp_rank in sorted({0, 1, cp_size // 2, cp_size - 1}):
        actual = cp_utils.build_dsv4_zigzag_cp_layout(
            cu, cp_rank, cp_size, 2 * segment_rows, ratio, window
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(thd_layout_kernels, "_require_cute", lambda *a, **k: None)
            patch.setattr(
                thd_layout_kernels,
                "_compressor_row_maps",
                packed_layout.compressor_row_maps_reference,
            )
            patch.setattr(
                thd_layout_kernels,
                "_compressor_rank_rows",
                packed_layout.compressor_rank_rows_reference,
            )
            expected = cp_utils.build_dsv4_zigzag_cp_layout(
                _cu(total), cp_rank, cp_size, 2 * segment_rows, ratio, window
            )
        assert torch.equal(actual.q_position_ids.cpu(), expected.q_position_ids)
        assert torch.equal(actual.kv_position_ids.cpu(), expected.kv_position_ids)
        kwargs = {}
        if ratio > 1:
            for field in thd_layout_kernels.CPCompressorLayout._fields[:8]:
                assert torch.equal(
                    getattr(actual.compressor_layout, field).cpu(),
                    getattr(expected.compressor_layout, field),
                ), field
            kwargs = dict(
                cu_seqlens_compressed=actual.compressor_layout.cu_seqlens_compressed,
                seq_to_rank_row=actual.compressor_layout.seq_to_rank_row,
                compressed_rows=cp_size * actual.compressor_layout.group_ids.shape[0],
            )
        args = (
            cu,
            actual.segment_starts[0],
            2 * segment_rows,
            d_window,
            window,
            ratio,
            total // ratio if ratio > 1 else 0,
        )
        segments = dict(segment_rows=segment_rows, second_global_start=actual.segment_starts[1])
        indices, lengths, _, _ = thd_layout_kernels.build_attention_indices(
            *args, output_alignment=output_alignment, **kwargs, **segments
        )
        reference, reference_lengths, _, _ = packed_layout._build_attention_indices(
            *args, output_alignment=output_alignment, **kwargs, **segments
        )
        width = max(indices.shape[1], reference.shape[1])
        pad = lambda t: torch.nn.functional.pad(t, (0, width - t.shape[1]), value=-1)
        assert torch.equal(_valid_first(pad(indices)), _valid_first(pad(reference)))
        assert torch.equal(lengths, reference_lengths)


def test_cuda_contiguous_attention_indices_unchanged():
    """Without segment args the kernel reproduces the one-segment oracle exactly."""
    _require_cute_cuda()
    cu = torch.tensor([0, 300, 1024, 2048], dtype=torch.int32, device="cuda")
    layout = thd_layout_kernels.build_cp_compressor_layout(cu, 512, 512, 4, 128)
    args = (cu, 512, 512, 128, 128, 128, 16)
    kwargs = dict(
        cu_seqlens_compressed=layout.cu_seqlens_compressed,
        seq_to_rank_row=layout.seq_to_rank_row,
        compressed_rows=4 * layout.group_ids.shape[0],
    )
    indices, lengths, _, _ = thd_layout_kernels.build_attention_indices(*args, **kwargs)
    reference, reference_lengths, _, _ = packed_layout._build_attention_indices(*args, **kwargs)
    assert torch.equal(_valid_first(indices), _valid_first(reference))
    assert torch.equal(lengths, reference_lengths)


@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (4, 1), (4, 3)])
def test_cuda_zigzag_projection_compaction_forward_backward(cp_size, cp_rank):
    """CompressorProjectionCompact gathers and scatters through the merged zigzag maps."""
    _require_cute_cuda()
    segment_rows, ratio, width = 256, 128, 64
    _, starts = cp_utils.thd_cp_zigzag_segments(2 * segment_rows, cp_rank, cp_size)
    layout = thd_layout_kernels.build_cp_zigzag_compressor_layout(
        _cu(2 * cp_size * segment_rows, "cuda"), starts, segment_rows, cp_size, ratio
    )
    values = [
        torch.randn(rows, 1, width, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for rows in (2 * segment_rows, 2 * segment_rows, 2 * ratio, 2 * ratio)
    ]
    compact = thd_layout_kernels.CompressorProjectionCompact.apply(*values, layout)
    for local, boundary, output in (
        (values[0], values[2], compact[0]),
        (values[1], values[3], compact[1]),
    ):
        source = torch.cat((boundary, local))
        expected = _take_rows(source, layout.compact_to_source)
        assert torch.equal(output, expected)
    grads = [torch.randn_like(output) for output in compact]
    actual_grads = torch.autograd.grad(compact, values, grads)
    for local_grad, boundary_grad, grad in (
        (actual_grads[0], actual_grads[2], grads[0]),
        (actual_grads[1], actual_grads[3], grads[1]),
    ):
        expected = _take_rows(grad, layout.source_to_compact)
        assert torch.equal(torch.cat((boundary_grad, local_grad)), expected)


def test_cuda_zigzag_layout_graph_replay_matches_eager():
    """Layout and index construction are capturable and replay to the eager result.

    Zigzag CP does not support CUDA graphs yet (config guard); this checks that the per-pack
    layout itself does not block a later H-layer graph.
    """
    _require_cute_cuda()
    cp_size, segment_rows, ratio = 4, 256, 128
    total = 2 * cp_size * segment_rows
    cu = _cu(total, "cuda")

    def run():
        layout = cp_utils.build_dsv4_zigzag_cp_layout(cu, 1, cp_size, 2 * segment_rows, ratio, 128)
        indices = thd_layout_kernels.build_attention_indices(
            cu,
            layout.segment_starts[0],
            2 * segment_rows,
            layout.d_window,
            128,
            ratio,
            total // ratio,
            cu_seqlens_compressed=layout.compressor_layout.cu_seqlens_compressed,
            seq_to_rank_row=layout.compressor_layout.seq_to_rank_row,
            compressed_rows=cp_size * layout.compressor_layout.group_ids.shape[0],
            segment_rows=segment_rows,
            second_global_start=layout.segment_starts[1],
        )
        return (layout.q_position_ids, layout.kv_position_ids, *layout.compressor_layout[:8]) + (
            indices[0],
            indices[1],
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    expected = run()
    for actual, reference in zip(captured, expected):
        assert torch.equal(actual, reference)


# =============================================================================
# CUDA, distributed: torchrun --nproc-per-node 2 or 4
# =============================================================================


def _distributed_cp_size():
    if not torch.cuda.is_available() or not dist.is_available():
        return 0
    return int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.skipif(_distributed_cp_size() < 2, reason="needs torchrun with >= 2 CUDA ranks")
class TestDSv4ZigzagCPDistributed:
    """Zigzag CP module parity on CP2/CP4 (``torchrun --nproc-per-node 2|4``)."""

    @pytest.fixture(scope="class", autouse=True)
    def model_parallel(self):
        from tests.unit_tests.test_utilities import Utils

        cp_size = Utils.world_size
        if cp_size not in (2, 4):
            pytest.skip("zigzag CP parity runs with 2 or 4 ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        yield cp_size
        Utils.destroy_model_parallel()

    def test_nccl_zigzag_boundary_exchange(self, model_parallel):
        from megatron.core.process_groups_config import ProcessGroupCollection

        cp_size = model_parallel
        group = ProcessGroupCollection.use_mpu_process_groups().cp
        cp_rank = group.rank()
        d_window, segment_rows, width = 128, 512, 256
        total = 2 * cp_size * segment_rows
        generator = torch.Generator(device="cuda").manual_seed(7)
        full = torch.randn(total, width, device="cuda", generator=generator)
        halo_grads = torch.randn(cp_size, 2 * d_window, width, device="cuda", generator=generator)
        rows = _zigzag_rows(cp_rank, cp_size, segment_rows, "cuda")
        local = full[rows].clone().unsqueeze(1).requires_grad_(True)
        halo = cp_utils.exchange_cp_zigzag_boundary_hidden(local, 0, d_window, group).squeeze(1)
        starts = (int(rows[0]), int(rows[segment_rows]))
        assert torch.equal(halo, _take_rows(full, _halo_rows(starts, d_window, "cuda")))
        halo.backward(halo_grads[cp_rank])
        expected_grad = torch.zeros_like(full)
        for peer in range(cp_size):
            peer_rows = _zigzag_rows(peer, cp_size, segment_rows, "cuda")
            peer_halo = _halo_rows((int(peer_rows[0]), int(peer_rows[segment_rows])), d_window)
            valid = peer_halo >= 0
            expected_grad[peer_halo[valid].cuda()] += halo_grads[peer][valid.cuda()]
        assert torch.equal(local.grad.squeeze(1), expected_grad[rows])

    @pytest.mark.parametrize("layer_number", [1, 2], ids=["window_only", "ratio_128"])
    @pytest.mark.parametrize("apply_rope_fusion", [True, False], ids=["fused_rope", "unfused_rope"])
    def test_zigzag_matches_contiguous_and_cp1(
        self, model_parallel, layer_number, apply_rope_fusion
    ):
        """Zigzag output and gradients match the CP1 reference and the contiguous partition."""
        from megatron.core.extensions.transformer_engine import HAVE_TE
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
            _build_attention,
        )
        from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
            _assert_cp_tensor_match,
            _copy_module_parameters,
            _dsv4_cp_fused_kernels_available,
            _make_dsv4_cp_config,
            _make_thd_packed_seq_params,
            _ReferenceCPGroup,
        )

        if not HAVE_TE:
            pytest.skip("transformer_engine not available")
        cp_size = model_parallel
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg.cp = _ReferenceCPGroup()
        cp_rank = pg.cp.rank()
        segment_rows = 512
        total = 2 * cp_size * segment_rows
        use_fused = _dsv4_cp_fused_kernels_available()
        common = dict(
            dsa_indexer_loss_coeff=0.0,
            use_fused_kernels=use_fused,
            apply_rope_fusion=apply_rope_fusion,
            csa_compress_ratios=[0, 128, 0, 128],
            csa_dense_mode=True,
        )
        configs = {
            "zigzag": _make_dsv4_cp_config(
                context_parallel_size=cp_size,
                cp_partition_mode="zigzag",
                dsv4_cp_zigzag=True,
                sequence_packing_scheduler=None,
                **common,
            ),
            "contiguous": _make_dsv4_cp_config(context_parallel_size=cp_size, **common),
            "cp1": _make_dsv4_cp_config(context_parallel_size=1, **common),
        }
        torch.manual_seed(1234 + layer_number)
        model_parallel_cuda_manual_seed(1234 + layer_number)
        modules = {
            name: _build_attention(
                config, layer_number=layer_number, pg_collection=ref_pg if name == "cp1" else pg
            ).cuda()
            for name, config in configs.items()
        }
        for name in ("contiguous", "cp1"):
            _copy_module_parameters(modules["zigzag"], modules[name])
        rows = {
            "zigzag": _zigzag_rows(cp_rank, cp_size, segment_rows, "cuda"),
            "contiguous": torch.arange(
                cp_rank * 2 * segment_rows, (cp_rank + 1) * 2 * segment_rows, device="cuda"
            ),
            "cp1": torch.arange(total, device="cuda"),
        }
        full_hidden = torch.randn(
            total, 1, configs["cp1"].hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        dist.broadcast(full_hidden, src=dist.get_global_rank(pg.cp, 0), group=pg.cp)
        full_grad = torch.randn_like(full_hidden)
        dist.broadcast(full_grad, src=dist.get_global_rank(pg.cp, 0), group=pg.cp)
        results = {}
        for name, module in modules.items():
            packed = _make_thd_packed_seq_params([total])
            packed.cp_partition_mode = "zigzag" if name == "zigzag" else "contiguous"
            hidden = full_hidden.index_select(0, rows[name]).detach().clone().requires_grad_(True)
            output, _ = module(hidden_states=hidden, attention_mask=None, packed_seq_params=packed)
            output.backward(full_grad.index_select(0, rows[name]))
            grads = {}
            for param_name, param in module.named_parameters():
                assert param.grad is not None, f"{name}: missing grad for {param_name}"
                grad = param.grad.detach().clone()
                if name != "cp1":
                    dist.all_reduce(grad, group=pg.cp)
                grads[param_name] = grad
            results[name] = (output.detach(), hidden.grad.detach(), grads)

        label = f"layer={layer_number}:rope_fusion={apply_rope_fusion}:fused={use_fused}"
        reference_output, reference_dx, reference_grads = results["cp1"]
        for name in ("zigzag", "contiguous"):
            output, dx, grads = results[name]
            _assert_cp_tensor_match(output, reference_output[rows[name]], f"{label}:{name}:output")
            _assert_cp_tensor_match(dx, reference_dx[rows[name]], f"{label}:{name}:dx")
            for param_name, grad in grads.items():
                _assert_cp_tensor_match(
                    grad, reference_grads[param_name], f"{label}:{name}:grad:{param_name}"
                )

        # Zigzag against contiguous in global row order. Per row, both partitions see the same
        # keys in the same order; only kv_proj runs a different row count, so the forward is
        # expected to be (nearly) bitwise and gradients to agree at BF16 level.
        def to_global(name, index):
            local = results[name][index].contiguous()
            gathered = [torch.empty_like(local) for _ in range(cp_size)]
            dist.all_gather(gathered, local, group=pg.cp)
            output = torch.empty(
                (total,) + tuple(local.shape[1:]), dtype=local.dtype, device="cuda"
            )
            for peer, values in enumerate(gathered):
                if name == "zigzag":
                    peer_rows = _zigzag_rows(peer, cp_size, segment_rows, "cuda")
                else:
                    peer_rows = torch.arange(
                        peer * 2 * segment_rows, (peer + 1) * 2 * segment_rows, device="cuda"
                    )
                output[peer_rows] = values
            return output

        for index, what, max_rel_l2, min_cosine in (
            (0, "output", 5e-3, 0.99999),
            (1, "dx", 1e-2, 0.9999),
        ):
            zigzag_global, contiguous_global = to_global("zigzag", index), to_global(
                "contiguous", index
            )
            equal_rows = (zigzag_global == contiguous_global).flatten(1).all(dim=1).float().mean()
            difference = (zigzag_global.double() - contiguous_global.double()).norm()
            rel_l2 = (difference / contiguous_global.double().norm().clamp_min(1e-30)).item()
            cosine = torch.nn.functional.cosine_similarity(
                zigzag_global.double().flatten(), contiguous_global.double().flatten(), dim=0
            ).item()
            if cp_rank == 0:
                print(
                    f"[{label}] zigzag vs contiguous {what}: bitwise-equal rows "
                    f"{equal_rows.item():.4f}, rel L2 {rel_l2:.3e}, cosine {cosine:.9f}"
                )
            assert (
                rel_l2 <= max_rel_l2 and cosine >= min_cosine
            ), f"{label}: zigzag vs contiguous {what}: rel L2 {rel_l2:.3e}, cosine {cosine:.9f}"


def _packed_document_zigzag_rows(padded_lengths, cp_rank, cp_size):
    """Independent oracle for TE's per-document two-chunk THD partition."""
    parts = []
    start = 0
    for length in padded_lengths:
        chunk, remainder = divmod(length, 2 * cp_size)
        assert remainder == 0
        for index in (cp_rank, 2 * cp_size - 1 - cp_rank):
            parts.append(torch.arange(start + index * chunk, start + (index + 1) * chunk))
        start += length
    return torch.cat(parts).cuda()


def _packed_h_error(actual, reference, label, limit, group, sharded):
    """Measure the entire CP group, including the worst single element, under one tolerance."""
    a, b = actual.detach().double(), reference.detach().double()
    assert a.shape == b.shape, label
    delta = a - b
    sums = torch.stack((delta.square().sum(), b.square().sum()))
    absolute = delta.abs().amax()
    if sharded:
        dist.all_reduce(sums, group=group)
        dist.all_reduce(absolute, op=dist.ReduceOp.MAX, group=group)
    relative = (sums[0] / sums[1].clamp_min(1e-30)).sqrt().item()
    if group.rank() == 0:
        print(
            f"[packed-H] group={dist.get_rank()} {label} "
            f"rel_l2={relative:.8g} max_abs={absolute.item():.8g} limit={limit}",
            flush=True,
        )
    assert relative <= limit, f"{label}: relative L2 {relative}, max absolute {absolute.item()}"


@pytest.mark.skipif(_distributed_cp_size() < 2, reason="needs torchrun with >= 2 CUDA ranks")
class TestPackedDSv4ZigzagCPDistributed:
    """Real H attention against independent one-document CP1 forwards and backwards."""

    @pytest.fixture(scope="class", autouse=True)
    def packed_cp(self):
        from tests.unit_tests.test_utilities import Utils

        cp_size = min(16, Utils.world_size)
        if cp_size not in (2, 4, 8, 16) or Utils.world_size % cp_size:
            pytest.skip("requires complete CP2/4/8/16 groups")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        yield cp_size
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer_number", [1, 2], ids=["window_only", "ratio_128"])
    def test_multi_document_forward_backward_and_isolation(self, packed_cp, layer_number):
        from megatron.core.extensions.transformer_engine import HAVE_TE
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
            _build_attention,
        )
        from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
            _copy_module_parameters,
            _dsv4_cp_fused_kernels_available,
            _make_dsv4_cp_config,
            _make_thd_packed_seq_params,
            _ReferenceCPGroup,
        )

        if not HAVE_TE:
            pytest.skip("requires Transformer Engine")
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg.cp = _ReferenceCPGroup()
        group, cp_size = pg.cp, packed_cp
        leader = dist.get_global_rank(group, 0)
        fused = _dsv4_cp_fused_kernels_available()
        # With CP16 the per-document chunks are 1, 3, 5, 11, 26, and 82 rows.
        # Every 128-row compressor block spans multiple chunks, and two docs are shorter
        # than one sliding window. CP groups get different document order and contents.
        lengths = [31, 95, 129, 337, 811, 2603]
        padded = [32, 96, 160, 352, 832, 2624]
        if leader:
            lengths.reverse()
            padded.reverse()
        total = sum(padded)
        local = total // cp_size
        packed_args = dict(
            dsa_indexer_loss_coeff=0.0,
            use_fused_kernels=fused,
            csa_compress_ratios=[0, 128, 0, 128],
            csa_dense_mode=True,
            cuda_graph_impl="none",
        )
        configs = {
            "zigzag": _make_dsv4_cp_config(
                context_parallel_size=cp_size,
                max_seqlen_per_dp_cp_rank=local,
                cp_partition_mode="zigzag",
                dsv4_cp_zigzag=True,
                sequence_packing_scheduler="dp_balanced",
                **packed_args,
            ),
            "contiguous": _make_dsv4_cp_config(
                context_parallel_size=cp_size, max_seqlen_per_dp_cp_rank=local, **packed_args
            ),
            "cp1": _make_dsv4_cp_config(context_parallel_size=1, **packed_args),
        }
        seed = 1234 + layer_number + leader
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        modules = {
            name: _build_attention(
                config, layer_number=layer_number, pg_collection=ref_pg if name == "cp1" else pg
            ).cuda()
            for name, config in configs.items()
        }
        for parameter in modules["zigzag"].parameters():
            dist.broadcast(parameter.data, src=leader, group=group)
        for name in ("contiguous", "cp1"):
            _copy_module_parameters(modules["zigzag"], modules[name])
        generator = torch.Generator(device="cuda").manual_seed(seed + 53)
        shape = (total, 1, configs["cp1"].hidden_size)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        valid = torch.cat([torch.arange(n, device="cuda") < v for n, v in zip(padded, lengths)])
        dy[~valid] = 0
        oracle_y, oracle_dx, oracle_grads = [], [], {}
        start = 0
        for length, span in zip(lengths, padded):
            model = modules["cp1"]
            model.zero_grad(set_to_none=True)
            sample = x[start : start + span].detach().requires_grad_(True)
            packed = _make_thd_packed_seq_params([length], [span])
            out = model(hidden_states=sample, attention_mask=None, packed_seq_params=packed)[0]
            out.backward(dy[start : start + span])
            oracle_y.append(out.detach())
            oracle_dx.append(sample.grad.detach())
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                if name not in oracle_grads:
                    oracle_grads[name] = parameter.grad.detach().float().clone()
                else:
                    oracle_grads[name].add_(parameter.grad.detach().float())
            start += span
        oracle_y = torch.cat(oracle_y)
        oracle_dx = torch.cat(oracle_dx)
        rows = {
            "contiguous": torch.arange(
                group.rank() * local, (group.rank() + 1) * local, device="cuda"
            ),
            "zigzag": _packed_document_zigzag_rows(padded, group.rank(), cp_size),
        }
        for name in ("contiguous", "zigzag"):
            packed = _make_thd_packed_seq_params(lengths, padded)
            packed.cp_partition_mode = name
            indices = rows[name]
            is_valid = valid[indices]
            model = modules[name]
            sample = x[indices].detach().requires_grad_(True)
            out = model(hidden_states=sample, attention_mask=None, packed_seq_params=packed)[0]
            out.backward(dy[indices])
            label = f"cp={cp_size} layer={layer_number} fused={fused} {name}"
            _packed_h_error(
                out[is_valid], oracle_y[indices][is_valid], label + "/y", 0.01, group, True
            )
            _packed_h_error(
                sample.grad[is_valid],
                oracle_dx[indices][is_valid],
                label + "/dX",
                0.02,
                group,
                True,
            )
            for key, parameter in model.named_parameters():
                assert parameter.grad is not None, key
                grad = parameter.grad.detach().float()
                dist.all_reduce(grad, group=group)
                _packed_h_error(grad, oracle_grads[key], label + "/grad/" + key, 0.05, group, False)
            if name == "zigzag":
                # Change exactly one document. Independently seed only its outputs for the VJP.
                # Every other document's forward must remain identical, and dX must be exactly zero.
                target = padded.index(352)
                first = sum(padded[:target])
                here = (indices >= first) & (indices < first + lengths[target])
                other = is_valid & ~here
                perturbed = sample.detach().clone()
                perturbed[here] *= -2
                perturbed.requires_grad_(True)
                changed = model(
                    hidden_states=perturbed, attention_mask=None, packed_seq_params=packed
                )[0]
                upstream = torch.zeros_like(changed)
                upstream[here] = dy[indices][here]
                delta_x = torch.autograd.grad(changed, perturbed, upstream)[0]
                leaks = torch.stack(
                    ((changed[other] != out.detach()[other]).sum(), delta_x[other].count_nonzero())
                )
                changed_rows = (changed[here] != out.detach()[here]).sum()
                dist.all_reduce(leaks, group=group)
                dist.all_reduce(changed_rows, group=group)
                assert not leaks.any() and changed_rows > 0, (
                    label,
                    leaks.tolist(),
                    changed_rows.item(),
                )
                if group.rank() == 0:
                    print(
                        f"[packed-H-isolation] group={leader} layer={layer_number} output_leaks=0 dx_leaks=0",
                        flush=True,
                    )
