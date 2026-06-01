# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsv4_cp import (
    _build_compressor_prep_compact,
    _contiguous_cp_range,
    _final_design_flat_idxs,
    _final_design_flat_idxs_for_indexer_loss,
    _pack_design_kv_full,
    exchange_left_boundary_tensor,
)


def test_contiguous_cp_range_requires_even_total_capacity():
    cu_seqlens = torch.tensor([0, 7], dtype=torch.int32)

    assert _contiguous_cp_range(cu_seqlens, cp_size=1, cp_rank=0) == (0, 7)

    with pytest.raises(RuntimeError, match="padded_total_tokens % cp_size"):
        _contiguous_cp_range(cu_seqlens, cp_size=2, cp_rank=0)


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_left_boundary_tensor(local, d_window=2, cp_group=None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_compressor_prep_compact_uses_only_complete_boundary_groups():
    cu_seqlens = torch.tensor([0, 192, 512], dtype=torch.int32)
    global_start = 384
    l_local = 128
    d_window = 128
    ratio = 128
    d_comp = 128
    hidden_size = 2

    local_positions = torch.arange(global_start, global_start + l_local, dtype=torch.float32)
    hidden_local = local_positions[:, None].repeat(1, hidden_size)
    boundary_positions = torch.arange(global_start - d_window, global_start, dtype=torch.float32)
    boundary_hidden = boundary_positions[:, None].repeat(1, hidden_size)

    hidden_compact, cu_compact, seq_ids, comp_ids, valid, c_cap = _build_compressor_prep_compact(
        hidden_local,
        boundary_hidden,
        cu_seqlens,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
    )

    expected_positions = torch.arange(320, 448, dtype=torch.float32)[:, None].repeat(
        1, hidden_size
    )
    assert c_cap == 2
    assert torch.equal(cu_compact.cpu(), torch.tensor([0, 0, 128], dtype=torch.int32))
    assert torch.equal(hidden_compact[:128].cpu(), expected_positions)
    assert torch.equal(hidden_compact[128:].cpu(), torch.zeros_like(hidden_compact[128:]).cpu())
    assert torch.equal(seq_ids.cpu(), torch.tensor([1, -1], dtype=torch.int32))
    assert torch.equal(comp_ids.cpu(), torch.tensor([1, -1], dtype=torch.int32))
    assert torch.equal(valid.cpu(), torch.tensor([True, False]))


def test_kv_full_pack_keeps_per_sequence_window_then_compressed_layout():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    global_start = 3
    l_local = 3
    d_window = 2

    kv_local = torch.tensor([[103.0], [104.0], [105.0]])
    boundary_kv = torch.tensor([[101.0], [102.0]])
    compressed_rank_major = torch.tensor([[900.0], [901.0], [902.0]])
    seq_ids_rank_major = torch.tensor([0, 1, 1], dtype=torch.int32)
    comp_ids_rank_major = torch.tensor([0, 0, 1], dtype=torch.int32)
    valid_rank_major = torch.tensor([True, True, False])

    kv_full, window_map, compressed_map = _pack_design_kv_full(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        global_start,
        l_local,
        d_window,
    )

    assert window_map == {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    assert compressed_map == {(0, 0): 3, (1, 0): 6}
    expected_prefix = torch.tensor([101.0, 102.0, 103.0, 900.0, 104.0, 105.0, 901.0])
    assert torch.equal(kv_full[:7].squeeze(-1), expected_prefix)
    assert torch.equal(kv_full[7:], torch.zeros_like(kv_full[7:]))


def test_final_design_flat_idxs_respect_sequence_local_compressed_ids():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}

    topk_idxs, topk_length = _final_design_flat_idxs(
        cu_seqlens,
        global_start=3,
        l_local=3,
        window_size=2,
        ratio=2,
        device=torch.device("cpu"),
        window_map=window_map,
        compressed_map=compressed_map,
        max_n_compressed=2,
    )

    expected = torch.tensor(
        [
            [1, 2, 3, -1],
            [4, -1, -1, -1],
            [4, 5, 6, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(topk_idxs, expected)
    assert torch.equal(topk_length, torch.tensor([3, 1, 3], dtype=torch.int32))


def test_final_design_flat_idxs_for_indexer_loss_preserves_rank_major_ids():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}
    logical_ids = torch.tensor([[0, -1], [-1, -1], [0, -1]], dtype=torch.int32)
    rank_major_ids = torch.tensor([[5, -1], [-1, -1], [7, -1]], dtype=torch.int32)

    topk_idxs, lowered_rank_major_ids = _final_design_flat_idxs_for_indexer_loss(
        cu_seqlens,
        global_start=3,
        l_local=3,
        window_size=2,
        device=torch.device("cpu"),
        window_map=window_map,
        compressed_map=compressed_map,
        indexer_topk_compressed_logical_ids=logical_ids,
        indexer_topk_rank_major_ids=rank_major_ids,
    )

    expected = torch.tensor(
        [
            [3, -1, 1, 2],
            [-1, -1, 4, -1],
            [6, -1, 4, 5],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(topk_idxs, expected)
    assert torch.equal(lowered_rank_major_ids, rank_major_ids)
