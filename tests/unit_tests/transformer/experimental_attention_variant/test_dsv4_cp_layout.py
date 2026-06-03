# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    build_compressor_prep_compact,
    build_cp_flat_idxs,
    build_cp_flat_idxs_for_indexer_loss,
    build_cp_indexer_topk_inputs,
    contiguous_cp_partition,
    exchange_left_boundary_tensor,
    pack_cp_kv_full,
)


def test_contiguous_cp_partition_requires_padded_total_divisible_by_cp_size():
    """Validate the fixed-size padded contiguous CP partition contract.

    Expected: a single CP rank owns the full padded range, while a padded token
    count that is not divisible by cp_size raises before CP layout construction builds
    local layouts. A failure here means CP could silently proceed with uneven
    rank padded-token ranges.
    """
    cu_seqlens_padded = torch.tensor([0, 7], dtype=torch.int32)

    assert contiguous_cp_partition(cu_seqlens_padded, cp_size=1, cp_rank=0) == (0, 7)

    with pytest.raises(RuntimeError, match="padded_total_tokens % cp_size"):
        contiguous_cp_partition(cu_seqlens_padded, cp_size=2, cp_rank=0)


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP boundary exchange contract.

    Expected: with cp_group=None, the fixed left boundary is zero-filled and its
    backward path contributes no gradient to the local tensor. A failure here
    means the CP path could invent boundary tokens or bogus local gradients.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_left_boundary_tensor(local, d_window=2, cp_group=None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_compressor_prep_compact_uses_only_complete_boundary_groups():
    """Validate compressor-prep compacting across a sequence boundary.

    Expected: for ratio=128, only the complete visible group [320, 448) is
    compacted; the partial boundary group [192, 320) is skipped and unused
    fixed capacity remains padded. A failure here means compressor prep may read
    outside the legal boundary window or emit wrong compressed metadata.
    """
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

    hidden_compact, cu_compact, seq_ids, comp_ids, valid, c_cap = build_compressor_prep_compact(
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
    """Validate the post-all-gather KV full packing order.

    Expected: each active sequence writes its local/boundary window first, then
    valid compressed entries, while invalid compressed entries are skipped and
    tail capacity stays zero. A failure here means final lowered indices could
    point at the wrong KV rows.
    """
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

    kv_full, window_map, compressed_map = pack_cp_kv_full(
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


def test_build_cp_indexer_topk_inputs_uses_local_trapezoid_cu_seqlens():
    """Validate CP-local indexer top-k input construction for a split sequence.

    Expected: the helper keeps only this rank's Q rows but builds a longer
    compressed-KV prefix for the same sequence, producing different Q/KV
    cu_seqlens. A failure here means the THD top-k kernel cannot receive the
    local trapezoid mask shape needed after CP partitioning.
    """
    ratio = 4
    cu_seqlens_q = torch.tensor([0, 32], dtype=torch.int32)
    cu_seqlens_compressed = torch.tensor([0, 8], dtype=torch.int32)
    q_indexer_local = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2)
    weights_indexer_local = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    k_indexer_seq_major = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    (
        q_topk,
        k_topk,
        weights_topk,
        cu_q_topk,
        cu_k_topk,
        max_q,
        max_k,
        local_row_ids,
    ) = build_cp_indexer_topk_inputs(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
        global_start=16,
        l_local=8,
        ratio=ratio,
    )

    assert torch.equal(q_topk, q_indexer_local)
    assert torch.equal(weights_topk, weights_indexer_local)
    assert torch.equal(k_topk, k_indexer_seq_major[:6])
    assert torch.equal(cu_q_topk, torch.tensor([0, 8], dtype=torch.int32))
    assert torch.equal(cu_k_topk, torch.tensor([0, 6], dtype=torch.int32))
    assert max_q == 8
    assert max_k == 6
    assert torch.equal(local_row_ids, torch.arange(8, dtype=torch.long))


def test_build_cp_flat_idxs_respect_sequence_local_compressed_ids():
    """Validate final idx lowering for the normal sparse-attention path.

    Expected: window ids and visible compressed ids lower to kv_full flat ids in
    the same sequence only, and topk_length counts valid entries per row. A
    failure here means sparse attention could attend across sequence boundaries
    or use an incorrect compact length.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}

    topk_idxs, topk_length = build_cp_flat_idxs(
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


def test_build_cp_flat_idxs_for_indexer_loss_preserves_rank_major_ids():
    """Validate final idx lowering for the sparse indexer-loss path.

    Expected: compressed top-k columns are lowered before window columns, and
    the rank-major compressed ids stay aligned with the selected compressed
    entries. A failure here means the fused sparse-loss kernel could compare
    indexer scores against the wrong compressed KV entries.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}
    logical_ids = torch.tensor([[0, -1], [-1, -1], [0, -1]], dtype=torch.int32)
    rank_major_ids = torch.tensor([[5, -1], [-1, -1], [7, -1]], dtype=torch.int32)

    topk_idxs, lowered_rank_major_ids = build_cp_flat_idxs_for_indexer_loss(
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
