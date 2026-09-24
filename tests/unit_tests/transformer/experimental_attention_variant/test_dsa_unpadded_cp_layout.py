# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unpadded context-parallel token positions and all-gather permutations."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_layout


@pytest.mark.parametrize("cp_size", [1, 2, 8, 64])
@pytest.mark.parametrize("local_rows", [0, 2, 6, 128])
def test_unpadded_layout_matches_packed_builder_on_every_rank(cp_size, local_rows):
    """Check every rank, changing geometry, and the complete inverse permutation."""
    device = torch.device("cpu")
    cu = torch.tensor([0, local_rows * cp_size], dtype=torch.int32)
    _, expected_reorder = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
        cu,
        cu,
        cp_size,
        0,
        device,
        local_output_size=local_rows,
        key_local_output_size=local_rows,
        global_output_size=local_rows * cp_size,
        query_cu_seqlens_cover_output=True,
        key_cu_seqlens_cover_output=True,
    )
    reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(local_rows, cp_size, device)
    gathered_positions = []
    for rank in range(cp_size):
        positions = dsa_layout.build_zigzag_cp_local_positions(
            local_rows * cp_size, cp_size, rank, device
        )
        expected_positions = dsa_layout.build_packed_allgather_cp_local_positions(
            cu, cp_size, rank, device, local_rows, cu_seqlens_cover_output=True
        )
        torch.testing.assert_close(positions, expected_positions, rtol=0, atol=0)
        torch.testing.assert_close(reorder, expected_reorder, rtol=0, atol=0)
        assert positions.dtype == reorder.dtype == torch.int64
        assert positions.is_contiguous() and reorder.is_contiguous()
        gathered_positions.append(positions)
    restored = torch.cat(gathered_positions).index_select(0, expected_reorder)
    torch.testing.assert_close(restored, torch.arange(local_rows * cp_size), rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [1, 2, 8, 64])
def test_unpadded_reorder_preserves_values_and_gradients(cp_size):
    """The inverse permutation must route each token's gradient back to its owner."""
    local_rows = 6
    device = torch.device("cpu")
    gathered_positions = torch.cat(
        [
            dsa_layout.build_zigzag_cp_local_positions(local_rows * cp_size, cp_size, rank, device)
            for rank in range(cp_size)
        ]
    )
    reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(local_rows, cp_size, device)
    global_values = torch.arange(local_rows * cp_size * 3, dtype=torch.float64).reshape(-1, 3)
    gathered_values = global_values.index_select(0, gathered_positions).requires_grad_()
    restored = gathered_values.index_select(0, reorder)
    torch.testing.assert_close(restored, global_values, rtol=0, atol=0)
    gradient = global_values.flip(0) / 7
    restored.backward(gradient)
    torch.testing.assert_close(
        gathered_values.grad, gradient.index_select(0, gathered_positions), rtol=0, atol=0
    )


@pytest.mark.parametrize("local_rows,cp_size", [(4, 0), (-2, 2), (3, 2)])
def test_unpadded_reorder_rejects_invalid_geometry(local_rows, cp_size):
    with pytest.raises(ValueError, match="Zigzag CP"):
        dsa_layout.build_zigzag_allgather_cp_key_reorder(local_rows, cp_size, torch.device("cpu"))


def test_unpadded_cp1_allows_odd_rows():
    positions = dsa_layout.build_zigzag_cp_local_positions(3, 1, 0, torch.device("cpu"))
    reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(3, 1, torch.device("cpu"))
    torch.testing.assert_close(positions, torch.arange(3), rtol=0, atol=0)
    torch.testing.assert_close(reorder, torch.arange(3), rtol=0, atol=0)


def test_unpadded_reorder_matches_independent_chunk_order():
    # CP4 ranks own paired chunks (0, 7), (1, 6), (2, 5), (3, 4).
    gathered_positions = torch.tensor([0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9])
    reorder = dsa_layout.build_zigzag_allgather_cp_key_reorder(4, 4, torch.device("cpu"))
    torch.testing.assert_close(
        reorder,
        torch.tensor([0, 1, 4, 5, 8, 9, 12, 13, 14, 15, 10, 11, 6, 7, 2, 3]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(gathered_positions[reorder], torch.arange(16), rtol=0, atol=0)
