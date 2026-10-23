# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from megatron.training.distillation.utils import (
    pad_and_stack_cu_seqlens,
    reassemble_cp_sequence,
    slice_tensor_for_cp_rank,
    unpack_indices,
    v2_pack_indices,
    v2_unpack_indices,
)

# ---------------------------------------------------------------------------
# v2 17th-bit index packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 3, 128), (5, 1, 7), (1, 1, 1), (2, 6, 65)])
def test_v2_pack_unpack_round_trip(shape):
    torch.manual_seed(0)
    indices = torch.randint(0, 2**17, shape, dtype=torch.long)
    low_bits, packed_bit_17 = v2_pack_indices(indices)
    restored = v2_unpack_indices(low_bits, packed_bit_17)
    assert torch.equal(restored, indices)


def test_v2_pack_indices_boundary_values():
    shape = (2, 3, 4)
    zeros = torch.zeros(shape, dtype=torch.long)
    low_bits, bit_17 = v2_pack_indices(zeros)
    assert torch.equal(v2_unpack_indices(low_bits, bit_17), zeros)
    assert not torch.from_numpy(np.unpackbits(bit_17.numpy())).any()

    all_high = torch.full(shape, 2**16, dtype=torch.long)
    low_bits, bit_17 = v2_pack_indices(all_high)
    assert torch.equal(v2_unpack_indices(low_bits, bit_17), all_high)
    assert torch.all(low_bits == 0)

    all_low_max = torch.full(shape, 2**16 - 1, dtype=torch.long)
    low_bits, bit_17 = v2_pack_indices(all_low_max)
    assert torch.equal(v2_unpack_indices(low_bits, bit_17), all_low_max)
    assert torch.all(low_bits == 2**16 - 1)


def test_v2_pack_indices_must_be_packed_at_monolith_level():
    torch.manual_seed(1)
    mb0 = torch.randint(0, 2**17, (3, 1, 5), dtype=torch.long)
    mb1 = torch.randint(0, 2**17, (3, 1, 5), dtype=torch.long)

    # Pack each microbatch separately and concatenate the packed bytes.
    _, bit_17_mb0 = v2_pack_indices(mb0)
    _, bit_17_mb1 = v2_pack_indices(mb1)
    packed_separately = torch.cat([bit_17_mb0, bit_17_mb1])

    # Pack the already-concatenated monolith once, as the code always does.
    monolith = torch.cat([mb0, mb1], dim=1)
    _, bit_17_monolith = v2_pack_indices(monolith)

    assert not torch.equal(packed_separately, bit_17_monolith)


def test_unpack_indices_v1_legacy():
    torch.manual_seed(2)
    indices = torch.randint(0, 2**17, (4, 1, 6), dtype=torch.long)
    low_bits = (indices & 0xFFFF).to(torch.uint16)
    bit_17 = (indices >> 16).bool()
    restored = unpack_indices(low_bits, bit_17)
    assert torch.equal(restored, indices)


# ---------------------------------------------------------------------------
# CP zigzag slicing / reassembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4])
@pytest.mark.parametrize("chunk_size", [1, 5])
def test_cp_slice_reassemble_round_trip(cp_size, chunk_size):
    seq_len = 2 * cp_size * chunk_size
    torch.manual_seed(3)
    full = torch.randn(seq_len, 2, 4)

    shards = [slice_tensor_for_cp_rank(full, rank, cp_size) for rank in range(cp_size)]
    reassembled = reassemble_cp_sequence(shards)
    assert torch.equal(reassembled, full)


def test_cp_slice_cp_size_one_is_identity():
    tensor = torch.randn(6, 2)
    result = slice_tensor_for_cp_rank(tensor, 0, 1)
    assert result is tensor


def test_cp_slice_rejects_non_divisible_seq_len():
    tensor = torch.randn(5, 2)
    with pytest.raises(ValueError):
        slice_tensor_for_cp_rank(tensor, 0, 2)


def test_cp_reassemble_rejects_mismatched_shapes():
    shard0 = torch.randn(4, 2)
    shard1 = torch.randn(4, 3)
    with pytest.raises(ValueError):
        reassemble_cp_sequence([shard0, shard1])


# ---------------------------------------------------------------------------
# CP zigzag slicing / reassembly -- packed (cu_seqlens / --sft) sequences
#
# Real Megatron-Core CP partitioning under packed data zigzags each document
# independently (_get_batch_on_this_cp_rank_per_document_balancing,
# megatron/core/utils.py), not the whole sequence at once. These tests cover
# slice_tensor_for_cp_rank/reassemble_cp_sequence's cu_seqlens-aware path,
# which mirrors that per-document semantics for CP-aware teacher-logit save
# and load.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_cp_slice_reassemble_round_trip_packed_1d(cp_size):
    torch.manual_seed(4)
    # Three documents of different lengths, each divisible by 2 * cp_size.
    doc_lens = [2 * cp_size, 4 * cp_size, 2 * cp_size]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0)))
    seq_len = int(cu_seqlens[-1])
    full = torch.randn(seq_len, 3, 4)

    shards = [
        slice_tensor_for_cp_rank(full, rank, cp_size, cu_seqlens=cu_seqlens)
        for rank in range(cp_size)
    ]
    reassembled = reassemble_cp_sequence(shards, cu_seqlens=cu_seqlens)
    assert torch.equal(reassembled, full)


def test_cp_slice_reassemble_round_trip_packed_2d_per_sample():
    torch.manual_seed(5)
    cp_size = 2
    # Two samples in the same load microbatch, each packed differently.
    cu_seqlens = torch.tensor(
        [
            [0, 8, 24, 24],  # sample 0: docs [0,8) [8,24), trailing pad repeat
            [0, 24, 24, 24],  # sample 1: one doc [0,24)
        ]
    )
    full = torch.randn(24, 2, 4)

    shards = [
        slice_tensor_for_cp_rank(full, rank, cp_size, cu_seqlens=cu_seqlens)
        for rank in range(cp_size)
    ]
    reassembled = reassemble_cp_sequence(shards, cu_seqlens=cu_seqlens)
    assert torch.equal(reassembled, full)

    # Sample 1 (single full-length document) must match the plain
    # whole-sequence zigzag exactly.
    plain = slice_tensor_for_cp_rank(full[:, 1:2], 0, cp_size)
    assert torch.equal(shards[0][:, 1:2], plain)


def test_cp_slice_packed_matches_hand_verified_reference():
    # Fixed regression anchor: hand-derived per-document zigzag positions for
    # cp_size=2, documents [0,8) and [8,24), confirmed against a standalone
    # repro run of this exact production code during the original bug
    # investigation (see PR description / commit history for derivation).
    cp_size = 2
    cu_seqlens = torch.tensor([0, 8, 24])
    full = torch.arange(24).float().view(24, 1, 1)

    rank0 = slice_tensor_for_cp_rank(full, 0, cp_size, cu_seqlens=cu_seqlens)
    rank1 = slice_tensor_for_cp_rank(full, 1, cp_size, cu_seqlens=cu_seqlens)

    assert rank0.squeeze().long().tolist() == [0, 1, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23]
    assert rank1.squeeze().long().tolist() == [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19]


def test_cp_slice_packed_rejects_non_divisible_document():
    cu_seqlens = torch.tensor([0, 5, 24])  # first document length 5, not divisible by 2*cp_size=4
    full = torch.randn(24, 1, 1)
    with pytest.raises(ValueError):
        slice_tensor_for_cp_rank(full, 0, 2, cu_seqlens=cu_seqlens)


def test_cp_slice_packed_single_document_matches_unpacked_path():
    """A single document spanning the whole sequence must match the
    cu_seqlens=None whole-sequence zigzag exactly."""
    torch.manual_seed(6)
    cp_size = 3
    seq_len = 2 * cp_size * 5
    full = torch.randn(seq_len, 1, 4)
    cu_seqlens = torch.tensor([0, seq_len])

    for rank in range(cp_size):
        packed = slice_tensor_for_cp_rank(full, rank, cp_size, cu_seqlens=cu_seqlens)
        unpacked = slice_tensor_for_cp_rank(full, rank, cp_size)
        assert torch.equal(packed, unpacked)


def test_cp_reshard_save_cp_ne_load_cp_matches_true_per_document_partition():
    """Reproduces the original bug repro as a regression test: reassembling
    at one CP size and reslicing at a different CP size must exactly match
    each load-rank's true per-document partition of the original sequence,
    not just round-trip with itself."""
    torch.manual_seed(7)
    cp_save, cp_load = 2, 4
    cu_seqlens = torch.tensor([0, 8, 24])  # both documents divisible by 2*cp_load=8
    seq_len = 24
    full = torch.arange(seq_len).float().view(seq_len, 1, 1)

    save_shards = [
        slice_tensor_for_cp_rank(full, r, cp_save, cu_seqlens=cu_seqlens) for r in range(cp_save)
    ]
    reconstructed = reassemble_cp_sequence(save_shards, cu_seqlens=cu_seqlens)
    assert torch.equal(reconstructed, full)

    for r in range(cp_load):
        reloaded = slice_tensor_for_cp_rank(reconstructed, r, cp_load, cu_seqlens=cu_seqlens)
        true_local = slice_tensor_for_cp_rank(full, r, cp_load, cu_seqlens=cu_seqlens)
        assert torch.equal(reloaded, true_local)


def test_pad_and_stack_cu_seqlens():
    rows_a = torch.tensor([[0, 8, 24]])
    rows_b = torch.tensor([[0, 24]])
    stacked = pad_and_stack_cu_seqlens([rows_a, rows_b])
    assert stacked.tolist() == [[0, 8, 24], [0, 24, 24]]


def test_pad_and_stack_cu_seqlens_equal_width_no_padding_needed():
    rows_a = torch.tensor([[0, 8, 24]])
    rows_b = torch.tensor([[0, 12, 24]])
    stacked = pad_and_stack_cu_seqlens([rows_a, rows_b])
    assert stacked.tolist() == [[0, 8, 24], [0, 12, 24]]
