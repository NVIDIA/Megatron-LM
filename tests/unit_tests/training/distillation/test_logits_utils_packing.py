# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from megatron.training.distillation.utils import (
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
