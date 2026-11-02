# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SharedPrefixLayout metadata against the dense rows it represents (CPU)."""

import pytest
import torch

from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout


def test_layout_prompt_multiplicities_equal_dense_gather_counts():
    layout = SharedPrefixLayout(3, (2, 5, 1))
    indices = torch.cat(layout.dense_branch_indices("cpu"))
    expected = torch.bincount(indices, minlength=layout.total_len).float()
    torch.testing.assert_close(
        layout.padded_token_multiplicities(layout.total_len, "cpu"), expected, rtol=0, atol=0
    )
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 3]


def test_cp_local_indices_follow_the_zigzag_split():
    """Rank r of C owns chunks r and 2C-r-1 of 2C; together the ranks own every index once."""
    assert SharedPrefixLayout.cp_local_indices(8, 2, 0, "cpu").tolist() == [0, 1, 6, 7]
    assert SharedPrefixLayout.cp_local_indices(8, 2, 1, "cpu").tolist() == [2, 3, 4, 5]
    for cp_size in (1, 2, 4):
        owned = [
            SharedPrefixForestLayout.cp_local_indices(16, cp_size, rank, "cpu")
            for rank in range(cp_size)
        ]
        assert all(indices.dtype == torch.long for indices in owned)
        assert torch.equal(torch.cat(owned).sort().values, torch.arange(16))


@pytest.mark.parametrize(
    "length, cp_size, cp_rank, match",
    [
        (8, 0, 0, "invalid CP geometry"),
        (8, 2, -1, "invalid CP geometry"),
        (8, 2, 2, "invalid CP geometry"),
        (10, 4, 0, r"divisible by 2 \* cp_size"),
    ],
)
def test_cp_local_indices_reject_invalid_geometry(length, cp_size, cp_rank, match):
    with pytest.raises(ValueError, match=match):
        SharedPrefixLayout.cp_local_indices(length, cp_size, cp_rank, "cpu")
