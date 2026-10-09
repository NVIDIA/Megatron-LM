# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SharedPrefixLayout metadata against the dense rows it represents (CPU)."""

import torch

from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout


def test_layout_prompt_multiplicities_equal_dense_gather_counts():
    layout = SharedPrefixLayout(3, (2, 5, 1))
    indices = torch.cat(layout.dense_branch_indices("cpu"))
    expected = torch.bincount(indices, minlength=layout.total_len).float()
    torch.testing.assert_close(
        layout.padded_token_multiplicities(layout.total_len, "cpu"), expected, rtol=0, atol=0
    )
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 3]
