# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared-prefix Mamba branch packing and merging (CPU, float64)."""

import pytest
import torch

from megatron.core.ssm.mamba_branch_layout import merge_mamba_branches, pack_mamba_branches


@pytest.mark.parametrize("tail", [0, 2, 3])
def test_mamba_branch_layout_gradients_match_slice_reference(tail):
    lengths = (2, 0, 5)
    values = torch.randn(10, 1, 2, dtype=torch.float64, requires_grad=True)
    reference = values.detach().clone().requires_grad_()
    actual = pack_mamba_branches(values, prefix_len=3, tail_len=tail, completion_lens=lengths)
    branches = reference.new_zeros(tail + 5, 3, 2)
    offset = 3
    for branch, length in enumerate(lengths):
        branches[:tail, branch] = reference[3 - tail : 3, 0]
        branches[tail : tail + length, branch] = reference[offset : offset + length, 0]
        offset += length
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    branches.backward(cotangent)
    torch.testing.assert_close(actual, branches, rtol=0, atol=0)
    torch.testing.assert_close(values.grad, reference.grad, rtol=0, atol=0)
    head = torch.randn(3 - tail, 1, 2, dtype=torch.float64, requires_grad=True)
    branch_input = actual.detach().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda h, b: merge_mamba_branches(h, b, tail_len=tail, completion_lens=lengths),
        (head, branch_input),
    )
