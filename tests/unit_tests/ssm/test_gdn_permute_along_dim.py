# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.gated_delta_net.common import (
    _build_head_perm_and_inverse_for_split_sections,
    _build_thd_cp_a2a_perm,
    _PermuteAlongDim,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _assert_matches_index_select(x, perm, inverse_perm, dim):
    reference_input = x.clone().requires_grad_(True)
    permuted_input = x.clone().requires_grad_(True)
    grad_output = torch.randn_like(x.index_select(dim, perm))

    reference = reference_input.index_select(dim, perm)
    permuted = _PermuteAlongDim.apply(permuted_input, perm, inverse_perm, dim)
    reference.backward(grad_output)
    permuted.backward(grad_output)

    assert torch.equal(permuted, reference)
    assert torch.equal(permuted_input.grad, reference_input.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_head_perm_matches_index_select(dtype):
    perm, inverse_perm = _build_head_perm_and_inverse_for_split_sections(
        (64, 64, 32, 8), 4, torch.device(DEVICE)
    )
    x = torch.randn(96, 2, perm.numel(), device=DEVICE, dtype=dtype)
    _assert_matches_index_select(x, perm, inverse_perm, -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_thd_cp_perm_matches_index_select(dtype):
    cu_seqlens = torch.tensor([0, 64, 192, 256], device=DEVICE, dtype=torch.int32)
    idx, inv = _build_thd_cp_a2a_perm(cu_seqlens, 4, int(cu_seqlens[-1]))
    x = torch.randn(idx.numel(), 1, 48, device=DEVICE, dtype=dtype)
    _assert_matches_index_select(x, idx, inv, 0)
    _assert_matches_index_select(x, inv, idx, 0)
