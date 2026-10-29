# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MTP rolling must follow physical ownership, including loss-mask and gradient seams."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import (
    _packed_seq_params_for_local_hsm_roll,
    roll_tensor,
    roll_tensor_precomputed_embeddings,
)
from tests.unit_tests.test_utilities import Utils


def _reference_roll(tensor, docs):
    result = torch.zeros_like(tensor)
    for start, valid_end in docs:
        result[start : valid_end - 1] = tensor[start + 1 : valid_end]
    return result


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("seq_dim", [0, -1])
def test_contiguous_mtp_roll_matches_global_tokens_masks_and_gradients(packed, seq_dim):
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        cp = parallel_state.get_context_parallel_group()
        # Document two crosses the rank boundary; document one has internal padding.
        docs = [(0, 16)] if not packed else [(0, 3), (4, 11), (12, 16)]
        params = (
            None
            if not packed
            else PackedSeqParams(
                qkv_format="thd",
                cp_partition_mode="contiguous",
                cu_seqlens_q=torch.tensor([0, 3, 10, 14], device="cuda", dtype=torch.int32),
                cu_seqlens_q_padded=torch.tensor([0, 4, 12, 16], device="cuda", dtype=torch.int32),
            )
        )
        full = torch.arange(1, 33, device="cuda", dtype=torch.float32).view(16, 2)
        full.requires_grad_()
        start = cp.rank() * 8
        local = full.detach()[start : start + 8].movedim(0, seq_dim).contiguous().requires_grad_()
        actual, expected = local, full
        mask = torch.ones_like(local)
        expected_mask = torch.ones_like(full)
        for _ in range(3):
            actual, total = roll_tensor(
                actual,
                dims=seq_dim,
                cp_group=cp,
                packed_seq_params=params,
                cp_partition_mode="contiguous",
            )
            mask, _ = roll_tensor(
                mask,
                dims=seq_dim,
                cp_group=cp,
                packed_seq_params=params,
                cp_partition_mode="contiguous",
            )
            expected = _reference_roll(expected, docs)
            expected_mask = _reference_roll(expected_mask, docs)
            torch.testing.assert_close(actual.movedim(seq_dim, 0), expected[start : start + 8])
            torch.testing.assert_close(mask.movedim(seq_dim, 0), expected_mask[start : start + 8])
            torch.testing.assert_close(total, actual.sum())
        weights = torch.arange(32, device="cuda", dtype=torch.float32).view(16, 2)
        (actual.movedim(seq_dim, 0) * weights[start : start + 8]).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(local.grad.movedim(seq_dim, 0), full.grad[start : start + 8])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("packed", [False, True])
def test_contiguous_precomputed_embeddings_roll_across_tp_cp(packed):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    try:
        cp = parallel_state.get_context_parallel_group()
        tp = parallel_state.get_tensor_model_parallel_group()
        params = (
            None
            if not packed
            else PackedSeqParams(
                qkv_format="thd",
                cp_partition_mode="contiguous",
                cu_seqlens_q=torch.tensor([0, 6, 16], device="cuda", dtype=torch.int32),
            )
        )
        docs = [(0, 16)] if not packed else [(0, 6), (6, 16)]
        full = torch.arange(32.0, device="cuda").view(16, 1, 2).requires_grad_()
        start = (cp.rank() * tp.size() + tp.rank()) * 4
        local = full.detach()[start : start + 4].clone().requires_grad_()
        actual, _ = roll_tensor_precomputed_embeddings(
            local,
            sp_group=tp,
            cp_group=cp,
            packed_seq_params=params,
            cp_partition_mode="contiguous",
            return_sum=False,
        )
        expected = _reference_roll(full, docs)
        torch.testing.assert_close(actual, expected[start : start + 4])
        weights = torch.arange(1.0, 33.0, device="cuda").view(16, 1, 2)
        (actual * weights[start : start + 4]).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(local.grad, full.grad[start : start + 4])
        if packed:
            local_params = _packed_seq_params_for_local_hsm_roll(params, 4, cp, tp)
            hsm, _ = roll_tensor(local.detach(), dims=0, packed_seq_params=local_params)
            # HSM's local-only roll intentionally clears every shard's final token.
            expected_hsm = expected.detach()[start : start + 4].clone()
            expected_hsm[-1] = 0
            torch.testing.assert_close(hsm, expected_hsm)
    finally:
        Utils.destroy_model_parallel()
