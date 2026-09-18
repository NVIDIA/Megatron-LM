# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed MTP token/mask and differentiable boundary parity across prediction depths."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel.sequence_roll import roll_contiguous, roll_contiguous_fields
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_contiguous_roll_matches_whole_pack_and_gradients(cp_size, monkeypatch):
    if Utils.world_size < cp_size:
        pytest.skip(f"requires {cp_size} ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        group = ProcessGroupCollection.use_mpu_process_groups().cp
        total, local = 96, 96 // cp_size
        physical = torch.tensor([0, 13, 13, 63, 96], device="cuda", dtype=torch.int32)
        real = torch.tensor([0, 11, 11, 56, 84], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(cu_seqlens_q=real, cu_seqlens_q_padded=physical, qkv_format="thd")
        tokens = torch.arange(total, device="cuda").view(1, -1)
        rows = slice(group.rank() * local, (group.rank() + 1) * local)
        value = tokens[:, rows]
        padding = value == 0
        calls = []
        original = dist.batch_isend_irecv

        def record(ops):
            calls.append(len(ops))
            return original(ops)

        monkeypatch.setattr(dist, "batch_isend_irecv", record)
        expected, expected_padding = tokens, tokens == 0
        for _ in range(3):
            value, padding = roll_contiguous_fields(
                (value, padding), group, packed, fill_values=(0, True)
            )
            expected = roll_contiguous(expected, -1, None, packed)
            expected_padding = roll_contiguous(expected_padding, -1, None, packed, True)
            assert torch.equal(value, expected[:, rows])
            assert torch.equal(padding, expected_padding[:, rows])
        assert len(calls) == (3 if cp_size > 1 else 0)
        # HSM carries gradients, so its reversed P2P must return the boundary gradient.
        hidden = tokens[:, rows].float().requires_grad_()
        full = tokens.float().requires_grad_()
        roll_contiguous(hidden, -1, group, packed).square().sum().backward()
        roll_contiguous(full, -1, None, packed).square().sum().backward()
        torch.testing.assert_close(hidden.grad, full.grad[:, rows])
    finally:
        Utils.destroy_model_parallel()
