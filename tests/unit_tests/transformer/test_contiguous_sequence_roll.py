# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Packed MTP token/mask and differentiable boundary parity across prediction depths."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel.sequence_roll import roll_contiguous, roll_contiguous_fields
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import roll_tensor
from tests.unit_tests.test_utilities import Utils


def test_roll_tensor_fill_value_contract():
    """Nonzero fill is explicit and applies only to the contiguous layout."""
    tensor = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="only with cp_layout='contiguous'"):
        roll_tensor(tensor, cp_layout="zigzag", fill_value=True)
    rolled, total = roll_tensor(tensor, cp_layout="contiguous", fill_value=True, return_sum=False)
    assert torch.equal(rolled, torch.tensor([[False, False, False, True]]))
    assert total is None


@pytest.mark.parametrize(
    "cp_size,sp_size",
    [(1, 1), (1, 2), (1, 4), (2, 1), (4, 1), (2, 2)],
    ids=["cp1-sp1", "cp1-sp2", "cp1-sp4", "cp2-sp1", "cp4-sp1", "cp2-sp2"],
)
@pytest.mark.parametrize("layout", ["dense", "packed", "trailing", "empty"])
def test_contiguous_roll_matches_whole_pack_and_gradients(cp_size, sp_size, monkeypatch, layout):
    if Utils.world_size % (cp_size * sp_size):
        pytest.skip(f"requires a world size divisible by CP{cp_size} x SP{sp_size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=sp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        group, sp = groups.cp, groups.tp
        total, local = 96, 96 // cp_size
        physical = torch.tensor([0, 13, 13, 63, 96], device="cuda", dtype=torch.int32)
        real = torch.tensor([0, 11, 11, 56, 84], device="cuda", dtype=torch.int32)
        if layout == "trailing":
            physical, real = physical[:2], real[:2]
        elif layout == "empty":
            physical, real = physical[:1], real[:1]
        packed = PackedSeqParams(cu_seqlens_q=real, cu_seqlens_q_padded=physical, qkv_format="thd")
        if layout == "dense":
            packed = None
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
            value, absent, padding = roll_contiguous_fields(
                (value, None, padding), group, packed, fill_values=(0, 0, True)
            )
            assert absent is None
            expected = roll_contiguous(expected, -1, None, packed)
            expected_padding = roll_contiguous(expected_padding, -1, None, packed, True)
            assert torch.equal(value, expected[:, rows])
            assert torch.equal(padding, expected_padding[:, rows])
        assert len(calls) == (3 if cp_size > 1 else 0)
        # HSM carries gradients, so its reversed P2P must return the boundary gradient.
        local //= sp_size
        owner = group.rank() * sp_size + sp.rank()
        rows = slice(owner * local, (owner + 1) * local)
        hidden = tokens[:, rows].float().requires_grad_()
        full = tokens.float().requires_grad_()
        output = roll_contiguous(hidden, -1, group, packed, fill_value=7, sp_group=sp)
        expected = roll_contiguous(full, -1, None, packed, fill_value=7)
        torch.testing.assert_close(output, expected[:, rows], atol=0, rtol=0)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(hidden.grad, full.grad[:, rows])
    finally:
        Utils.destroy_model_parallel()


def test_dense_fields_do_not_build_a_boundary_mask():
    tokens = torch.arange(1, 9).view(1, -1)
    padding = torch.zeros_like(tokens, dtype=torch.bool)
    with patch("torch.arange", wraps=torch.arange) as arange:
        actual, absent, actual_padding = roll_contiguous_fields(
            (tokens, None, padding), None, fill_values=(0, 0, True)
        )
    assert absent is None
    torch.testing.assert_close(actual, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]]))
    torch.testing.assert_close(actual_padding, torch.tensor([[False] * 7 + [True]]))
    arange.assert_not_called()
