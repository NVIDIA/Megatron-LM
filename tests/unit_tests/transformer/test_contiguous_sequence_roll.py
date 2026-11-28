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
from tests.unit_tests.process_group_test_utils import cleanup_process_groups
from tests.unit_tests.test_utilities import Utils


def test_roll_tensor_fill_value_contract():
    """Nonzero fill is explicit and applies only to the contiguous layout."""
    tensor = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="only with cp_layout='contiguous'"):
        roll_tensor(tensor, cp_layout="zigzag", fill_value=True)
    rolled, total = roll_tensor(tensor, cp_layout="contiguous", fill_value=True, return_sum=False)
    assert torch.equal(rolled, torch.tensor([[False, False, False, True]]))
    assert total is None


def _reference_roll(tensor, params, dims=-1, fill_value=0):
    """Shift within each full-sequence segment independently of the distributed roll."""
    tensor = tensor.movedim(dims, 0)
    if params is None:
        starts, lengths = [0], [tensor.size(0)]
    else:
        physical = params.cu_seqlens_q_padded
        if physical is None:
            physical = params.cu_seqlens_q
        starts = physical.tolist()
        lengths = params.cu_seqlens_q.diff().tolist() + [tensor.size(0) - starts[-1]]
    result = torch.full_like(tensor, fill_value)
    for start, length in zip(starts, lengths):
        if length > 1:
            result[start : start + length - 1] = tensor[start + 1 : start + length]
    return result.movedim(0, dims)


@pytest.mark.parametrize(
    "cp_size,sp_size,dims",
    [(1, 1, 0), (1, 2, 1), (1, 4, -1), (2, 1, -1), (4, 1, 0), (2, 2, 1)],
    ids=["cp1-sp1", "cp1-sp2", "cp1-sp4", "cp2-sp1", "cp4-sp1", "cp2-sp2"],
)
@pytest.mark.parametrize("layout", ["dense", "packed", "padded", "trailing", "empty"])
def test_contiguous_roll_matches_whole_pack_and_gradients(
    cp_size, sp_size, dims, monkeypatch, layout
):
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
        if layout == "packed":
            real = physical
        elif layout == "trailing":
            physical, real = physical[:2], real[:2]
        elif layout == "empty":
            physical, real = physical[:1], real[:1]
        packed = PackedSeqParams(cu_seqlens_q=real, cu_seqlens_q_padded=physical, qkv_format="thd")
        if layout == "dense":
            packed = None
        lane = groups.dp.rank()
        tokens = torch.arange(total, device="cuda").view(1, -1) + lane * total
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
            expected = _reference_roll(expected, packed)
            expected_padding = _reference_roll(expected_padding, packed, fill_value=True)
            assert torch.equal(value, expected[:, rows])
            assert torch.equal(padding, expected_padding[:, rows])
        assert len(calls) == (3 if cp_size > 1 else 0)
        local //= sp_size
        owner = group.rank() * sp_size + sp.rank()
        full = torch.arange(total * 6, device="cuda", dtype=torch.float32).view(total, 2, 3)
        full = (full + lane * full.numel()).movedim(0, dims).requires_grad_()
        hidden = full.detach().narrow(dims, owner * local, local).clone().requires_grad_()
        output, expected = hidden, full
        for _ in range(3):
            output = roll_contiguous(output, dims, group, packed, fill_value=7, sp_group=sp)
            expected = _reference_roll(expected, packed, dims, fill_value=7)
            torch.testing.assert_close(
                output, expected.narrow(dims, owner * local, local), atol=0, rtol=0
            )
        dout = torch.arange(full.numel(), device=full.device).view_as(full).float() + 1 + lane
        output.backward(dout.narrow(dims, owner * local, local))
        expected.backward(dout)
        torch.testing.assert_close(
            hidden.grad, full.grad.narrow(dims, owner * local, local), atol=0, rtol=0
        )
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
