# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardPackedSummary,
    CPBackwardUnpackedSummary,
    CPForwardPackedSummary,
    _all_gather_backward_summary,
    _all_gather_forward_summary,
    _slice_backward_summary,
    _slice_forward_summary,
    build_packed_sequence_cp_metadata,
)


class _FakeGroup:
    def size(self) -> int:
        return 2


def test_unpacked_backward_summary_gather_and_slice(monkeypatch):
    transition = torch.randn(2, 3)
    state_grad = torch.randn(2, 3, 4)

    def fake_all_gather_into_tensor(output, input_, group):
        output.copy_(input_.expand_as(output))

    monkeypatch.setattr(torch.distributed, "_coalescing_manager", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)

    gathered = _all_gather_backward_summary(
        CPBackwardUnpackedSummary(transition=transition, local_state_grad=state_grad), _FakeGroup()
    )
    following = _slice_backward_summary(gathered, slice(1, 2))

    assert following.transition.shape == (1, *transition.shape)
    assert following.local_state_grad.shape == (1, *state_grad.shape)
    torch.testing.assert_close(following.transition[0], transition)
    torch.testing.assert_close(following.local_state_grad[0], state_grad)


def test_packed_summary_gather_and_slice(monkeypatch):
    forward_packed = torch.randn(2, 3, 4)
    backward_packed = torch.randn(2, 3, 4)
    gathered_inputs = []

    def fake_all_gather_into_tensor(output, input_, group):
        gathered_inputs.append(input_)
        output.copy_(input_.expand_as(output))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)
    group = _FakeGroup()

    gathered_forward = _all_gather_forward_summary(
        CPForwardPackedSummary(packed=forward_packed), group
    )
    preceding = _slice_forward_summary(gathered_forward, slice(0, 1))
    gathered_backward = _all_gather_backward_summary(
        CPBackwardPackedSummary(packed=backward_packed), group
    )
    following = _slice_backward_summary(gathered_backward, slice(1, 2))

    assert len(gathered_inputs) == 2
    assert gathered_inputs[0].data_ptr() == forward_packed.data_ptr()
    assert gathered_inputs[1].data_ptr() == backward_packed.data_ptr()
    torch.testing.assert_close(preceding.packed[0], forward_packed)
    torch.testing.assert_close(following.packed[0], backward_packed)


@pytest.mark.parametrize(
    "cp_rank,expected_bounds,expected_cu_seqlens",
    [(0, (0, 1), [0, 4]), (1, (0, 2), [0, 1, 4]), (2, (1, 3), [0, 1, 4]), (3, (2, 3), [0, 4])],
)
def test_build_packed_sequence_cp_metadata(cp_rank, expected_bounds, expected_cu_seqlens):
    global_seq_idx = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32), torch.tensor([5, 4, 7])
    ).unsqueeze(0)

    metadata = build_packed_sequence_cp_metadata(global_seq_idx, cp_rank=cp_rank, cp_size=4)

    assert (metadata.preceding_rank_start, metadata.following_rank_stop - 1) == expected_bounds
    assert (
        metadata.local_seq_idx.untyped_storage().data_ptr()
        == global_seq_idx.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(
        metadata.local_seq_idx, global_seq_idx[:, cp_rank * 4 : (cp_rank + 1) * 4]
    )
    torch.testing.assert_close(
        metadata.local_cu_seqlens, torch.tensor(expected_cu_seqlens, dtype=torch.int32)
    )
