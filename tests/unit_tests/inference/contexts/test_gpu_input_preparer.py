# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.gpu_input_preparer import GpuInputPreparer
from megatron.core.inference.contexts.gpu_input_state import GpuSampledTokenState
from megatron.core.inference.engines.dynamic_step import (
    DynamicStepContextSnapshot,
    DynamicStepId,
    DynamicStepRequestPlan,
    StepInputPlan,
)


class FakeGpuView:
    def __init__(
        self,
        *,
        slot_id: int,
        max_tokens: int = 8,
        max_requests: int = 4,
        max_blocks: int = 8,
    ):
        self.current_snapshot_slot_id = slot_id
        self.token_to_input_ids = torch.zeros(max_tokens, dtype=torch.int64, device="cuda")
        self.token_to_pos_ids = torch.full(
            (max_tokens,), -1, dtype=torch.int64, device="cuda"
        )
        self.token_to_request_idx = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device="cuda"
        )
        self.token_to_position_in_request = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device="cuda"
        )
        self.token_to_local_position_within_kv_block = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device="cuda"
        )
        self.token_to_block_idx = torch.full(
            (max_tokens,), -1, dtype=torch.int32, device="cuda"
        )
        self.request_query_lengths = torch.zeros(
            max_requests, dtype=torch.int32, device="cuda"
        )
        self.request_kv_length_offsets = (
            torch.arange(max_requests, dtype=torch.int32, device="cuda") * 4 + 5
        )
        self.mha_query_lengths = torch.zeros(max_requests, dtype=torch.int32, device="cuda")
        self.mha_kv_seq_lengths = torch.zeros(max_requests, dtype=torch.int32, device="cuda")
        self.mha_cu_query_seq_lengths = torch.zeros(
            max_requests + 1, dtype=torch.int32, device="cuda"
        )
        self.mha_cu_kv_seq_lengths = torch.zeros(
            max_requests + 1, dtype=torch.int32, device="cuda"
        )
        self.mha_block_table = torch.arange(
            max_requests * max_blocks, dtype=torch.int32, device="cuda"
        ).view(max_requests, max_blocks)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU input preparer tests")


def _snapshot(slot_id: int = 0) -> DynamicStepContextSnapshot:
    step_id = DynamicStepId(1)
    return DynamicStepContextSnapshot(
        step_id=step_id,
        snapshot_slot_id=slot_id,
        request_plan=DynamicStepRequestPlan(step_id=step_id),
        gpu_view=FakeGpuView(slot_id=slot_id),
    )


def _seed_expected_decode_metadata(
    gpu_view: FakeGpuView, plan: StepInputPlan, *, block_size_tokens: int
) -> None:
    request_slots = torch.tensor(plan.decode_request_slots, dtype=torch.long, device="cuda")
    destinations = torch.tensor(
        plan.decode_input_destination_indices, dtype=torch.long, device="cuda"
    )
    tokens_per_request = plan.speculative_width + 1
    offsets = torch.arange(tokens_per_request, dtype=torch.long, device="cuda")
    token_indices = (destinations[:, None] + offsets[None, :]).reshape(-1)
    positions = (
        gpu_view.request_kv_length_offsets[request_slots].to(torch.long)[:, None]
        + offsets[None, :]
    ).reshape(-1)
    positions_i32 = positions.to(torch.int32)

    gpu_view.token_to_pos_ids.index_copy_(0, token_indices, positions)
    gpu_view.token_to_position_in_request.index_copy_(0, token_indices, positions_i32)
    gpu_view.token_to_local_position_within_kv_block.index_copy_(
        0, token_indices, torch.remainder(positions_i32, block_size_tokens)
    )
    gpu_view.token_to_request_idx.index_copy_(
        0, token_indices, request_slots.to(torch.int32).repeat_interleave(tokens_per_request)
    )
    block_columns = torch.div(positions, block_size_tokens, rounding_mode="floor").to(torch.long)
    block_ids = gpu_view.mha_block_table[
        request_slots.repeat_interleave(tokens_per_request), block_columns
    ]
    gpu_view.token_to_block_idx.index_copy_(0, token_indices, block_ids)

    query_lengths = torch.full(
        (len(plan.decode_request_slots),), tokens_per_request, dtype=torch.int32, device="cuda"
    )
    gpu_view.request_query_lengths.index_copy_(0, request_slots, query_lengths)
    gpu_view.mha_query_lengths.index_copy_(0, request_slots, query_lengths)
    kv_seq_lengths = gpu_view.request_kv_length_offsets[request_slots] + query_lengths
    gpu_view.mha_kv_seq_lengths.index_copy_(0, request_slots, kv_seq_lengths)
    gpu_view.mha_cu_query_seq_lengths[: len(plan.decode_request_slots) + 1].copy_(
        torch.arange(
            len(plan.decode_request_slots) + 1, dtype=torch.int32, device="cuda"
        )
        * tokens_per_request
    )
    gpu_view.mha_cu_kv_seq_lengths[0].zero_()
    gpu_view.mha_cu_kv_seq_lengths[1 : len(plan.decode_request_slots) + 1].copy_(
        torch.cumsum(kv_seq_lengths, dim=0)
    )


def test_gpu_input_preparer_scatters_decode_tokens_into_snapshot():
    _require_cuda()
    stream = torch.cuda.Stream()
    sample_state = GpuSampledTokenState(
        max_requests=4, num_speculative_tokens=0, device="cuda", stream=stream
    )
    sample_state.record(
        sampled_tokens=torch.tensor([101, 102, 103, 104], dtype=torch.int64, device="cuda"),
        active_request_count=4,
    )
    snapshot = _snapshot(slot_id=0)
    plan = StepInputPlan(
        step_id=snapshot.step_id,
        snapshot_slot_id=0,
        decode_request_slots=(0, 1, 2),
        decode_input_destination_indices=(0, 2, 4),
        debug_expected_input_ids=torch.tensor([101, 0, 102, 0, 103], dtype=torch.int64),
    )
    _seed_expected_decode_metadata(snapshot.gpu_view, plan, block_size_tokens=4)

    ready_event = GpuInputPreparer(
        stream=stream, block_size_tokens=4, debug_enabled=True
    ).prepare(
        snapshot, plan, sample_state
    )

    torch.cuda.current_stream().wait_event(ready_event)
    assert snapshot.gpu_view.token_to_input_ids[:5].tolist() == [101, 0, 102, 0, 103]
    assert snapshot.gpu_view.token_to_pos_ids[[0, 2, 4]].tolist() == [5, 9, 13]
    assert snapshot.gpu_view.token_to_request_idx[[0, 2, 4]].tolist() == [0, 1, 2]
    assert snapshot.gpu_view.token_to_block_idx[[0, 2, 4]].tolist() == [1, 10, 19]


def test_gpu_input_preparer_scatters_speculative_tokens():
    _require_cuda()
    stream = torch.cuda.Stream()
    sample_state = GpuSampledTokenState(
        max_requests=2, num_speculative_tokens=2, device="cuda", stream=stream
    )
    sample_state.record(
        sampled_tokens=torch.tensor([11, 12], dtype=torch.int64, device="cuda"),
        active_request_count=2,
        sampled_mtp_tokens=torch.tensor([[21, 22], [31, 32]], dtype=torch.int64, device="cuda"),
        accepted_token_counts=torch.tensor([2, 1], dtype=torch.int64, device="cuda"),
    )
    snapshot = _snapshot(slot_id=0)
    plan = StepInputPlan(
        step_id=snapshot.step_id,
        snapshot_slot_id=0,
        decode_request_slots=(0, 1),
        decode_input_destination_indices=(0, 3),
        speculative_width=2,
        debug_expected_input_ids=torch.tensor([11, 21, 31, 12, 22, 32], dtype=torch.int64),
    )
    _seed_expected_decode_metadata(snapshot.gpu_view, plan, block_size_tokens=4)

    ready_event = GpuInputPreparer(
        stream=stream, block_size_tokens=4, debug_enabled=True
    ).prepare(
        snapshot, plan, sample_state
    )

    torch.cuda.current_stream().wait_event(ready_event)
    assert snapshot.gpu_view.token_to_input_ids[:6].tolist() == [11, 21, 31, 12, 22, 32]
    assert snapshot.gpu_view.token_to_pos_ids[:6].tolist() == [5, 6, 7, 9, 10, 11]
    assert snapshot.gpu_view.token_to_local_position_within_kv_block[:6].tolist() == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]


def test_gpu_input_preparer_rejects_mismatched_snapshot_slot():
    _require_cuda()
    stream = torch.cuda.Stream()
    sample_state = GpuSampledTokenState(
        max_requests=1, num_speculative_tokens=0, device="cuda", stream=stream
    )
    sample_state.record(
        sampled_tokens=torch.tensor([7], dtype=torch.int64, device="cuda"),
        active_request_count=1,
    )
    snapshot = _snapshot(slot_id=0)
    plan = StepInputPlan(
        step_id=snapshot.step_id,
        snapshot_slot_id=1,
        decode_request_slots=(0,),
        decode_input_destination_indices=(0,),
    )

    with pytest.raises(RuntimeError, match="targets snapshot slot"):
        GpuInputPreparer(stream=stream, block_size_tokens=4).prepare(
            snapshot, plan, sample_state
        )


def test_gpu_input_preparer_rejects_snapshot_view_bound_to_another_slot():
    _require_cuda()
    stream = torch.cuda.Stream()
    sample_state = GpuSampledTokenState(
        max_requests=1, num_speculative_tokens=0, device="cuda", stream=stream
    )
    sample_state.record(
        sampled_tokens=torch.tensor([7], dtype=torch.int64, device="cuda"),
        active_request_count=1,
    )
    snapshot = _snapshot(slot_id=0)
    snapshot.gpu_view.current_snapshot_slot_id = 1
    plan = StepInputPlan(
        step_id=snapshot.step_id,
        snapshot_slot_id=0,
        decode_request_slots=(0,),
        decode_input_destination_indices=(0,),
    )

    with pytest.raises(RuntimeError, match="bound to slot"):
        GpuInputPreparer(stream=stream, block_size_tokens=4).prepare(
            snapshot, plan, sample_state
        )
