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
    def __init__(self, *, slot_id: int, max_tokens: int = 8):
        self.current_snapshot_slot_id = slot_id
        self.token_to_input_ids = torch.zeros(max_tokens, dtype=torch.int64, device="cuda")


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

    ready_event = GpuInputPreparer(stream=stream, debug_enabled=True).prepare(
        snapshot, plan, sample_state
    )

    torch.cuda.current_stream().wait_event(ready_event)
    assert snapshot.gpu_view.token_to_input_ids[:5].tolist() == [101, 0, 102, 0, 103]


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

    ready_event = GpuInputPreparer(stream=stream, debug_enabled=True).prepare(
        snapshot, plan, sample_state
    )

    torch.cuda.current_stream().wait_event(ready_event)
    assert snapshot.gpu_view.token_to_input_ids[:6].tolist() == [11, 21, 31, 12, 22, 32]


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
        GpuInputPreparer(stream=stream).prepare(snapshot, plan, sample_state)


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
        GpuInputPreparer(stream=stream).prepare(snapshot, plan, sample_state)
