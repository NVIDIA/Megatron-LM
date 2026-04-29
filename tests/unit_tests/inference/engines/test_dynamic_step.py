# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

from megatron.core.inference.engines.dynamic_step import (
    AsyncStepOutput,
    DynamicAsyncPipeline,
    DynamicStepContextSnapshot,
    DynamicStepGpuLaunch,
    DynamicStepId,
    DynamicStepRequestPlan,
    ResourceReservation,
    SnapshotSlotHandle,
    StepInputPlan,
    StepJournalEntry,
    StepRetirementResult,
    assert_snapshot_gpu_view_bound,
)


class FakeRetirementService:
    def __init__(self):
        self.shutdown_drains = 0
        self.suspend_drains = 0
        self.reused_request_ids = []

    def drain_for_shutdown(self):
        self.shutdown_drains += 1

    def drain_for_suspend(self):
        self.suspend_drains += 1

    def drain_for_request_reuse(self, request_id):
        self.reused_request_ids.append(request_id)


class FakePipelineEngine:
    def __init__(self, *, max_launches=1, lookahead_allowed=True):
        self.context = SimpleNamespace(snapshot_pool=object())
        self.max_launches = max_launches
        self.lookahead_allowed = lookahead_allowed
        self.forward_calls = []
        self.bookkeep_calls = []

    async def async_forward(self):
        step_id = len(self.forward_calls)
        self.forward_calls.append(step_id)
        return ({"step": step_id}, {"dynamic_step_id": step_id}, float(step_id))

    async def async_bookkeep(self, step_result, context_state, step_time):
        self.bookkeep_calls.append((step_result, context_state, step_time))
        return {"retired_step": context_state["dynamic_step_id"]}

    def has_async_overlap_launch_work(self):
        return len(self.forward_calls) < self.max_launches

    def can_launch_async_overlap_lookahead(self):
        return self.lookahead_allowed


def test_dynamic_async_pipeline_queue_depth_one_retires_immediately():
    engine = FakePipelineEngine()
    retirement_service = FakeRetirementService()
    pipeline = DynamicAsyncPipeline(
        engine=engine, retirement_service=retirement_service, queue_depth=1
    )

    result = asyncio.run(pipeline.step())

    assert result == {"retired_step": 0}
    assert engine.forward_calls == [0]
    assert [call[1]["dynamic_step_id"] for call in engine.bookkeep_calls] == [0]
    assert pipeline.pending_launch_count == 0
    assert pipeline.snapshot_pool is engine.context.snapshot_pool


def test_dynamic_async_pipeline_drains_before_shutdown_hooks():
    engine = FakePipelineEngine()
    retirement_service = FakeRetirementService()
    pipeline = DynamicAsyncPipeline(
        engine=engine, retirement_service=retirement_service, queue_depth=1
    )

    pipeline.drain_for_shutdown()
    pipeline.drain_for_suspend()
    pipeline.drain_for_request_reuse(7)

    assert retirement_service.shutdown_drains == 1
    assert retirement_service.suspend_drains == 1
    assert retirement_service.reused_request_ids == [7]


def test_dynamic_async_pipeline_queue_depth_two_launches_before_retirement():
    engine = FakePipelineEngine(max_launches=3)
    pipeline = DynamicAsyncPipeline(
        engine=engine, retirement_service=FakeRetirementService(), queue_depth=2
    )

    first = asyncio.run(pipeline.step())
    second = asyncio.run(pipeline.step())
    final = asyncio.run(pipeline.step())

    assert first == {"retired_step": 0}
    assert second == {"retired_step": 1}
    assert final == {"retired_step": 2}
    assert engine.forward_calls == [0, 1, 2]
    assert [call[1]["dynamic_step_id"] for call in engine.bookkeep_calls] == [0, 1, 2]
    assert pipeline.pending_launch_count == 0


def test_dynamic_async_pipeline_queue_depth_two_falls_back_when_lookahead_blocked():
    engine = FakePipelineEngine(max_launches=2, lookahead_allowed=False)
    pipeline = DynamicAsyncPipeline(
        engine=engine, retirement_service=FakeRetirementService(), queue_depth=2
    )

    first = asyncio.run(pipeline.step())
    second = asyncio.run(pipeline.step())

    assert first == {"retired_step": 0}
    assert second == {"retired_step": 1}
    assert engine.forward_calls == [0, 1]
    assert [call[1]["dynamic_step_id"] for call in engine.bookkeep_calls] == [0, 1]
    assert pipeline.pending_launch_count == 0


def test_dynamic_step_records_are_frozen_and_copy_mutable_inputs():
    step_id = DynamicStepId(3)
    request_ids = ["request-0"]
    placeholders = {"request-0": 1}

    plan = DynamicStepRequestPlan(
        step_id=step_id,
        active_request_ids=request_ids,
        decode_request_ids=request_ids,
        placeholder_token_counts=placeholders,
    )
    request_ids.append("request-1")
    placeholders["request-0"] = 5

    assert plan.active_request_ids == ("request-0",)
    assert plan.placeholder_token_counts["request-0"] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.active_request_ids = ()
    with pytest.raises(TypeError):
        plan.placeholder_token_counts["request-2"] = 1


def test_empty_step_contract_construction():
    step_id = DynamicStepId(0)
    request_plan = DynamicStepRequestPlan(step_id=step_id)
    snapshot_handle = SnapshotSlotHandle(step_id=step_id, snapshot_slot_id=0)
    input_plan = StepInputPlan(step_id=step_id, snapshot_slot_id=0)
    context_snapshot = DynamicStepContextSnapshot(
        step_id=step_id, snapshot_slot_id=0, request_plan=request_plan
    )
    gpu_launch = DynamicStepGpuLaunch(step_id=step_id, snapshot_slot_id=0)
    step_output = AsyncStepOutput(step_id=step_id, snapshot_slot_id=0)
    journal = StepJournalEntry(step_id=step_id, snapshot_slot_id=0)
    retirement = StepRetirementResult(step_id=step_id, snapshot_slot_id=0)

    assert request_plan.active_request_ids == ()
    assert snapshot_handle.metadata_ready_event is None
    assert input_plan.decode_request_slots == ()
    assert input_plan.decode_input_destination_indices == ()
    assert input_plan.prefill_prompt_token_ranges == ()
    assert input_plan.speculative_width == 0
    assert context_snapshot.request_plan is request_plan
    assert gpu_launch.compute_done_event is None
    assert step_output.output_ready_event is None
    assert journal.resources_waiting_on_snapshot == ()
    assert retirement.committed_request_ids == ()


def test_decode_only_step_contract_construction():
    step_id = DynamicStepId(1)
    request_plan = DynamicStepRequestPlan(
        step_id=step_id,
        active_request_ids=("decode-0", "decode-1"),
        decode_request_ids=("decode-0", "decode-1"),
    )
    input_plan = StepInputPlan(
        step_id=step_id,
        snapshot_slot_id=0,
        request_ids=request_plan.active_request_ids,
        decode_request_slots=(0, 1),
        decode_request_ids=request_plan.decode_request_ids,
        decode_input_destination_indices=(0, 1),
    )
    reservation = ResourceReservation(
        step_id=step_id,
        request_id="decode-0",
        snapshot_slot_id=0,
        kv_block_ids=(7,),
        prefix_cache_refcount_deltas={"block-7": 1},
    )
    journal = StepJournalEntry(
        step_id=step_id,
        snapshot_slot_id=0,
        active_request_ids=request_plan.active_request_ids,
        reserved_kv_blocks=reservation.kv_block_ids,
        decode_input_destination_indices=input_plan.decode_input_destination_indices,
        resources_waiting_on_snapshot=(reservation,),
    )

    assert request_plan.prefill_request_ids == ()
    assert input_plan.decode_request_slots == (0, 1)
    assert input_plan.decode_input_destination_indices == (0, 1)
    assert reservation.kv_block_ids == (7,)
    assert journal.resources_waiting_on_snapshot == (reservation,)


def test_mixed_prefill_decode_step_contract_construction():
    step_id = DynamicStepId(2)
    request_plan = DynamicStepRequestPlan(
        step_id=step_id,
        active_request_ids=("decode-0", "prefill-0"),
        decode_request_ids=("decode-0",),
        prefill_request_ids=("prefill-0",),
    )
    input_plan = StepInputPlan(
        step_id=step_id,
        snapshot_slot_id=1,
        request_ids=request_plan.active_request_ids,
        decode_request_slots=(0,),
        decode_request_ids=request_plan.decode_request_ids,
        prefill_request_ids=request_plan.prefill_request_ids,
        decode_input_destination_indices=(0,),
        prefill_prompt_token_ranges=((1, 4),),
    )
    context_snapshot = DynamicStepContextSnapshot(
        step_id=step_id,
        snapshot_slot_id=1,
        request_plan=request_plan,
        metadata_ready_event="metadata-ready",
        input_ready_event="input-ready",
    )
    gpu_launch = DynamicStepGpuLaunch(
        step_id=step_id,
        snapshot_slot_id=1,
        metadata_ready_event=context_snapshot.metadata_ready_event,
        input_ready_event=input_plan.input_ready_event,
        compute_done_event="compute-done",
        gpu_view=context_snapshot.gpu_view,
    )

    assert request_plan.decode_request_ids == ("decode-0",)
    assert request_plan.prefill_request_ids == ("prefill-0",)
    assert input_plan.prefill_prompt_token_ranges == ((1, 4),)
    assert context_snapshot.snapshot_slot_id == 1
    assert gpu_launch.compute_done_event == "compute-done"


def test_snapshot_gpu_view_debug_guard_detects_mutable_context_reads():
    step_id = DynamicStepId(8)
    snapshot_view = object()
    live_view = object()
    snapshot = DynamicStepContextSnapshot(
        step_id=step_id,
        snapshot_slot_id=0,
        request_plan=DynamicStepRequestPlan(step_id=step_id),
        gpu_view=snapshot_view,
    )

    assert_snapshot_gpu_view_bound(snapshot, live_view, debug_enabled=False)
    assert_snapshot_gpu_view_bound(snapshot, snapshot_view, debug_enabled=True)
    with pytest.raises(RuntimeError, match="prepared snapshot"):
        assert_snapshot_gpu_view_bound(snapshot, live_view, debug_enabled=True)


def test_speculative_placeholder_step_contract_construction():
    step_id = DynamicStepId(4)
    request_plan = DynamicStepRequestPlan(
        step_id=step_id,
        active_request_ids=("spec-0",),
        decode_request_ids=("spec-0",),
        speculative_request_ids=("spec-0",),
        placeholder_token_counts={"spec-0": 4},
    )
    reservation = ResourceReservation(
        step_id=step_id,
        request_id="spec-0",
        snapshot_slot_id=2,
        kv_block_ids=(10, 11),
        mamba_slot_ids=(3,),
    )
    journal = StepJournalEntry(
        step_id=step_id,
        snapshot_slot_id=2,
        active_request_ids=request_plan.active_request_ids,
        placeholder_token_counts=request_plan.placeholder_token_counts,
        reserved_kv_blocks=reservation.kv_block_ids,
        reserved_mamba_slots=reservation.mamba_slot_ids,
        resources_waiting_on_snapshot=(reservation,),
    )
    output = AsyncStepOutput(
        step_id=step_id,
        snapshot_slot_id=2,
        compute_done_event="compute-done",
        output_ready_event="output-ready",
        accepted_token_counts_cpu=(2,),
    )

    assert request_plan.speculative_request_ids == ("spec-0",)
    assert request_plan.placeholder_token_counts["spec-0"] == 4
    assert journal.placeholder_token_counts["spec-0"] == 4
    assert output.accepted_token_counts_cpu == (2,)


def test_step_input_plan_rejects_mismatched_decode_destinations():
    with pytest.raises(ValueError, match="same length"):
        StepInputPlan(
            step_id=DynamicStepId(5),
            snapshot_slot_id=0,
            decode_request_slots=(0, 1),
            decode_input_destination_indices=(0,),
        )


def test_step_input_plan_rejects_invalid_prefill_ranges():
    with pytest.raises(ValueError, match="invalid token range"):
        StepInputPlan(
            step_id=DynamicStepId(6),
            snapshot_slot_id=0,
            prefill_prompt_token_ranges=((3, 2),),
        )


@pytest.mark.parametrize("record_factory", [DynamicStepId])
def test_step_ids_reject_negative_values(record_factory):
    with pytest.raises(ValueError):
        record_factory(-1)


@pytest.mark.parametrize(
    "record_factory",
    [
        lambda step_id: SnapshotSlotHandle(step_id=step_id, snapshot_slot_id=-1),
        lambda step_id: StepInputPlan(step_id=step_id, snapshot_slot_id=-1),
        lambda step_id: DynamicStepContextSnapshot(
            step_id=step_id,
            snapshot_slot_id=-1,
            request_plan=DynamicStepRequestPlan(step_id=step_id),
        ),
        lambda step_id: DynamicStepGpuLaunch(step_id=step_id, snapshot_slot_id=-1),
        lambda step_id: AsyncStepOutput(step_id=step_id, snapshot_slot_id=-1),
        lambda step_id: ResourceReservation(
            step_id=step_id, request_id="request-0", snapshot_slot_id=-1
        ),
        lambda step_id: StepJournalEntry(step_id=step_id, snapshot_slot_id=-1),
        lambda step_id: StepRetirementResult(step_id=step_id, snapshot_slot_id=-1),
    ],
)
def test_snapshot_owners_reject_negative_slot_ids(record_factory):
    with pytest.raises(ValueError):
        record_factory(DynamicStepId(0))
