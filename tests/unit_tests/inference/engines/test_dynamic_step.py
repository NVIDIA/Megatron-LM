# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses

import pytest

from megatron.core.inference.engines.dynamic_step import (
    AsyncStepOutput,
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
