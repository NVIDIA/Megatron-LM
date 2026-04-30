# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the typed contract structures introduced in v3 plan commit 2.

These types are pure dataclasses and an in-memory transaction journal — no
GPU work, no model construction. The tests assert basic instantiation,
defaults, and lifecycle invariants.
"""

import pytest

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.engines.async_pipeline_types import (
    AsyncStepOutput,
    DynamicStepLaunch,
    DynamicStepPlan,
    DynamicStepSnapshot,
    Reservation,
    ReservationState,
    ResourceKind,
    StepInputPlan,
    StepRetirementResult,
)
from megatron.core.inference.engines.transaction_journal import JournalEntry, TransactionJournal

# ---------------------------------------------------------------------------
# Async pipeline types
# ---------------------------------------------------------------------------


class TestReservation:
    def test_default_state_is_reserved(self):
        r = Reservation(
            journal_id=0,
            resource_kind=ResourceKind.KV_BLOCK,
            resource_handle=42,
        )
        assert r.state is ReservationState.RESERVED
        assert r.must_outlast_snapshot_step_id == -1

    def test_state_transitions(self):
        r = Reservation(
            journal_id=1,
            resource_kind=ResourceKind.MAMBA_SLOT,
            resource_handle="slot-7",
        )
        r.state = ReservationState.COMMITTED
        assert r.state is ReservationState.COMMITTED
        r.state = ReservationState.ROLLED_BACK
        assert r.state is ReservationState.ROLLED_BACK


class TestStepInputPlan:
    def test_speculative_width_zero(self):
        plan = StepInputPlan(
            decode_request_slots=(0, 1, 2),
            decode_token_destination_indices=(10, 11, 12),
            prefill_cpu_token_ranges=(),
            speculative_width=0,
            previous_sample_source_step=None,
        )
        assert plan.speculative_width == 0
        assert plan.previous_sample_source_step is None
        assert len(plan.decode_request_slots) == 3


class TestDynamicStepPlan:
    def test_construct_with_empty_plan(self):
        input_plan = StepInputPlan(
            decode_request_slots=(),
            decode_token_destination_indices=(),
            prefill_cpu_token_ranges=(),
            speculative_width=0,
            previous_sample_source_step=None,
        )
        plan = DynamicStepPlan(
            step_id=42,
            request_slots=(),
            placeholder_deltas=(),
            input_plan=input_plan,
            resource_reservation_ids=[],
            intended_batch_dimensions=None,
        )
        assert plan.step_id == 42


class TestDynamicStepSnapshot:
    def test_default_events_are_none(self):
        snap = DynamicStepSnapshot(
            step_id=0,
            buffer_pool_slot=0,
            active_request_slot_view=None,
            attention_metadata=None,
            mamba_metadata=None,
            graph_match=None,
        )
        assert snap.metadata_ready_event is None
        assert snap.input_ready_event is None
        assert snap.cpu_owner_step_count == 0


class TestAsyncStepOutput:
    def test_default_collections_independent(self):
        a = AsyncStepOutput(step_id=0)
        b = AsyncStepOutput(step_id=1)
        a.source_gpu_tensors["x"] = 1
        # Default factories must create independent dicts.
        assert "x" not in b.source_gpu_tensors

    def test_has_payload_default_false(self):
        out = AsyncStepOutput(step_id=0)
        assert out.has_payload("sampled_tokens") is False
        out.payload_metadata["sampled_tokens"] = True
        assert out.has_payload("sampled_tokens") is True


class TestDynamicStepLaunch:
    def test_default_journal_id(self):
        snap = DynamicStepSnapshot(
            step_id=5,
            buffer_pool_slot=0,
            active_request_slot_view=None,
            attention_metadata=None,
            mamba_metadata=None,
            graph_match=None,
        )
        launch = DynamicStepLaunch(step_id=5, snapshot=snap)
        assert launch.journal_id == -1
        assert launch.output is None


class TestStepRetirementResult:
    def test_default_counts_zero(self):
        r = StepRetirementResult(step_id=0)
        assert r.reservation_commit_count == 0
        assert r.reservation_rollback_count == 0
        assert r.discarded_lookahead_token_count == 0
        assert r.finished_request_records == []


# ---------------------------------------------------------------------------
# Transaction journal
# ---------------------------------------------------------------------------


class TestTransactionJournal:
    def test_begin_step_is_idempotent(self):
        j = TransactionJournal()
        a = j.begin_step_transaction(7)
        b = j.begin_step_transaction(7)
        assert a is b
        assert j.open_step_count() == 1

    def test_journal_ids_are_unique(self):
        j = TransactionJournal()
        ids = {j.issue_journal_id() for _ in range(100)}
        assert len(ids) == 100

    def test_commit_marks_reservations_committed(self):
        j = TransactionJournal()
        j.begin_step_transaction(0)
        r = Reservation(
            journal_id=j.issue_journal_id(),
            resource_kind=ResourceKind.KV_BLOCK,
            resource_handle=99,
        )
        j.record_resource_reservation(0, r)
        entry = j.commit_step_transaction(0)
        assert entry.reservations[0].state is ReservationState.COMMITTED
        assert j.open_step_count() == 0

    def test_rollback_marks_reservations_rolled_back(self):
        j = TransactionJournal()
        j.begin_step_transaction(0)
        r = Reservation(
            journal_id=j.issue_journal_id(),
            resource_kind=ResourceKind.MAMBA_SLOT,
            resource_handle=0,
        )
        j.record_resource_reservation(0, r)
        entry = j.rollback_step_transaction(0)
        assert entry.reservations[0].state is ReservationState.ROLLED_BACK
        assert j.open_step_count() == 0

    def test_double_commit_raises(self):
        j = TransactionJournal()
        j.begin_step_transaction(0)
        j.commit_step_transaction(0)
        with pytest.raises(KeyError):
            j.commit_step_transaction(0)

    def test_open_step_ids_sorted(self):
        j = TransactionJournal()
        for sid in [5, 1, 7, 3]:
            j.begin_step_transaction(sid)
        assert j.open_step_ids() == [1, 3, 5, 7]

    def test_journal_entry_defaults(self):
        e = JournalEntry(step_id=0)
        assert e.snapshot_buffer_id == -1
        assert e.graph_batch_dimensions is None
        assert e.reservations == []


# ---------------------------------------------------------------------------
# Inference config wiring
# ---------------------------------------------------------------------------


class TestInferenceConfigAsyncOverlapKnobs:
    def test_defaults_select_async_overlap(self):
        cfg = InferenceConfig()
        assert cfg.enable_async_overlap is True
        assert cfg.async_overlap_queue_size == 2
        assert cfg.async_overlap_debug_checks is False
        assert cfg.cuda_graph_capture_mode == "warmup_only"
        assert cfg.cuda_graph_memory_budget_bytes is None
        assert cfg.cuda_graph_max_captures is None

    def test_knobs_are_settable(self):
        cfg = InferenceConfig(
            enable_async_overlap=True,
            async_overlap_queue_size=1,
            async_overlap_debug_checks=True,
            cuda_graph_memory_budget_bytes=2 * 1024 * 1024 * 1024,
            cuda_graph_capture_mode="on_first_use",
            cuda_graph_max_captures=64,
        )
        assert cfg.enable_async_overlap is True
        assert cfg.async_overlap_queue_size == 1
        assert cfg.cuda_graph_capture_mode == "on_first_use"
