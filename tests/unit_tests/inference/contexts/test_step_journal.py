# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.step_journal import StepJournal


class FakeGpuView:
    current_snapshot_slot_id = 0


@dataclass(frozen=True)
class FakeResourceReservation:
    kv_block_ids: tuple[int, ...] = field(default_factory=tuple)
    mamba_slot_ids: tuple[int, ...] = field(default_factory=tuple)
    prefix_cache_refcount_deltas: dict[str, int] = field(default_factory=dict)


def _fake_dynamic_context():
    context = object.__new__(DynamicInferenceContext)
    context.step_journal = StepJournal()
    context.total_request_count = 3
    context.paused_request_count = 1
    context.request_ids = torch.tensor([100, 101, 102], dtype=torch.int64)
    context.gpu_view = FakeGpuView()
    return context


def test_step_journal_records_and_commits_entry():
    journal = StepJournal()

    entry = journal.begin_step_journal(
        3, request_slots_before=(0, 1), active_request_ids=(10, 11)
    )
    assert entry.step_id == 3
    assert entry.request_slots_before == (0, 1)
    assert journal.open_step_ids == (3,)

    journal.record_placeholder_delta(3, 10, 1)
    journal.record_snapshot_slot(3, 2, cuda_graph_batch_dimensions=4)
    journal.record_request_slot_transition(
        3,
        request_slots_after=(0, 1, 2),
        active_request_ids=(10, 11, 12),
        decode_input_destination_indices=(5, 6),
    )

    committed = journal.commit_step_journal(3)

    assert committed.snapshot_slot_id == 2
    assert committed.placeholder_token_counts["10"] == 1
    assert committed.request_slots_after == (0, 1, 2)
    assert committed.active_request_ids == ("10", "11", "12")
    assert committed.decode_input_destination_indices == (5, 6)
    assert committed.cuda_graph_batch_dimensions == 4
    assert journal.open_entry_count == 0
    assert journal.get_committed_entry(3) == committed


def test_step_journal_records_resource_reservation():
    journal = StepJournal()
    journal.begin_step_journal(4)

    reservation = FakeResourceReservation(
        kv_block_ids=(7, 8),
        mamba_slot_ids=(2,),
        prefix_cache_refcount_deltas={"prefix": 1},
    )
    entry = journal.record_resource_reservation(4, reservation)

    assert entry.reserved_kv_blocks == (7, 8)
    assert entry.reserved_mamba_slots == (2,)
    assert entry.prefix_cache_refcount_deltas["prefix"] == 1
    assert entry.resources_waiting_on_snapshot == (reservation,)


def test_step_journal_rolls_back_open_entries():
    journal = StepJournal()
    journal.begin_step_journal(1)
    journal.begin_step_journal(2)

    rolled_back = journal.rollback_all_open(reason="test")

    assert [entry.step_id for entry in rolled_back] == [1, 2]
    assert journal.open_step_ids == ()
    assert journal.get_rolled_back_entry(1).step_id == 1
    assert journal.get_rolled_back_entry(2).step_id == 2


def test_step_journal_rejects_duplicate_or_terminal_entry():
    journal = StepJournal()
    journal.begin_step_journal(9)

    with pytest.raises(RuntimeError, match="already open"):
        journal.begin_step_journal(9)

    journal.commit_step_journal(9)
    with pytest.raises(RuntimeError, match="already terminal"):
        journal.begin_step_journal(9)


def test_dynamic_context_journal_api_captures_live_request_state():
    context = _fake_dynamic_context()

    context.begin_step_journal(5)
    context.record_snapshot_slot(5, 3)
    committed = context.commit_step_journal(5)

    assert committed.request_slots_before == (0, 1, 2)
    assert committed.request_slots_after == (0, 1, 2)
    assert committed.active_request_ids == ("101", "102")
    assert committed.snapshot_slot_id == 3
    assert context.step_journal.open_entry_count == 0


def test_dynamic_context_record_api_is_noop_without_open_journal():
    context = _fake_dynamic_context()

    assert context.record_snapshot_slot(99, 1) is None
    assert context.record_placeholder_delta(99, 100, 1) is None
