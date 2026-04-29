# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.dynamic_ledgers import (
    DynamicRequestLedgers,
    RequestLedgerState,
)


class FakeDynamicContext:
    def __init__(self):
        self.total_request_count = 3
        self.paused_request_count = 1
        self.active_token_count = 5
        self.chunked_prefill_request_id = -1
        self.request_ids = torch.tensor([10, 11, 12, -1], dtype=torch.int32)
        self.request_query_lengths = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
        self.request_in_prefill_status_tensor = torch.tensor([0, 1, 0, -1], dtype=torch.int32)


def test_request_ledger_state_snapshots_context_values():
    context = FakeDynamicContext()

    state = RequestLedgerState.from_context(context)

    assert state.total_request_count == 3
    assert state.paused_request_count == 1
    assert state.active_token_count == 5
    assert state.request_ids == (10, 11, 12)
    assert state.request_query_lengths == (1, 2, 3)
    assert state.request_in_prefill_status == (0, 1, 0)


def test_request_ledger_returns_per_request_state():
    state = RequestLedgerState.from_context(FakeDynamicContext())

    paused = state.get_request_state(10)
    active_prefill = state.get_request_state(11)

    assert paused.slot == 0
    assert paused.is_paused is True
    assert paused.in_prefill is False
    assert active_prefill.slot == 1
    assert active_prefill.is_paused is False
    assert active_prefill.in_prefill is True
    assert state.get_request_state(99) is None


def test_queue_depth_one_sync_keeps_ledgers_identical():
    ledgers = DynamicRequestLedgers()
    context = FakeDynamicContext()

    ledgers.sync_from_context_for_queue_depth_one(context)

    assert ledgers.get_committed_request_state() == ledgers.get_optimistic_request_state()
    assert ledgers.get_committed_request_state(12).query_length == 3
    assert ledgers.get_optimistic_request_state(12).query_length == 3
    ledgers.assert_committed_matches_optimistic()


def test_ledger_debug_assertion_detects_divergence():
    ledgers = DynamicRequestLedgers()
    context = FakeDynamicContext()
    ledgers.sync_from_context_for_queue_depth_one(context)
    ledgers.optimistic = RequestLedgerState()

    with pytest.raises(AssertionError, match="diverged"):
        ledgers.assert_committed_matches_optimistic()
