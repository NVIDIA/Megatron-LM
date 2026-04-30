# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for cross-rank optimistic-ledger validation (v3 plan
commit 28).

Plan validation: distributed test that artificially injects a one-step
request-set difference on one rank; assert
OptimisticLedgerDivergenceError is raised within one step, with both
ranks' journal entries logged. The distributed correctness portion is
exercised by the engine's distributed integration tests; here we
verify the hash + assertion primitives in isolation.
"""

import pytest

from megatron.core.inference.engines.async_pipeline_types import Reservation
from megatron.core.inference.engines.dynamic_engine import (
    OptimisticLedgerDivergenceError,
    assert_no_optimistic_ledger_divergence,
    compute_optimistic_ledger_hash,
)
from megatron.core.inference.engines.transaction_journal import JournalEntry


def _entry(step_id: int, slot_map: dict, placeholders: dict) -> JournalEntry:
    e = JournalEntry(step_id=step_id)
    e.request_slot_map_after = dict(slot_map)
    e.placeholder_deltas = dict(placeholders)
    return e


class TestOptimisticLedgerHash:
    def test_hash_is_deterministic(self):
        a = _entry(0, {0: 1, 1: 2}, {0: 1, 1: 1})
        b = _entry(0, {0: 1, 1: 2}, {0: 1, 1: 1})
        assert compute_optimistic_ledger_hash(a) == compute_optimistic_ledger_hash(b)

    def test_hash_changes_with_step_id(self):
        a = _entry(0, {0: 1}, {0: 1})
        b = _entry(1, {0: 1}, {0: 1})
        assert compute_optimistic_ledger_hash(a) != compute_optimistic_ledger_hash(b)

    def test_hash_changes_with_slot_map(self):
        a = _entry(0, {0: 1}, {0: 1})
        b = _entry(0, {0: 2}, {0: 1})
        assert compute_optimistic_ledger_hash(a) != compute_optimistic_ledger_hash(b)

    def test_hash_changes_with_placeholder_deltas(self):
        a = _entry(0, {0: 1}, {0: 1})
        b = _entry(0, {0: 1}, {0: 2})
        assert compute_optimistic_ledger_hash(a) != compute_optimistic_ledger_hash(b)

    def test_hash_unaffected_by_dict_iteration_order(self):
        a = _entry(0, {1: 5, 2: 6, 3: 7}, {3: 1, 1: 2})
        b = _entry(0, {3: 7, 1: 5, 2: 6}, {1: 2, 3: 1})
        assert compute_optimistic_ledger_hash(a) == compute_optimistic_ledger_hash(b)


class TestDivergenceAssertion:
    def test_zero_xor_is_no_op(self):
        e = _entry(0, {0: 1}, {0: 1})
        h = compute_optimistic_ledger_hash(e)
        # No error raised on zero xor.
        assert_no_optimistic_ledger_divergence(h, reduced_xor=0, journal_entry=e)

    def test_nonzero_xor_raises_with_journal_entry_in_message(self):
        e = _entry(0, {0: 1}, {0: 1})
        with pytest.raises(OptimisticLedgerDivergenceError) as exc_info:
            assert_no_optimistic_ledger_divergence(
                local_hash=1234, reduced_xor=42, journal_entry=e
            )
        msg = str(exc_info.value)
        assert "step 0" in msg
        assert "reduced_xor=42" in msg
        assert "local_hash=1234" in msg
