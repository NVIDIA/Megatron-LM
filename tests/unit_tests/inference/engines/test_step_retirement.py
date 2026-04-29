# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.inference.engines.step_retirement import StepRetirementService


class FakeContext:
    enable_prefix_caching = False

    def __init__(self):
        self.commit_calls = []
        self.rollback_reasons = []

    def commit_step_journal(self, step_id):
        self.commit_calls.append(step_id)

    def rollback_all_open_step_journals(self, *, reason):
        self.rollback_reasons.append(reason)


class FakeEngine:
    def __init__(self, *, queue_depth=1):
        self.context = FakeContext()
        self.async_overlap_queue_depth = queue_depth
        self.async_overlap_debug_counters = SimpleNamespace()
        self._current_dynamic_step_id = -1
        self.failed_request_ids = []
        self.use_coordinator = False
        self.logging_step_interval = 0
        self.metrics_writer = None

    def _step_nvtx_label(self, name, step_id=None):
        del step_id
        return name


def _context_state(step_id):
    return {"dynamic_step_id": step_id, "kv_stats": None}


def test_submit_step_retires_queue_depth_one_immediately():
    engine = FakeEngine(queue_depth=1)
    service = StepRetirementService(engine)

    result = service.submit_step(None, _context_state(0), 0.0)

    assert result["active_request_ids"] == []
    assert result["finished_request_records"] == []
    assert service.pending_count == 0
    assert engine.context.commit_calls == [0]
    assert engine.async_overlap_debug_counters.retirement_backlog == 0
    assert engine.async_overlap_debug_counters.max_retirement_backlog == 1


def test_retirement_backpressure_limits_pending_items():
    engine = FakeEngine(queue_depth=1)
    service = StepRetirementService(engine)

    service.enqueue_step(None, _context_state(0), 0.0)

    assert service.pending_count == 1
    assert engine.async_overlap_debug_counters.retirement_backlog == 1
    with pytest.raises(RuntimeError, match="backlog is full"):
        service.enqueue_step(None, _context_state(1), 0.0)

    result = service.drain_next()

    assert result["finished_request_records"] == []
    assert service.pending_count == 0
    assert engine.context.commit_calls == [0]


def test_shutdown_drain_retires_pending_before_rollback():
    engine = FakeEngine(queue_depth=1)
    service = StepRetirementService(engine)
    service.enqueue_step(None, _context_state(0), 0.0)

    service.drain_for_shutdown()

    assert service.pending_count == 0
    assert engine.context.commit_calls == [0]
    assert engine.context.rollback_reasons == ["shutdown_drain"]
