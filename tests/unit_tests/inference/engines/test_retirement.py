# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``StepRetirementService`` (v3 plan commit 5).

Covers the validation list from the plan: drain on shutdown, drain on
suspend, cancellation while a placeholder exists (queue depth 1 here, so
"cancellation" maps to draining a single in-flight entry), and request-id
reuse.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput
from megatron.core.inference.engines.retirement import StepRetirementService


@dataclass
class _FakeEvent:
    """Test fake for ``torch.cuda.Event`` used by AsyncStepOutput.

    ``query()`` returns ``ready``; ``synchronize()`` flips ``ready`` true so
    callers can verify the service blocks on the event.
    """

    ready: bool = False
    sync_calls: int = 0

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.sync_calls += 1
        self.ready = True


def _build_output(step_id: int, ready: bool = True) -> AsyncStepOutput:
    out = AsyncStepOutput(step_id=step_id)
    out.d2h_done_event = _FakeEvent(ready=ready)
    return out


@dataclass
class _Recorder:
    """Captures finalize-callback invocations and renders the public response."""

    finalize_calls: list = field(default_factory=list)

    async def finalize(self, step_result, context_state, step_time):
        self.finalize_calls.append((step_result, context_state, step_time))
        return {"step": step_result, "step_time": step_time}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStepRetirementService:
    def test_retire_runs_callback_and_syncs_event(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        out = _build_output(step_id=0, ready=False)
        result = _run(svc.retire(out, ("res", {"k": 1}, 0.5)))
        assert result == {"step": "res", "step_time": 0.5}
        assert rec.finalize_calls == [("res", {"k": 1}, 0.5)]
        assert out.d2h_done_event.sync_calls == 1

    def test_retire_none_output_runs_callback(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        result = _run(svc.retire(None, (None, {}, 0.0)))
        assert result == {"step": None, "step_time": 0.0}
        assert rec.finalize_calls == [(None, {}, 0.0)]

    def test_retire_all_ready_drains_only_ready_prefix(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        out0 = _build_output(0, ready=True)
        out1 = _build_output(1, ready=False)
        out2 = _build_output(2, ready=True)
        svc.enqueue(0, out0, ("r0", {}, 0.0))
        svc.enqueue(1, out1, ("r1", {}, 0.0))
        svc.enqueue(2, out2, ("r2", {}, 0.0))
        results = _run(svc.retire_all_ready())
        assert [r["step"] for r in results] == ["r0"]
        # out1 still in flight blocks out2's retirement (ordering).
        assert svc.inflight_count == 2
        # Now mark out1 ready; entire prefix drains in order.
        out1.d2h_done_event.ready = True
        results = _run(svc.retire_all_ready())
        assert [r["step"] for r in results] == ["r1", "r2"]
        assert svc.inflight_count == 0

    def test_drain_on_shutdown_retires_all_in_step_order(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        for sid in range(3):
            svc.enqueue(sid, _build_output(sid, ready=False), (f"r{sid}", {}, 0.0))
        results = _run(svc.drain())
        assert [r["step"] for r in results] == ["r0", "r1", "r2"]
        assert svc.inflight_count == 0

    def test_drain_on_suspend_blocks_until_finalized(self):
        """Drain on suspend ignores readiness and synchronizes each event."""
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        outs = [_build_output(i, ready=False) for i in range(2)]
        for sid, out in enumerate(outs):
            svc.enqueue(sid, out, (f"r{sid}", {}, 0.0))
        _run(svc.drain())
        for out in outs:
            assert out.d2h_done_event.sync_calls == 1

    def test_close_blocks_further_enqueue(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        svc.close()
        assert svc.closed is True
        with pytest.raises(RuntimeError):
            svc.enqueue(0, None, (None, {}, 0.0))

    def test_request_id_reuse_drains_referencing_entries(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        svc.enqueue(
            0, _build_output(0, ready=True), ("r0", {}, 0.0), request_ids=(7,)
        )
        svc.enqueue(
            1, _build_output(1, ready=False), ("r1", {}, 0.0), request_ids=(8,)
        )
        # Reuse id 7: only the first entry references it; that one entry drains.
        drained = _run(svc.await_request_id_release(7))
        assert [r["step"] for r in drained] == ["r0"]
        assert svc.inflight_count == 1
        # Reuse id 99 (not referenced anywhere): no-op.
        drained = _run(svc.await_request_id_release(99))
        assert drained == []
        assert svc.inflight_count == 1

    def test_referenced_request_ids(self):
        rec = _Recorder()
        svc = StepRetirementService(rec.finalize)
        svc.enqueue(
            0, _build_output(0, ready=False), (None, {}, 0.0), request_ids=(1, 2)
        )
        svc.enqueue(
            1, _build_output(1, ready=False), (None, {}, 0.0), request_ids=(2, 3)
        )
        assert svc.referenced_request_ids() == {1, 2, 3}
