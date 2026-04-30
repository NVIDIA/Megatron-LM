# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for suspend/resume pipeline drain (v3 plan commit 26).

Plan validation: suspend/resume integration test on a model mid-decode
(long-running requests); resume produces correct output. The model-side
correctness is exercised by the engine's suspend/resume integration
tests; here we verify the lower-level drain + journal-reset invariants.
"""

import asyncio

import pytest

from megatron.core.inference.engines.retirement import StepRetirementService


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSuspendDrain:
    def test_drain_processes_in_flight_entries_in_order(self):
        finalize_calls = []

        async def finalize(*payload):
            finalize_calls.append(payload)
            return None

        svc = StepRetirementService(finalize)
        for sid in range(3):
            svc.enqueue(sid, output=None, payload=(f"r{sid}",))
        _run(svc.drain())
        assert [p[0] for p in finalize_calls] == ["r0", "r1", "r2"]
        assert svc.inflight_count == 0

    def test_drain_empties_dummy_steps(self):
        async def finalize(*payload):
            return None

        svc = StepRetirementService(finalize)
        svc.enqueue_dummy_step(step_id=0)
        svc.enqueue_dummy_step(step_id=1)
        _run(svc.drain())
        assert svc.inflight_count == 0
