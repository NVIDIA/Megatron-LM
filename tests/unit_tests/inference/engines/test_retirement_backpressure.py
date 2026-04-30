# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for retirement-service backpressure (v3 plan commit 19).

Plan validation: test that artificially blocks retirement (sleep) and
asserts the engine blocks rather than crashing on pool exhaustion.
"""

import asyncio

import pytest

from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput
from megatron.core.inference.engines.retirement import StepRetirementService


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRetirementBackpressure:
    def test_wait_for_capacity_drains_until_under_limit(self):
        """When the in-flight count is at or above ``max_inflight``,
        wait_for_capacity drains FIFO entries until count < max_inflight
        so the caller can safely launch a new step without exceeding the
        pool."""
        finalize_calls = []

        async def finalize(*payload):
            finalize_calls.append(payload)
            return {"step": payload[0]}

        svc = StepRetirementService(finalize)
        for sid in range(3):
            svc.enqueue(sid, output=None, payload=(f"r{sid}",))
        # max_inflight=2 → drain entries until inflight_count < 2.
        # Starting at 3 entries, two drains land at count=1.
        results = _run(svc.wait_for_capacity(max_inflight=2))
        assert [r["step"] for r in results] == ["r0", "r1"]
        assert svc.inflight_count == 1

    def test_wait_for_capacity_no_op_when_below_limit(self):
        """Below the limit the call returns immediately with no drains."""
        async def finalize(*payload):
            return None

        svc = StepRetirementService(finalize)
        svc.enqueue(0, output=None, payload=(None,))
        results = _run(svc.wait_for_capacity(max_inflight=2))
        assert results == []
        assert svc.inflight_count == 1

    def test_artificial_retirement_block_never_crashes(self):
        """Simulate a retirement that sleeps. wait_for_capacity awaits the
        slow finalize without exception (the engine blocks, doesn't crash
        on pool exhaustion)."""
        finalize_calls = []

        async def slow_finalize(*payload):
            await asyncio.sleep(0.0)
            finalize_calls.append(payload)
            return None

        svc = StepRetirementService(slow_finalize)
        for sid in range(3):
            svc.enqueue(sid, output=None, payload=(f"r{sid}",))
        # max_inflight=3 → drain only the oldest so count<3.
        _run(svc.wait_for_capacity(max_inflight=3))
        assert len(finalize_calls) == 1
        assert svc.inflight_count == 2
