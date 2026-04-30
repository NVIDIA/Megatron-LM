# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for dummy-step queue accounting (v3 plan commit 25).

Plan validation: distributed integration test with coordinator + EP=8 +
PP=4; long decode run; output matches single-node serial. The
distributed correctness portion is exercised by the engine's PP/EP
integration tests; here we verify the queue-depth-synchronizer
primitive: dummy steps occupy queue slots so PP/EP ranks don't diverge
in queue depth.
"""

import asyncio

import pytest

from megatron.core.inference.engines.retirement import StepRetirementService


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDummyStep:
    def test_enqueue_dummy_step_occupies_a_slot(self):
        async def finalize(*payload):
            return None

        svc = StepRetirementService(finalize)
        svc.enqueue_dummy_step(step_id=42)
        assert svc.inflight_count == 1

    def test_dummy_step_drains_through_drain_call(self):
        async def finalize(*payload):
            return {"step": payload[0]}

        svc = StepRetirementService(finalize)
        svc.enqueue_dummy_step(step_id=0)
        svc.enqueue_dummy_step(step_id=1)
        results = _run(svc.drain())
        assert len(results) == 2
        assert svc.inflight_count == 0

    def test_dummy_steps_count_against_capacity(self):
        """wait_for_capacity must drain dummy entries to make room for a
        real step launch."""
        finalize_calls = []

        async def finalize(*payload):
            finalize_calls.append(payload)
            return None

        svc = StepRetirementService(finalize)
        svc.enqueue_dummy_step(step_id=0)
        svc.enqueue_dummy_step(step_id=1)
        _run(svc.wait_for_capacity(max_inflight=2))
        assert svc.inflight_count == 1
        assert len(finalize_calls) == 1

    def test_close_blocks_further_dummy_enqueue(self):
        async def finalize(*payload):
            return None

        svc = StepRetirementService(finalize)
        svc.close()
        with pytest.raises(RuntimeError):
            svc.enqueue_dummy_step(step_id=0)
