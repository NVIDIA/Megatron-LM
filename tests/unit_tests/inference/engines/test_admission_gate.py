# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the DP coordinator step-boundary admission gate (v3
plan commit 29).

Plan validation: distributed test with simulated coordinator delays;
ranks block at the boundary as expected; no
OptimisticLedgerDivergenceError is raised under any non-fault scenario.
The full distributed test runs over real coordinator processes; here we
verify the gate's await/publish contract on a single rank.
"""

import asyncio

import pytest

from megatron.core.inference.engines.admission_gate import StepBoundaryAdmissionGate


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestStepBoundaryAdmissionGate:
    def test_publish_then_wait_returns_admission_set(self):
        gate = StepBoundaryAdmissionGate()
        gate.publish(step_id=0, request_ids=[10, 20, 30])
        admitted = _run(gate.wait_for_admission(step_id=0))
        assert admitted == (10, 20, 30)

    def test_wait_then_publish_unblocks_waiter(self):
        gate = StepBoundaryAdmissionGate()

        async def scenario():
            wait_task = asyncio.create_task(gate.wait_for_admission(step_id=1))
            # Yield once so the waiter registers.
            await asyncio.sleep(0)
            assert not wait_task.done()
            gate.publish(step_id=1, request_ids=[7, 8])
            return await wait_task

        admitted = _run(scenario())
        assert admitted == (7, 8)

    def test_wait_with_timeout_raises_when_unpublished(self):
        gate = StepBoundaryAdmissionGate()
        with pytest.raises(asyncio.TimeoutError):
            _run(gate.wait_for_admission(step_id=2, timeout=0.05))

    def test_is_admitted_reflects_publish(self):
        gate = StepBoundaryAdmissionGate()
        assert gate.is_admitted(step_id=0) is False
        gate.publish(step_id=0, request_ids=[1])
        assert gate.is_admitted(step_id=0) is True

    def test_known_step_ids_sorted(self):
        gate = StepBoundaryAdmissionGate()
        gate.publish(step_id=5, request_ids=[])
        gate.publish(step_id=1, request_ids=[])
        gate.publish(step_id=3, request_ids=[])
        assert gate.known_step_ids() == [1, 3, 5]

    def test_reset_clears_state(self):
        gate = StepBoundaryAdmissionGate()
        gate.publish(step_id=0, request_ids=[42])
        gate.reset()
        assert gate.known_step_ids() == []
        assert gate.is_admitted(step_id=0) is False

    def test_publish_idempotent_with_same_step_id(self):
        gate = StepBoundaryAdmissionGate()
        gate.publish(step_id=0, request_ids=[1, 2])
        gate.publish(step_id=0, request_ids=[1, 2])
        admitted = _run(gate.wait_for_admission(step_id=0))
        assert admitted == (1, 2)
