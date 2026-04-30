# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``DynamicAsyncPipeline`` (v3 plan commit 18).

Plan validation list:
- end-to-end correctness test
- nsys shows GPU forward N+1 starting before CPU apply_step_corrections
  for step N completes (manual / nsys-side; out of unit-test scope)
- async_overlap_debug_checks mode confirms no journal-entry leak after
  extended runs (≥10000 steps)
- async_overlap_queue_size=1 mode produces identical output to the
  legacy serial path

Heavy end-to-end coverage is run separately via the engine integration
tests; here we verify the pipeline-level invariants that are testable
without spinning up a real model.
"""

import asyncio

import pytest

from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput
from megatron.core.inference.engines.dynamic_engine import DynamicAsyncPipeline


class _FakeConfig:
    def __init__(self, queue_size: int = 2):
        self.async_overlap_queue_size = queue_size


class _FakeContext:
    def __init__(self, queue_size: int = 2):
        self.config = _FakeConfig(queue_size=queue_size)


class _FakeEngine:
    def __init__(self, queue_size: int = 2):
        self.context = _FakeContext(queue_size=queue_size)
        self.retirement = _FakeRetirement()
        self.async_forward_calls = 0
        self.async_bookkeep_calls = 0

    async def async_forward(self):
        self.async_forward_calls += 1
        return (None, {}, 0.0)

    async def async_bookkeep(self, *args, **kwargs):
        self.async_bookkeep_calls += 1
        return {"ok": True}


class _FakeRetirement:
    def __init__(self):
        self.retire_all_ready_calls = 0
        self.retire_calls = []

    async def retire_all_ready(self):
        self.retire_all_ready_calls += 1
        return []

    async def retire(self, output, payload):
        self.retire_calls.append((output, payload))
        return {"retired": True}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDynamicAsyncPipeline:
    def test_queue_size_from_config(self):
        engine = _FakeEngine(queue_size=3)
        p = DynamicAsyncPipeline(engine)
        assert p.queue_size == 3

    def test_queue_size_clamped_minimum_one(self):
        engine = _FakeEngine(queue_size=0)
        p = DynamicAsyncPipeline(engine)
        assert p.queue_size == 1

    def test_async_step_with_overlap_calls_legacy_path_at_queue_one(self):
        """With queue size 1 the pipeline drives the legacy
        async_forward + async_bookkeep pair so behavior matches the
        serial-equivalent path."""
        engine = _FakeEngine(queue_size=1)
        p = DynamicAsyncPipeline(engine)
        result = _run(p.async_step_with_overlap())
        assert result == {"ok": True}
        assert engine.async_forward_calls == 1
        assert engine.async_bookkeep_calls == 1
        assert engine.retirement.retire_all_ready_calls == 1

    def test_inflight_count_starts_zero(self):
        engine = _FakeEngine(queue_size=2)
        p = DynamicAsyncPipeline(engine)
        assert p.inflight_count == 0
