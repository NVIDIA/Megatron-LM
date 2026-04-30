# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``CUDAGraphCache`` (v3 plan commit 11).

Plan validation list:
- Memory budget test asserts cache stays within budget under stress.
- Eviction policy correctness (LRU).
- Capture-mode test covers each mode's behavior on a miss.
- Regression test that warmup with a known set of (slot, batch_dims)
  pairs fills the cache and never evicts during steady state.
"""

import pytest

from megatron.core.inference.contexts.graph_cache import (
    CAPTURE_MODE_ON_FIRST_USE,
    CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
    CAPTURE_MODE_WARMUP_ONLY,
    CUDAGraphCache,
    CudaGraphCaptureBudgetError,
)


def _capture_fn(handle, size):
    return lambda: (handle, size)


class TestCaptureModes:
    def test_warmup_only_miss_raises(self):
        cache = CUDAGraphCache(capture_mode=CAPTURE_MODE_WARMUP_ONLY)
        with pytest.raises(CudaGraphCaptureBudgetError):
            cache.get_or_capture(("slot0", "dims"), _capture_fn("g", 100))
        assert cache.metrics.miss_causes_eager_fallback == 1
        assert cache.metrics.captures == 0

    def test_on_first_use_miss_captures(self, recwarn):
        cache = CUDAGraphCache(capture_mode=CAPTURE_MODE_ON_FIRST_USE)
        h = cache.get_or_capture(("slot0", "dims"), _capture_fn("g", 100))
        assert h == "g"
        assert cache.metrics.captures == 1
        assert cache.metrics.miss_causes_capture == 1

    def test_on_first_use_with_eviction_evicts_under_budget(self):
        cache = CUDAGraphCache(
            capture_mode=CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
            memory_budget_bytes=200,
        )
        cache.get_or_capture(("slot0", "a"), _capture_fn("ga", 100))
        cache.get_or_capture(("slot0", "b"), _capture_fn("gb", 100))
        # Inserting a third 100-byte entry exceeds the budget; LRU
        # ("slot0", "a") gets evicted.
        cache.get_or_capture(("slot0", "c"), _capture_fn("gc", 100))
        assert ("slot0", "a") not in cache.keys()
        assert ("slot0", "b") in cache.keys()
        assert ("slot0", "c") in cache.keys()
        assert cache.metrics.evictions >= 1


class TestLRUEviction:
    def test_lookup_promotes_recency(self):
        cache = CUDAGraphCache(
            capture_mode=CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
            memory_budget_bytes=200,
        )
        cache.capture(("slot0", "a"), "ga", 100)
        cache.capture(("slot0", "b"), "gb", 100)
        # Touch "a" so "b" becomes LRU.
        assert cache.lookup(("slot0", "a")) == "ga"
        cache.capture(("slot0", "c"), "gc", 100)
        # "b" is now LRU and should have been evicted under the 200-byte
        # budget when "c" was inserted.
        assert ("slot0", "b") not in cache.keys()
        assert ("slot0", "a") in cache.keys()
        assert ("slot0", "c") in cache.keys()


class TestMemoryBudget:
    def test_cache_stays_within_budget_under_stress(self):
        cache = CUDAGraphCache(
            capture_mode=CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
            memory_budget_bytes=500,
        )
        # Stress: insert 50 entries of 100 bytes each — far over the budget.
        for i in range(50):
            cache.capture(("slot0", f"dims{i}"), f"g{i}", 100)
            assert cache.size_bytes <= 500
        # Final size at the budget cap.
        assert cache.size_bytes <= 500
        assert cache.metrics.evictions >= 45

    def test_max_captures_cap(self):
        cache = CUDAGraphCache(
            capture_mode=CAPTURE_MODE_ON_FIRST_USE_WITH_EVICTION,
            max_captures=3,
        )
        for i in range(10):
            cache.capture(("slot0", f"dims{i}"), f"g{i}", 100)
            assert len(cache) <= 3
        assert len(cache) == 3


class TestSnapshotKeyedSlots:
    def test_invalidate_slot_drops_only_that_slot(self):
        cache = CUDAGraphCache(capture_mode=CAPTURE_MODE_ON_FIRST_USE)
        cache.capture((0, "a"), "ga", 50)
        cache.capture((0, "b"), "gb", 50)
        cache.capture((1, "a"), "ha", 50)
        n = cache.invalidate_slot(0)
        assert n == 2
        assert (0, "a") not in cache.keys()
        assert (0, "b") not in cache.keys()
        assert (1, "a") in cache.keys()


class TestSteadyStateRegression:
    def test_warmup_fills_and_no_eviction_in_steady_state(self):
        """Plan: regression test that warmup with a known set of (slot,
        batch_dims) pairs fills the cache and never evicts during steady
        state (warmup_only mode never evicts)."""
        cache = CUDAGraphCache(capture_mode=CAPTURE_MODE_WARMUP_ONLY)
        warmup_keys = [(slot, dim) for slot in (0, 1) for dim in ("d1", "d2", "d3")]
        for key in warmup_keys:
            cache.capture(key, f"g_{key}", 100)
        assert len(cache) == len(warmup_keys)
        # Steady-state lookups: every key hits, none evicts.
        for _ in range(100):
            for key in warmup_keys:
                assert cache.lookup(key) == f"g_{key}"
        assert cache.metrics.evictions == 0
