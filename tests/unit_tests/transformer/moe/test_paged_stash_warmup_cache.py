# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe.paged_stash import PagedStashManager, paged_stash_reset


def _manager(allocate):
    manager = SimpleNamespace(
        iteration=0,
        status="begin",
        stash_buffers=None,
        overflow=Mock(),
        host_spill=Mock(),
        vp_size=1,
        paged_tensors_to_stash=[],
        paged_tensors_stash_in_progress=[],
    )

    def allocate_buffers(**kwargs):
        allocate()
        manager.stash_buffers = {}

    manager.allocate_stash_buffers = allocate_buffers
    return manager


@pytest.mark.parametrize("capturing", [False, True])
def test_warmup_cache_release_precedes_initial_stash_allocation(monkeypatch, capturing):
    calls = []
    manager = _manager(lambda: calls.append("allocate"))
    monkeypatch.setattr(PagedStashManager, "get_instance", lambda: manager)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    paged_stash_reset(enabled=False)
    paged_stash_reset()  # Record the first iteration's activation sizes.
    assert calls == []
    paged_stash_reset()  # Transition from recording to persistent storage.
    expected = ["allocate"] if capturing else ["synchronize", "empty_cache", "allocate"]
    assert calls == expected
    paged_stash_reset()  # Ordinary iterations must not flush the cache.
    assert calls == expected
    manager.stash_buffers = None  # Fallback reallocation is not another initial warmup.
    paged_stash_reset()
    assert calls == expected + ["allocate"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_initial_stash_releases_eager_cache_without_changing_live_tensors(monkeypatch):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.memory.get_allocator_backend() != "native":
        pytest.skip("Reserved-memory assertion requires the native caching allocator")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    canary = torch.arange(1024, device="cuda", dtype=torch.float32)
    stash = []
    reserved_at_allocation = []

    def allocate():
        reserved_at_allocation.append(torch.cuda.memory_reserved())
        stash.append(torch.empty(2**21, device="cuda", dtype=torch.uint8))

    manager = _manager(allocate)
    monkeypatch.setattr(PagedStashManager, "get_instance", lambda: manager)
    paged_stash_reset()

    # Earlier tests may leave a live tensor in a cached segment with a large free remainder.
    # The allocator would carve the eager tensor out of that remainder, and empty_cache() never
    # returns a segment that still holds a live block. Requesting more than all cached free
    # memory forces the eager tensor into a segment of its own, which empty_cache() can return.
    cached_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    eager_bytes = max(2**26, (cached_free // 2**21 + 1) * 2**21)
    reserved_without_eager = torch.cuda.memory_reserved()
    eager = torch.empty(eager_bytes, device="cuda", dtype=torch.uint8)
    assert (
        torch.cuda.memory_reserved() - reserved_without_eager >= eager_bytes
    ), "eager tensor must occupy its own segment"
    del eager
    torch.cuda.synchronize()

    paged_stash_reset()

    assert len(stash) == 1
    # The eager footprint is back with the driver before the persistent stash is allocated.
    assert reserved_at_allocation[0] <= reserved_without_eager
    assert torch.equal(canary.cpu(), torch.arange(1024, dtype=torch.float32))
