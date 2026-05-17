# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator

TOTAL_COUNT = 10
PAUSED_COUNT = 2
MAX_REQUESTS = 8
MAX_BLOCKS_PER_REQ = 4


def _make_context(
    paused_request_count=0,
    total_request_count=0,
    request_kv_block_counts=None,
    request_to_kv_block_ids=None,
):
    """Build a minimal DynamicInferenceContext-like fake for the allocator."""
    if request_kv_block_counts is None:
        request_kv_block_counts = torch.zeros(MAX_REQUESTS, dtype=torch.int32)
    if request_to_kv_block_ids is None:
        request_to_kv_block_ids = -torch.ones((MAX_REQUESTS, MAX_BLOCKS_PER_REQ), dtype=torch.int32)
    return SimpleNamespace(
        paused_request_count=paused_request_count,
        total_request_count=total_request_count,
        request_kv_block_counts=request_kv_block_counts,
        request_to_kv_block_ids=request_to_kv_block_ids,
    )


def test_allocate_release_reset_round_trip_no_prefix_caching():
    """End-to-end exercise of the no-prefix-caching path: allocate from the
    bag (popping IDs off the top), release returns them, reset rewinds.

    Also covers the surrounding invariants the allocator must preserve:
    total_avail bookkeeping, the active_count >= 1 assertion at init, the
    is_memory_available fast-path + no-eviction fallback, and the noop
    behaviour of release([]).
    """
    ctx = _make_context()

    # The init's active_count >= 1 assertion fires when paused saturates the pool.
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, total_count=3, paused_count=2)  # active = 0

    a = KVBlockAllocator(ctx, total_count=TOTAL_COUNT, paused_count=PAUSED_COUNT)
    # Initial state: TOTAL_COUNT - 1 (dummy block) available, nothing used.
    assert a.total_avail == TOTAL_COUNT - 1
    assert a.get_total_used() == 0
    # is_memory_available short-circuits True when free pool has enough.
    assert a.is_memory_available(5) is True

    # Allocate 3 → pop IDs off the top of the bag.
    ids = a.allocate_memory_blocks(3)
    assert ids is not None and ids.numel() == 3
    assert a.total_avail == TOTAL_COUNT - 1 - 3

    # Empty release is a no-op; non-empty release returns IDs to the bag.
    before = a.total_avail
    a.release_memory_blocks(torch.tensor([], dtype=torch.int32))
    assert a.total_avail == before
    a.release_memory_blocks(ids)
    assert a.total_avail == before + 3

    # Free pool exhausted: without prefix caching there's no eviction path,
    # so both is_memory_available and allocate_memory_blocks return failure.
    small_alloc = KVBlockAllocator(ctx, total_count=4, paused_count=1)  # total_avail = 3
    assert small_alloc.is_memory_available(5) is False
    assert small_alloc.allocate_memory_blocks(5) is None

    # reset rewinds the bag back to arange(total_count) and clears routing state.
    a.allocate_memory_blocks(4)
    a.reset()
    assert a.total_avail == TOTAL_COUNT - 1
    assert a.block_bag.tolist() == list(range(TOTAL_COUNT))
    assert a.block_routing == {}


@pytest.mark.parametrize(
    "scope,paused,total,counts,expected_active,expected_paused",
    [
        # active_used = sum over [paused:total]; paused_used = sum over [:paused].
        ("nonempty", 1, 4, [1, 2, 3, 4, 0, 0, 0, 0], 9, 1),
        ("paused_only", 2, 2, [5, 7, 0, 0, 0, 0, 0, 0], 0, 12),
    ],
)
def test_block_usage_counts_no_prefix_caching(
    scope, paused, total, counts, expected_active, expected_paused
):
    """get_active_used / get_paused_used sum request_kv_block_counts over the
    [paused:total] and [:paused] slices respectively."""
    ctx = _make_context(
        paused_request_count=paused,
        total_request_count=total,
        request_kv_block_counts=torch.tensor(counts, dtype=torch.int32),
    )
    a = KVBlockAllocator(ctx, total_count=TOTAL_COUNT, paused_count=3)
    assert a.get_active_used() == expected_active
    assert a.get_paused_used() == expected_paused
    assert a.get_active_avail() == a.active_count - expected_active
    assert a.get_paused_avail() == a.paused_count - expected_paused


@pytest.mark.parametrize(
    "policy,expect_timestamps",
    [(PrefixCachingEvictionPolicy.LRU, True), (PrefixCachingEvictionPolicy.REF_ZERO, False)],
)
def test_prefix_caching_state_layout(policy, expect_timestamps):
    """Prefix-caching mode allocates block_hashes (initially -1) and ref_counts
    (initially 0). LRU policy also allocates timestamps; REF_ZERO does not."""
    a = KVBlockAllocator(
        _make_context(),
        total_count=8,
        paused_count=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=policy,
    )
    assert (a.block_hashes == -1).all().item()
    assert (a.block_ref_counts == 0).all().item()
    assert a.kv_hash_to_block_id == {}
    assert hasattr(a, "block_timestamps") is expect_timestamps


def test_prefix_caching_allocate_and_hash_registration():
    """allocate_memory_blocks initialises ref_count=1; register_kv_block_hashes
    populates both block_hashes[] and the kv_hash_to_block_id dict; the
    `is_memory_available` short-circuit returns False under REF_ZERO when
    the free pool can't satisfy and no cached blocks are evictable."""
    a = KVBlockAllocator(
        _make_context(),
        total_count=8,
        paused_count=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )

    # Newly allocated blocks have ref_count == 1.
    ids = a.allocate_memory_blocks(2)
    assert (a.block_ref_counts[ids] == 1).all().item()

    # Hash registration populates both the tensor and the dict.
    a.register_kv_block_hashes(block_ids=[1, 3], block_hashes=[111, 333])
    assert a.block_hashes[1].item() == 111
    assert a.block_hashes[3].item() == 333
    assert a.kv_hash_to_block_id == {111: 1, 333: 3}

    # Empty inputs are a no-op (avoids zero-element tensor construction).
    a.register_kv_block_hashes(block_ids=[], block_hashes=[])
    assert a.kv_hash_to_block_id == {111: 1, 333: 3}

    # REF_ZERO has no eviction path when the free pool is short.
    small = KVBlockAllocator(
        _make_context(),
        total_count=4,
        paused_count=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )
    assert small.is_memory_available(5) is False


@pytest.mark.parametrize(
    "paused,total,active_assignments,paused_assignments,expected_active,expected_paused",
    [
        # active rows [1:3] reference {2,3,4,5}; no paused rows assigned.
        (1, 3, {1: [2, 3, -1, -1], 2: [3, 4, 5, -1]}, {}, 4, 0),
        # paused rows [:2] reference {1,2,3}; no active rows assigned.
        (2, 2, {}, {0: [1, 2, -1, -1], 1: [1, 3, -1, -1]}, 0, 3),
    ],
)
def test_block_usage_counts_with_prefix_caching(
    paused, total, active_assignments, paused_assignments, expected_active, expected_paused
):
    """With prefix caching, get_active_used / get_paused_used count UNIQUE
    block IDs (since multiple requests can reference the same cached block)."""
    request_to_kv = -torch.ones((MAX_REQUESTS, MAX_BLOCKS_PER_REQ), dtype=torch.int32)
    for row_idx, ids in {**active_assignments, **paused_assignments}.items():
        request_to_kv[row_idx] = torch.tensor(ids, dtype=torch.int32)
    ctx = _make_context(
        paused_request_count=paused,
        total_request_count=total,
        request_to_kv_block_ids=request_to_kv,
    )
    a = KVBlockAllocator(ctx, total_count=TOTAL_COUNT, paused_count=3, enable_prefix_caching=True)
    assert a.get_active_used() == expected_active
    assert a.get_paused_used() == expected_paused
