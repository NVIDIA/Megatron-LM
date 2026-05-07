# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator


def _make_context(
    paused_request_count=0,
    total_request_count=0,
    request_kv_block_counts=None,
    request_to_kv_block_ids=None,
):
    """Build a minimal DynamicInferenceContext-like fake for the allocator."""
    if request_kv_block_counts is None:
        request_kv_block_counts = torch.zeros(8, dtype=torch.int32)
    if request_to_kv_block_ids is None:
        request_to_kv_block_ids = -torch.ones((8, 4), dtype=torch.int32)
    return SimpleNamespace(
        paused_request_count=paused_request_count,
        total_request_count=total_request_count,
        request_kv_block_counts=request_kv_block_counts,
        request_to_kv_block_ids=request_to_kv_block_ids,
    )


class TestKVBlockAllocatorNoPrefixCaching:

    def test_init_basic_invariants(self):
        """Constructor records counts, reserves dummy block, and seeds the block bag."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2, enable_prefix_caching=False)
        assert a.total_count == 10
        assert a.total_avail == 9  # total_count - 1 (dummy)
        assert a.paused_count == 2
        assert a.active_count == 7  # 10 - 2 - 1 dummy
        assert a.dummy_block_idx == 9
        # block_bag is arange(total_count).
        assert a.block_bag.tolist() == list(range(10))
        # No prefix-caching state when disabled.
        assert not hasattr(a, "block_hashes")
        assert a.block_routing == {}

    def test_init_active_count_must_be_at_least_one(self):
        """An allocator with paused_count >= total_count - 1 hits the active>=1 assertion."""
        ctx = _make_context()
        with pytest.raises(AssertionError):
            KVBlockAllocator(ctx, total_count=3, paused_count=2)  # active = 0

    def test_str_reports_usage(self):
        """__str__ formats current usage counts."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        s = str(a)
        assert "total" in s
        assert "active" in s
        assert "paused" in s

    def test_get_total_used_starts_at_zero(self):
        """get_total_used = total_count - total_avail - 1 (dummy); zero before any allocation."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        assert a.get_total_used() == 0

    def test_is_memory_available_fast_path(self):
        """is_memory_available short-circuits True when free pool has enough."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        assert a.is_memory_available(5) is True

    def test_is_memory_available_returns_false_without_caching(self):
        """Without prefix caching, no eviction path; returns False when free pool exhausted."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=4, paused_count=1)  # total_avail = 3
        assert a.is_memory_available(5) is False

    def test_allocate_memory_blocks_basic(self):
        """allocate_memory_blocks pops num_blocks IDs from the top of the bag."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        # total_avail = 9 after init
        ids = a.allocate_memory_blocks(3)
        assert ids is not None
        assert ids.numel() == 3
        assert a.total_avail == 6  # 9 - 3
        # IDs allocated from positions [6:9]
        expected = a.block_bag[6:9].tolist()
        assert ids.tolist() == expected

    def test_allocate_memory_blocks_returns_none_when_insufficient(self):
        """allocate_memory_blocks returns None when free pool is too small (no caching)."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=4, paused_count=1)
        assert a.allocate_memory_blocks(5) is None

    def test_release_memory_blocks_returns_to_pool(self):
        """release_memory_blocks puts IDs back in the bag and bumps total_avail."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        ids = a.allocate_memory_blocks(3)
        avail_after_alloc = a.total_avail
        a.release_memory_blocks(ids)
        assert a.total_avail == avail_after_alloc + 3

    def test_release_memory_blocks_empty_is_noop(self):
        """release_memory_blocks([]) does not modify state."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        before = a.total_avail
        a.release_memory_blocks(torch.tensor([], dtype=torch.int32))
        assert a.total_avail == before

    def test_reset_restores_initial_state(self):
        """reset resets total_avail and the block bag to a fresh arange."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        a.allocate_memory_blocks(4)
        a.reset()
        assert a.total_avail == 9  # total_count - 1
        assert a.block_bag.tolist() == list(range(10))
        assert a.block_routing == {}

    def test_get_active_used_no_prefix_caching(self):
        """get_active_used sums request_kv_block_counts over the active range."""
        ctx = _make_context(
            paused_request_count=1,
            total_request_count=4,
            request_kv_block_counts=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0], dtype=torch.int32),
        )
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        # active range = [1:4], sum = 2+3+4 = 9
        assert a.get_active_used() == 9

    def test_get_paused_used_no_prefix_caching(self):
        """get_paused_used sums request_kv_block_counts over the paused prefix."""
        ctx = _make_context(
            paused_request_count=2,
            request_kv_block_counts=torch.tensor([5, 7, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        )
        a = KVBlockAllocator(ctx, total_count=10, paused_count=3)
        assert a.get_paused_used() == 12  # 5 + 7

    def test_get_active_and_paused_avail(self):
        """get_active_avail / get_paused_avail = capacity - used."""
        ctx = _make_context()
        a = KVBlockAllocator(ctx, total_count=10, paused_count=2)
        # No requests => used == 0; avail == capacity.
        assert a.get_active_avail() == a.active_count
        assert a.get_paused_avail() == a.paused_count


class TestKVBlockAllocatorWithPrefixCaching:

    def test_init_creates_prefix_caching_state(self):
        """Prefix-caching mode allocates block_hashes, ref_counts, and the hash map."""
        ctx = _make_context()
        a = KVBlockAllocator(
            ctx,
            total_count=8,
            paused_count=2,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        )
        # block_hashes initialised to -1 (uncomputed) for all blocks.
        assert a.block_hashes.shape == (8,)
        assert (a.block_hashes == -1).all().item()
        assert a.block_ref_counts.shape == (8,)
        assert (a.block_ref_counts == 0).all().item()
        # LRU policy adds timestamps.
        assert hasattr(a, "block_timestamps")
        assert a.block_timestamps.shape == (8,)
        # Hash map starts empty.
        assert a.kv_hash_to_block_id == {}

    def test_init_ref_zero_policy_no_timestamps(self):
        """REF_ZERO eviction policy does NOT allocate the LRU timestamps array."""
        ctx = _make_context()
        a = KVBlockAllocator(
            ctx,
            total_count=8,
            paused_count=2,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        assert not hasattr(a, "block_timestamps")

    def test_is_memory_available_ref_zero_no_eviction(self):
        """In REF_ZERO mode, no eviction path: returns False when pool is short."""
        ctx = _make_context()
        a = KVBlockAllocator(
            ctx,
            total_count=4,
            paused_count=1,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        assert a.is_memory_available(5) is False

    def test_register_kv_block_hashes_records_mappings(self):
        """register_kv_block_hashes populates block_hashes and the kv_hash_to_block_id dict."""
        ctx = _make_context()
        a = KVBlockAllocator(
            ctx, total_count=8, paused_count=2, enable_prefix_caching=True
        )
        a.register_kv_block_hashes(block_ids=[1, 3], block_hashes=[111, 333])
        assert a.block_hashes[1].item() == 111
        assert a.block_hashes[3].item() == 333
        assert a.kv_hash_to_block_id[111] == 1
        assert a.kv_hash_to_block_id[333] == 3

    def test_register_kv_block_hashes_empty_is_noop(self):
        """register_kv_block_hashes with empty inputs does nothing."""
        ctx = _make_context()
        a = KVBlockAllocator(
            ctx, total_count=8, paused_count=2, enable_prefix_caching=True
        )
        a.register_kv_block_hashes(block_ids=[], block_hashes=[])
        assert a.kv_hash_to_block_id == {}

    def test_allocate_initializes_ref_count_to_one(self):
        """In prefix-caching mode, allocated blocks start with ref_count=1."""
        ctx = _make_context()
        # REF_ZERO policy avoids touching context.prefix_cache_lru_clock during allocate.
        a = KVBlockAllocator(
            ctx,
            total_count=8,
            paused_count=2,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
        )
        ids = a.allocate_memory_blocks(2)
        assert (a.block_ref_counts[ids] == 1).all().item()

    def test_get_active_used_with_prefix_caching(self):
        """With prefix caching, get_active_used counts unique block ids in the active rows."""
        request_to_kv = -torch.ones((8, 4), dtype=torch.int32)
        # Active rows [1:3], with non-trivial block ids.
        request_to_kv[1] = torch.tensor([2, 3, -1, -1], dtype=torch.int32)
        request_to_kv[2] = torch.tensor([3, 4, 5, -1], dtype=torch.int32)
        ctx = _make_context(
            paused_request_count=1,
            total_request_count=3,
            request_to_kv_block_ids=request_to_kv,
        )
        a = KVBlockAllocator(
            ctx, total_count=10, paused_count=2, enable_prefix_caching=True
        )
        # Unique active block ids = {2, 3, 4, 5} = 4 entries.
        assert a.get_active_used() == 4

    def test_get_paused_used_with_prefix_caching(self):
        """With prefix caching, get_paused_used counts unique block ids in the paused rows."""
        request_to_kv = -torch.ones((8, 4), dtype=torch.int32)
        request_to_kv[0] = torch.tensor([1, 2, -1, -1], dtype=torch.int32)
        request_to_kv[1] = torch.tensor([1, 3, -1, -1], dtype=torch.int32)
        ctx = _make_context(
            paused_request_count=2,
            request_to_kv_block_ids=request_to_kv,
        )
        a = KVBlockAllocator(
            ctx, total_count=10, paused_count=3, enable_prefix_caching=True
        )
        # Unique paused block ids = {1, 2, 3} = 3 entries.
        assert a.get_paused_used() == 3
