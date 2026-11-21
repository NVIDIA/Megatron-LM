# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import heapq
import random

import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.prefix_cache_block_state import PrefixCacheBlockState


def _random_lru_state(rng, num_blocks):
    """Register a random prefix forest; pinned blocks keep their whole ancestor chain pinned,
    as running requests do. Timestamps come from a small range so many of them tie."""
    state = PrefixCacheBlockState(num_blocks, PrefixCachingEvictionPolicy.LRU)
    parents = [-1 if b == 0 or rng.random() < 0.2 else rng.randrange(b) for b in range(num_blocks)]
    registered = []
    for b in range(num_blocks):
        if rng.random() < 0.9 and (parents[b] == -1 or parents[b] in registered):
            registered.append(b)
    state.stamp_block_hashes(registered, [1000 + b for b in registered])
    state.record_parent_chain(registered, [parents[b] for b in registered])
    state.block_timestamps[registered] = torch.tensor(
        [rng.randint(0, 5) for _ in registered], dtype=torch.int64
    )
    pinned = set()
    for b in registered:
        if rng.random() < 0.1:
            while b != -1 and b not in pinned:
                pinned.add(b)
                b = parents[b]
    state.block_ref_counts[registered] = torch.tensor(
        [1 if b in pinned else 0 for b in registered], dtype=torch.int32
    )
    cached = [b for b in registered if b not in pinned]
    return state, {b: parents[b] for b in cached}, cached


def _reference_eviction_order(state, cached_parents, num_blocks_needed):
    """Peel every cached leaf through one heap keyed by (timestamp, block id)."""
    children = {b: 0 for b in cached_parents}
    for p in cached_parents.values():
        if p in children:
            children[p] += 1
    ts = state.block_timestamps.tolist()
    heap = [(ts[b], b) for b, count in children.items() if count == 0]
    heapq.heapify(heap)
    order = []
    while len(order) < num_blocks_needed:
        _, b = heapq.heappop(heap)
        order.append(b)
        p = cached_parents[b]
        if p in children:
            children[p] -= 1
            if children[p] == 0:
                heapq.heappush(heap, (ts[p], p))
    return order


def test_find_lru_evictable_matches_full_leaf_peel_order():
    """The evicted blocks and their order match a peel over all cached leaves, including
    timestamp ties, pinned ancestors, and requests for fewer blocks than there are leaves."""
    rng = random.Random(0)
    for _ in range(200):
        state, cached_parents, cached = _random_lru_state(rng, rng.randint(2, 80))
        for k in {1, 2, rng.randint(1, max(1, len(cached))), len(cached)}:
            evicted = state.find_lru_evictable(k)
            if k > len(cached):
                assert evicted is None
                continue
            assert evicted.dtype == torch.int64
            assert evicted.tolist() == _reference_eviction_order(state, cached_parents, k)
        assert state.find_lru_evictable(len(cached) + 1) is None
        assert state.find_lru_evictable(0).tolist() == []


def test_find_lru_evictable_takes_newly_exposed_parent_before_younger_leaves():
    """A parent exposed by evicting its only child is older than the remaining leaves, so it
    is the next victim even though it was not a leaf when the call started.

        A(ts 1) -> B(ts 2)      C(ts 5)  D(ts 6)  E(ts 7)
    """
    state = PrefixCacheBlockState(8, PrefixCachingEvictionPolicy.LRU)
    blocks = [0, 1, 2, 3, 4]  # A, B, C, D, E
    state.stamp_block_hashes(blocks, [10, 20, 30, 40, 50])
    state.record_parent_chain(blocks, [-1, 0, -1, -1, -1])
    state.block_timestamps[blocks] = torch.tensor([1, 2, 5, 6, 7], dtype=torch.int64)
    state.block_ref_counts[blocks] = 0

    assert state.find_lru_evictable(2).tolist() == [1, 0]
    assert state.find_lru_evictable(3).tolist() == [1, 0, 2]
