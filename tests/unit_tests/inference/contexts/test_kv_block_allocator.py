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
    prefix_cache_lru_clock=0,
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
        prefix_cache_lru_clock=prefix_cache_lru_clock,
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
    assert (a.block_parent_hashes == 0).all().item()
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

    # Hash registration populates both the tensor and the dict. Parent hashes are
    # optional and default to 0 (root) when omitted.
    a.register_kv_block_hashes(block_ids=[1, 3], block_hashes=[111, 333])
    assert a.block_hashes[1].item() == 111
    assert a.block_hashes[3].item() == 333
    assert a.block_parent_hashes[1].item() == 0
    assert a.block_parent_hashes[3].item() == 0
    assert a.kv_hash_to_block_id == {111: 1, 333: 3}

    # When supplied, parent hashes are recorded per block.
    a.register_kv_block_hashes(block_ids=[2, 4], block_hashes=[222, 444], parent_hashes=[111, 222])
    assert a.block_parent_hashes[2].item() == 111
    assert a.block_parent_hashes[4].item() == 222

    # Mismatched parent-hash length is rejected.
    with pytest.raises(AssertionError):
        a.register_kv_block_hashes(block_ids=[5], block_hashes=[555], parent_hashes=[1, 2])

    # Empty inputs are a no-op (avoids zero-element tensor construction).
    a.register_kv_block_hashes(block_ids=[], block_hashes=[])
    assert a.kv_hash_to_block_id == {111: 1, 333: 3, 222: 2, 444: 4}

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


# ---------------------------------------------------------------------------
# LRU eviction: parent-chain safety
# ---------------------------------------------------------------------------


def _lru_allocator(total_count=16, paused_count=1):
    """LRU-mode prefix-caching allocator over a fresh fake context."""
    return KVBlockAllocator(
        _make_context(),
        total_count=total_count,
        paused_count=paused_count,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )


def _seed_cached_chain(a, block_ids, hashes, parents, timestamps):
    """Register a chain of cached (ref_count == 0) blocks with explicit LRU
    timestamps, bypassing the allocation path to control the layout directly."""
    a.register_kv_block_hashes(block_ids=block_ids, block_hashes=hashes, parent_hashes=parents)
    ids = torch.tensor(block_ids, dtype=torch.int64)
    a.block_ref_counts[ids] = 0  # cached / evictable
    a.block_timestamps[ids] = torch.tensor(timestamps, dtype=torch.int64)
    # Mark the blocks as out of the free pool so _deregister_blocks (which pushes
    # them back) keeps total_avail bookkeeping consistent.
    a.total_avail -= len(block_ids)


def _assert_prefix_invariant(a):
    """Every cached block must have its parent cached too (or be a root). This is
    exactly the invariant _find_kv_match_count relies on."""
    present = set(a.kv_hash_to_block_id.keys())
    for block_hash, block_id in a.kv_hash_to_block_id.items():
        parent = a.block_parent_hashes[block_id].item()
        if parent != 0:
            assert parent in present, (
                f"dangling child: block {block_id} (hash {block_hash}) parent "
                f"{parent} not cached"
            )


def test_evict_lru_never_orphans_a_child():
    """Regression: with chunked prefill an ancestor block can end up OLDER than
    its descendant. A naive oldest-first eviction would evict the parent and leave
    a dangling child; leaf-only eviction must evict the child instead."""
    a = _lru_allocator()
    # Chain b0 -> b1 -> b2. Parent b1 (ts=1) is older than child b2 (ts=5).
    _seed_cached_chain(
        a, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[0, 10, 20], timestamps=[1, 1, 5]
    )

    assert a.evict_lru_blocks(1) is True
    # The leaf (b2, hash 30) is evicted, not the older parent b1 (hash 20).
    assert a.kv_hash_to_block_id == {10: 0, 20: 1}
    assert a.block_hashes[2].item() == -1
    assert a.block_parent_hashes[2].item() == 0
    _assert_prefix_invariant(a)


def test_evict_lru_cascades_up_the_chain():
    """Evicting more blocks than there are leaves walks up the chain from the
    deepest descendant, always keeping the retained set descendant-closed."""
    a = _lru_allocator()
    _seed_cached_chain(
        a, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[0, 10, 20], timestamps=[1, 1, 5]
    )

    assert a.evict_lru_blocks(2) is True
    # b2 then b1 evicted; only the root b0 remains.
    assert a.kv_hash_to_block_id == {10: 0}
    _assert_prefix_invariant(a)


def test_evict_lru_normal_lru_order_when_leaf_is_oldest():
    """When the oldest block is already a leaf (the common partial-match case,
    where ancestors are refreshed and descendants are stale), plain LRU order
    applies and the oldest leaf is evicted first."""
    a = _lru_allocator()
    # Ancestors refreshed (ts=9); descendant stale (ts=3) and is the leaf.
    _seed_cached_chain(
        a, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[0, 10, 20], timestamps=[9, 9, 3]
    )

    assert a.evict_lru_blocks(1) is True
    assert a.kv_hash_to_block_id == {10: 0, 20: 1}
    _assert_prefix_invariant(a)


def test_evict_lru_branching_prefix_tree():
    """A shared parent with two divergent children (branching prefixes) must keep
    the parent cached until BOTH children are evicted."""
    a = _lru_allocator()
    # b0 is the parent of both b1 and b2 (e.g. prompts "P+X" and "P+Y").
    _seed_cached_chain(
        a, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[0, 10, 10], timestamps=[1, 2, 8]
    )

    # Evicting one block takes a leaf (b1, the older child), never the parent.
    assert a.evict_lru_blocks(1) is True
    assert a.kv_hash_to_block_id == {10: 0, 30: 2}
    _assert_prefix_invariant(a)

    # Evicting the second child leaves only the parent.
    assert a.evict_lru_blocks(1) is True
    assert a.kv_hash_to_block_id == {10: 0}
    _assert_prefix_invariant(a)


def test_evict_lru_insufficient_cached_blocks_returns_false():
    """When fewer cached blocks exist than requested, eviction fails without
    touching the cache."""
    a = _lru_allocator()
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[0, 10], timestamps=[1, 2])
    assert a.evict_lru_blocks(3) is False
    assert a.kv_hash_to_block_id == {10: 0, 20: 1}


def test_evict_lru_terminates_on_cyclic_parent_graph():
    """Safety bound: a hash collision could make the parent graph cyclic, which
    would otherwise make the subtree-max / depth recurrence spin forever. The
    num_cached iteration cap must let eviction terminate and still evict the
    requested number of blocks (a hang would time this test out)."""
    a = _lru_allocator()
    # 2-cycle: block 0's parent hash is 20 (block 1) and block 1's parent hash is
    # 10 (block 0). register_kv_block_hashes never produces this — we seed it
    # directly to model the pathological collision case.
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[20, 10], timestamps=[1, 2])
    assert int(a.get_evictable_block_count()) == 2

    # Terminates (no hang) and evicts exactly one of the two cached blocks.
    assert a.evict_lru_blocks(1) is True
    assert int(a.get_evictable_block_count()) == 1
    assert len(a.kv_hash_to_block_id) == 1

    # A longer 3-cycle also terminates and can be fully evicted.
    b = _lru_allocator()
    _seed_cached_chain(
        b, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[30, 10, 20], timestamps=[1, 2, 3]
    )
    assert b.evict_lru_blocks(3) is True
    assert b.kv_hash_to_block_id == {}


def test_is_memory_available_excludes_reserved_evictable():
    """reserved_evictable removes soon-to-be-pinned cached blocks from the
    evictable capacity, so availability matches what allocation can satisfy
    once those blocks (e.g. prefix matches) are pinned."""
    a = _lru_allocator(total_count=6, paused_count=1)
    # Drain the free pool: every block is allocated (ref_count == 1), none free.
    a.allocate_memory_blocks(a.total_avail)
    assert a.total_avail == 0
    # Mark two blocks as cached/evictable, mirroring an LRU release: ref_count
    # drops to 0 and the hash is retained, but the block stays out of the free
    # pool (total_avail unchanged).
    a.register_kv_block_hashes(block_ids=[0, 1], block_hashes=[10, 20], parent_hashes=[0, 10])
    a.block_ref_counts[torch.tensor([0, 1])] = 0
    assert a.total_avail == 0
    assert int(a.get_evictable_block_count()) == 2

    # Both evictable blocks count toward availability by default.
    assert a.is_memory_available(2) is True
    # Reserving one (it will be pinned) leaves only one usable for the request.
    assert a.is_memory_available(2, reserved_evictable=1) is False
    assert a.is_memory_available(1, reserved_evictable=1) is True
    # Reserving all evictable blocks leaves nothing to satisfy a new block.
    assert a.is_memory_available(1, reserved_evictable=2) is False


def test_evict_lru_preserves_invariant_under_random_chains():
    """Property test: across many randomized multi-chain layouts and eviction
    counts, the retained cache always satisfies the parent-chain invariant."""
    torch.manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(2, 10, (1,)).item())
        a = _lru_allocator(total_count=n + 4)
        block_ids = list(range(n))
        # Build a forest: block k's parent is a random earlier block or a root.
        hashes = [100 + k for k in range(n)]
        parents = []
        for k in range(n):
            if k == 0 or int(torch.randint(0, 2, (1,)).item()) == 0:
                parents.append(0)  # root
            else:
                parents.append(hashes[int(torch.randint(0, k, (1,)).item())])
        timestamps = torch.randint(1, 20, (n,)).tolist()
        _seed_cached_chain(a, block_ids, hashes, parents, timestamps)

        k_evict = int(torch.randint(1, n + 1, (1,)).item())
        assert a.evict_lru_blocks(k_evict) is True
        assert len(a.kv_hash_to_block_id) == n - k_evict
        _assert_prefix_invariant(a)
