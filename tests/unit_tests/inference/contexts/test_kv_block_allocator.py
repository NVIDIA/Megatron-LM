# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator

POOL_SIZE = 10
PAUSED_LIMIT = 2
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
    pool_avail bookkeeping, paused-limit headroom validation, the computed
    allocatable count, the is_memory_available fast-path + no-eviction fallback,
    and the noop behaviour of release([]).
    """
    ctx = _make_context()

    # The paused limit must leave one usable non-dummy block for liveness.
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=3, paused_limit=2)
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=3, paused_limit=-1)
    with pytest.raises(AssertionError):
        KVBlockAllocator(ctx, pool_size=1, paused_limit=0)

    a = KVBlockAllocator(ctx, pool_size=POOL_SIZE, paused_limit=PAUSED_LIMIT)
    # Initial state: POOL_SIZE - 1 (dummy block) available, nothing used.
    assert a.pool_avail == POOL_SIZE - 1
    assert a.get_allocatable_count() == POOL_SIZE - 1
    assert a.get_total_used() == 0
    assert not hasattr(a, "active_count")
    assert not hasattr(a, "get_active_avail")
    assert not hasattr(a, "get_paused_avail")
    assert not hasattr(a, "get_allocatable_block_count")
    assert str(a) == "blocks: occupied 0/9; allocatable 9; active-used 0; paused-used 0/2"
    # is_memory_available short-circuits True when free pool has enough.
    assert a.is_memory_available(5) is True

    # Allocate 3 → pop IDs off the top of the bag.
    ids = a.allocate_memory_blocks(3)
    assert ids is not None and ids.numel() == 3
    assert a.pool_avail == POOL_SIZE - 1 - 3
    assert a.get_allocatable_count() == POOL_SIZE - 1 - 3

    # Empty release is a no-op; non-empty release returns IDs to the bag.
    before = a.pool_avail
    a.release_memory_blocks(torch.tensor([], dtype=torch.int32))
    assert a.pool_avail == before
    a.release_memory_blocks(ids)
    assert a.pool_avail == before + 3
    assert a.get_allocatable_count() == before + 3

    # Free pool exhausted: without prefix caching there's no eviction path,
    # so both is_memory_available and allocate_memory_blocks return failure.
    small_alloc = KVBlockAllocator(ctx, pool_size=4, paused_limit=1)
    assert small_alloc.pool_avail == 3
    assert small_alloc.get_allocatable_count() == 3
    assert small_alloc.is_memory_available(5) is False
    assert small_alloc.allocate_memory_blocks(5) is None

    # reset rewinds the bag back to arange(pool_size) and clears routing state.
    a.allocate_memory_blocks(4)
    a.reset()
    assert a.pool_avail == POOL_SIZE - 1
    assert a.get_allocatable_count() == POOL_SIZE - 1
    assert a.block_bag.tolist() == list(range(POOL_SIZE))
    assert a.block_routing == {}


def test_reset_under_inference_mode_preserves_mutable_block_bag():
    allocator = KVBlockAllocator(_make_context(), pool_size=8, paused_limit=0)
    original_block_bag = allocator.block_bag

    with torch.inference_mode():
        allocator.reset()

    blocks = allocator.allocate_memory_blocks(1)
    allocator.release_memory_blocks(blocks)

    assert allocator.block_bag is original_block_bag
    assert allocator.pool_avail == 7


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
    a = KVBlockAllocator(ctx, pool_size=POOL_SIZE, paused_limit=3)
    assert a.get_active_used() == expected_active
    assert a.get_paused_used() == expected_paused


@pytest.mark.parametrize(
    "policy,expect_timestamps",
    [(PrefixCachingEvictionPolicy.LRU, True), (PrefixCachingEvictionPolicy.REF_ZERO, False)],
)
def test_prefix_caching_state_layout(policy, expect_timestamps):
    """Prefix-caching mode allocates block_hashes (initially -1) and ref_counts
    (initially 0). LRU policy also allocates timestamps and the persisted
    prefix-forest bookkeeping (block_parent_id / block_child_count); REF_ZERO
    does not."""
    a = KVBlockAllocator(
        _make_context(),
        pool_size=8,
        paused_limit=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=policy,
    )
    assert (a.block_hashes == -1).all().item()
    assert (a.block_ref_counts == 0).all().item()
    assert a.kv_hash_to_block_id == {}
    assert hasattr(a, "block_timestamps") is expect_timestamps
    assert hasattr(a, "block_parent_id") is expect_timestamps
    assert hasattr(a, "block_child_count") is expect_timestamps
    if expect_timestamps:
        assert (a.block_parent_id == -1).all().item()
        assert (a.block_child_count == 0).all().item()


def test_prefix_caching_allocate_and_hash_registration():
    """allocate_memory_blocks initialises ref_count=1; register_kv_block_hashes
    populates both block_hashes[] and the kv_hash_to_block_id dict; the
    `is_memory_available` short-circuit returns False under REF_ZERO when
    the free pool can't satisfy and no cached blocks are evictable."""
    a = KVBlockAllocator(
        _make_context(),
        pool_size=8,
        paused_limit=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )

    # Newly allocated blocks have ref_count == 1.
    ids = a.allocate_memory_blocks(2)
    assert (a.block_ref_counts[ids] == 1).all().item()

    # Hash registration populates both the tensor and the dict. Parent hashes are
    # ignored under REF_ZERO (they only drive LRU eviction ordering), so this mode
    # keeps no per-block parent bookkeeping.
    a.register_kv_block_hashes(block_ids=[1, 3], block_hashes=[111, 333])
    assert a.block_hashes[1].item() == 111
    assert a.block_hashes[3].item() == 333
    assert not hasattr(a, "block_parent_id")
    assert a.kv_hash_to_block_id == {111: 1, 333: 3}

    # Supplying parent hashes is accepted (and ignored) under REF_ZERO.
    a.register_kv_block_hashes(block_ids=[2, 4], block_hashes=[222, 444], parent_hashes=[111, 222])

    # Mismatched parent-hash length is rejected regardless of policy.
    with pytest.raises(AssertionError):
        a.register_kv_block_hashes(block_ids=[5], block_hashes=[555], parent_hashes=[1, 2])

    # Empty inputs are a no-op (avoids zero-element tensor construction).
    a.register_kv_block_hashes(block_ids=[], block_hashes=[])
    assert a.kv_hash_to_block_id == {111: 1, 333: 3, 222: 2, 444: 4}

    # REF_ZERO has no eviction path when the free pool is short.
    small = KVBlockAllocator(
        _make_context(),
        pool_size=4,
        paused_limit=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )
    assert small.pool_avail == 3
    assert small.get_allocatable_count() == 3
    assert small.is_memory_available(5) is False


@pytest.mark.parametrize(
    "policy", [PrefixCachingEvictionPolicy.REF_ZERO, PrefixCachingEvictionPolicy.LRU]
)
def test_release_shared_block_aggregates_duplicate_references(policy):
    """Releasing a shared block once per request decrements every reference but
    returns the physical block to the free pool at most once."""
    a = KVBlockAllocator(
        _make_context(),
        pool_size=6,
        paused_limit=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=policy,
    )
    block = a.allocate_memory_blocks(1)
    block_id = int(block.item())
    raw_avail_after_allocate = a.pool_avail
    a.register_kv_block_hashes(block_ids=[block_id], block_hashes=[111])

    # Model two requests sharing the same registered prefix block, then release
    # both request references in one batched call.
    a.block_ref_counts[block_id] = 2
    a.release_memory_blocks(block.repeat(2))

    assert a.block_ref_counts[block_id].item() == 0
    if policy == PrefixCachingEvictionPolicy.REF_ZERO:
        assert a.block_hashes[block_id].item() == -1
        assert a.pool_avail == raw_avail_after_allocate + 1
        assert a.get_total_used() == 0
    else:
        # LRU keeps the physical block outside the free pool but exposes it
        # through get_allocatable_count because it is now evictable.
        assert a.block_hashes[block_id].item() == 111
        assert a.pool_avail == raw_avail_after_allocate
        assert a.get_allocatable_count() == raw_avail_after_allocate + 1
        assert a.get_total_used() == 1

    # In either policy the allocator tracks exactly one allocatable physical
    # copy, whether raw-free or evictable.
    assert a.get_allocatable_count() == raw_avail_after_allocate + 1


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
    a = KVBlockAllocator(ctx, pool_size=POOL_SIZE, paused_limit=3, enable_prefix_caching=True)
    assert a.get_active_used() == expected_active
    assert a.get_paused_used() == expected_paused


def test_release_shared_block_decrements_once_per_owner():
    """A shared prefix block appears once per finishing owner in a batched
    release: each occurrence must decrement (scatter-accumulate), and a block
    reaching ref 0 with a duplicated ID is freed/deregistered exactly once."""
    # REF_ZERO: three owners of a shared block finish in stages, with a private
    # block mixed into the final batch.
    a = KVBlockAllocator(
        _make_context(),
        pool_size=8,
        paused_limit=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )
    ids = a.allocate_memory_blocks(2)  # ref_count == 1 each
    shared, private = int(ids[0]), int(ids[1])
    a.register_kv_block_hashes(block_ids=[shared], block_hashes=[111])
    a.block_ref_counts[shared] += 2  # two more owners pin the shared block -> ref 3
    avail0 = a.pool_avail

    # One owner finishes alone: ref 3 -> 2, nothing freed yet.
    a.release_memory_blocks(torch.tensor([shared], dtype=torch.int32))
    assert a.block_ref_counts[shared].item() == 2
    assert a.pool_avail == avail0
    assert 111 in a.kv_hash_to_block_id

    # The final two owners and the private request finish in one batch: the
    # shared block appears twice and both decrements must land (ref 2 -> 0).
    a.release_memory_blocks(torch.tensor([shared, private, shared], dtype=torch.int32))
    assert a.block_ref_counts[shared].item() == 0
    assert a.block_ref_counts[private].item() == 0
    # Two distinct blocks return to the pool; the shared one only once (not twice).
    assert a.pool_avail == avail0 + 2
    assert 111 not in a.kv_hash_to_block_id  # deregistered exactly once
    free_region = a.block_bag[: a.pool_avail].tolist()
    assert len(set(free_region)) == len(free_region)  # no double-returned id

    # LRU: a hashed shared block released by both owners in one batch must hit
    # ref 0 (becoming evictable), not stall at 1 with a leaked reference. A hashed
    # block stays cached for reuse rather than returning to the pool.
    lru = KVBlockAllocator(
        _make_context(),
        pool_size=8,
        paused_limit=2,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    lshared = int(lru.allocate_memory_blocks(1)[0])
    lru.register_kv_block_hashes(block_ids=[lshared], block_hashes=[333], parent_hashes=[0])
    lru.block_ref_counts[lshared] += 1  # second owner -> ref 2
    lru.release_memory_blocks(torch.tensor([lshared, lshared], dtype=torch.int32))
    assert lru.block_ref_counts[lshared].item() == 0
    assert int(lru.get_evictable_block_count()) == 1
    assert lru.block_hashes[lshared].item() == 333  # kept cached, not pool-returned


# ---------------------------------------------------------------------------
# LRU eviction: parent-chain safety
# ---------------------------------------------------------------------------


def _lru_allocator(pool_size=16, paused_limit=1):
    """LRU-mode prefix-caching allocator over a fresh fake context."""
    return KVBlockAllocator(
        _make_context(),
        pool_size=pool_size,
        paused_limit=paused_limit,
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
    # them back) keeps pool_avail bookkeeping consistent.
    a.pool_avail -= len(block_ids)


def _assert_prefix_invariant(a):
    """Every cached block must have its parent cached too (or be a root). This is
    exactly the invariant _find_kv_match_count relies on."""
    cached_ids = set(a.kv_hash_to_block_id.values())
    for block_hash, block_id in a.kv_hash_to_block_id.items():
        parent_id = a.block_parent_id[block_id].item()
        if parent_id >= 0:
            assert parent_id in cached_ids, (
                f"dangling child: block {block_id} (hash {block_hash}) parent "
                f"block {parent_id} not cached"
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
    assert a.block_parent_id[2].item() == -1
    # Evicting the leaf drops it from its parent's child count.
    assert a.block_child_count[1].item() == 0
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


def test_evict_lru_cached_child_with_pinned_parent_treated_as_root():
    """Multi-turn / agentic case: a shared prefix block S0 stays pinned by an
    active request (ref_count > 0) while a descendant S1 from a finished turn is
    cached (ref_count == 0). S0 is not in the candidate set, so S1's parent
    resolves to -1 (S1 is treated as a forest root) and is safely evicted by
    normal LRU. The pinned parent must never be touched, even when it is the
    oldest block of all."""
    a = _lru_allocator()
    # Chain S0 -> S1, plus an unrelated cached root SX.
    a.register_kv_block_hashes(
        block_ids=[0, 1, 2], block_hashes=[10, 20, 30], parent_hashes=[0, 10, 0]
    )
    ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    # S0 pinned (active request), S1 and SX cached/evictable. S0 is the OLDEST
    # (ts=0) — a pin-blind oldest-first eviction would wrongly take it and orphan
    # nothing here, but in general orphan its children.
    a.block_ref_counts[ids] = torch.tensor([1, 0, 0], dtype=torch.int32)
    a.block_timestamps[ids] = torch.tensor([0, 1, 9], dtype=torch.int64)
    a.pool_avail -= 3

    # Only S1 and SX are candidates; the pinned S0 is excluded.
    assert int(a.get_evictable_block_count()) == 2

    # Evict one: S1 (ts=1) is the oldest candidate and a leaf; evicted first.
    assert a.evict_lru_blocks(1) is True
    assert a.kv_hash_to_block_id == {10: 0, 30: 2}  # S0 (pinned) + SX survive
    assert a.block_ref_counts[0].item() == 1  # parent still pinned
    assert a.block_hashes[0].item() == 10  # parent hash intact
    assert a.block_hashes[1].item() == -1  # child deregistered
    _assert_prefix_invariant(a)

    # Evict again: only SX remains as a candidate; S0 stays pinned throughout.
    assert a.evict_lru_blocks(1) is True
    assert a.kv_hash_to_block_id == {10: 0}
    assert a.block_ref_counts[0].item() == 1
    # The pinned parent can never be evicted, so a third eviction fails.
    assert a.evict_lru_blocks(1) is False


def test_evict_lru_partial_chain_eviction_peels_from_leaf_keeping_root():
    """Evicting fewer blocks than a chain's length peels from the leaf end, even
    when the root is the least-recently-used block.

    Chain A -> B -> C with the root A oldest (ts 1 < 2 < 3); evict 2. Eviction
    proceeds leaf-first, so C then B are removed and the root A is retained. The
    retained cache stays descendant-closed (no cached block is left with an
    evicted parent).
    """
    a = _lru_allocator()
    _seed_cached_chain(
        a, block_ids=[0, 1, 2], hashes=[10, 20, 30], parents=[0, 10, 20], timestamps=[1, 2, 3]
    )

    assert a.evict_lru_blocks(2) is True
    # Leaf C and its parent B are evicted; the root A survives despite being oldest.
    assert a.kv_hash_to_block_id == {10: 0}
    assert a.block_hashes[0].item() == 10  # root A retained
    assert a.block_hashes[1].item() == -1  # B deregistered
    assert a.block_hashes[2].item() == -1  # C deregistered
    _assert_prefix_invariant(a)


def test_evict_lru_insufficient_cached_blocks_returns_false():
    """When fewer cached blocks exist than requested, eviction fails without
    touching the cache."""
    a = _lru_allocator()
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[0, 10], timestamps=[1, 2])
    assert a.evict_lru_blocks(3) is False
    assert a.kv_hash_to_block_id == {10: 0, 20: 1}


def test_evict_lru_keeps_hottest_leaf_over_cold_interior_parent():
    """Optimality: leaf-peeling must retain the single most-recently-used block
    even when reaching it means evicting a colder interior parent elsewhere. A
    block is kept only for its own recency, never because a hot descendant props
    it up, so the hot leaf E survives while the colder interior block B is evicted.

        A(ts 1) -> B(ts 2) -> C(ts 5)
                          +-> F(ts 3)
                +-> D(ts 3) -> E(ts 5)
    """
    a = _lru_allocator(pool_size=8)
    # hashes: A=10, B=20, C=30, F=40, D=50, E=60
    _seed_cached_chain(
        a,
        block_ids=[0, 1, 2, 3, 4, 5],
        hashes=[10, 20, 30, 40, 50, 60],
        parents=[0, 10, 20, 20, 10, 50],
        timestamps=[1, 2, 5, 3, 3, 5],
    )

    assert a.evict_lru_blocks(3) is True
    # Evicted F(3), C(5), then B(2) once childless. Retains A, D, and the hottest
    # block E -- never evicting E in favor of the colder interior B.
    assert a.kv_hash_to_block_id == {10: 0, 50: 4, 60: 5}
    assert a.block_hashes[5].item() == 60  # hottest leaf E retained
    assert a.block_hashes[1].item() == -1  # cold interior B evicted
    _assert_prefix_invariant(a)


def test_register_existing_block_is_idempotent_and_keeps_parent_evictable():
    """Re-registering an already registered block must not disturb the prefix
    chain. Callers can re-offer a cached block they matched earlier (a prefill
    chunk boundary landing inside a matched block makes its slot part of the
    next chunk's registration span), and a second child increment on that
    block's parent is unrecoverable: the child can only be deregistered once, so
    the parent never reaches child_count == 0, is never an evictable leaf, and
    the leaf peel in evict_lru_blocks runs out of candidates while still
    counting the parent as cached.

        A(ts 1) -> B(ts 2)      B re-registered with its existing hash

    Both blocks are cached and evicting both must succeed.
    """
    a = _lru_allocator()
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[0, 10], timestamps=[1, 2])
    assert a.block_child_count[0].item() == 1

    # Re-register the child exactly as it stands: same block, hash and parent.
    a.register_kv_block_hashes(block_ids=[1], block_hashes=[20], parent_hashes=[10])

    # The chain is unchanged -- one child on the parent, not two.
    assert a.block_child_count[0].item() == 1
    assert a.block_child_count[1].item() == 0
    assert a.block_parent_id[1].item() == 0
    assert a.kv_hash_to_block_id == {10: 0, 20: 1}

    # Both cached blocks stay reachable by the leaf peel: B is evicted first,
    # which makes A childless and evictable in turn.
    assert int(a.get_evictable_block_count()) == 2
    assert a.evict_lru_blocks(2) is True
    assert a.kv_hash_to_block_id == {}
    _assert_prefix_invariant(a)


def test_register_mixed_batch_skips_only_the_already_registered_blocks():
    """A batch that mixes an already registered block with new ones registers
    the new blocks normally. The skipped block still resolves as a parent for
    its successor in the same batch, so the chain stays connected."""
    a = _lru_allocator()
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[0, 10], timestamps=[1, 2])

    # Block 1 is already registered; blocks 2 and 3 extend the chain past it.
    a.register_kv_block_hashes(
        block_ids=[1, 2, 3], block_hashes=[20, 30, 40], parent_hashes=[10, 20, 30]
    )
    a.block_ref_counts[torch.tensor([2, 3])] = 0
    a.pool_avail -= 2

    assert a.kv_hash_to_block_id == {10: 0, 20: 1, 30: 2, 40: 3}
    assert a.block_parent_id[2].item() == 1  # resolved through the skipped block
    assert a.block_parent_id[3].item() == 2
    assert a.block_child_count.tolist()[:4] == [1, 1, 1, 0]

    # The whole chain peels leaf-first without stalling.
    assert a.evict_lru_blocks(4) is True
    assert a.kv_hash_to_block_id == {}
    _assert_prefix_invariant(a)


def test_register_rejects_hash_change_on_a_registered_block():
    """Registering a live block under a hash other than the one it holds would
    overwrite its recorded parent while leaving the previous parent's child count
    raised. That is a bookkeeping error, not a no-op, and must fail loudly."""
    a = _lru_allocator()
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[0, 10], timestamps=[1, 2])

    with pytest.raises(AssertionError, match="different hash"):
        a.register_kv_block_hashes(block_ids=[1], block_hashes=[99], parent_hashes=[10])

    # A deregistered block is free to take a new hash.
    assert a.evict_lru_blocks(1) is True
    a.pool_avail -= 1
    a.register_kv_block_hashes(block_ids=[1], block_hashes=[99], parent_hashes=[10])
    assert a.kv_hash_to_block_id == {10: 0, 99: 1}
    assert a.block_child_count[0].item() == 1


def test_evict_lru_asserts_on_cyclic_parent_graph():
    """The parent graph is assumed acyclic (a forest). A hash collision producing
    a cycle exposes no leaf, so the peel cannot collect enough blocks; this is a
    bug and must fail loudly rather than silently under-evict."""
    a = _lru_allocator()
    # 2-cycle: block 0's parent hash is 20 (block 1) and block 1's parent hash is
    # 10 (block 0). register_kv_block_hashes never produces this — we seed it
    # directly to model the pathological collision case.
    _seed_cached_chain(a, block_ids=[0, 1], hashes=[10, 20], parents=[20, 10], timestamps=[1, 2])
    assert int(a.get_evictable_block_count()) == 2

    with pytest.raises(AssertionError):
        a.evict_lru_blocks(1)


def test_is_memory_available_excludes_soon_to_be_pinned_blocks():
    """potential_matched_count removes soon-to-be-pinned cached blocks from the
    evictable capacity, so availability matches what allocation can satisfy
    once those blocks (e.g. prefix matches) are pinned."""
    a = _lru_allocator(pool_size=6, paused_limit=1)
    # Drain the free pool: every block is allocated (ref_count == 1), none free.
    a.allocate_memory_blocks(a.pool_avail)
    assert a.pool_avail == 0
    assert a.get_allocatable_count() == 0
    # Mark two blocks as cached/evictable, mirroring an LRU release: ref_count
    # drops to 0 and the hash is retained, but the block stays out of the free
    # pool (pool_avail unchanged).
    a.register_kv_block_hashes(block_ids=[0, 1], block_hashes=[10, 20], parent_hashes=[0, 10])
    a.block_ref_counts[torch.tensor([0, 1])] = 0
    assert a.pool_avail == 0
    assert a.get_allocatable_count() == 2
    assert int(a.get_evictable_block_count()) == 2

    # Both evictable blocks count toward the computed allocatable count.
    assert a.is_memory_available(2) is True
    # Excluding one (it will be pinned) leaves only one usable for the request.
    assert a.is_memory_available(2, potential_matched_count=1) is False
    assert a.is_memory_available(1, potential_matched_count=1) is True
    # Excluding all evictable blocks leaves nothing to satisfy a new block.
    assert a.is_memory_available(1, potential_matched_count=2) is False

    # Allocation must evict the cached blocks into the raw free pool before
    # popping them, even though get_allocatable_count already reports their
    # capacity.
    allocated = a.allocate_memory_blocks(2)
    assert allocated is not None and allocated.numel() == 2
    assert a.pool_avail == 0
    assert a.get_allocatable_count() == 0
    assert a.kv_hash_to_block_id == {}


def _reference_leaf_peel(block_ids, hashes, parents, timestamps, k_evict):
    """Independent, straightforward greedy reference: repeatedly evict the
    currently-evictable leaf with the oldest (timestamp, block_id). Returns the
    set of evicted block ids. Used to pin the optimal eviction choice."""
    hash_to_id = dict(zip(hashes, block_ids))
    ts = dict(zip(block_ids, timestamps))
    child_count = {b: 0 for b in block_ids}
    parent_of = {}
    for b, p in zip(block_ids, parents):
        pid = hash_to_id.get(p)
        parent_of[b] = pid
        if pid is not None:
            child_count[pid] += 1

    import heapq as _heapq

    heap = [(ts[b], b) for b in block_ids if child_count[b] == 0]
    _heapq.heapify(heap)
    evicted = set()
    while heap and len(evicted) < k_evict:
        _, b = _heapq.heappop(heap)
        evicted.add(b)
        pid = parent_of[b]
        if pid is not None:
            child_count[pid] -= 1
            if child_count[pid] == 0:
                _heapq.heappush(heap, (ts[pid], pid))
    return evicted


def test_evict_lru_preserves_invariant_under_random_chains():
    """Property test: across many randomized multi-chain layouts and eviction
    counts, eviction (a) preserves the parent-chain invariant and (b) evicts
    exactly the optimal leaf-peel set (matched against an independent
    reference)."""
    torch.manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(2, 10, (1,)).item())
        a = _lru_allocator(pool_size=n + 4)
        block_ids = list(range(n))
        # Build a forest: block k's parent is a random earlier block or a root.
        hashes = [100 + k for k in range(n)]
        parents = []
        for k in range(n):
            if k == 0 or int(torch.randint(0, 2, (1,)).item()) == 0:
                parents.append(0)  # root
            else:
                parents.append(hashes[int(torch.randint(0, k, (1,)).item())])
        # Distinct timestamps so the optimal evicted set is unique and the
        # reference comparison is exact (no tie-break ambiguity).
        timestamps = torch.randperm(50)[:n].add(1).tolist()
        _seed_cached_chain(a, block_ids, hashes, parents, timestamps)

        k_evict = int(torch.randint(1, n + 1, (1,)).item())
        expected_evicted = _reference_leaf_peel(block_ids, hashes, parents, timestamps, k_evict)

        assert a.evict_lru_blocks(k_evict) is True
        retained = set(a.kv_hash_to_block_id.values())
        assert retained == set(block_ids) - expected_evicted
        assert len(retained) == n - k_evict
        _assert_prefix_invariant(a)
