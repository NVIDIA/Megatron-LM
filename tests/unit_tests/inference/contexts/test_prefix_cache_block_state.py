# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.prefix_cache_block_state import PrefixCacheBlockState


def _ids(values):
    """Build the int64 CPU id tensor `PrefixCacheBlockState` expects."""
    return torch.tensor(values, dtype=torch.int64, device="cpu")


class TestPrefixCacheBlockState:
    @pytest.mark.internal
    @pytest.mark.parametrize(
        "policy",
        [PrefixCachingEvictionPolicy.LRU, PrefixCachingEvictionPolicy.REF_ZERO],
    )
    def test_allocate_stamp_match_release_lifecycle(self, policy):
        """End-to-end lifecycle: allocate -> stamp hashes -> prefix-match -> release.

        Drives every method that runs under both policies; the policy-specific
        return shape of `on_release_compute_pool_returns` is then asserted
        separately for REF_ZERO and LRU.
        """
        s = PrefixCacheBlockState(total_count=8, eviction_policy=policy)
        assert s.is_lru == (policy == PrefixCachingEvictionPolicy.LRU)

        # Initial state: ref_counts all 0, hashes all -1, nothing evictable.
        assert torch.equal(s.block_ref_counts, torch.zeros(8, dtype=torch.int32))
        assert torch.equal(s.block_hashes, torch.full((8,), -1, dtype=torch.int64))
        assert int(s.get_evictable_block_count()) == 0
        assert s.extra_blocks_available() == 0

        # Allocate blocks [1, 3, 5]: ref_count = 1 each; timestamps stamped under LRU.
        s.on_allocate(_ids([1, 3, 5]), lru_clock=10)
        assert s.block_ref_counts[1] == 1
        assert s.block_ref_counts[3] == 1
        assert s.block_ref_counts[5] == 1
        if s.is_lru:
            assert s.block_timestamps[1] == 10
            assert s.block_timestamps[3] == 10
            assert s.block_timestamps[5] == 10
        else:
            # REF_ZERO never allocates the timestamp tensor.
            assert s.block_timestamps is None

        # Stamp hashes for blocks 1 and 3. Block 5 stays uncomputed (hash == -1).
        s.stamp_block_hashes([1, 3], [111, 333])
        assert s.block_hashes[1] == 111
        assert s.block_hashes[3] == 333
        assert s.block_hashes[5] == -1

        # Empty stamp is a silent no-op.
        s.stamp_block_hashes([], [])
        assert s.block_hashes[1] == 111

        # A second caller reuses [1, 3] via a prefix match: ref count -> 2.
        s.on_prefix_match(_ids([1, 3]), lru_clock=20)
        assert s.block_ref_counts[1] == 2
        assert s.block_ref_counts[3] == 2
        if s.is_lru:
            assert s.block_timestamps[1] == 20
            assert s.block_timestamps[3] == 20

        # Nothing is evictable while every cached block has a live owner.
        assert int(s.get_evictable_block_count()) == 0
        assert s.extra_blocks_available() == 0

        # First caller releases [1, 3, 5]: counts decrement to [.., 1, .., 1, .., 0, ..].
        blocks_for_pool, hashes_to_drop = s.on_release_compute_pool_returns(_ids([1, 3, 5]))
        if policy == PrefixCachingEvictionPolicy.REF_ZERO:
            # Only block 5 (now ref==0) returns; its hash is the -1 sentinel.
            assert blocks_for_pool.tolist() == [5]
            assert hashes_to_drop == [-1]
            # Block 5 stays fully reset; 1 and 3 unchanged in state besides ref count.
            assert s.block_hashes[5] == -1
            assert s.block_ref_counts[5] == 0
        else:
            # LRU: only unregistered (no-hash) zero-ref blocks come back. Block 5 qualifies;
            # 1 and 3 (still ref==1) do not.
            assert blocks_for_pool.tolist() == [5]
            assert hashes_to_drop == []
            assert s.block_ref_counts[5] == 0
            assert s.block_hashes[5] == -1

        # Second caller releases [1, 3]: both go to ref_count 0.
        blocks_for_pool, hashes_to_drop = s.on_release_compute_pool_returns(_ids([1, 3]))
        if policy == PrefixCachingEvictionPolicy.REF_ZERO:
            # Both zero-ref AND deregistered: returned for pool + their hashes for the registry.
            assert sorted(blocks_for_pool.tolist()) == [1, 3]
            assert sorted(hashes_to_drop) == [111, 333]
            assert s.block_hashes[1] == -1
            assert s.block_hashes[3] == -1
            # REF_ZERO never accumulates an evictable reservoir.
            assert int(s.get_evictable_block_count()) == 0
            assert s.extra_blocks_available() == 0
        else:
            # LRU: registered zero-ref blocks stay cached for later reuse; nothing returned now.
            assert blocks_for_pool.numel() == 0
            assert hashes_to_drop == []
            # Hashes survive — that's the whole point of LRU caching.
            assert s.block_hashes[1] == 111
            assert s.block_hashes[3] == 333
            # Now both blocks are in the evictable reservoir.
            assert int(s.get_evictable_block_count()) == 2
            assert s.extra_blocks_available() == 2

    @pytest.mark.internal
    def test_lru_eviction_picks_oldest(self):
        """`find_lru_evictable` / `try_lru_evict_for_pool` return the oldest cached blocks.

        Covers the LRU selection ordering, the "not enough victims" branch, and the
        state-reset side-effect of `try_lru_evict_for_pool`.
        """
        s = PrefixCacheBlockState(total_count=6, eviction_policy=PrefixCachingEvictionPolicy.LRU)

        # Allocate four blocks at distinct timestamps and stamp them.
        for bid, ts, h in [(0, 5, 50), (1, 9, 90), (2, 2, 20), (3, 7, 70)]:
            s.on_allocate(_ids([bid]), lru_clock=ts)
            s.stamp_block_hashes([bid], [h])

        # Release all four; under LRU they stay cached (registered zero-ref).
        s.on_release_compute_pool_returns(_ids([0, 1, 2, 3]))
        assert int(s.get_evictable_block_count()) == 4

        # Oldest two are block 2 (ts=2) then block 0 (ts=5).
        victims = s.find_lru_evictable(num_blocks_needed=2)
        assert victims is not None
        assert victims.tolist() == [2, 0]

        # Asking for more than available -> None (never a partial answer).
        assert s.find_lru_evictable(num_blocks_needed=5) is None

        # `try_lru_evict_for_pool` deregisters the chosen victims and returns their hashes.
        result = s.try_lru_evict_for_pool(num_blocks_needed=2)
        assert result is not None
        evicted, hashes = result
        assert evicted.tolist() == [2, 0]
        assert sorted(hashes) == [20, 50]
        # Evicted blocks are reset.
        for bid in (0, 2):
            assert s.block_hashes[bid] == -1
            assert s.block_ref_counts[bid] == 0
            assert s.block_timestamps[bid] == 0
        # The other two stay cached.
        assert int(s.get_evictable_block_count()) == 2

        # Not enough remaining -> try_lru_evict_for_pool returns None.
        assert s.try_lru_evict_for_pool(num_blocks_needed=10) is None

    @pytest.mark.internal
    def test_ref_zero_has_no_lru_machinery(self):
        """REF_ZERO disables every LRU-specific code path.

        - `block_timestamps` is never allocated.
        - `update_timestamps` is a silent no-op (does not crash on the missing tensor).
        - `extra_blocks_available` is hard-coded to 0 even if a cached reservoir exists.
        - `try_lru_evict_for_pool` returns None unconditionally.
        - `find_lru_evictable` asserts under REF_ZERO.
        """
        s = PrefixCacheBlockState(total_count=4, eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        assert s.is_lru is False
        assert s.block_timestamps is None

        # Silent no-op — must not touch the None tensor.
        s.update_timestamps(_ids([0, 1]), lru_clock=42)

        # Force a state that *looks* evictable; the policy still refuses to count it.
        s.block_ref_counts[0] = 0
        s.block_hashes[0] = 123
        assert int(s.get_evictable_block_count()) == 1
        assert s.extra_blocks_available() == 0

        # No LRU eviction path under REF_ZERO.
        assert s.try_lru_evict_for_pool(num_blocks_needed=1) is None
        with pytest.raises(AssertionError):
            s.find_lru_evictable(num_blocks_needed=1)

    @pytest.mark.internal
    def test_reset_clears_all_state(self):
        """`reset` zeroes ref counts and timestamps and refills hashes with -1."""
        s = PrefixCacheBlockState(total_count=4, eviction_policy=PrefixCachingEvictionPolicy.LRU)
        s.on_allocate(_ids([0, 1]), lru_clock=99)
        s.stamp_block_hashes([0, 1], [11, 22])

        s.reset()
        assert torch.equal(s.block_hashes, torch.full((4,), -1, dtype=torch.int64))
        assert torch.equal(s.block_ref_counts, torch.zeros(4, dtype=torch.int32))
        assert torch.equal(s.block_timestamps, torch.zeros(4, dtype=torch.int64))

    @pytest.mark.internal
    def test_deregister_blocks_returns_hashes_and_resets_state(self):
        """`deregister_blocks` returns the hashes to drop and resets the per-block state.

        Doubles as coverage of the empty-input fast path that callers rely on.
        """
        s = PrefixCacheBlockState(total_count=4, eviction_policy=PrefixCachingEvictionPolicy.LRU)
        s.on_allocate(_ids([0, 2]), lru_clock=5)
        s.stamp_block_hashes([0, 2], [100, 200])

        # Empty input -> empty result and no state touched.
        assert s.deregister_blocks(_ids([])) == []
        assert s.block_hashes[0] == 100

        hashes = s.deregister_blocks(_ids([0, 2]))
        assert sorted(hashes) == [100, 200]
        for bid in (0, 2):
            assert s.block_hashes[bid] == -1
            assert s.block_ref_counts[bid] == 0
            assert s.block_timestamps[bid] == 0
