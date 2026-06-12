# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.core.inference.contexts.prefix_cache_registry import PrefixCacheRegistry


class TestPrefixCacheRegistry:
    @pytest.mark.internal
    def test_kv_lifecycle(self):
        r = PrefixCacheRegistry()

        # Register three parent-chained blocks; the full list is a hit.
        r.register_kv([1, 2, 3], [100, 200, 300])
        assert r.match_kv_prefix([100, 200, 300]) == ([1, 2, 3], 300)

        # Partial match walks back from the end past the unknown hash.
        assert r.match_kv_prefix([100, 200, 999]) == ([1, 2], 200)

        # No overlap -> empty prefix.
        assert r.match_kv_prefix([999, 998]) == ([], 0)

        # Eviction filters the -1 sentinel and ignores non-present hashes.
        assert r.evict_kv([-1, 200, 555]) == {200}
        assert r.match_kv_prefix([100]) == ([1], 100)

        r.clear_kv()
        assert r.match_kv_prefix([100, 300]) == ([], 0)

    @pytest.mark.internal
    def test_mamba_lifecycle_with_callback(self):
        r = PrefixCacheRegistry()
        captured: list = []
        r.set_mamba_evict_callback(captured.extend)

        # `hash <= 0` is the uncomputed sentinel; `register_mamba` skips it.
        r.register_mamba([1, 2, 3, 4], [10, 0, -1, 40])
        assert r.mamba_hash_to_block_id == {10: 1, 40: 4}

        # Farthest hit at index 2 (`40`) -> returns 3.
        assert r.match_mamba_farthest([10, 99, 40]) == 3
        # Bounded scan over [0, 2) excludes the farther hit and finds `10`.
        assert r.find_mamba_backoff([10, 99, 40], 2) == 1

        # Evict one present hash + the -1 sentinel; callback receives only the real block.
        assert r.evict_mamba([-1, 10]) == {10}
        assert captured == [1]

        # Evicting a non-present hash is a no-op and does not fire the callback.
        captured.clear()
        assert r.evict_mamba([999]) == set()
        assert captured == []

        r.clear_mamba()
        assert r.match_mamba_farthest([40]) == 0

    @pytest.mark.internal
    def test_kv_evict_cascades_to_mamba(self):
        r = PrefixCacheRegistry()
        captured: list = []
        r.set_mamba_evict_callback(captured.extend)

        # Block 2 sits in both caches; block 1 is KV-only; block 3 is KV-only.
        r.register_kv([1, 2, 3], [10, 20, 30])
        r.register_mamba([2], [20])

        # Evicting a KV-only block has no Mamba overlap, so no cascade.
        r.evict_kv([10])
        assert captured == []

        # Evicting the shared block clears the Mamba entry and fires the callback.
        assert r.evict_kv([20]) == {20}
        assert r.mamba_hash_to_block_id == {}
        assert captured == [2]

    @pytest.mark.internal
    def test_reset_preserves_callback(self):
        r = PrefixCacheRegistry()
        captured: list = []
        r.set_mamba_evict_callback(captured.extend)

        r.register_kv([1], [10])
        r.register_mamba([1], [10])
        r.reset()
        assert r.kv_hash_to_block_id == {}
        assert r.mamba_hash_to_block_id == {}

        # Callback survives the reset.
        r.register_mamba([2], [20])
        r.evict_mamba([20])
        assert captured == [2]
