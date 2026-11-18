# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure scheduler-prefix contracts shared by eager and chunked Mamba admission."""

from types import SimpleNamespace

import pytest

from megatron.core.inference.contexts.prefix_cache_registry import PrefixCacheRegistry
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


def _mamba_allocator(cached):
    """Build an allocator stub whose prefix-cache registry holds `cached` Mamba blocks."""
    if cached is None:
        return None
    registry = PrefixCacheRegistry()
    registry.register_mamba(list(cached.values()), list(cached.keys()))
    return SimpleNamespace(registry=registry)


@pytest.mark.parametrize(
    "hashes,cached,expected",
    [
        ([], {}, 0),
        ([11, 22, 33], {}, 0),
        ([11, 22, 33], {11: 0}, 1),
        ([11, 22, 33], {11: 0, 33: 1}, 3),
        ([11], None, 0),
    ],
)
def test_scheduler_finds_farthest_cached_mamba_boundary(hashes, cached, expected):
    allocator = _mamba_allocator(cached)
    engine = SimpleNamespace(context=SimpleNamespace(mamba_slot_allocator=allocator))
    request = SimpleNamespace(precomputed_block_hashes=hashes)
    assert DynamicInferenceEngine._find_mamba_match_count(engine, request) == expected
