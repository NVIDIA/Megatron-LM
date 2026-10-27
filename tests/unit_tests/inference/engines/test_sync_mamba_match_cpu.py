# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure scheduler-prefix contracts shared by eager and chunked Mamba admission."""

from types import SimpleNamespace

import pytest

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


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
    allocator = None if cached is None else SimpleNamespace(hash_to_block_id=cached)
    engine = SimpleNamespace(context=SimpleNamespace(mamba_slot_allocator=allocator))
    request = SimpleNamespace(precomputed_block_hashes=hashes)
    assert DynamicInferenceEngine._find_mamba_match_count(engine, request) == expected
