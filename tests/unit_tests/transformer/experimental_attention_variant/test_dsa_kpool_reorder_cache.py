# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The packed KPool pool reorder is built once per pack layout and reused by every DSA layer."""

import torch

from megatron.core.transformer.experimental_attention_variant import dsa_layout

# Two documents, each a multiple of 2 * cp_size * pool_size tokens.
HOST_CU = [0, 64, 192]
CPU = torch.device("cpu")


def _build(cp_rank, cu=HOST_CU):
    return dsa_layout.build_packed_kpool_pool_reorder_from_host(
        cu, cp_size=4, cp_rank=cp_rank, pool_size=4, device=CPU
    )


def test_same_layout_returns_the_cached_table():
    dsa_layout._KPOOL_POOL_REORDER_CACHE.clear()
    first = _build(1)
    second = _build(1)
    assert first is second


def test_cached_table_equals_a_fresh_build_and_differs_by_layout():
    dsa_layout._KPOOL_POOL_REORDER_CACHE.clear()
    cached = _build(2)
    dsa_layout._KPOOL_POOL_REORDER_CACHE.clear()
    fresh = _build(2)
    assert torch.equal(cached, fresh)
    other = _build(2, cu=[0, 128, 192])
    assert other is not fresh


def test_cache_is_bounded():
    dsa_layout._KPOOL_POOL_REORDER_CACHE.clear()
    for end in range(64, 64 + 32 * 10, 32):
        _build(0, cu=[0, 32, end + 32])
    assert len(dsa_layout._KPOOL_POOL_REORDER_CACHE) <= dsa_layout._KPOOL_POOL_REORDER_CACHE_SIZE
