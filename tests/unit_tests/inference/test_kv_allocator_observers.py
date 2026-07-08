from unittest.mock import Mock

import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator


def test_allocator_notifies_observer_without_replacing_legacy_callback():
    context = Mock()
    allocator = KVBlockAllocator(
        context=context,
        total_count=8,
        paused_count=0,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )
    removed = Mock()
    legacy = Mock()
    allocator.add_blocks_deregistered_observer(removed)
    allocator.on_blocks_deregistered = legacy

    blocks = allocator.allocate_memory_blocks(2)
    allocator.register_kv_block_hashes(blocks.tolist(), [101, 202])
    allocator.release_memory_blocks(blocks)

    legacy.assert_called_once()
    removed.assert_called_once()
