# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock, patch

import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.dynamic_context import DynamoHelper
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator


def test_allocator_notifies_observer_without_replacing_legacy_callback():
    # prefix_cache_epoch dates each registered block; the real context always has it.
    context = Mock(prefix_cache_epoch=0)
    allocator = KVBlockAllocator(
        context,
        8,
        0,
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


def test_listener_failure_does_not_interrupt_block_deregistration():
    # prefix_cache_epoch dates each registered block; the real context always has it.
    context = Mock(prefix_cache_epoch=0)
    allocator = KVBlockAllocator(
        context,
        8,
        0,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO,
    )
    helper = DynamoHelper()
    failing_listener = Mock(side_effect=RuntimeError("publisher unavailable"))
    healthy_listener = Mock()
    helper.add_kv_event_listener(failing_listener)
    helper.add_kv_event_listener(healthy_listener)
    allocator.add_blocks_deregistered_observer(helper.on_kv_blocks_deregistered)

    blocks = allocator.allocate_memory_blocks(2)

    def assert_allocator_committed(_kind, _payload):
        block_ids = blocks.to(torch.int64)
        assert torch.all(allocator.block_hashes[block_ids] == -1)
        assert torch.all(allocator.block_ref_counts[block_ids] == 0)

    healthy_listener.side_effect = assert_allocator_committed
    allocator.register_kv_block_hashes(blocks.tolist(), [101, 202])
    with patch(
        "megatron.core.inference.contexts.dynamic_context.logging.exception"
    ) as log_exception:
        allocator.release_memory_blocks(blocks)

    assert not allocator.kv_hash_to_block_id
    assert torch.all(allocator.block_hashes[blocks.to(torch.int64)] == -1)
    assert torch.all(allocator.block_ref_counts[blocks.to(torch.int64)] == 0)
    failing_listener.assert_called_once()
    healthy_listener.assert_called_once()
    assert failing_listener.call_args.args[0] == "removed"
    assert healthy_listener.call_args.args[0] == "removed"
    assert set(failing_listener.call_args.args[1]["block_hashes"]) == {101, 202}
    assert set(healthy_listener.call_args.args[1]["block_hashes"]) == {101, 202}
    log_exception.assert_called_once_with("KV-event listener failed while handling %r", "removed")
