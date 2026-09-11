# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext, DynamoHelper
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState


def _context_with_listener():
    context = DynamicInferenceContext.__new__(DynamicInferenceContext)
    context.dynamo_helper = DynamoHelper()
    listener = Mock()
    context.dynamo_helper.add_kv_event_listener(listener)
    return context, listener


def test_stored_event_is_published_only_after_forward_completion():
    context, listener = _context_with_listener()
    payload = {"block_hashes": [101], "token_ids": [1, 2]}

    context.dynamo_helper.queue_kv_stored_event(payload)

    listener.assert_not_called()
    context.dynamo_helper.publish_pending_kv_stored_events()
    listener.assert_called_once_with("stored", payload)
    context.dynamo_helper.publish_pending_kv_stored_events()
    listener.assert_called_once_with("stored", payload)


def test_cache_clear_discards_unpublished_stored_events():
    context, listener = _context_with_listener()

    context.dynamo_helper.queue_kv_stored_event({"block_hashes": [101]})
    context.dynamo_helper.notify_kv_cache_cleared()

    listener.assert_called_once_with("cleared", {})
    context.dynamo_helper.publish_pending_kv_stored_events()
    listener.assert_called_once_with("cleared", {})


def test_dummy_reset_preserves_prefix_cache_without_publishing_clear():
    context, listener = _context_with_listener()
    context.enable_prefix_caching = True
    context.reset_tensors = Mock()
    context.reset_metadata = Mock()
    context.step_count = 17
    context.prefix_cache_lru_clock = 11
    context.mamba_slot_allocator = Mock()
    context.dynamo_helper.queue_kv_stored_event({"block_hashes": [101]})

    context.reset(preserve_prefix_cache=True, preserve_counters=True)
    context.dynamo_helper.publish_pending_kv_stored_events()

    listener.assert_not_called()
    context.reset_tensors.assert_called_once_with()
    context.reset_metadata.assert_called_once_with(
        preserve_prefix_cache=True, preserve_counters=True
    )
    context.mamba_slot_allocator.reset.assert_not_called()
    assert context.step_count == 17
    assert context.prefix_cache_lru_clock == 11


def test_full_reset_publishes_clear_after_cache_reset():
    context, listener = _context_with_listener()
    order = []
    context.enable_prefix_caching = True
    context.reset_tensors = Mock(side_effect=lambda: order.append("reset_tensors"))
    context.reset_metadata = Mock(side_effect=lambda **_kwargs: order.append("reset_metadata"))
    context.step_count = 17
    context.prefix_cache_lru_clock = 11
    context.mamba_slot_allocator = Mock()
    context.mamba_slot_allocator.reset.side_effect = lambda: order.append("reset_mamba")
    listener.side_effect = lambda kind, _payload: order.append(kind)

    context.reset()

    assert order == ["reset_tensors", "reset_metadata", "reset_mamba", "cleared"]


@pytest.mark.parametrize(
    ("cache_mode", "expect_clear"),
    [
        (KVCacheManagementMode.PERSIST, False),
        (KVCacheManagementMode.OFFLOAD, False),
        (KVCacheManagementMode.RECOMPUTE, True),
    ],
)
def test_suspend_clears_only_recomputed_cache(cache_mode, expect_clear):
    context, listener = _context_with_listener()
    order = []
    context.kv_cache_management_mode = cache_mode
    context.static_kv_memory_pointers = True
    context.deallocate_inference_state_buffers = Mock(
        side_effect=lambda: order.append("deallocate")
    )
    listener.side_effect = lambda kind, _payload: order.append(kind)
    context.dynamo_helper.queue_kv_stored_event({"block_hashes": [101]})

    engine = object.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.context = context
    engine.controller = SimpleNamespace(_async_sched_logits=Mock())
    engine.unified_memory_level = 0
    engine.requests = {}
    engine.waiting_request_ids = deque()
    engine.use_coordinator = False
    engine._vision_embedding_cache = {}
    engine._vision_embedding_cache_bytes = 0

    with (
        mock.patch.object(DynamicInferenceEngine, "suspend_resume_ctx", return_value=nullcontext()),
        mock.patch("megatron.core.inference.engines.dynamic_engine.InferenceMode.unset_active"),
    ):
        engine.suspend()
    context.dynamo_helper.publish_pending_kv_stored_events()

    expected_order = ["deallocate", "cleared"] if expect_clear else ["deallocate"]
    assert order == expected_order
    assert engine.controller._async_sched_logits.clear.call_count == int(expect_clear)


def test_reset_metadata_can_preserve_prefix_allocator():
    context = DynamicInferenceContext.__new__(DynamicInferenceContext)
    context.enable_prefix_caching = True
    context.reset_attention_state = Mock()
    context.reset_mamba_state = Mock()
    context.kv_block_allocator = Mock()
    context.request_to_kv_block_ids = Mock()

    context.reset_metadata(preserve_prefix_cache=True)

    context.reset_attention_state.assert_called_once_with()
    context.reset_mamba_state.assert_called_once_with()
    context.kv_block_allocator.reset.assert_not_called()
    context.request_to_kv_block_ids.fill_.assert_called_once_with(-1)


def test_next_forward_can_discard_events_left_by_a_failed_forward():
    context, listener = _context_with_listener()

    context.dynamo_helper.queue_kv_stored_event({"block_hashes": [101]})
    context.dynamo_helper.discard_pending_kv_stored_events()
    context.dynamo_helper.publish_pending_kv_stored_events()

    listener.assert_not_called()


@pytest.mark.asyncio
async def test_async_forward_discards_before_scheduling_and_publishes_after_forward(monkeypatch):
    context, listener = _context_with_listener()
    context.step_count = 0
    context.prefix_cache_lru_clock = 0
    context.active_token_count = 0
    context.chunked_prefill_request_id = -1
    context.num_prefill_requests = 0
    context.config = SimpleNamespace(async_sched_mode=AsyncScheduleMode.LEGACY)
    context.dynamo_helper.queue_kv_stored_event({"block_hashes": [7]})

    order = []
    payload = {"block_hashes": [101], "token_ids": [1, 2]}
    discard = context.dynamo_helper.discard_pending_kv_stored_events
    publish = context.dynamo_helper.publish_pending_kv_stored_events

    def discard_pending():
        order.append("discard")
        discard()

    def schedule():
        order.append("schedule")
        context.dynamo_helper.queue_kv_stored_event(payload)

    async def forward(**kwargs):
        order.append("forward")
        assert kwargs == {}
        listener.assert_not_called()
        return SimpleNamespace(decode_only=False, output={"output": True})

    def publish_pending():
        order.append("publish")
        publish()

    context.dynamo_helper.discard_pending_kv_stored_events = discard_pending
    context.dynamo_helper.publish_pending_kv_stored_events = publish_pending

    engine = object.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.context = context
    engine.logging_step_interval = 0
    engine.schedule_waiting_requests = schedule
    engine.controller = SimpleNamespace(async_generate_output_tokens_dynamic_batch=forward)

    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.nvtx_range_push", lambda *_: None
    )
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.nvtx_range_pop", lambda *_: None
    )

    result, _, _ = await DynamicInferenceEngine.async_forward(engine)

    assert result == {"output": True}
    assert order == ["discard", "schedule", "forward", "publish"]
    listener.assert_called_once_with("stored", payload)
