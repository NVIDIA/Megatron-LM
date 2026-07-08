from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState


def _context_with_listener():
    context = DynamicInferenceContext.__new__(DynamicInferenceContext)
    listener = Mock()
    context._kv_event_listeners = [listener]
    context._pending_kv_stored_events = []
    return context, listener


def test_stored_event_is_published_only_after_forward_completion():
    context, listener = _context_with_listener()
    payload = {"block_hashes": [101], "token_ids": [1, 2]}

    context._pending_kv_stored_events.append(payload)

    listener.assert_not_called()
    context.publish_pending_kv_stored_events()
    listener.assert_called_once_with("stored", payload)
    assert context._pending_kv_stored_events == []


def test_cache_clear_discards_unpublished_stored_events():
    context, listener = _context_with_listener()

    context._pending_kv_stored_events.append({"block_hashes": [101]})
    context.notify_kv_cache_cleared()

    listener.assert_called_once_with("cleared", {})
    assert context._pending_kv_stored_events == []


def test_dummy_reset_preserves_prefix_cache_without_publishing_clear():
    context, listener = _context_with_listener()
    context.enable_prefix_caching = True
    context.reset_tensors = Mock()
    context.reset_metadata = Mock()
    context.step_count = 17
    context.prefix_cache_lru_clock = 11
    context.mamba_slot_allocator = Mock()
    context._pending_kv_stored_events.append({"block_hashes": [101]})

    context.reset(preserve_prefix_cache=True)

    listener.assert_not_called()
    context.reset_tensors.assert_called_once_with()
    context.reset_metadata.assert_called_once_with(preserve_prefix_cache=True)
    context.mamba_slot_allocator.reset.assert_not_called()
    assert context._pending_kv_stored_events == []
    assert context.step_count == 17
    assert context.prefix_cache_lru_clock == 11


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

    context._pending_kv_stored_events.append({"block_hashes": [101]})
    context.discard_pending_kv_stored_events()

    listener.assert_not_called()
    assert context._pending_kv_stored_events == []


@pytest.mark.asyncio
async def test_async_forward_discards_before_scheduling_and_publishes_after_forward(monkeypatch):
    context, listener = _context_with_listener()
    context.step_count = 0
    context.prefix_cache_lru_clock = 0
    context.active_token_count = 0
    context.is_decode_only = lambda: False
    context._pending_kv_stored_events.append({"block_hashes": [7]})

    order = []
    payload = {"block_hashes": [101], "token_ids": [1, 2]}
    discard = context.discard_pending_kv_stored_events
    publish = context.publish_pending_kv_stored_events

    def discard_pending():
        order.append("discard")
        discard()

    def schedule():
        order.append("schedule")
        assert context._pending_kv_stored_events == []
        context._pending_kv_stored_events.append(payload)

    async def forward():
        order.append("forward")
        listener.assert_not_called()
        return {"output": True}

    def publish_pending():
        order.append("publish")
        publish()

    context.discard_pending_kv_stored_events = discard_pending
    context.publish_pending_kv_stored_events = publish_pending

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
