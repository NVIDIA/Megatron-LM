# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.async_txn import (
    AsyncTxnDiagnostics,
    AsyncTxnSkipReason,
    StepTxn,
    TxnRetireQueue,
    classify_decode_child_launch,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeEvent:
    def __init__(self, *, done: bool = True):
        self.done = done
        self.synchronized = 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synchronized += 1
        self.done = True


class FakeLaunchContext:
    def __init__(
        self,
        *,
        decode_only: bool = True,
        chunked_prefill_request_id: int = -1,
        paused_request_count: int = 0,
    ):
        self._decode_only = decode_only
        self.chunked_prefill_request_id = chunked_prefill_request_id
        self.paused_request_count = paused_request_count
        self.total_request_count = 1
        self.num_speculative_tokens = 0
        self.block_size_tokens = 4
        self.request_last_kv_block_offset = torch.tensor([0], dtype=torch.int32)
        self.request_ids = torch.tensor([101], dtype=torch.int32)
        self.kv_block_allocator = SimpleNamespace(is_memory_available=lambda count: True)

    def is_decode_only(self) -> bool:
        return self._decode_only


class FakeController:
    def __init__(self):
        self.barrier_reasons = []

    async def async_generate_output_tokens_dynamic_batch(
        self, *, async_launch_barrier_reason=None, profile_async_child_forward=False
    ):
        del profile_async_child_forward
        self.barrier_reasons.append(async_launch_barrier_reason)
        return None


class FakeAsyncForwardContext:
    def __init__(self):
        self.async_scheduling = True
        self.step_count = 0
        self.prefix_cache_lru_clock = 0
        self.max_requests = 8
        self.total_request_count = 1
        self.paused_request_count = 0
        self.active_token_count = 1

    def is_decode_only(self) -> bool:
        return True


def _make_controller(context):
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    controller.model_config = SimpleNamespace(expert_model_parallel_size=1, num_moe_experts=None)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    return controller


def _make_engine_for_barriers(context=None):
    engine = object.__new__(DynamicInferenceEngine)
    engine.context = context or SimpleNamespace(
        async_scheduling=True, chunked_prefill_request_id=-1
    )
    engine.state = EngineState.RUNNING
    engine._next_async_launch_barrier_reason = None
    return engine


def test_admission_defers_child_launch_even_without_prepared_child():
    context = FakeLaunchContext()
    controller = _make_controller(context)

    reason = controller._async_child_launch_skip_reason(
        None,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
        async_launch_barrier_reason=AsyncTxnSkipReason.PENDING_ADMISSION,
    )

    assert reason == AsyncTxnSkipReason.PENDING_ADMISSION


def test_chunked_prefill_disables_and_reenables_async_barrier():
    context = SimpleNamespace(async_scheduling=True, chunked_prefill_request_id=77)
    engine = _make_engine_for_barriers(context)

    reason = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=False
    )
    context.chunked_prefill_request_id = -1
    resumed_reason = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=False
    )

    assert reason == AsyncTxnSkipReason.CHUNKED_PREFILL
    assert resumed_reason is None


@pytest.mark.parametrize(
    "reason",
    [
        AsyncTxnSkipReason.RESUME_BARRIER,
        AsyncTxnSkipReason.EVICT_BARRIER,
        AsyncTxnSkipReason.FORCE_PAUSE_BARRIER,
        AsyncTxnSkipReason.GRAPH_RECAPTURE_BARRIER,
    ],
)
def test_explicit_barrier_reasons_are_consumed_once(reason):
    engine = _make_engine_for_barriers()

    DynamicInferenceEngine._request_next_async_launch_barrier(engine, reason)
    first = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=False
    )
    second = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=False
    )

    assert first == reason
    assert second is None


def test_log_interval_barrier_skips_only_the_intended_step():
    engine = _make_engine_for_barriers()

    first = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=True
    )
    second = DynamicInferenceEngine._consume_async_launch_barrier_reason(
        engine, admitted_requests=False, will_log_this_step=False
    )

    assert first == AsyncTxnSkipReason.LOG_INTERVAL_BARRIER
    assert second is None


def test_resume_evict_and_overflow_pause_are_concrete_launch_gates():
    paused_context = FakeLaunchContext(paused_request_count=1)
    launchable_context = FakeLaunchContext()

    paused = classify_decode_child_launch(paused_context, async_enabled=True)
    resume = classify_decode_child_launch(
        launchable_context, async_enabled=True, resume_barrier=True
    )
    evict = classify_decode_child_launch(
        launchable_context, async_enabled=True, evict_barrier=True
    )
    force_pause = classify_decode_child_launch(
        launchable_context, async_enabled=True, force_pause_barrier=True
    )

    assert paused.reason == AsyncTxnSkipReason.PAUSED_REQUESTS
    assert resume.reason == AsyncTxnSkipReason.RESUME_BARRIER
    assert evict.reason == AsyncTxnSkipReason.EVICT_BARRIER
    assert force_pause.reason == AsyncTxnSkipReason.FORCE_PAUSE_BARRIER


def test_async_forward_passes_pending_admission_barrier_to_controller():
    context = FakeAsyncForwardContext()
    controller = FakeController()
    engine = _make_engine_for_barriers(context)
    engine.controller = controller
    engine.schedule_waiting_requests = lambda: True
    engine.logging_step_interval = 0
    engine.is_decode_only = None

    result, context_state, step_time = asyncio.run(
        DynamicInferenceEngine.async_forward(engine)
    )

    assert result is None
    assert context_state["active_token_count"] == 1
    assert step_time == 0.0
    assert context.step_count == 1
    assert context.prefix_cache_lru_clock == 1
    assert controller.barrier_reasons == [AsyncTxnSkipReason.PENDING_ADMISSION]


def test_drain_synchronizes_inflight_forward_and_keeps_consumable_child_by_default():
    diagnostics = AsyncTxnDiagnostics(enabled=True)
    retire_queue = TxnRetireQueue(diagnostics)
    released = []
    retire_queue.enqueue(FakeEvent(done=True), lambda: released.append("retired"))
    context = SimpleNamespace(async_txn_retire_queue=retire_queue)
    controller = _make_controller(context)
    forward_event = FakeEvent(done=False)
    sample_event = FakeEvent(done=False)
    launched = StepTxn(
        step_id=9,
        request_ids=(101,),
        forward_done_event=forward_event,
        sample_done_event=sample_event,
        launched=True,
    )
    controller._async_launched_child_txn = launched
    controller._async_prepared_child_txn = StepTxn(step_id=10, request_ids=(101,))

    controller.drain_async_transactions()

    assert forward_event.synchronized == 1
    assert sample_event.synchronized == 1
    assert released == ["retired"]
    assert controller._async_prepared_child_txn is None
    assert controller._async_launched_child_txn is launched

    controller.drain_async_transactions(clear_launched=True)

    assert controller._async_launched_child_txn is None
