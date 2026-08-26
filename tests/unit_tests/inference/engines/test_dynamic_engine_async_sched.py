# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.contexts.dynamic_context import DynamoHelper
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState, _get_decode_only_log_state
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    DynamicBatchControllerStepResult,
)


def _make_engine(
    async_sched_mode=AsyncScheduleMode.ASYNC, engine_cls=DynamicInferenceEngine, **overrides
):
    engine = engine_cls.__new__(engine_cls)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
        num_prefill_requests=0,
        total_request_count=0,
        max_requests=8,
        can_prepare_requests=mock.Mock(return_value=True),
        active_token_count=0,
        max_tokens=8,
        chunked_prefill_request_id=-1,
    )
    model_config = SimpleNamespace(
        expert_model_parallel_size=1, num_moe_experts=None, moe_enable_routing_replay=False
    )
    engine.context = context
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config)),
        num_mtp_depths=0,
    )
    engine.enable_chunked_prefill = False
    engine.num_speculative_tokens = 0
    engine.materialize_only_last_token_logits = True

    for name, value in overrides.items():
        if name.startswith("context_"):
            setattr(context, name.removeprefix("context_"), value)
        elif name.startswith("controller_"):
            setattr(engine.controller, name.removeprefix("controller_"), value)
        elif name.startswith("model_config_"):
            setattr(model_config, name.removeprefix("model_config_"), value)
        else:
            setattr(engine, name, value)
    return engine


@pytest.mark.parametrize(
    "overrides, should_raise",
    [
        ({"async_sched_mode": AsyncScheduleMode.LEGACY, "num_speculative_tokens": 1}, False),
        ({}, False),
        ({"enable_chunked_prefill": True}, False),
        ({"num_speculative_tokens": 1}, True),
        ({"num_speculative_tokens": 1, "controller_num_mtp_depths": 1}, False),
        ({"context_is_hybrid_model": True}, False),
        (
            {
                "context_is_hybrid_model": True,
                "num_speculative_tokens": 1,
                "controller_num_mtp_depths": 1,
                "model_config_expert_model_parallel_size": 2,
                "model_config_num_moe_experts": 4,
            },
            False,
        ),
        ({"context_enable_prefix_caching": True}, False),
        (
            {
                "enable_chunked_prefill": True,
                "context_enable_prefix_caching": True,
                "context_is_hybrid_model": True,
            },
            False,
        ),
        ({"materialize_only_last_token_logits": False}, False),
        ({"model_config_expert_model_parallel_size": 2}, False),
        ({"model_config_num_moe_experts": 4}, False),
        ({"model_config_moe_enable_routing_replay": True}, True),
    ],
)
def test_validate_async_sched_support_for_config(overrides, should_raise):
    """Ensure engine config validation accepts only supported async scheduling configs."""
    engine = _make_engine(**overrides)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_config()
    else:
        engine._validate_async_sched_support_for_config()


@pytest.mark.parametrize(
    "can_prepare, has_waiting, availability, expected",
    [
        (False, False, (False, False, False), False),
        (True, False, (True, True, True), True),
        (True, True, (False, True, True), True),
        (True, True, (True, True, True), False),
    ],
)
def test_should_run_async_sched_overlap(can_prepare, has_waiting, availability, expected):
    """The overlap probe observes prefill eligibility without admitting the request."""
    engine = _make_engine()
    engine.context.can_prepare_requests.return_value = can_prepare
    engine.context.check_availability = mock.Mock(return_value=availability)
    engine.waiting_request_ids = deque([10] if has_waiting else [])
    request = SimpleNamespace(remaining_prompt_tokens=[1, 2], cg_wait_iters=3)
    engine.get_request = mock.Mock(return_value=request)
    engine._cg_admission_gating_active = mock.Mock(return_value=False)

    assert engine._should_run_async_sched_overlap() is expected
    engine.context.can_prepare_requests.assert_called_once_with()
    assert list(engine.waiting_request_ids) == ([10] if has_waiting else [])
    assert request.cg_wait_iters == 3


def test_async_sched_overlap_probe_uses_non_mutating_cuda_graph_match():
    """A scheduling probe does not update CUDA-graph wait accounting."""
    engine = _make_engine()
    engine.context.active_token_count = 2
    engine.context.num_prefill_requests = 0
    engine.context.num_decode_requests = 2
    engine.context.check_availability = mock.Mock(return_value=(True, True, True))
    engine._cg_admission_gating_active = mock.Mock(return_value=True)
    engine._matches_cg_admission = mock.Mock(return_value=False)
    engine._cg_admission_check = mock.Mock()
    request = SimpleNamespace(remaining_prompt_tokens=[1, 2], cg_wait_iters=7)

    assert not engine._can_schedule_non_chunked_prefill(request, record_cg_wait=False)
    engine._matches_cg_admission.assert_called_once()
    engine._cg_admission_check.assert_not_called()
    assert request.cg_wait_iters == 7


@pytest.mark.parametrize(
    "availability, active_token_count, chunked_prefill_request_id, expected",
    [
        ((True, False, True), 7, -1, True),
        ((False, False, True), 7, 10, True),
        ((True, True, False), 7, -1, False),
        ((True, True, True), 8, -1, False),
    ],
)
def test_can_schedule_chunked_prefill(
    availability, active_token_count, chunked_prefill_request_id, expected
):
    """The chunk probe requires request, KV-cache, and partial-token capacity."""
    engine = _make_engine(enable_chunked_prefill=True)
    engine.context.active_token_count = active_token_count
    engine.context.chunked_prefill_request_id = chunked_prefill_request_id
    engine.context.check_availability = mock.Mock(return_value=availability)
    request = SimpleNamespace(request_id=10)

    assert engine._can_schedule_chunked_prefill(request) is expected


def test_async_sched_overlap_probe_routes_schedulable_chunk_to_no_overlap():
    """A schedulable chunk is admitted only after no-overlap lifecycle bookkeeping."""
    engine = _make_engine(enable_chunked_prefill=True)
    engine.context.active_token_count = 2
    engine.context.check_availability = mock.Mock(return_value=(True, False, True))
    engine.waiting_request_ids = deque([10])
    engine.get_request = mock.Mock(return_value=SimpleNamespace(request_id=10))

    assert not engine._should_run_async_sched_overlap()


def test_ready_handoff_uses_safe_no_overlap_admission_point():
    """A completed import cannot join the batch before pending logits are consumed."""

    engine = _make_engine(engine_cls=DisaggDynamicInferenceEngine)
    engine._initialize_disaggregation_state()
    engine.waiting_request_ids = deque()
    engine._pending_kv_imports.append(SimpleNamespace(request_id=7, resume_tokens=[55]))
    engine._handoff_completion_notifications[7] = False

    assert not engine._should_run_async_sched_overlap()

    # Keep overlap enabled while the active batch is full; the normal lifecycle
    # boundary will select no-overlap once capacity becomes available.
    engine.context.total_request_count = engine.context.max_requests
    assert engine._should_run_async_sched_overlap()


@pytest.mark.parametrize(
    "mode, run_async_overlap, decode_only, primer_only, expected_schedule_calls, "
    "expected_nvtx_range",
    [
        (
            AsyncScheduleMode.LEGACY,
            None,
            DecodeOnly(consumed=False, launched=False),
            False,
            1,
            "Prefill",
        ),
        (
            AsyncScheduleMode.LEGACY,
            None,
            DecodeOnly(consumed=True, launched=True),
            False,
            1,
            "Decode",
        ),
        (
            AsyncScheduleMode.ASYNC,
            True,
            DecodeOnly(consumed=True, launched=True),
            False,
            0,
            "AsyncOverlap",
        ),
        (
            AsyncScheduleMode.ASYNC,
            False,
            DecodeOnly(consumed=False, launched=True),
            False,
            0,
            "AsyncNoOverlap",
        ),
        (
            AsyncScheduleMode.ASYNC,
            False,
            DecodeOnly(consumed=None, launched=False),
            True,
            0,
            "AsyncNoOverlap",
        ),
        (
            AsyncScheduleMode.ASYNC,
            False,
            DecodeOnly(consumed=True, launched=None),
            False,
            0,
            "AsyncNoOverlap",
        ),
    ],
)
def test_async_forward_routes_one_controller_iteration(
    mode, run_async_overlap, decode_only, primer_only, expected_schedule_calls, expected_nvtx_range
):
    """Primer-only work crosses the engine boundary without an internal controller loop."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.state = EngineState.RUNNING
    engine.logging_step_interval = 0
    engine.metrics_writer = None
    engine.schedule_waiting_requests = mock.Mock()
    engine._should_run_async_sched_overlap = mock.Mock(return_value=run_async_overlap)
    engine.context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=mode),
        step_count=4,
        prefix_cache_lru_clock=7,
        active_token_count=2,
        num_prefill_requests=1 if expected_nvtx_range == "Prefill" else 0,
        chunked_prefill_request_id=17,
        is_decode_only=mock.Mock(return_value=decode_only.launched),
        dynamo_helper=DynamoHelper(),
    )
    output = None if primer_only else {"sample": "tokens"}
    engine.controller = SimpleNamespace(
        async_generate_output_tokens_dynamic_batch=mock.AsyncMock(
            return_value=DynamicBatchControllerStepResult(
                decode_only=decode_only, output=output, primer_only=primer_only
            )
        )
    )

    with (
        mock.patch(
            "megatron.core.inference.engines.dynamic_engine.nvtx_range_push"
        ) as nvtx_range_push,
        mock.patch(
            "megatron.core.inference.engines.dynamic_engine.nvtx_range_pop"
        ) as nvtx_range_pop,
    ):
        result, context_state, _ = asyncio.run(engine.async_forward())

    assert result is output
    assert context_state["decode_only"] == decode_only
    assert context_state["chunked_prefill_request_id"] == 17
    assert engine.decode_only == decode_only
    assert not hasattr(engine, "is_decode_only")
    assert engine.context.step_count == 5
    assert engine.context.prefix_cache_lru_clock == 8
    assert engine.schedule_waiting_requests.call_count == expected_schedule_calls
    nvtx_range_push.assert_called_once_with(expected_nvtx_range)
    nvtx_range_pop.assert_called_once_with(expected_nvtx_range)
    if mode == AsyncScheduleMode.LEGACY:
        engine._should_run_async_sched_overlap.assert_not_called()
        engine.controller.async_generate_output_tokens_dynamic_batch.assert_awaited_once_with()
    else:
        engine._should_run_async_sched_overlap.assert_called_once_with()
        engine.controller.async_generate_output_tokens_dynamic_batch.assert_awaited_once_with(
            run_async_overlap=run_async_overlap,
            schedule_waiting_requests=(
                None if run_async_overlap else engine.schedule_waiting_requests
            ),
        )
    engine.context.is_decode_only.assert_not_called()


def test_async_bookkeep_uses_consumed_chunked_prefill_request_id():
    """Post-processing classifies output using the chunk ID from its consumed forward."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.track_paused_request_events = False
    engine.post_process_requests = mock.Mock(return_value=([10], []))
    engine.failed_request_ids = set()
    engine.requests = {}
    engine.use_coordinator = False
    engine.context = SimpleNamespace(enable_prefix_caching=False, step_count=1)
    engine.logging_step_interval = 0
    engine.num_speculative_tokens = 0
    step_result = {
        "active_request_ids": [10],
        "finished_request_ids": [],
        "sample": [20],
        "accepted_tokens": None,
        "log_probs": None,
        "cuda_graph_request_count": None,
    }
    context_state = {
        "active_token_count": 4,
        "step_count": 0,
        "chunked_prefill_request_id": 10,
        "kv_stats": None,
    }

    with (
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_push"),
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_pop"),
    ):
        asyncio.run(engine.async_bookkeep(step_result, context_state, 0.0))

    assert (
        engine.post_process_requests.call_args.kwargs["consumed_chunked_prefill_request_id"] == 10
    )


@pytest.mark.parametrize(
    "mode, decode_only, expected",
    [
        (
            AsyncScheduleMode.LEGACY,
            DecodeOnly(consumed=False, launched=False),
            ("non-decode", False),
        ),
        (AsyncScheduleMode.LEGACY, DecodeOnly(consumed=True, launched=True), ("decode", True)),
        (
            AsyncScheduleMode.ASYNC,
            DecodeOnly(consumed=False, launched=False),
            ("non-decode", False),
        ),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=True, launched=True), ("decode", True)),
        (
            AsyncScheduleMode.ASYNC,
            DecodeOnly(consumed=False, launched=True),
            ("decode (prev: non-decode)", True),
        ),
        (
            AsyncScheduleMode.ASYNC,
            DecodeOnly(consumed=True, launched=False),
            ("non-decode (prev: decode)", False),
        ),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=None, launched=False), ("non-decode", False)),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=None, launched=True), ("decode", True)),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=False, launched=None), ("non-decode", False)),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=True, launched=None), ("decode", True)),
        (AsyncScheduleMode.ASYNC, DecodeOnly(consumed=None, launched=None), ("idle", None)),
    ],
)
def test_get_decode_only_log_state(mode, decode_only, expected):
    """Console logging reports transitions and colors the latest available phase."""
    assert _get_decode_only_log_state(mode, decode_only) == expected


def test_base_engine_rejects_kv_handoff_commands():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)

    assert InferenceStateHandoffMixin not in DynamicInferenceEngine.mro()
    assert engine.pending_kv_import_count == 0
    with pytest.raises(RuntimeError, match="SUBMIT_REQUEST_WITH_KV"):
        engine.add_request_with_kv_handoff(1, [], SamplingParams(), {}, [])
    with pytest.raises(RuntimeError, match="RELEASE_KV"):
        engine.release_handoff_blocks(1)


def test_disagg_engine_resolves_handoff_methods_from_mixin():
    assert DisaggDynamicInferenceEngine.mro()[:3] == [
        DisaggDynamicInferenceEngine,
        InferenceStateHandoffMixin,
        DynamicInferenceEngine,
    ]
    assert (
        DisaggDynamicInferenceEngine.add_request_with_kv_handoff
        is InferenceStateHandoffMixin.add_request_with_kv_handoff
    )
