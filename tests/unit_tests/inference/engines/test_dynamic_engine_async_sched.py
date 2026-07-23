# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DynamicBatchControllerStepResult,
)


def _make_engine(async_sched_mode=AsyncScheduleMode.ASYNC, **overrides):
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
        num_prefill_requests=0,
        can_prepare_requests=mock.Mock(return_value=True),
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
        ({"enable_chunked_prefill": True}, True),
        ({"num_speculative_tokens": 1}, True),
        ({"num_speculative_tokens": 1, "controller_num_mtp_depths": 1}, False),
        ({"context_is_hybrid_model": True}, True),
        ({"context_enable_prefix_caching": True}, True),
        ({"materialize_only_last_token_logits": False}, True),
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
    "async_sched_mode, sampling_params, should_raise",
    [
        (AsyncScheduleMode.LEGACY, SamplingParams(top_k=0, top_p=0.5), False),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=1, top_p=0.5), True),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=1, top_p=0.0, return_log_probs=True), True),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=1, top_p=0.0, top_n_logprobs=1), True),
        (AsyncScheduleMode.ASYNC, SamplingParams(top_k=1, top_p=0.0, stop_words=["END"]), True),
    ],
)
def test_validate_async_sched_support_for_request(async_sched_mode, sampling_params, should_raise):
    """Ensure engine request validation accepts only supported async scheduling requests."""
    engine = _make_engine(async_sched_mode=async_sched_mode)
    request = SimpleNamespace(sampling_params=sampling_params)

    if should_raise:
        with pytest.raises(ValueError, match="Async scheduling"):
            engine._validate_async_sched_support_for_request(request)
    else:
        engine._validate_async_sched_support_for_request(request)


def test_add_request_runs_async_sched_request_validation():
    """Ensure request validation is called before mutating engine request state."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine._validate_async_sched_support_for_request = mock.Mock(
        side_effect=RuntimeError("validated")
    )
    request = SimpleNamespace(request_id=10)

    with pytest.raises(RuntimeError, match="validated"):
        engine._add_request(request)

    engine._validate_async_sched_support_for_request.assert_called_once_with(request)


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
    "mode, run_async_overlap, primer_only, expected_schedule_calls",
    [
        (AsyncScheduleMode.LEGACY, None, False, 1),
        (AsyncScheduleMode.ASYNC, True, False, 0),
        (AsyncScheduleMode.ASYNC, False, True, 0),
    ],
)
def test_async_forward_routes_one_controller_iteration(
    mode, run_async_overlap, primer_only, expected_schedule_calls
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
        is_decode_only=mock.Mock(return_value=True),
    )
    output = None if primer_only else {"sample": "tokens"}
    engine.controller = SimpleNamespace(
        async_generate_output_tokens_dynamic_batch=mock.AsyncMock(
            return_value=DynamicBatchControllerStepResult(output=output, primer_only=primer_only)
        )
    )

    with (
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_push"),
        mock.patch("megatron.core.inference.engines.dynamic_engine.nvtx_range_pop"),
    ):
        result, _, _ = asyncio.run(engine.async_forward())

    assert result is output
    assert engine.context.step_count == 5
    assert engine.context.prefix_cache_lru_clock == 8
    assert engine.schedule_waiting_requests.call_count == expected_schedule_calls
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
