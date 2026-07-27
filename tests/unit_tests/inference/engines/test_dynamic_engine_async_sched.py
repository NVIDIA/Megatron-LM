# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams


def _make_engine(async_sched_mode=AsyncScheduleMode.SERIAL, **overrides):
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
    )
    model_config = SimpleNamespace(
        expert_model_parallel_size=1, num_moe_experts=None, moe_enable_routing_replay=False
    )
    engine.context = context
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config))
    )
    engine.num_speculative_tokens = 0
    engine.materialize_only_last_token_logits = True

    for name, value in overrides.items():
        if name.startswith("context_"):
            setattr(context, name.removeprefix("context_"), value)
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
        ({"num_speculative_tokens": 1}, True),
        ({"context_is_hybrid_model": True}, True),
        ({"context_enable_prefix_caching": True}, True),
        ({"materialize_only_last_token_logits": False}, True),
        ({"model_config_expert_model_parallel_size": 2}, True),
        ({"model_config_num_moe_experts": 4}, True),
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
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0), False),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=0, top_p=0.0), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.5), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, return_log_probs=True), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, top_n_logprobs=1), True),
        (AsyncScheduleMode.SERIAL, SamplingParams(top_k=1, top_p=0.0, stop_words=["END"]), True),
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
