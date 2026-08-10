# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import os
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import (
    AsyncScheduleMode,
    CudaGraphSizingDistribution,
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamoHelper
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState, _get_decode_only_log_state
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    AsyncScheduleLogitsState,
    DecodeOnly,
    DynamicBatchControllerStepResult,
    TextGenerationController,
)
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.test_utilities import Utils


def _make_engine(async_sched_mode=AsyncScheduleMode.ASYNC, **overrides):
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    context = SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=async_sched_mode),
        is_hybrid_model=False,
        enable_prefix_caching=False,
        num_prefill_requests=0,
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


@pytest.mark.parametrize("track_paused_request_events", [False, True])
def test_async_bookkeep_uses_consumed_chunked_prefill_request_id(track_paused_request_events):
    """Bookkeeping uses consumed-forward state and records requested pause events."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.track_paused_request_events = track_paused_request_events
    engine.post_process_requests = mock.Mock(return_value=([10], []))
    paused_request = DynamicInferenceRequest(request_id=11)
    engine.get_request = mock.Mock(return_value=paused_request)
    engine.failed_request_ids = set()
    engine.requests = {}
    engine.use_coordinator = False
    engine.context = SimpleNamespace(enable_prefix_caching=False, step_count=1)
    engine.logging_step_interval = 0
    engine.num_speculative_tokens = 0
    step_result = {
        "active_request_ids": [10],
        "finished_request_ids": [],
        "newly_paused_request_ids": torch.tensor([11]),
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
    assert [event.type for event in paused_request.events] == (
        [DynamicInferenceEventType.PAUSE] if track_paused_request_events else []
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


# Pairwise coverage below is intentionally declarative.  A flag only receives credit when the
# owning scenario proves both async overlap and feature-specific runtime behavior.
_UNIT = "unit"
_FUNCTIONAL = "functional"
_NEGATIVE = "negative"
_OUT_OF_SCOPE = "out_of_scope"

_INFERENCE_CONFIG_DISPOSITIONS = {
    _UNIT: {
        "async_sched_mode",
        "block_size_tokens",
        "buffer_size_gb",
        "cuda_graph_all_prefills",
        "cuda_graph_max_tokens",
        "cuda_graph_mixed_prefill_count",
        "cuda_graph_sizing_distribution",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "kv_cache_management_mode",
        "mamba_inference_state_config",
        "mamba_memory_ratio",
        "materialize_only_last_token_logits",
        "max_requests",
        "max_sequence_length",
        "max_tokens",
        "num_cuda_graphs",
        "num_speculative_tokens",
        "offset_sampling_seed_by_dp_rank",
        "paused_buffer_size_gb",
        "prefix_caching_eviction_policy",
        "request_metadata_types",
        "sampling_backend",
        "static_kv_memory_pointers",
        "track_generated_token_events",
        "track_paused_request_events",
        "use_cuda_graphs_for_non_decode_steps",
        "use_flashinfer_fused_rope",
        "logprobs_mode",
    },
    _FUNCTIONAL: {
        "disable_ep_consensus",
        "prefix_caching_mamba_gb",
        "prefix_caching_coordinator_policy",
        "prefix_caching_routing_alpha",
        "unified_memory_level",
        "use_synchronous_zmq_collectives",
    },
    _NEGATIVE: set(),
    _OUT_OF_SCOPE: {
        "ep_consensus_interval",
        "logging_step_interval",
        "metrics_writer",
        "pg_collection",
        "verbose",
    },
}

_SAMPLING_PARAM_DISPOSITIONS = {
    _UNIT: {
        "add_BOS",
        "detokenize_stop_sequence",
        "num_tokens_to_generate",
        "num_tokens_total",
        "return_log_probs",
        "skip_prompt_log_probs",
        "stop_words",
        "temperature",
        "termination_id",
        "top_k",
        "top_n_logprobs",
        "top_p",
    },
    _FUNCTIONAL: {"return_prompt_tokens", "streaming", "streaming_interval"},
    _NEGATIVE: set(),
    _OUT_OF_SCOPE: {"return_prompt_top_n_logprobs", "return_segments"},
}


@dataclass(frozen=True)
class _AsyncPairScenario:
    """One unique owner for a set of async-scheduling pairs."""

    name: str
    pairs: tuple[str, ...]
    config: dict[str, object] = field(default_factory=dict)
    sampling: tuple[dict[str, object], ...] = ()
    signals: tuple[str, ...] = ()
    request_profile: str = "staggered"
    prerequisite: str | None = None
    atol: float = 1.0e-3
    parity: str = "exact"
    exact_top_n: bool = True


_BASE_PAIR_CONFIG = {
    "num_requests": 0,
    "max_sequence_length": 40,
    "context_max_requests": 4,
    "context_max_tokens": 32,
    "num_tokens_to_generate": 10,
    "num_gap_steps": 0,
    "sampling_backend": "torch",
    "top_k": 1,
}


def _pair_scenario(name, *pairs, **kwargs):
    """Build a compact scenario declaration."""
    return _AsyncPairScenario(name=name, pairs=pairs, **kwargs)


_ASYNC_PAIR_SCENARIOS = (
    _pair_scenario(
        "dense-eager-events",
        "execution:eager",
        "model:gpt",
        "precision:bf16",
        "sampling:torch",
        "kv:persist",
        "kv:static-pointers",
        "events:generated-token",
        "metadata:explicit-request-schema",
        "seed:dp-offset",
        "logits:last-token",
        config={
            "track_generated_token_events": True,
            "inference_config_overrides": {
                "request_metadata_types": DynamicInferenceRequest.get_metadata_types()
            },
        },
        signals=(
            "dp-offset",
            "events",
            "last-logits",
            "metadata-schema",
            "persist",
            "suspend-resume",
            "static-pointers",
            "torch-backend",
        ),
    ),
    _pair_scenario(
        "chunked-capacity",
        "prefill:chunked",
        "capacity:max-requests",
        "capacity:max-tokens",
        "capacity:block-size",
        "capacity:buffer",
        "capacity:paused-buffer",
        config={
            "enable_chunked_prefill": True,
            "context_max_tokens": 8,
            "context_block_size_tokens": 512,
            "context_buffer_size_gb": 0.02,
            "context_paused_buffer_size_gb": 0.004,
        },
        signals=("capacity", "chunked", "paused-buffer-allocation"),
    ),
    _pair_scenario(
        "prefix-ref-zero",
        "prefix:enabled",
        "prefix:ref-zero",
        config={
            "enable_prefix_caching": True,
            "max_sequence_length": 544,
            "context_block_size_tokens": 256,
            "context_max_tokens": 768,
        },
        request_profile="prefix",
        signals=("prefix-hit", "ref-zero"),
    ),
    _pair_scenario(
        "prefix-lru-chunked",
        "prefix:lru",
        "interaction:prefix-chunk-pressure",
        config={
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_sequence_length": 544,
            "context_block_size_tokens": 256,
            "context_max_tokens": 384,
            "inference_config_overrides": {
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.LRU
            },
        },
        request_profile="prefix",
        signals=("chunked", "lru", "prefix-hit"),
    ),
    _pair_scenario(
        "graph-decode-block-exponential",
        "graph:block-scope",
        "graph:decode-only",
        "graph:exponential",
        config={
            "num_cuda_graphs": 4,
            "force_build_cuda_graphs": True,
            "use_cuda_graphs_for_non_decode_steps": False,
            "inference_cuda_graph_scope": InferenceCudaGraphScope.block,
            "inference_config_overrides": {
                "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.EXPONENTIAL,
                "cuda_graph_max_tokens": 16,
            },
        },
        signals=("cuda-graph", "graph-decode-config"),
    ),
    _pair_scenario(
        "graph-bounded-mixed-prefill",
        "graph:bounded-prefill",
        "graph:max-token-ceiling",
        config={
            "num_cuda_graphs": 4,
            "force_build_cuda_graphs": True,
            "inference_config_overrides": {
                "cuda_graph_max_tokens": 16,
                "cuda_graph_mixed_prefill_count": 2,
            },
        },
        signals=("cuda-graph", "graph-bounded-config"),
    ),
    _pair_scenario(
        "graph-mixed-layer-linear",
        "graph:layer-scope",
        "graph:mixed-prefill",
        "graph:linear",
        "graph:all-prefills",
        "graph:mixed-count",
        config={
            "num_cuda_graphs": 4,
            "force_build_cuda_graphs": True,
            "cuda_graph_all_prefills": True,
            "inference_cuda_graph_scope": InferenceCudaGraphScope.layer,
            "inference_config_overrides": {
                "cuda_graph_mixed_prefill_count": 2,
                "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.LINEAR,
                "cuda_graph_max_tokens": 24,
            },
        },
        signals=("cuda-graph", "graph-mixed-config"),
    ),
    _pair_scenario(
        "sampling-temperature-top-k",
        "sampling:temperature",
        "sampling:top-k",
        sampling=({"temperature": 0.7, "top_k": 8},),
        signals=("sampled",),
        parity="reproducible",
    ),
    _pair_scenario(
        "sampling-top-p",
        "sampling:top-p",
        sampling=({"temperature": 1.1, "top_k": 0, "top_p": 0.85},),
        signals=("sampled",),
        parity="reproducible",
    ),
    _pair_scenario(
        "sampling-top-k-top-p",
        "sampling:top-k-top-p",
        config={"sampling_backend": "flashinfer"},
        sampling=({"temperature": 0.8, "top_k": 12, "top_p": 0.9},),
        signals=("flashinfer", "sampled"),
        prerequisite="flashinfer",
        parity="reproducible",
    ),
    _pair_scenario(
        "raw-prompt-top-n-logprobs",
        "logprobs:raw",
        "logprobs:prompt",
        "logprobs:top-n",
        "logits:full",
        config={"return_log_probs": True, "materialize_only_last_token_logits": False},
        sampling=(
            {"return_log_probs": True, "top_n_logprobs": 2},
            {"return_log_probs": True, "top_n_logprobs": 5},
        ),
        signals=("full-logits", "logprobs", "top-n"),
    ),
    _pair_scenario(
        "processed-skip-prompt-logprobs",
        "logprobs:processed",
        "logprobs:skip-prompt",
        config={
            "return_log_probs": True,
            "skip_prompt_log_probs": True,
            "logprobs_mode": "processed_logprobs",
        },
        sampling=({"return_log_probs": True, "skip_prompt_log_probs": True, "top_n_logprobs": 3},),
        signals=("logprobs", "top-n", "processed-logprobs"),
    ),
    _pair_scenario(
        "flashinfer-sampling",
        "sampling:flashinfer",
        config={"sampling_backend": "flashinfer"},
        sampling=({"temperature": 0.8, "top_k": 0, "top_p": 0.0},),
        signals=("flashinfer", "sampled"),
        prerequisite="flashinfer",
        parity="reproducible",
    ),
    _pair_scenario(
        "length-zero-and-total",
        "length:num-generate",
        "length:num-total",
        "length:zero-output",
        request_profile="lengths",
        signals=("lengths",),
    ),
    _pair_scenario("string-prompt-bos", "prompt:add-bos", request_profile="bos", signals=("bos",)),
    _pair_scenario(
        "termination-values",
        "termination:explicit-eos",
        "termination:disabled",
        request_profile="termination",
        signals=("termination",),
    ),
    _pair_scenario(
        "stop-sequence-keep",
        "termination:stop-sequence-keep",
        request_profile="stop-keep",
        signals=("stop",),
    ),
    _pair_scenario(
        "stop-sequence-strip",
        "termination:stop-sequence-strip",
        request_profile="stop-strip",
        signals=("stop",),
    ),
    _pair_scenario(
        "mtp-depth-one",
        "speculation:mtp-depth-one",
        config={"num_speculative_tokens": 1},
        signals=("mtp",),
    ),
    _pair_scenario(
        "mtp-graph-heterogeneous-logprobs",
        "speculation:mtp-depth-two",
        "interaction:mtp-graph-metadata-compaction",
        config={
            "num_speculative_tokens": 2,
            "num_cuda_graphs": 4,
            "force_build_cuda_graphs": True,
            "materialize_only_last_token_logits": False,
        },
        sampling=(
            {"return_log_probs": True, "skip_prompt_log_probs": False, "top_n_logprobs": 2},
            {"return_log_probs": False, "skip_prompt_log_probs": True},
            {"return_log_probs": True, "skip_prompt_log_probs": True, "top_n_logprobs": 4},
        ),
        signals=("cuda-graph", "logprobs", "mtp", "top-n"),
        # MTP changes the forward batch shape; near-tied non-selected alternatives
        # may exchange the final top-N slot while selected-token parity remains exact.
        exact_top_n=False,
    ),
    _pair_scenario(
        "hybrid-mamba",
        "model:hybrid-mamba",
        "interaction:mamba-state-compaction",
        "mamba:memory-ratio",
        config={
            "model_provider": "hybrid",
            "inference_config_overrides": {"mamba_memory_ratio": 0.5},
        },
        signals=("hybrid", "mamba-memory-ratio"),
        prerequisite="mamba",
        atol=5.0e-3,
    ),
    _pair_scenario(
        "transformer-engine",
        "implementation:transformer-engine",
        config={"transformer_impl": "transformer_engine"},
        signals=("transformer-engine",),
        prerequisite="transformer-engine",
    ),
    _pair_scenario(
        "inference-optimized",
        "implementation:inference-optimized",
        config={"transformer_impl": "inference_optimized"},
        signals=("inference-optimized",),
    ),
    _pair_scenario(
        "fp8-transformer-engine",
        "precision:fp8",
        config={"fp8": True},
        signals=("fp8",),
        prerequisite="fp8",
        atol=5.0e-3,
        parity="reproducible",
    ),
    _pair_scenario(
        "flashinfer-fused-rope",
        "kernel:flashinfer-fused-rope",
        config={
            "hidden_size": 64,
            "position_embedding_type": "rope",
            "inference_config_overrides": {"use_flashinfer_fused_rope": True},
        },
        signals=("fused-rope",),
        prerequisite="flashinfer",
        parity="reproducible",
    ),
    _pair_scenario(
        "swa-off-by-one-sink",
        "attention:swa-all-layers",
        "attention:off-by-one-sink",
        config={"window_size": (4, 0), "softmax_type": "off-by-one"},
        signals=("swa", "softmax-sink"),
    ),
    _pair_scenario(
        "alternating-swa-learnable-sink",
        "attention:swa-alternating",
        "attention:learnable-sink",
        config={"window_size": (4, 0), "window_attn_skip_freq": 2, "softmax_type": "learnable"},
        signals=("swa", "softmax-sink"),
    ),
    _pair_scenario(
        "shared-dp-seed",
        "seed:shared-across-dp",
        config={"inference_config_overrides": {"offset_sampling_seed_by_dp_rank": False}},
        sampling=({"temperature": 0.8, "top_k": 8},),
        signals=("shared-seed", "sampled"),
        parity="reproducible",
    ),
)

_ASYNC_PARALLEL_SCENARIOS = (
    _pair_scenario(
        "tp2-pp2-sp-dp2",
        "topology:tp",
        "topology:pp",
        "topology:sp",
        "topology:dp",
        config={
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "sequence_parallel": True,
        },
        signals=("parallel",),
    ),
    _pair_scenario(
        "moe-ep2-nccl",
        "model:moe",
        "topology:ep",
        "dispatcher:nccl",
        "interaction:moe-ep-ordering",
        config={"expert_model_parallel_size": 2},
        signals=("moe", "parallel"),
    ),
    _pair_scenario(
        "optimized-tp2-sp",
        "interaction:optimized-tp-sp",
        config={
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "parallel"),
    ),
)

_NEGATIVE_PAIR_OWNERS = {
    "unsupported:routing-replay": "test_async_negative_routing_replay",
    "unsupported:mtp-depth-mismatch": "test_async_negative_mtp_depth_mismatch",
    "unsupported:processed-logprobs-mtp": "test_async_negative_processed_logprobs_mtp",
    "unsupported:skip-bookkeeping": "test_async_negative_skip_bookkeeping",
}

_FOCUSED_PAIR_OWNERS = {
    "events:paused-request": "test_async_bookkeep_uses_consumed_chunked_prefill_request_id",
    "metadata:heterogeneous-survivor-compaction": (
        "test_async_compaction_preserves_all_request_metadata"
    ),
    "lifecycle:reset-clears-pending-logits": "test_async_reset_clears_pending_logits",
    "lifecycle:recompute-clears-pending-logits": (
        "test_async_suspend_pending_logits_lifecycle[recompute-clears]"
    ),
    "lifecycle:persist-preserves-pending-logits": (
        "test_async_suspend_pending_logits_lifecycle[persist-preserves]"
    ),
    "lifecycle:offload-preserves-pending-logits": (
        "test_async_suspend_pending_logits_lifecycle[offload-preserves]"
    ),
}

# This is intentionally independent of the owner tables above: adding an owner
# does not silently redefine what "complete" means, and removing one leaves a
# precise coverage-hole failure.
_REQUIRED_ASYNC_PAIR_UNIVERSE = frozenset(
    {
        "attention:learnable-sink",
        "attention:off-by-one-sink",
        "attention:swa-all-layers",
        "attention:swa-alternating",
        "capacity:block-size",
        "capacity:buffer",
        "capacity:max-requests",
        "capacity:max-tokens",
        "capacity:paused-buffer",
        "dispatcher:nccl",
        "events:generated-token",
        "events:paused-request",
        "execution:eager",
        "graph:all-prefills",
        "graph:block-scope",
        "graph:bounded-prefill",
        "graph:decode-only",
        "graph:exponential",
        "graph:layer-scope",
        "graph:linear",
        "graph:max-token-ceiling",
        "graph:mixed-count",
        "graph:mixed-prefill",
        "implementation:inference-optimized",
        "implementation:transformer-engine",
        "interaction:mamba-state-compaction",
        "interaction:moe-ep-ordering",
        "interaction:mtp-graph-metadata-compaction",
        "interaction:optimized-tp-sp",
        "interaction:prefix-chunk-pressure",
        "kernel:flashinfer-fused-rope",
        "kv:persist",
        "kv:static-pointers",
        "length:num-generate",
        "length:num-total",
        "length:zero-output",
        "lifecycle:offload-preserves-pending-logits",
        "lifecycle:persist-preserves-pending-logits",
        "lifecycle:recompute-clears-pending-logits",
        "lifecycle:reset-clears-pending-logits",
        "logits:full",
        "logits:last-token",
        "logprobs:processed",
        "logprobs:prompt",
        "logprobs:raw",
        "logprobs:skip-prompt",
        "logprobs:top-n",
        "mamba:memory-ratio",
        "metadata:explicit-request-schema",
        "metadata:heterogeneous-survivor-compaction",
        "model:gpt",
        "model:hybrid-mamba",
        "model:moe",
        "precision:bf16",
        "precision:fp8",
        "prefill:chunked",
        "prefix:enabled",
        "prefix:lru",
        "prefix:ref-zero",
        "prompt:add-bos",
        "sampling:flashinfer",
        "sampling:temperature",
        "sampling:top-k",
        "sampling:top-k-top-p",
        "sampling:top-p",
        "sampling:torch",
        "seed:dp-offset",
        "seed:shared-across-dp",
        "speculation:mtp-depth-one",
        "speculation:mtp-depth-two",
        "termination:disabled",
        "termination:explicit-eos",
        "termination:stop-sequence-keep",
        "termination:stop-sequence-strip",
        "topology:dp",
        "topology:ep",
        "topology:pp",
        "topology:sp",
        "topology:tp",
        "unsupported:mtp-depth-mismatch",
        "unsupported:processed-logprobs-mtp",
        "unsupported:routing-replay",
        "unsupported:skip-bookkeeping",
    }
)

_UNIT_FIELD_PAIR_OWNERS = {
    "InferenceConfig": {
        "async_sched_mode": "execution:eager",
        "block_size_tokens": "capacity:block-size",
        "buffer_size_gb": "capacity:buffer",
        "cuda_graph_all_prefills": "graph:all-prefills",
        "cuda_graph_max_tokens": "graph:max-token-ceiling",
        "cuda_graph_mixed_prefill_count": "graph:mixed-count",
        "cuda_graph_sizing_distribution": "graph:linear",
        "enable_chunked_prefill": "prefill:chunked",
        "enable_prefix_caching": "prefix:enabled",
        "kv_cache_management_mode": "kv:persist",
        "logprobs_mode": "logprobs:processed",
        "mamba_inference_state_config": "interaction:mamba-state-compaction",
        "mamba_memory_ratio": "mamba:memory-ratio",
        "materialize_only_last_token_logits": "logits:full",
        "max_requests": "capacity:max-requests",
        "max_sequence_length": "length:num-total",
        "max_tokens": "capacity:max-tokens",
        "num_cuda_graphs": "graph:block-scope",
        "num_speculative_tokens": "speculation:mtp-depth-two",
        "offset_sampling_seed_by_dp_rank": "seed:shared-across-dp",
        "paused_buffer_size_gb": "capacity:paused-buffer",
        "prefix_caching_eviction_policy": "prefix:lru",
        "request_metadata_types": "metadata:explicit-request-schema",
        "sampling_backend": "sampling:flashinfer",
        "static_kv_memory_pointers": "kv:static-pointers",
        "track_generated_token_events": "events:generated-token",
        "track_paused_request_events": "events:paused-request",
        "use_cuda_graphs_for_non_decode_steps": "graph:decode-only",
        "use_flashinfer_fused_rope": "kernel:flashinfer-fused-rope",
    },
    "SamplingParams": {
        "add_BOS": "prompt:add-bos",
        "detokenize_stop_sequence": "termination:stop-sequence-keep",
        "num_tokens_to_generate": "length:num-generate",
        "num_tokens_total": "length:num-total",
        "return_log_probs": "logprobs:raw",
        "skip_prompt_log_probs": "logprobs:skip-prompt",
        "stop_words": "termination:stop-sequence-strip",
        "temperature": "sampling:temperature",
        "termination_id": "termination:explicit-eos",
        "top_k": "sampling:top-k",
        "top_n_logprobs": "logprobs:top-n",
        "top_p": "sampling:top-p",
    },
}

_COORDINATOR_PROFILE = Path(
    "tests/functional_tests/test_cases/gpt/"
    "gpt_dynamic_inference_tp1_pp1_dp8_583m_async_sched_zmq/model_config.yaml"
)
_UVM_PROFILE = Path(
    "tests/functional_tests/test_cases/gpt/"
    "gpt_dynamic_inference_tp1_pp1_dp8_583m_async_sched_uvm_persist_zmq/model_config.yaml"
)
_HYBRID_PROFILE = Path(
    "tests/functional_tests/test_cases/hybrid/"
    "hybrid_dynamic_inference_tp1_pp1_dp8_2b_async_sched_async/model_config.yaml"
)
_HTTP_PROFILE = Path(
    "tests/functional_tests/test_cases/gpt/"
    "gpt_inference_server_smoke_tp1_pp1_dp8_583m/serve_smoke.py"
)
_FUNCTIONAL_FIELD_OWNERS = {
    "InferenceConfig": {
        "disable_ep_consensus": (_COORDINATOR_PROFILE, "--inference-disable-ep-consensus"),
        "prefix_caching_coordinator_policy": (
            _COORDINATOR_PROFILE,
            "--inference-dynamic-batching-prefix-caching-coordinator-policy",
        ),
        "prefix_caching_mamba_gb": (
            _HYBRID_PROFILE,
            "--inference-dynamic-batching-prefix-caching-mamba-gb",
        ),
        "prefix_caching_routing_alpha": (
            _COORDINATOR_PROFILE,
            "--inference-dynamic-batching-prefix-caching-routing-alpha",
        ),
        "unified_memory_level": (_UVM_PROFILE, "--inference-dynamic-batching-unified-memory-level"),
        "use_synchronous_zmq_collectives": (
            _COORDINATOR_PROFILE,
            "--inference-use-synchronous-zmq-collectives",
        ),
    },
    "SamplingParams": {
        "return_prompt_tokens": (_HTTP_PROFILE, '"prompt_token_ids"'),
        "streaming": (_HTTP_PROFILE, '"stream": True'),
        "streaming_interval": (_HTTP_PROFILE, '"streaming_interval": 2'),
    },
}


def _flatten_dispositions(dispositions):
    return [field_name for fields in dispositions.values() for field_name in fields]


@pytest.mark.parametrize(
    ("owner", "declared", "dispositions"),
    [
        ("InferenceConfig", InferenceConfig.__annotations__, _INFERENCE_CONFIG_DISPOSITIONS),
        ("SamplingParams", SamplingParams.__annotations__, _SAMPLING_PARAM_DISPOSITIONS),
    ],
)
def test_async_pairwise_field_dispositions_are_closed(owner, declared, dispositions):
    """Every inference-facing field has one deliberate coverage disposition."""
    classified = _flatten_dispositions(dispositions)
    assert len(classified) == len(set(classified)), f"duplicate {owner} dispositions"
    assert set(classified) == set(declared), (
        f"{owner} disposition drift: missing={set(declared) - set(classified)}, "
        f"unknown={set(classified) - set(declared)}"
    )


@pytest.mark.parametrize(
    ("owner", "dispositions"),
    [
        ("InferenceConfig", _INFERENCE_CONFIG_DISPOSITIONS),
        ("SamplingParams", _SAMPLING_PARAM_DISPOSITIONS),
    ],
)
def test_async_pairwise_fields_have_concrete_owners(owner, dispositions):
    """Unit fields name a pair owner; functional fields name an active profile signal."""
    unit_owners = _UNIT_FIELD_PAIR_OWNERS[owner]
    assert set(unit_owners) == dispositions[_UNIT]
    assert set(unit_owners.values()) <= _REQUIRED_ASYNC_PAIR_UNIVERSE

    functional_owners = _FUNCTIONAL_FIELD_OWNERS[owner]
    assert set(functional_owners) == dispositions[_FUNCTIONAL]
    repo_root = Path(__file__).parents[4]
    for profile, activation in functional_owners.values():
        assert activation in (repo_root / profile).read_text()


def test_async_pair_owners_are_closed_and_unique():
    """Every supported or rejected async pair has exactly one named owner."""
    owners = {}
    scenario_names = set()
    for scenario in (*_ASYNC_PAIR_SCENARIOS, *_ASYNC_PARALLEL_SCENARIOS):
        assert scenario.name not in scenario_names
        scenario_names.add(scenario.name)
        assert scenario.signals, f"{scenario.name} has no runtime activation signal"
        assert scenario.parity in {"exact", "reproducible"}
        for pair in scenario.pairs:
            assert pair not in owners, f"{pair} owned by both {owners[pair]} and {scenario.name}"
            owners[pair] = scenario.name
    for pair, owner in _NEGATIVE_PAIR_OWNERS.items():
        assert pair not in owners
        assert owner.startswith("test_async_negative_")
        owners[pair] = owner
    for pair, owner in _FOCUSED_PAIR_OWNERS.items():
        assert pair not in owners
        assert owner.startswith("test_async_")
        owners[pair] = owner
    assert set(owners) == _REQUIRED_ASYNC_PAIR_UNIVERSE


_PROMPT_LENGTHS = (9, 5, 12, 7, 10, 6, 11)
_OUTPUT_LENGTHS = (10, 5, 9, 7, 11, 6, 8)
_REQUEST_WAVES = ((0, 1, 2), (3, 4), (5, 6))


def _check_scenario_prerequisite(scenario):
    if scenario.prerequisite == "flashinfer":
        pytest.importorskip("flashinfer")
    elif scenario.prerequisite == "mamba":
        available, reason = _check_mamba_sequence_packing_support()
        if not available:
            pytest.skip(reason)
    elif scenario.prerequisite == "transformer-engine":
        if not is_te_min_version("2.2.0"):
            pytest.skip("Transformer Engine 2.2.0 is required")
    elif scenario.prerequisite == "fp8":
        available, reason = check_fp8_support()
        if not available:
            pytest.skip(reason)


def _sampling_params_for_request(scenario, request_id, prompt_length):
    output_length = _OUTPUT_LENGTHS[request_id]
    kwargs = {
        "num_tokens_to_generate": output_length,
        "termination_id": -1,
        "temperature": scenario.config.get("temperature", 1.0),
        "top_k": scenario.config.get("top_k", 1),
        "top_p": scenario.config.get("top_p", 0.0),
        "return_log_probs": scenario.config.get("return_log_probs", False),
        "skip_prompt_log_probs": scenario.config.get("skip_prompt_log_probs", False),
    }
    if scenario.sampling:
        kwargs.update(scenario.sampling[request_id % len(scenario.sampling)])
    if scenario.request_profile == "lengths":
        if request_id == 0:
            kwargs["num_tokens_to_generate"] = 0
        elif request_id == 1:
            kwargs["num_tokens_to_generate"] = None
            kwargs["num_tokens_total"] = prompt_length + output_length
    elif scenario.request_profile == "bos":
        kwargs["add_BOS"] = True
    elif scenario.request_profile == "termination":
        # The probe run keeps termination disabled. The parity runs replace
        # request 1's value with a token observed from that deterministic probe.
        kwargs["termination_id"] = -1
    return SamplingParams(**kwargs)


def _make_scenario_requests(env, scenario, stop_tokens=None, termination_token=None):
    controller = env.engine.controller
    controller.tokenizer.detokenize = lambda tokens, **kwargs: (
        f"tok_{tokens[0]}" if tokens else ""
    )
    if scenario.request_profile.startswith("stop"):
        controller.tokenizer.bos = None
        controller.tokenizer.tokenize = lambda text: [int(token) for token in text.split()]

    requests = []
    for request_id, prompt_length in enumerate(_PROMPT_LENGTHS):
        raw_prompt = None
        if scenario.request_profile == "prefix":
            shared = torch.arange(512, dtype=torch.int64, device="cuda") % 97
            tail = torch.full(
                (request_id % 7 + 1,), request_id + 1, dtype=torch.int64, device="cuda"
            )
            prompt_tokens = torch.cat((shared, tail))
        elif scenario.request_profile == "bos":
            controller.tokenizer.bos = 98
            controller.tokenizer.tokenize = lambda text: [int(token) for token in text.split()]
            raw_prompt = " ".join(str((request_id + token) % 90) for token in range(prompt_length))
            prompt_tokens = torch.tensor(
                controller.tokenize_prompt(controller.tokenizer, raw_prompt, add_BOS=False),
                dtype=torch.int64,
                device="cuda",
            )
        else:
            prompt_tokens = (
                torch.arange(prompt_length, dtype=torch.int64, device="cuda") + request_id * 13
            ) % 97

        sampling_params = _sampling_params_for_request(scenario, request_id, len(prompt_tokens))
        if termination_token is not None and request_id == 1:
            sampling_params.termination_id = termination_token
        if stop_tokens is not None and request_id == 1:
            sampling_params.stop_words = [" ".join(str(token) for token in stop_tokens)]
            sampling_params.detokenize_stop_sequence = scenario.request_profile == "stop-keep"
        requests.append(
            DynamicInferenceRequest(
                request_id=request_id,
                prompt=raw_prompt,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                block_size_tokens=env.engine.context.block_size_tokens,
                enable_prefix_caching=env.engine.context.enable_prefix_caching,
            )
        )
    return requests


def _instrument_async_phases(controller, runtime):
    for method_name in (
        "_run_async_sched_step_no_overlap",
        "_run_async_sched_step_overlap",
        "_run_async_sched_step_overlap_mtp",
    ):
        original = getattr(controller, method_name)

        async def counted(*args, _name=method_name, _original=original, **kwargs):
            runtime[_name] += 1
            return await _original(*args, **kwargs)

        setattr(controller, method_name, counted)


def _as_float_list(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return list(values)


def _snapshot_requests(requests):
    snapshots = []
    for request in requests:
        snapshots.append(
            {
                "tokens": list(request.generated_tokens),
                "status": request.status,
                "prompt_logprobs": _as_float_list(request.prompt_log_probs),
                "generated_logprobs": _as_float_list(request.generated_log_probs),
                "prompt_top_n": request.prompt_top_n_logprobs,
                "generated_top_n": request.generated_top_n_logprobs,
                "events": [event.type for event in request.events],
            }
        )
    return snapshots


def _assert_top_n_parity(actual, expected, atol, exact):
    assert (actual is None) == (expected is None)
    if actual is None:
        return
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        assert all(isinstance(token, str) for token in actual_row)
        values = torch.tensor(list(actual_row.values()))
        assert not torch.isnan(values).any()
        assert not torch.isposinf(values).any()
        assert torch.isfinite(values).any()
        if exact:
            assert actual_row.keys() == expected_row.keys()
            assert list(actual_row.values()) == pytest.approx(
                list(expected_row.values()), rel=0, abs=atol
            )


def _assert_request_parity(
    actual_requests, expected, atol, compare_events=False, exact_numerics=True, exact_top_n=False
):
    assert len(actual_requests) == len(expected)
    for request, reference in zip(actual_requests, expected):
        assert request.status == reference["status"] == Status.COMPLETED
        assert len(request.generated_tokens) == len(reference["tokens"])
        prompt_logprobs = _as_float_list(request.prompt_log_probs) or []
        generated_logprobs = _as_float_list(request.generated_log_probs) or []
        assert len(prompt_logprobs) == len(reference["prompt_logprobs"] or [])
        assert len(generated_logprobs) == len(reference["generated_logprobs"] or [])
        if exact_numerics:
            assert request.generated_tokens == reference["tokens"]
            assert prompt_logprobs == pytest.approx(
                reference["prompt_logprobs"] or [], rel=0, abs=atol
            )
            assert generated_logprobs == pytest.approx(
                reference["generated_logprobs"] or [], rel=0, abs=atol
            )
        else:
            assert all(0 <= token < 100 for token in request.generated_tokens)
        assert torch.isfinite(torch.tensor(prompt_logprobs)).all()
        assert torch.isfinite(torch.tensor(generated_logprobs)).all()
        _assert_top_n_parity(
            request.prompt_top_n_logprobs, reference["prompt_top_n"], atol, exact_top_n
        )
        _assert_top_n_parity(
            request.generated_top_n_logprobs, reference["generated_top_n"], atol, exact_top_n
        )
        if compare_events:
            assert [event.type for event in request.events] == reference["events"]


class _AsyncPairwiseHarness(_DynamicInferenceEngineTestBase):
    """Shared real-engine runner for the async pair owners."""

    @classmethod
    def _run_scenario_mode(cls, scenario, mode, stop_tokens=None, termination_token=None):
        config = dict(_BASE_PAIR_CONFIG)
        config.update(scenario.config)
        config["async_sched_mode"] = mode
        env = cls._build_test_env(_DynamicEngineTestConfig(**config))
        env.requests = _make_scenario_requests(
            env, scenario, stop_tokens=stop_tokens, termination_token=termination_token
        )
        runtime = Counter()
        if mode == AsyncScheduleMode.ASYNC:
            _instrument_async_phases(env.engine.controller, runtime)

        def step():
            result = env.engine.step_modern()
            runtime["steps"] += 1
            runtime["cuda_graph_steps"] += int(env.engine.context.using_cuda_graph_this_step())
            runtime["max_paused"] = max(
                runtime["max_paused"], env.engine.context.paused_request_count
            )
            for record in result["finished_request_records"]:
                request = record.merge()
                env.requests[request.request_id] = request
                if request.request_id == 1 and any(
                    other.request_id != 1 and other.status not in (Status.COMPLETED, Status.FAILED)
                    for other in env.requests
                ):
                    runtime["middle_finished_with_survivors"] += 1

        for wave_idx, request_ids in enumerate(_REQUEST_WAVES):
            while wave_idx > 0 and not any(
                request.generated_tokens and request.status not in (Status.COMPLETED, Status.FAILED)
                for request in env.requests
            ):
                assert (
                    env.engine.has_unfinished_requests()
                ), f"{scenario.name} drained before the next request wave could arrive"
                step()
            if wave_idx > 0:
                runtime["arrival_during_decode"] += 1
            for request_id in request_ids:
                request = env.requests[request_id]
                prompt = request.prompt if request.prompt is not None else request.prompt_tokens
                env.engine.add_request(request_id, prompt, request.sampling_params)
                env.requests[request_id] = env.engine.get_request(request_id)
                env.requests[request_id].state = "pending"
            runtime["waves"] += 1
            runtime["max_waiting"] = max(
                runtime["max_waiting"], len(env.engine.waiting_request_ids)
            )
            step()
            if wave_idx == 0 and "suspend-resume" in scenario.signals:
                memory_ptr = env.engine.context.memory_buffer.data_ptr()
                env.engine.suspend()
                env.engine.resume()
                runtime["suspend-resume"] += 1
                runtime["static-pointer-preserved"] += int(
                    env.engine.context.memory_buffer.data_ptr() == memory_ptr
                )

        while env.engine.has_unfinished_requests():
            step()
            assert runtime["steps"] < 400, f"{scenario.name} did not converge"
        return env, runtime

    @staticmethod
    def _assert_runtime_signals(env, scenario, runtime, stop_tokens, termination_token):
        context = env.engine.context
        controller = env.engine.controller
        model_config = controller.inference_wrapped_model.model.config

        assert runtime["waves"] == 3
        assert len(env.requests) > context.max_requests
        assert runtime["max_waiting"] > 0
        assert runtime["arrival_during_decode"] > 0
        assert runtime["middle_finished_with_survivors"] > 0
        assert context.async_sched_step_count > 0
        assert context.async_sched_compaction_step_count > 0
        assert runtime["_run_async_sched_step_no_overlap"] > 0
        assert (
            runtime["_run_async_sched_step_overlap"] + runtime["_run_async_sched_step_overlap_mtp"]
            > 0
        )
        assert not controller._async_sched_logits.is_valid
        assert context.total_request_count == context.active_token_count == 0

        signals = set(scenario.signals)
        if "torch-backend" in signals:
            assert context.config.sampling_backend == "torch"
        if "persist" in signals:
            assert context.kv_cache_management_mode == KVCacheManagementMode.PERSIST
        if "static-pointers" in signals:
            assert context.static_kv_memory_pointers
        if "suspend-resume" in signals:
            assert runtime["suspend-resume"] == runtime["static-pointer-preserved"] == 1
        if "dp-offset" in signals:
            assert context.config.offset_sampling_seed_by_dp_rank
        if "last-logits" in signals:
            assert context.config.materialize_only_last_token_logits
        if "events" in signals:
            for request in env.requests:
                event_types = [event.type for event in request.events]
                assert event_types.count(DynamicInferenceEventType.GENERATED_TOKEN) == len(
                    request.generated_tokens
                )
        if "metadata-schema" in signals:
            assert (
                context.config.request_metadata_types
                == DynamicInferenceRequest.get_metadata_types()
            )
            assert set(context.active_request_metadata) == {
                label for label, _ in context.config.request_metadata_types
            }
        if "chunked" in signals:
            assert context.enable_chunked_prefill
            assert any(len(request.prompt_tokens) > context.max_tokens for request in env.requests)
        if "capacity" in signals:
            assert context.max_requests == 4
            assert context.max_tokens == 8
            assert context.block_size_tokens == 512
            assert context.kv_block_allocator.paused_limit > 0
        if "paused-buffer-allocation" in signals:
            assert context.config.paused_buffer_size_gb == pytest.approx(0.004)
            assert context.kv_block_allocator.paused_limit > 0
        if "prefix-hit" in signals:
            assert env.engine._prefix_cache_hits > 0
            assert env.engine._prefill_tokens_skipped > 0
        if "ref-zero" in signals:
            assert context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
        if "lru" in signals:
            assert context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU
        if "cuda-graph" in signals:
            assert context.cuda_graph_batch_dimensions_list
            assert runtime["cuda_graph_steps"] > 0
        if "graph-decode-config" in signals:
            assert not context.use_cuda_graphs_for_non_decode_steps
            assert context.config.cuda_graph_sizing_distribution == (
                CudaGraphSizingDistribution.EXPONENTIAL
            )
            assert context.config.cuda_graph_max_tokens == 16
            assert all(
                dimensions.prefill_req_count == 0
                for dimensions in context.cuda_graph_batch_dimensions_list
            )
        if "graph-mixed-config" in signals:
            assert context.use_cuda_graphs_for_non_decode_steps
            assert context.config.cuda_graph_all_prefills
            assert context.config.cuda_graph_mixed_prefill_count == 2
            assert (
                context.config.cuda_graph_sizing_distribution == CudaGraphSizingDistribution.LINEAR
            )
            assert max(context.cuda_graph_token_counts) == context.max_tokens
            assert any(
                dimensions.prefill_req_count > 0
                for dimensions in context.cuda_graph_batch_dimensions_list
            )
        if "graph-bounded-config" in signals:
            assert context.use_cuda_graphs_for_non_decode_steps
            assert not context.config.cuda_graph_all_prefills
            assert context.config.cuda_graph_max_tokens == 16
            assert max(context.cuda_graph_token_counts) == 16
            assert any(
                dimensions.prefill_req_count > 0
                for dimensions in context.cuda_graph_batch_dimensions_list
            )
        if "sampled" in signals:
            assert all(request.generated_tokens for request in env.requests)
        if "logprobs" in signals:
            for request in env.requests:
                if request.sampling_params.return_log_probs:
                    assert request.generated_log_probs is not None
                    assert len(request.generated_log_probs) == len(request.generated_tokens)
                else:
                    assert request.prompt_log_probs is None
                    assert request.generated_log_probs is None
        if "full-logits" in signals:
            assert not context.config.materialize_only_last_token_logits
        if "top-n" in signals:
            for request in env.requests:
                if request.sampling_params.top_n_logprobs > 0:
                    assert request.generated_top_n_logprobs
                    assert len(request.generated_top_n_logprobs) == len(request.generated_tokens)
                    for token, logprob, top_n in zip(
                        request.generated_tokens,
                        request.generated_log_probs,
                        request.generated_top_n_logprobs,
                    ):
                        token_text = controller.tokenizer.detokenize([token])
                        assert 0 < len(top_n) <= request.sampling_params.top_n_logprobs
                        assert token_text in top_n
                        assert top_n[token_text] == pytest.approx(logprob, rel=0, abs=0.1)
                    if request.sampling_params.skip_prompt_log_probs:
                        assert not request.prompt_top_n_logprobs
                    else:
                        assert len(request.prompt_top_n_logprobs) == len(request.prompt_tokens) - 1
        if "processed-logprobs" in signals:
            assert context.config.logprobs_mode == "processed_logprobs"
        if "flashinfer" in signals:
            assert context.config.sampling_backend == "flashinfer"
        if "lengths" in signals:
            assert env.requests[0].generated_tokens == []
            assert len(env.requests[1].generated_tokens) == _OUTPUT_LENGTHS[1]
        if "bos" in signals:
            assert all(request.sampling_params.add_BOS for request in env.requests)
            assert all(request.prompt_tokens[0].item() == 98 for request in env.requests)
        if "termination" in signals:
            assert {request.sampling_params.termination_id for request in env.requests} == {
                -1,
                termination_token,
            }
            request = env.requests[1]
            assert termination_token is not None
            assert request.generated_tokens[-1] == termination_token
            assert len(request.generated_tokens) < _OUTPUT_LENGTHS[1]
        if "stop" in signals:
            request = env.requests[1]
            assert stop_tokens is not None
            assert len(request.generated_tokens) < _OUTPUT_LENGTHS[1]
            kept = request.generated_tokens[-len(stop_tokens) :] == list(stop_tokens)
            assert kept is (scenario.request_profile == "stop-keep")
        if "mtp" in signals:
            assert env.engine._spec_steps > 0
            assert int(env.engine._spec_tokens_proposed_per_pos.sum()) > 0
        if "hybrid" in signals:
            assert context.is_hybrid_model
            assert hasattr(context, "mamba_conv_states_shape")
        if "mamba-memory-ratio" in signals:
            assert context.config.mamba_memory_ratio == 0.5
        if "transformer-engine" in signals:
            assert model_config.transformer_impl == "transformer_engine"
        if "inference-optimized" in signals:
            assert model_config.transformer_impl == "inference_optimized"
        if "fp8" in signals:
            assert model_config.fp8 is not None
        if "fused-rope" in signals:
            assert context.use_flashinfer_fused_rope
            assert model_config.hidden_size // model_config.num_attention_heads == 16
            model = controller.inference_wrapped_model.model
            assert model.rotary_pos_emb.inv_freq.is_cuda
            assert model.rotary_pos_emb_cache[context.max_sequence_length].is_cuda
        if "swa" in signals:
            assert model_config.window_size == scenario.config["window_size"]
        if "softmax-sink" in signals:
            assert model_config.softmax_type == scenario.config["softmax_type"]
        if "shared-seed" in signals:
            assert not context.config.offset_sampling_seed_by_dp_rank
            assert controller.sampling_rng.initial_seed() == model_config.inference_sampling_seed
            generated = torch.tensor(
                [token for request in env.requests for token in request.generated_tokens],
                dtype=torch.int64,
                device="cuda",
            )
            gathered = [
                torch.empty_like(generated)
                for _ in range(torch.distributed.get_world_size(controller.dp_group))
            ]
            torch.distributed.all_gather(gathered, generated, group=controller.dp_group)
            assert all(torch.equal(peer, generated) for peer in gathered)
        if "dp-offset" in signals:
            expected_seed = model_config.inference_sampling_seed + torch.distributed.get_rank(
                group=controller.dp_group
            )
            assert controller.sampling_rng.initial_seed() == expected_seed
        if "parallel" in signals:
            assert (
                model_config.tensor_model_parallel_size > 1
                or model_config.pipeline_model_parallel_size > 1
                or model_config.expert_model_parallel_size > 1
            )
        if "moe" in signals:
            assert model_config.num_moe_experts is not None
            assert context._nccl_ep_dispatcher

    @classmethod
    def _assert_scenario_pair(cls, scenario):
        stop_tokens = None
        termination_token = None
        if scenario.request_profile.startswith("stop"):
            probe_env, _ = cls._run_scenario_mode(scenario, AsyncScheduleMode.LEGACY)
            stop_tokens = tuple(probe_env.requests[1].generated_tokens[1:3])
            assert len(stop_tokens) == 2
            del probe_env
            gc.collect()
            delete_cuda_graphs()
            torch.cuda.empty_cache()
        elif scenario.request_profile == "termination":
            probe_env, _ = cls._run_scenario_mode(scenario, AsyncScheduleMode.LEGACY)
            termination_token = probe_env.requests[1].generated_tokens[0]
            assert termination_token != -1
            del probe_env
            gc.collect()
            delete_cuda_graphs()
            torch.cuda.empty_cache()

        legacy_env, _ = cls._run_scenario_mode(
            scenario,
            AsyncScheduleMode.LEGACY,
            stop_tokens=stop_tokens,
            termination_token=termination_token,
        )
        expected = _snapshot_requests(legacy_env.requests)
        del legacy_env
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

        async_env, runtime = cls._run_scenario_mode(
            scenario,
            AsyncScheduleMode.ASYNC,
            stop_tokens=stop_tokens,
            termination_token=termination_token,
        )
        _assert_request_parity(
            async_env.requests,
            expected,
            scenario.atol,
            compare_events="events" in scenario.signals,
            exact_numerics=scenario.parity == "exact",
            exact_top_n=scenario.parity == "exact" and scenario.exact_top_n,
        )
        cls._assert_runtime_signals(async_env, scenario, runtime, stop_tokens, termination_token)
        if scenario.parity == "reproducible":
            async_expected = _snapshot_requests(async_env.requests)
            del async_env
            gc.collect()
            delete_cuda_graphs()
            torch.cuda.empty_cache()
            repeat_env, repeat_runtime = cls._run_scenario_mode(
                scenario,
                AsyncScheduleMode.ASYNC,
                stop_tokens=stop_tokens,
                termination_token=termination_token,
            )
            _assert_request_parity(
                repeat_env.requests,
                async_expected,
                scenario.atol,
                compare_events="events" in scenario.signals,
                exact_top_n=True,
            )
            cls._assert_runtime_signals(
                repeat_env, scenario, repeat_runtime, stop_tokens, termination_token
            )


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestAsyncSchedulePairwise(_AsyncPairwiseHarness):
    """Async scheduling crossed with every major single-topology inference feature."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        _set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("scenario", _ASYNC_PAIR_SCENARIOS, ids=lambda case: case.name)
    @torch.inference_mode()
    def test_async_matches_legacy_for_owned_pair(self, scenario):
        _check_scenario_prerequisite(scenario)
        try:
            self._assert_scenario_pair(scenario)
        finally:
            gc.collect()
            delete_cuda_graphs()
            torch.cuda.empty_cache()


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestAsyncSchedulePairwiseParallel(_AsyncPairwiseHarness):
    """Pair owners that require a non-default distributed topology."""

    @pytest.mark.parametrize("scenario", _ASYNC_PARALLEL_SCENARIOS, ids=lambda case: case.name)
    @torch.inference_mode()
    def test_async_matches_legacy_for_parallel_pair(self, scenario):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if scenario.name == "tp2-pp2-sp-dp2" and world_size != 8:
            pytest.skip("the TP2/PP2/DP2 owner requires exactly eight GPUs")
        if world_size < 4:
            pytest.skip("parallel async pair owners require at least four GPUs")
        _check_scenario_prerequisite(scenario)
        config = scenario.config
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=config.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=config.get("pipeline_model_parallel_size", 1),
            expert_model_parallel_size=config.get("expert_model_parallel_size", 1),
            expert_tensor_parallel_size=1,
        )
        try:
            assert parallel_state.get_tensor_model_parallel_world_size() == config.get(
                "tensor_model_parallel_size", 1
            )
            assert parallel_state.get_pipeline_model_parallel_world_size() == config.get(
                "pipeline_model_parallel_size", 1
            )
            assert parallel_state.get_expert_model_parallel_world_size() == config.get(
                "expert_model_parallel_size", 1
            )
            if scenario.name == "tp2-pp2-sp-dp2":
                assert parallel_state.get_data_parallel_world_size() == 2
            self._assert_scenario_pair(scenario)
        finally:
            gc.collect()
            delete_cuda_graphs()
            torch.cuda.empty_cache()
            _set_rounder(64)
            Utils.destroy_model_parallel()


def test_async_compaction_preserves_all_request_metadata():
    """Middle-row completion compacts every metadata field into survivor order."""
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.num_speculative_tokens = 1
    controller._enable_cuda_graph = False
    controller._all_logits_cuda = torch.arange(24).reshape(1, 6, 4)
    controller._async_sched_logits = AsyncScheduleLogitsState()
    controller._async_sched_logits.set_pending(3, torch.arange(6))

    metadata = {
        "temperature": torch.tensor([0.5, 0.7, 0.9]),
        "top_k": torch.tensor([2, 4, 8]),
        "top_p": torch.tensor([0.1, 0.2, 0.3]),
        "termination_id": torch.tensor([90, 91, 92]),
        "return_log_probs": torch.tensor([True, False, True]),
        "skip_prompt_log_probs": torch.tensor([False, True, True]),
        "top_n_logprobs": torch.tensor([2, 0, 5]),
        "custom_metadata": torch.tensor([101, 202, 303]),
    }
    gpu_view = SimpleNamespace(
        temperature=metadata["temperature"].clone(),
        top_k=metadata["top_k"].clone(),
        top_p=metadata["top_p"].clone(),
    )
    context = SimpleNamespace(active_request_metadata=metadata, gpu_view=gpu_view)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    expected = {label: values[[0, 2]].clone() for label, values in metadata.items()}

    controller._compact_async_sched_logits(torch.tensor([0, 2]))

    for label, values in expected.items():
        assert torch.equal(context.active_request_metadata[label][:2], values), label
    assert torch.equal(gpu_view.temperature[:2], expected["temperature"])
    assert torch.equal(gpu_view.top_k[:2], expected["top_k"])
    assert torch.equal(gpu_view.top_p[:2], expected["top_p"])
    assert torch.equal(
        controller._all_logits_cuda, torch.arange(24).reshape(1, 6, 4)[:, [0, 1, 4, 5]]
    )
    assert torch.equal(controller._async_sched_logits.token_row_indices, torch.tensor([0, 1, 4, 5]))


def _controller_with_pending_logits():
    controller = TextGenerationController.__new__(TextGenerationController)
    controller._async_sched_logits = AsyncScheduleLogitsState()
    controller._async_sched_logits.set_pending(4, torch.tensor([0, 1, 2]))
    return controller


def test_async_reset_clears_pending_logits():
    """Engine reset cannot expose logits produced for the previous request batch."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(reset=mock.Mock())
    engine.controller = _controller_with_pending_logits()
    engine.num_speculative_tokens = 1
    engine._loop = None

    with (
        mock.patch(
            "megatron.core.inference.engines.dynamic_engine.torch.distributed.get_rank",
            return_value=0,
        ),
        mock.patch(
            "megatron.core.inference.engines.dynamic_engine.torch.cuda.Event",
            return_value=mock.Mock(),
        ),
    ):
        engine.reset()

    assert not engine.controller._async_sched_logits.is_valid
    assert engine.controller._async_sched_logits.token_row_indices is None


@pytest.mark.parametrize(
    ("mode", "preserve_pending"),
    [
        pytest.param(KVCacheManagementMode.PERSIST, True, id="persist-preserves"),
        pytest.param(KVCacheManagementMode.OFFLOAD, True, id="offload-preserves"),
        pytest.param(KVCacheManagementMode.RECOMPUTE, False, id="recompute-clears"),
    ],
)
def test_async_suspend_pending_logits_lifecycle(mode, preserve_pending):
    """Only recomputation invalidates a forward result pending across suspend."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = _controller_with_pending_logits()
    engine.context = SimpleNamespace(
        deallocate_inference_state_buffers=mock.Mock(),
        dynamo_helper=None,
        kv_cache_management_mode=mode,
        static_kv_memory_pointers=True,
    )
    engine.state = EngineState.RUNNING
    engine.unified_memory_level = 0
    engine.waiting_request_ids = deque()
    engine.requests = {}
    engine.use_coordinator = False

    with (
        mock.patch.object(DynamicInferenceEngine, "suspend_resume_ctx", return_value=nullcontext()),
        mock.patch("megatron.core.inference.engines.dynamic_engine.InferenceMode.unset_active"),
    ):
        engine.suspend()

    assert engine.controller._async_sched_logits.is_valid is preserve_pending
    if preserve_pending:
        assert torch.equal(
            engine.controller._async_sched_logits.token_row_indices, torch.tensor([0, 1, 2])
        )
    else:
        assert engine.controller._async_sched_logits.token_row_indices is None


def test_async_negative_routing_replay():
    engine = _make_engine(
        model_config_num_moe_experts=4, model_config_moe_enable_routing_replay=True
    )
    with pytest.raises(ValueError, match="routing replay"):
        engine._validate_async_sched_support_for_config()


def test_async_negative_mtp_depth_mismatch():
    engine = _make_engine(num_speculative_tokens=2, controller_num_mtp_depths=1)
    with pytest.raises(ValueError, match="one MTP depth"):
        engine._validate_async_sched_support_for_config()


def test_async_negative_processed_logprobs_mtp():
    with pytest.raises(ValueError, match="processed_logprobs.*speculative decoding"):
        InferenceConfig(
            async_sched_mode=AsyncScheduleMode.ASYNC,
            logprobs_mode="processed_logprobs",
            num_speculative_tokens=1,
        )


def test_async_negative_skip_bookkeeping():
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=SimpleNamespace(
            config=SimpleNamespace(async_sched_mode=AsyncScheduleMode.ASYNC)
        )
    )
    with pytest.raises(AssertionError, match="requires request bookkeeping"):
        asyncio.run(controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True))
