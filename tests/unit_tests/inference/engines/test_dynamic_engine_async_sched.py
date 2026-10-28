# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import os
from collections import Counter, deque
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import (
    AsyncScheduleMode,
    CudaGraphSizingDistribution,
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext, DynamoHelper
from megatron.core.inference.disaggregation.engine import (
    DisaggDynamicInferenceEngine,
    StateHandoffDynamicInferenceEngine,
)
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import PendingKvImport
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState, _get_decode_only_log_state
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicVLMInferenceRequest,
    Status,
)
from megatron.core.inference.sampling_params import MIN_SAMPLING_TEMPERATURE, SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    AsyncScheduleForwardState,
    DecodeOnly,
    DynamicBatchControllerStepResult,
    TextGenerationController,
)
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.transformer.cuda_graphs import CudaGraphManager, delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import reset_rounder as _reset_rounder
from tests.unit_tests.test_utilities import Utils


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
        ({"model_config_moe_enable_routing_replay": True}, False),
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
    engine._pending_kv_imports.append(
        SimpleNamespace(request_id=7, resume_tokens=[55], cancel_requested=False)
    )
    engine._handoff_completion_notifications[7] = (False, True)

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
    finalize_requests = (
        engine.controller.async_generate_output_tokens_dynamic_batch.await_args.kwargs[
            "finalize_requests"
        ]
    )
    assert finalize_requests.func == engine._finalize_finished_requests
    assert finalize_requests.args == (
        {},
        {key: value for key, value in context_state.items() if key != "kv_stats"},
        False,
    )
    if output is not None:
        assert finalize_requests.args[0] is output["completed_requests"]
    if mode == AsyncScheduleMode.LEGACY:
        engine._should_run_async_sched_overlap.assert_not_called()
        engine.controller.async_generate_output_tokens_dynamic_batch.assert_awaited_once_with(
            finalize_requests=finalize_requests
        )
    else:
        engine._should_run_async_sched_overlap.assert_called_once_with()
        engine.controller.async_generate_output_tokens_dynamic_batch.assert_awaited_once_with(
            run_async_overlap=run_async_overlap,
            schedule_waiting_requests=(
                None if run_async_overlap else engine.schedule_waiting_requests
            ),
            finalize_requests=finalize_requests,
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


# Only list a pair when the harness below observes its production call/state
# transition during both async phase shapes and checks parity or an exact invariant.
@dataclass(frozen=True)
class _AsyncPairScenario:
    """A runtime-backed async-scheduling interaction scenario."""

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
        "logits:last-token",
        config={"track_generated_token_events": True},
        signals=(
            "eager",
            "events",
            "gpt",
            "last-logits",
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
        config={"enable_chunked_prefill": True, "context_max_tokens": 8},
        signals=("capacity", "chunked"),
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
        "interaction:prefix-hit-with-chunking",
        config={
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_sequence_length": 544,
            "context_block_size_tokens": 256,
            "context_max_tokens": 384,
            "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.LRU,
        },
        request_profile="prefix",
        signals=("chunked", "prefix-hit"),
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
            "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.EXPONENTIAL,
            "cuda_graph_max_tokens": 16,
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
            "cuda_graph_max_tokens": 16,
            "cuda_graph_mixed_prefill_count": 2,
        },
        signals=("cuda-graph", "graph-bounded-config"),
    ),
    _pair_scenario(
        "graph-mixed-layer-linear",
        "graph:layer-scope",
        "graph:mixed-prefill",
        "graph:linear",
        "graph:mixed-count",
        config={
            "num_cuda_graphs": 4,
            "force_build_cuda_graphs": True,
            "cuda_graph_all_prefills": True,
            "inference_cuda_graph_scope": InferenceCudaGraphScope.layer,
            "cuda_graph_mixed_prefill_count": 2,
            "cuda_graph_sizing_distribution": CudaGraphSizingDistribution.LINEAR,
            "cuda_graph_max_tokens": 24,
        },
        signals=("cuda-graph", "graph-mixed-config"),
    ),
    _pair_scenario(
        "sampling-temperature-top-k",
        "sampling:temperature",
        "sampling:top-k",
        sampling=({"temperature": 0.7, "top_k": 8},),
        signals=("sampled", "temperature-filter", "top-k-filter"),
        parity="reproducible",
    ),
    _pair_scenario(
        "sampling-top-p",
        "sampling:top-p",
        sampling=({"temperature": 1.1, "top_k": 0, "top_p": 0.85},),
        signals=("sampled", "temperature-filter", "top-p-filter"),
        parity="reproducible",
    ),
    _pair_scenario(
        "sampling-top-k-top-p",
        "sampling:top-k-top-p",
        config={"sampling_backend": "flashinfer"},
        sampling=({"temperature": 0.8, "top_k": 12, "top_p": 0.9},),
        signals=("flashinfer", "sampled", "temperature-filter", "top-k-filter", "top-p-filter"),
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
        # Different forward batch shapes may exchange a near-tied, non-selected
        # final candidate; the harness still requires exact async-repeat top-N.
        exact_top_n=False,
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
        "mtp-repeated-all-accept",
        "speculation:mtp-repeated-layer",
        "speculation:real-accept",
        "kv:mtp-draft-cache",
        config={
            "mtp_use_repeated_layer": True,
            "num_speculative_tokens": 2,
            "zero_model_parameters": True,
        },
        signals=("mtp", "mtp-accept-all", "mtp-repeated"),
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
        signals=("cuda-graph", "logprobs", "metadata-compaction", "mtp", "top-n"),
        # MTP changes the forward batch shape; near-tied non-selected alternatives
        # may exchange the final top-N slot while selected-token parity remains exact.
        exact_top_n=False,
    ),
    _pair_scenario(
        "vlm-precomputed-media",
        "model:vlm",
        "interaction:live-image-embedding-injection",
        "lifecycle:stale-media-opt-in",
        config={"model_provider": "vlm", "allow_stale_multimodal_embeddings": True},
        signals=("suspend-resume", "vlm", "vlm-stale-media"),
        request_profile="vlm",
        atol=5.0e-3,
    ),
    _pair_scenario(
        "hybrid-mamba",
        "model:hybrid-mamba",
        "interaction:mamba-state-compaction",
        config={"model_provider": "hybrid"},
        signals=("hybrid", "metadata-compaction"),
        prerequisite="mamba",
        atol=5.0e-3,
    ),
    _pair_scenario(
        "hybrid-gdn",
        "model:hybrid-gdn",
        "interaction:gdn-state-compaction",
        config={"model_provider": "hybrid", "ssm_mixer": "gdn"},
        signals=("gdn", "hybrid", "metadata-compaction"),
        prerequisite="gdn",
        atol=5.0e-3,
    ),
    _pair_scenario(
        "hybrid-gdp",
        "model:hybrid-gdp",
        "interaction:gdp-state-compaction",
        config={"model_provider": "hybrid", "ssm_mixer": "gdp"},
        signals=("gdp", "hybrid", "metadata-compaction"),
        prerequisite="gdp",
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
        config={"fp8": True, "hidden_size": 128},
        signals=("fp8",),
        prerequisite="fp8",
        atol=5.0e-3,
        parity="reproducible",
    ),
    _pair_scenario(
        "mxfp8-inference-optimized-moe",
        "precision:mxfp8",
        "model:moe",
        "implementation:inference-optimized",
        config={
            "fp8": True,
            "fp8_recipe": "mxfp8",
            "hidden_size": 128,
            "num_moe_experts": 2,
            "moe_grouped_gemm": True,
            "use_moe_layer_spec": True,
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "moe", "mxfp8"),
        prerequisite="mxfp8",
        atol=0.0,
    ),
    # Four heads of width 64 exercise FlashInfer's supported fused-RoPE kernel.
    # Width 16 crosses half-head boundaries in its vectorized NeoX rotation.
    _pair_scenario(
        "flashinfer-fused-rope",
        "kernel:flashinfer-fused-rope",
        config={
            "hidden_size": 256,
            "position_embedding_type": "rope",
            "use_flashinfer_fused_rope": True,
        },
        signals=("fused-rope",),
        prerequisite="flashinfer",
        parity="reproducible",
    ),
    _pair_scenario(
        "flash-attention-3",
        "attention:flash-v3",
        config={"flash_attention_version": 3},
        signals=("flash-attention-version",),
        prerequisite="fa3",
    ),
    _pair_scenario(
        "flash-attention-4",
        "attention:flash-v4",
        config={"flash_attention_version": 4, "hidden_size": 256},
        signals=("flash-attention-version",),
        prerequisite="fa4",
    ),
    _pair_scenario(
        "swa-off-by-one-sink",
        "attention:swa-all-layers",
        "attention:off-by-one-sink",
        config={"window_size": (4, 0), "softmax_type": "off-by-one"},
        signals=("softmax-sink", "swa-all"),
    ),
    _pair_scenario(
        "alternating-swa-learnable-sink",
        "attention:swa-alternating",
        "attention:learnable-sink",
        config={"window_size": (4, 0), "window_attn_skip_freq": 2, "softmax_type": "learnable"},
        signals=("softmax-sink", "swa-alternating"),
    ),
)

_ASYNC_SUSPEND_RESUME_SCENARIOS = (
    _pair_scenario(
        "offload-suspend-resume",
        "kv:offload",
        config={"kv_cache_management_mode": "offload", "static_kv_memory_pointers": False},
        signals=("offload", "suspend-resume"),
    ),
    _pair_scenario(
        "recompute-suspend-resume",
        "kv:recompute",
        config={"kv_cache_management_mode": "recompute", "static_kv_memory_pointers": False},
        signals=("recompute", "suspend-resume"),
    ),
)


_CUDA_GRAPH_LIFECYCLE_SCENARIOS = (
    _pair_scenario(
        "persist-static-suspend-resume",
        "kv:persist",
        "kv:static-pointers",
        signals=("context-reset", "persist", "static-pointers", "suspend-resume"),
    ),
    _pair_scenario(
        "persist-static-weight-refit",
        "kv:persist",
        "kv:static-pointers",
        "lifecycle:weight-refit",
        config={
            "enable_prefix_caching": True,
            "materialize_only_last_token_logits": False,
            "return_log_probs": True,
        },
        sampling=({"return_log_probs": True},),
        signals=("logprobs", "persist", "static-pointers", "suspend-resume", "weight-refit"),
    ),
    _pair_scenario(
        "recompute-static-tms-suspend-resume",
        "kv:recompute",
        "memory:torch-memory-saver",
        "kv:static-pointers",
        config={"kv_cache_management_mode": "recompute", "static_kv_memory_pointers": True},
        signals=("recompute", "static-pointers", "suspend-resume", "tms"),
        prerequisite="torch-memory-saver",
    ),
)


_ASYNC_PARALLEL_SCENARIOS = (
    _pair_scenario(
        "tp2-pp2-sp-dp2",
        "topology:tp",
        "topology:pp",
        "topology:sp",
        "topology:dp",
        "seed:offset-by-dp-rank",
        config={
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "sequence_parallel": True,
        },
        sampling=({"temperature": 0.8, "top_k": 8},),
        signals=("dp-offset", "parallel", "sampled", "temperature-filter", "top-k-filter"),
        parity="reproducible",
    ),
    _pair_scenario(
        "moe-ep2-nccl",
        "model:moe",
        "topology:ep",
        "dispatcher:nccl",
        "interaction:moe-ep-ordering",
        config={
            "expert_model_parallel_size": 2,
            "use_moe_layer_spec": True,
            "inference_moe_token_dispatcher_type": "nccl",
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "moe", "nccl-dispatch", "parallel"),
    ),
    _pair_scenario(
        "moe-ep2-nvls",
        "model:moe",
        "topology:ep",
        "dispatcher:nvls",
        "interaction:nondecode-graph-ep",
        config={
            "expert_model_parallel_size": 2,
            "moe_grouped_gemm": True,
            "use_moe_layer_spec": True,
            "inference_moe_token_dispatcher_type": "nvls",
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "moe", "nvls-dispatch", "parallel"),
    ),
    _pair_scenario(
        "moe-ep2-etp2-local-allgather",
        "model:moe",
        "topology:ep",
        "topology:etp",
        "dispatcher:allgather",
        config={
            "expert_model_parallel_size": 2,
            "expert_tensor_parallel_size": 2,
            "num_moe_experts": 4,
            "use_moe_layer_spec": True,
        },
        signals=("etp", "moe", "parallel"),
    ),
    _pair_scenario(
        "optimized-tp2-sp",
        "interaction:optimized-tp-sp",
        "seed:shared-across-dp",
        config={
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "transformer_impl": "inference_optimized",
            "offset_sampling_seed_by_dp_rank": False,
        },
        sampling=({"temperature": 0.8, "top_k": 8},),
        signals=(
            "inference-optimized",
            "parallel",
            "sampled",
            "shared-seed",
            "temperature-filter",
            "top-k-filter",
        ),
        parity="reproducible",
    ),
)


_SCENARIOS_BY_NAME = {case.name: case for case in _ASYNC_PAIR_SCENARIOS + _ASYNC_PARALLEL_SCENARIOS}


def _variant(base_name, name, *pairs, config=None, signals=(), **changes):
    """Derive a new case from an upstream one, leaving the upstream definition unchanged."""
    base = _SCENARIOS_BY_NAME[base_name]
    return replace(
        base,
        name=name,
        pairs=base.pairs + pairs,
        config=base.config | (config or {}),
        signals=base.signals + signals,
        **changes,
    )


# The harness follows handed-off requests only in asynchronous runs, so this case is
# compared with and without graphs, both scheduled asynchronously.
_CUDA_GRAPH_HANDOFF_SCENARIOS = (
    _pair_scenario(
        "disagg-decode-handoff",
        "disaggregation:decode-handoff",
        "kv:synthetic-import",
        config={"enable_prefix_caching": True, "engine_cls": StateHandoffDynamicInferenceEngine},
        signals=("disagg-handoff",),
        request_profile="handoff",
    ),
)


# CUDA-graph comparisons of upstream cases with extra settings or checks.
_CUDA_GRAPH_VARIANT_SCENARIOS = (
    _variant(
        "graph-decode-block-exponential",
        "graph-decode-block-warmup-one",
        config={"cuda_graph_warmup_steps": 1, "flash_attention_version": 2},
        signals=("flash-attention-version", "graph-warmup", "last-logits"),
    ),
    _variant(
        "graph-mixed-layer-linear",
        "graph-mixed-layer-all-prefill",
        "graph:all-prefill",
        signals=("graph-all-prefill",),
    ),
    _variant(
        "sampling-temperature-top-k",
        "sampling-request-seeds",
        "sampling:request-seed",
        sampling=(
            {"temperature": 0.7, "top_k": 8, "seed": 17},
            {"temperature": 0.7, "top_k": 8, "seed": 29},
        ),
        signals=("request-seed",),
    ),
    _variant("flashinfer-fused-rope", "flashinfer-fused-rope-exact", parity="exact"),
    _variant(
        "fp8-transformer-engine",
        "fp8-transformer-engine-delayed",
        "precision:fp8-delayed",
        config={"fp8_recipe": "delayed"},
        signals=("fixed-history-logits",),
        atol=0.0,
        parity="exact",
    ),
)


_CUDA_GRAPH_DIFFERENTIAL_SCENARIOS = (
    tuple(
        scenario
        for scenario in _ASYNC_PAIR_SCENARIOS
        if scenario.name
        in {
            "prefix-lru-chunked",
            "prefix-ref-zero",
            "chunked-capacity",
            "graph-bounded-mixed-prefill",
            "sampling-top-p",
            "sampling-top-k-top-p",
            "raw-prompt-top-n-logprobs",
            "processed-skip-prompt-logprobs",
            "flashinfer-sampling",
            "length-zero-and-total",
            "string-prompt-bos",
            "termination-values",
            "stop-sequence-keep",
            "stop-sequence-strip",
            "mtp-depth-one",
            "mtp-repeated-all-accept",
            "mtp-graph-heterogeneous-logprobs",
            "vlm-precomputed-media",
            "hybrid-mamba",
            "hybrid-gdn",
            "hybrid-gdp",
            "transformer-engine",
            "inference-optimized",
            "flash-attention-3",
            "flash-attention-4",
            "swa-off-by-one-sink",
            "alternating-swa-learnable-sink",
        }
    )
    + _CUDA_GRAPH_VARIANT_SCENARIOS
    + _CUDA_GRAPH_HANDOFF_SCENARIOS
    + _CUDA_GRAPH_LIFECYCLE_SCENARIOS
)


# Main's plain EP2 NCCL case and the all-gather ETP case run eagerly here.
_ASYNC_PARALLEL_GRAPH_SCENARIOS = tuple(
    scenario
    for scenario in _ASYNC_PARALLEL_SCENARIOS
    if scenario.name not in {"tp2-pp2-sp-dp2", "moe-ep2-nccl", "moe-ep2-etp2-local-allgather"}
) + (
    _variant(
        "moe-ep2-nccl",
        "moe-ep2-nccl-routing-replay",
        config={"moe_grouped_gemm": True, "moe_enable_routing_replay": True},
        signals=("routing-replay",),
    ),
)

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
    elif scenario.prerequisite == "gdn":
        from megatron.core.ssm.gated_delta_net import HAVE_FLA

        if not HAVE_FLA:
            pytest.skip("GDN requires FLA")
    elif scenario.prerequisite == "gdp":
        from tests.unit_tests.inference.engines.ssm_test_helpers import (
            skip_if_sequence_packing_not_available,
        )

        skip_if_sequence_packing_not_available("gdp")
    elif scenario.prerequisite == "transformer-engine":
        if not is_te_min_version("2.2.0"):
            pytest.skip("Transformer Engine 2.2.0 is required")
    elif scenario.prerequisite == "fp8":
        available, reason = check_fp8_support()
        if not available:
            pytest.skip(reason)
    elif scenario.prerequisite == "torch-memory-saver":
        pytest.importorskip("torch_memory_saver")
    elif scenario.prerequisite == "mxfp8":
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("MXFP8 parameter storage requires Blackwell")
    elif scenario.prerequisite in ("fa3", "fa4"):
        from megatron.core.transformer import attention

        available = attention.HAVE_FA3 if scenario.prerequisite == "fa3" else attention.HAVE_FA4
        if not available:
            pytest.skip(f"{scenario.prerequisite.upper()} is not installed")


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
        request_kwargs = dict(
            request_id=request_id,
            prompt=raw_prompt,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            block_size_tokens=env.engine.context.block_size_tokens,
            enable_prefix_caching=env.engine.context.enable_prefix_caching,
        )
        if scenario.request_profile == "vlm":
            image_token_mask = torch.full_like(prompt_tokens, -1)
            image_token_mask[1:3] = torch.arange(2, device="cuda")
            hidden_size = env.engine.controller.inference_wrapped_model.model.config.hidden_size
            image_embeddings = (
                torch.arange(2 * hidden_size, dtype=torch.float32, device="cuda")
                .reshape(2, 1, hidden_size)
                .add_(request_id)
                .mul_(1.0e-2)
                .to(torch.bfloat16)
            )
            request = DynamicVLMInferenceRequest(
                **request_kwargs,
                num_img_embeddings_per_tile=2,
                imgs=torch.empty(0, device="cuda"),
                num_tiles=torch.ones(1, dtype=torch.int32, device="cuda"),
                decoder_seq_length=0,
                image_embeddings=image_embeddings,
                image_token_mask=image_token_mask,
                media_tokens_preexpanded=True,
            )
        else:
            request = DynamicInferenceRequest(**request_kwargs)
        requests.append(request)
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


def _instrument_attention_runtime(module, runtime):
    """Observe the window and sink branches taken by dynamic attention."""
    original_offset = module._get_inference_softmax_offset

    def traced_offset():
        offset = original_offset()
        runtime["softmax-offset-calls"] += 1
        runtime["softmax-offset-produced"] += int(offset is not None)
        return offset

    module._get_inference_softmax_offset = traced_offset

    original_flash = module.flash_decode_and_prefill

    def traced_flash(*args, **kwargs):
        if is_layer_window_attention(
            module.config.window_size, module.config.window_attn_skip_freq, module.layer_number
        ):
            runtime["swa-kernel-calls"] += 1
        else:
            runtime["full-attention-kernel-calls"] += 1
        return original_flash(*args, **kwargs)

    module.flash_decode_and_prefill = traced_flash

    for method_name in (
        "_apply_sink_softmax_correction_varlen",
        "_apply_sink_softmax_correction_bshd",
    ):
        original_correction = getattr(module, method_name)

        def traced_correction(*args, _original=original_correction, **kwargs):
            runtime["sink-correction-calls"] += 1
            return _original(*args, **kwargs)

        setattr(module, method_name, traced_correction)


def _instrument_parallel_runtime(module, runtime):
    """Observe executed tensor partitions and their inference collectives."""
    class_name = type(module).__name__
    weight = getattr(module, "weight", None)
    tp_group = getattr(module, "tp_group", None)
    tp_size = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1

    if tp_size > 1 and weight is not None and "ColumnParallelLinear" in class_name:
        output_size = getattr(module, "output_size", getattr(module, "out_features", None))
        if output_size is not None and weight.shape[0] * tp_size == output_size:
            runtime["tp-column-partitions-installed"] += 1
    if tp_size > 1 and weight is not None and "RowParallelLinear" in class_name:
        input_size = getattr(module, "input_size", getattr(module, "in_features", None))
        if input_size is not None and weight.shape[1] * tp_size == input_size:
            runtime["tp-row-partitions-installed"] += 1

    if class_name == "ColumnParallelLinear" and tp_size > 1 and module.sequence_parallel:
        from megatron.core.tensor_parallel import layers as tp_layers

        original_forward = module.forward

        def traced_forward(*args, **kwargs):
            original_collective = tp_layers.gather_from_sequence_parallel_region

            def traced_collective(input_, *collective_args, **collective_kwargs):
                output = original_collective(input_, *collective_args, **collective_kwargs)
                runtime["tp-collective:gather_from_sequence_parallel_region"] += 1
                runtime["tp-sp-gather-dimensions"] += int(
                    output.shape[0] == input_.shape[0] * tp_size
                )
                return output

            tp_layers.gather_from_sequence_parallel_region = traced_collective
            try:
                return original_forward(*args, **kwargs)
            finally:
                tp_layers.gather_from_sequence_parallel_region = original_collective

        module.forward = traced_forward

    if class_name == "RowParallelLinear" and tp_size > 1:
        from megatron.core.tensor_parallel import layers as tp_layers

        original_forward = module.forward

        def traced_forward(*args, **kwargs):
            collective_name = (
                "reduce_scatter_to_sequence_parallel_region"
                if module.sequence_parallel
                else "reduce_from_tensor_model_parallel_region"
            )
            original_collective = getattr(tp_layers, collective_name)

            def traced_collective(input_, *collective_args, **collective_kwargs):
                output = original_collective(input_, *collective_args, **collective_kwargs)
                runtime[f"tp-collective:{collective_name}"] += 1
                if module.sequence_parallel:
                    runtime["tp-sp-reduce-scatter-dimensions"] += int(
                        output.shape[0] * tp_size == input_.shape[0]
                    )
                return output

            setattr(tp_layers, collective_name, traced_collective)
            try:
                return original_forward(*args, **kwargs)
            finally:
                setattr(tp_layers, collective_name, original_collective)

        module.forward = traced_forward

    if (
        class_name in {"InferenceColumnParallelLinear", "InferenceLayerNormColumnParallelLinear"}
        and tp_size > 1
    ):
        original_all_gather = module._all_gather

        def traced_all_gather(*args, **kwargs):
            input_ = args[0]
            output = original_all_gather(*args, **kwargs)
            runtime["tp-collective:optimized-all-gather"] += 1
            runtime["tp-sp-gather-dimensions"] += int(output.shape[0] == input_.shape[0] * tp_size)
            return output

        module._all_gather = traced_all_gather

    if class_name == "InferenceRowParallelLinear" and tp_size > 1:
        original_reduce_scatter = module._matmul_reduce_scatter

        def traced_reduce_scatter(*args, **kwargs):
            input_ = args[0]
            output = original_reduce_scatter(*args, **kwargs)
            runtime["tp-collective:optimized-reduce-scatter"] += 1
            runtime["tp-sp-reduce-scatter-dimensions"] += int(
                output.shape[0] * tp_size == input_.shape[0]
            )
            return output

        module._matmul_reduce_scatter = traced_reduce_scatter


def _instrument_nccl_dispatch_runtime(module, runtime):
    """Observe inference MoE dispatch, combine, and their ordering."""
    dispatcher = getattr(module, "_inference_token_dispatcher", None)
    if dispatcher is None or type(dispatcher).__name__ != "NCCLAllGatherDispatcher":
        return
    runtime["nccl-dispatchers-installed"] += 1
    original_dispatch = dispatcher.token_dispatch
    original_combine = dispatcher.token_combine

    def traced_dispatch(*args, **kwargs):
        result = original_dispatch(*args, **kwargs)
        runtime["nccl-token-dispatches"] += 1
        runtime["nccl-dispatch-inflight"] += 1
        return result

    def traced_combine(*args, **kwargs):
        runtime["nccl-combine-before-dispatch"] += int(runtime["nccl-dispatch-inflight"] <= 0)
        result = original_combine(*args, **kwargs)
        runtime["nccl-token-combines"] += 1
        runtime["nccl-dispatch-inflight"] -= 1
        return result

    dispatcher.token_dispatch = traced_dispatch
    dispatcher.token_combine = traced_combine


def _instrument_cuda_graph_replays(model, context, runtime):
    for manager in (module for module in model.modules() if isinstance(module, CudaGraphManager)):
        owner = getattr(getattr(manager, "func", None), "__name__", "forward")
        for runner in manager.cudagraph_runners:
            if getattr(runner, "_pairwise_runtime_trace", None) == id(runtime):
                continue
            original_replay = runner.replay_graph_capture
            base_module = runner.base_module
            replayed_modules = tuple(
                base_module.modules() if hasattr(base_module, "modules") else (base_module,)
            )
            replayed_module_names = tuple(
                sorted({type(module).__name__ for module in replayed_modules})
            )
            fp8_runtime_enabled = bool(runner.fp8_enabled)

            def traced_replay(
                is_first_microbatch,
                args,
                kwargs,
                _original=original_replay,
                _runner=runner,
                _module_names=replayed_module_names,
                _fp8_runtime_enabled=fp8_runtime_enabled,
                _owner=owner,
            ):
                if runtime["cuda-graph-capture-active"]:
                    return _original(is_first_microbatch, args, kwargs)
                real = context.batch_dimensions
                padded = context.padded_batch_dimensions
                dimension = (padded.token_count, padded.prefill_req_count, padded.decode_req_count)
                step_key = ("cuda-graph-replay-step", context.step_count, *dimension)
                runtime["cuda-graph-runner-replays"] += 1
                runtime["cuda-graph-fp8-replays"] += int(_fp8_runtime_enabled)
                runtime["cuda-graph-real-token-scalar-matches"] += int(
                    int(context.gpu_view.real_token_count.item()) == real.token_count
                )
                if padded.token_count > real.token_count:
                    padding = context.gpu_view.token_to_input_ids[
                        real.token_count : padded.token_count
                    ]
                    runtime["cuda-graph-padded-replays"] += 1
                    runtime["cuda-graph-zero-input-padding"] += int(
                        not torch.count_nonzero(padding).item()
                    )
                for module_name in _module_names:
                    runtime[("cuda-graph-replayed-module", module_name)] += 1
                runtime[("cuda-graph-replay-dimension", *dimension)] += 1
                if runtime[step_key] == 0:
                    active_slice = slice(context.paused_request_count, context.total_request_count)
                    runtime[
                        (
                            "cuda-graph-live-input",
                            dimension,
                            (real.token_count, real.prefill_req_count, real.decode_req_count),
                            tuple(context.request_ids[active_slice].tolist()),
                            tuple(
                                context.token_to_input_ids[: context.active_token_count].tolist()
                            ),
                        )
                    ] += 1
                runtime[step_key] += 1
                runtime[("cuda-graph-replay-owner", _owner, context.step_count)] += 1
                outputs = _original(is_first_microbatch, args, kwargs)
                for output in _runner.get_tensors(outputs):
                    runtime[("cuda-graph-output-pointer", output.data_ptr())] += 1
                return outputs

            runner.replay_graph_capture = traced_replay
            runner._pairwise_runtime_trace = id(runtime)
            runtime[("cuda-graph-warmup-steps", runner.num_warmup_steps)] += 1


def _instrument_scenario_runtime(env, scenario, runtime, teacher_tokens=None):
    """Record production calls and state transitions used as pairwise evidence."""
    controller = env.engine.controller
    context = env.engine.context
    _instrument_async_phases(controller, runtime)
    model = controller.inference_wrapped_model.model
    _instrument_cuda_graph_replays(model, context, runtime)
    if "cuda-graph" in scenario.signals:
        original_create_cuda_graphs = env.engine.create_cuda_graphs

        def traced_create_cuda_graphs(*args, **kwargs):
            runtime["cuda-graph-create-calls"] += 1
            runtime["cuda-graph-capture-active"] += 1
            try:
                result = original_create_cuda_graphs(*args, **kwargs)
            finally:
                runtime["cuda-graph-capture-active"] -= 1
            _instrument_cuda_graph_replays(model, context, runtime)
            return result

        env.engine.create_cuda_graphs = traced_create_cuda_graphs

    if "vlm" in scenario.signals:
        wrapper = controller.inference_wrapped_model
        original_vlm_forward = wrapper._forward_dynamic

        def traced_vlm_forward(inference_input):
            mask = inference_input.get("image_token_mask")
            embeddings = inference_input.get("image_embeddings")
            if mask is not None and embeddings is not None and bool((mask >= 0).any()):
                runtime["vlm-image-forwards"] += 1
                runtime["vlm-image-token-count"] += int((mask >= 0).sum().item())
                runtime[("vlm-image-step", context.step_count)] += 1
            return original_vlm_forward(inference_input)

        wrapper._forward_dynamic = traced_vlm_forward

    if "disagg-handoff" in scenario.signals:
        original_finalize_handoff = env.engine._finalize_kv_handoff_import

        def traced_finalize_handoff(pending):
            result = original_finalize_handoff(pending)
            request = env.engine.get_request(pending.request_id)
            env.requests[pending.request_id] = request
            runtime[("disagg-import", pending.request_id, pending.resume_tokens[0])] += 1
            runtime["disagg-prefill-tokens-skipped"] += request.num_cached_tokens
            return result

        env.engine._finalize_kv_handoff_import = traced_finalize_handoff

    if "routing-replay" in scenario.signals:
        original_capture_routing = controller._capture_async_sched_routing

        def traced_capture_routing():
            result = original_capture_routing()
            if result is not None:
                runtime[("routing-record-step", context.step_count)] += 1
            return result

        controller._capture_async_sched_routing = traced_capture_routing

    original_forward = controller._dynamic_step_forward_logits

    def traced_forward(*args, **kwargs):
        replays_before = runtime["cuda-graph-runner-replays"]
        real_dimensions = context.batch_dimensions
        if context.prefix_cache_hits > 0:
            runtime[("prefix-hit-forward-step", context.step_count)] += 1
            runtime[
                "prefix-hit-forward-skipped-tokens"
            ] += context.prefix_cache_prefill_skipped_tokens
        if context.chunked_prefill_request_id >= 0:
            runtime[("chunked-forward-step", context.step_count)] += 1
        runtime[
            (
                "model-step-dimension",
                context.step_count,
                real_dimensions.token_count,
                real_dimensions.prefill_req_count,
                real_dimensions.decode_req_count,
            )
        ] += 1
        if context.using_cuda_graph_this_step() and runtime["cuda-graph-capture-active"]:
            runtime["cuda-graph-capture-forwards"] += 1
        elif context.using_cuda_graph_this_step():
            dimensions = context.padded_batch_dimensions
            runtime[
                (
                    "cuda-graph-dimension",
                    dimensions.token_count,
                    dimensions.prefill_req_count,
                    dimensions.decode_req_count,
                )
            ] += 1
            runtime[f"cuda-graph-scope:{context.inference_cuda_graph_scope.name}"] += 1
            runtime["cuda-graph-prefill-forwards"] += int(dimensions.prefill_req_count > 0)
            runtime["cuda-graph-decode-forwards"] += int(dimensions.decode_req_count > 0)
            runtime["cuda-graph-all-prefill-forwards"] += int(
                dimensions.prefill_req_count > 0 and dimensions.decode_req_count == 0
            )
            runtime["cuda-graph-mixed-forwards"] += int(
                dimensions.prefill_req_count > 0 and dimensions.decode_req_count > 0
            )
        if controller.model_is_pipeline_parallel:
            from megatron.core.inference.text_generation_controllers import (
                text_generation_controller as controller_module,
            )

            original_broadcast = controller_module.broadcast_from_last_pipeline_stage

            def traced_broadcast(*broadcast_args, **broadcast_kwargs):
                result = original_broadcast(*broadcast_args, **broadcast_kwargs)
                runtime["pipeline-logits-broadcasts"] += 1
                return result

            controller_module.broadcast_from_last_pipeline_stage = traced_broadcast
            try:
                result = original_forward(*args, **kwargs)
            finally:
                controller_module.broadcast_from_last_pipeline_stage = original_broadcast
        else:
            result = original_forward(*args, **kwargs)
        runtime["model-forward"] += 1
        runtime[f"logits-dtype:{controller._all_logits_cuda.dtype}"] += 1
        if "fixed-history-logits" in scenario.signals and not runtime["cuda-graph-capture-active"]:
            active_count = context.total_request_count - context.paused_request_count
            active_slice = slice(context.paused_request_count, context.total_request_count)
            request_ids = context.request_ids[active_slice].tolist()
            last_idxs = context.gpu_view.active_request_last_token_idxs[:active_count].long()
            positions = args[1][0, last_idxs].tolist()
            input_tokens = args[0][0, last_idxs].tolist()
            logits = controller._all_logits_cuda[0, :active_count].detach().float().cpu()
            blocks = context.gpu_view.token_to_block_idx[last_idxs].long()
            offsets = context.gpu_view.token_to_local_position_within_kv_block[last_idxs].long()
            last_kv = context.memory_buffer[:, :, blocks, offsets].detach().float().cpu()
            replayed = runtime["cuda-graph-runner-replays"] > replays_before
            snapshots = runtime.setdefault("fixed-history-logit-snapshots", {})
            for i, (request_id, position, input_token, row) in enumerate(
                zip(request_ids, positions, input_tokens, logits)
            ):
                key = (request_id, position)
                assert key not in snapshots, f"duplicate fixed-history logit row {key}"
                snapshots[key] = (input_token, replayed, row.clone(), last_kv[:, :, i])
        if context.config.materialize_only_last_token_logits:
            runtime["last-logits-forward"] += 1
        else:
            runtime["full-logits-forward"] += 1
        return result

    controller._dynamic_step_forward_logits = traced_forward

    original_sample_kernel = controller._sampling.sample_kernel

    def traced_sample_kernel(*args, **kwargs):
        active_count = context.total_request_count - context.paused_request_count
        metadata = context.active_request_metadata
        runtime["sampling-kernel"] += 1
        runtime[f"sampling-backend:{controller._sampling_backend}"] += 1
        runtime["metadata-consumed"] += int(active_count > 0)
        sampling_class = type(controller._sampling).__name__

        if sampling_class == "TorchSampling":
            from megatron.core.inference.sampling import torch_sampling
            from megatron.core.inference.sampling.torch_sampling import TorchSampling

            original_filter_logits = TorchSampling.filter_logits

            def traced_filter_logits(logits, temperature, top_k, top_p, **filter_kwargs):
                runtime["temperature-filter"] += int(temperature != 1.0)
                runtime["top-k-filter"] += int(top_k >= 1)
                runtime["top-p-filter"] += int(top_p > 0.0)
                filtered = original_filter_logits(
                    logits, temperature, top_k, top_p, **filter_kwargs
                )
                finite = torch.isfinite(filtered)
                scaled = logits.clone()
                if temperature != 1.0:
                    scaled.div_(max(temperature, MIN_SAMPLING_TEMPERATURE))
                runtime["sampling-filter-rows"] += finite.shape[0]
                runtime["sampling-filter-nonempty-rows"] += int(finite.any(dim=-1).sum().item())
                runtime["temperature-scale-exact-rows"] += (
                    int(torch.all(filtered[finite] == scaled[finite]).item()) * finite.shape[0]
                )
                if top_k >= 1:
                    threshold = torch.topk(scaled, top_k)[0][..., -1, None]
                    runtime["top-k-support-exact-rows"] += (
                        int(torch.equal(finite, scaled >= threshold)) * finite.shape[0]
                    )
                elif 0.0 < top_p < 1.0:
                    sorted_probabilities = torch.sort(scaled, descending=True)[0].softmax(
                        dim=-1, dtype=torch.float32
                    )
                    expected_counts = (
                        (sorted_probabilities.cumsum(dim=-1) <= top_p).sum(dim=-1) + 1
                    ).clamp(max=scaled.shape[-1])
                    runtime["top-p-support-exact-rows"] += (
                        int(torch.equal(finite.sum(dim=-1), expected_counts)) * finite.shape[0]
                    )
                return filtered

            original_seed_noise = torch_sampling.fill_request_seed_noise

            def traced_seed_noise(*seed_args, **seed_kwargs):
                runtime["request-seed-kernel"] += 1
                return original_seed_noise(*seed_args, **seed_kwargs)

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        TorchSampling, "filter_logits", new=staticmethod(traced_filter_logits)
                    )
                )
                if original_seed_noise is not None:
                    stack.enter_context(
                        mock.patch.object(
                            torch_sampling, "fill_request_seed_noise", new=traced_seed_noise
                        )
                    )
                sampled = original_sample_kernel(*args, **kwargs)
                if "fixed-history-logits" in scenario.signals:
                    active_slice = slice(context.paused_request_count, context.total_request_count)
                    request_ids = context.request_ids[active_slice].tolist()
                    if teacher_tokens is not None:
                        forced = [
                            teacher_tokens[request_id][runtime[("teacher-token-index", request_id)]]
                            for request_id in request_ids
                        ]
                        sampled[:active_count].copy_(sampled.new_tensor(forced))
                    for request_id in request_ids:
                        runtime[("teacher-token-index", request_id)] += 1
                return sampled

        if sampling_class == "FlashInferSampling":
            from megatron.core.inference.sampling import flashinfer_sampling

            flashinfer_kernels = {
                "sampling_from_logits": (),
                "top_k_sampling_from_probs": ("top-k-filter",),
                "top_p_sampling_from_probs": ("top-p-filter",),
                "top_k_top_p_sampling_from_logits": ("top-k-filter", "top-p-filter"),
            }

            def traced_flashinfer_kernel(name, original):
                def traced(*kernel_args, **kernel_kwargs):
                    runtime[f"flashinfer-kernel:{name}"] += 1
                    for signal in flashinfer_kernels[name]:
                        runtime[signal] += 1
                    runtime["temperature-filter"] += int(
                        bool((metadata["temperature"][:active_count] != 1.0).any())
                    )
                    sampled = original(*kernel_args, **kernel_kwargs)
                    if name == "sampling_from_logits":
                        valid = (sampled >= 0) & (sampled < kernel_args[0].shape[-1])
                    else:
                        values = kernel_args[0]
                        probabilities = (
                            torch.softmax(values, dim=-1)
                            if name == "top_k_top_p_sampling_from_logits"
                            else values
                        )
                        sorted_probabilities, sorted_ids = probabilities.sort(
                            dim=-1, descending=True
                        )
                        keep = torch.ones_like(sorted_probabilities, dtype=torch.bool)
                        if name in {
                            "top_k_sampling_from_probs",
                            "top_k_top_p_sampling_from_logits",
                        }:
                            top_k = kernel_args[1].clamp(min=1, max=probabilities.shape[-1])
                            ranks = torch.arange(
                                probabilities.shape[-1], device=probabilities.device
                            )
                            keep &= ranks.unsqueeze(0) < top_k.unsqueeze(1)
                        if name in {
                            "top_p_sampling_from_probs",
                            "top_k_top_p_sampling_from_logits",
                        }:
                            top_p = kernel_args[
                                2 if name == "top_k_top_p_sampling_from_logits" else 1
                            ]
                            nucleus_probabilities = sorted_probabilities * keep
                            nucleus_probabilities /= nucleus_probabilities.sum(dim=-1, keepdim=True)
                            keep &= (
                                nucleus_probabilities.cumsum(dim=-1) - nucleus_probabilities
                            ) < top_p.unsqueeze(1)
                        valid = ((sorted_ids == sampled.unsqueeze(1)) & keep).any(dim=-1)
                    runtime["flashinfer-support-rows"] += sampled.numel()
                    runtime["flashinfer-valid-support-rows"] += int(valid.sum().item())
                    return sampled

                return traced

            with ExitStack() as stack:
                for name in flashinfer_kernels:
                    original = getattr(flashinfer_sampling.flashinfer.sampling, name)
                    stack.enter_context(
                        mock.patch.object(
                            flashinfer_sampling.flashinfer.sampling,
                            name,
                            new=traced_flashinfer_kernel(name, original),
                        )
                    )
                return original_sample_kernel(*args, **kwargs)

        raise AssertionError(f"Unexpected sampling backend class: {sampling_class}")

    controller._sampling.sample_kernel = traced_sample_kernel

    original_calculate_log_probs_tensors = context.calculate_log_probs_tensors

    def traced_calculate_log_probs_tensors(*args, **kwargs):
        runtime["log-probs-calculations"] += 1
        runtime[f"log-probs-mode:{context.config.logprobs_mode}"] += 1
        return original_calculate_log_probs_tensors(*args, **kwargs)

    context.calculate_log_probs_tensors = traced_calculate_log_probs_tensors

    original_log_probs_kernel = controller._sampling.log_probs_kernel

    def traced_log_probs_kernel(*args, **kwargs):
        runtime["log-probs-kernel"] += 1
        return original_log_probs_kernel(*args, **kwargs)

    controller._sampling.log_probs_kernel = traced_log_probs_kernel

    original_compact = controller._compact_async_sched_forward

    def traced_compact(survivor_idxs):
        if survivor_idxs.numel() > 0:
            runtime["metadata-compactions"] += 1
            runtime["non-prefix-survivor-compactions"] += int(
                not torch.equal(
                    survivor_idxs, torch.arange(survivor_idxs.numel(), device=survivor_idxs.device)
                )
            )
        return original_compact(survivor_idxs)

    controller._compact_async_sched_forward = traced_compact

    allocator = context.kv_block_allocator
    original_deregister = allocator._deregister_blocks

    def traced_deregister(block_ids):
        runtime["prefix-blocks-deregistered"] += block_ids.numel()
        return original_deregister(block_ids)

    allocator._deregister_blocks = traced_deregister

    if "fused-rope" in scenario.signals:
        original_fused_rope = context.apply_fused_qk_rotary_emb

        def traced_fused_rope(*args, **kwargs):
            runtime["fused-rope-kernel"] += 1
            return original_fused_rope(*args, **kwargs)

        context.apply_fused_qk_rotary_emb = traced_fused_rope

    if "flash-attention-version" in scenario.signals:
        for module in model.modules():
            if type(module).__name__ != "SelfAttention":
                continue
            original_resolve_flash_version = module._resolve_flash_version

            def traced_resolve_flash_version(_original=original_resolve_flash_version):
                use_fa4, use_fa3 = _original()
                version = 4 if use_fa4 else (3 if use_fa3 else 2)
                runtime[("flash-attention-version", version)] += 1
                return use_fa4, use_fa3

            module._resolve_flash_version = traced_resolve_flash_version

    if "hybrid" in scenario.signals:
        original_commit_mamba = controller._commit_mamba_intermediate_states
        original_resolve_requests = context.resolve_requests

        def traced_commit_mamba():
            runtime["mamba-state-commits"] += 1
            return original_commit_mamba()

        def traced_resolve_requests(active_requests_mask):
            before = context.mamba_metadata.request_to_mamba_state_idx[
                : context.total_request_count
            ].clone()
            result = original_resolve_requests(active_requests_mask)
            survivor_idxs = result[1]
            identity = torch.arange(survivor_idxs.numel(), device=survivor_idxs.device)
            if survivor_idxs.numel() > 0 and not torch.equal(survivor_idxs, identity):
                after = context.mamba_metadata.request_to_mamba_state_idx[: survivor_idxs.numel()]
                runtime["mamba-state-compactions"] += int(torch.equal(after, before[survivor_idxs]))
            return result

        controller._commit_mamba_intermediate_states = traced_commit_mamba
        context.resolve_requests = traced_resolve_requests

    watched_modules = {
        "gpt": lambda name, module: name == "GPTModel",
        "transformer-engine": (
            lambda name, module: name.startswith("TE") or "transformer_engine" in module
        ),
        "inference-optimized": (
            lambda name, module: name.startswith("Inference") and "inference_layers" in module
        ),
        "fp8": lambda name, module: name.startswith("TE") or "transformer_engine" in module,
        "hybrid": (
            lambda name, module: name in {"MambaMixer", "GatedDeltaNet", "GatedDeltaProductMixer"}
        ),
        "moe": lambda name, module: name in {"MoELayer", "InferenceTopKRouter"},
        "parallel": (
            lambda name, module: "ParallelLinear" in name
            or name in {"ColumnParallelLinear", "RowParallelLinear"}
        ),
    }
    active_tags = set(scenario.signals) & watched_modules.keys()
    for module in model.modules():
        class_name = type(module).__name__
        module_name = type(module).__module__
        if "softmax-sink" in scenario.signals and class_name == "SelfAttention":
            _instrument_attention_runtime(module, runtime)
        if "parallel" in scenario.signals and (
            "ColumnParallelLinear" in class_name or "RowParallelLinear" in class_name
        ):
            _instrument_parallel_runtime(module, runtime)
        if "nccl-dispatch" in scenario.signals and class_name == "MoELayer":
            _instrument_nccl_dispatch_runtime(module, runtime)
        for tag in active_tags:
            if watched_modules[tag](class_name, module_name):

                def record_module(_module, _inputs, _tag=tag):
                    runtime[f"module-forward:{_tag}"] += 1
                    if _tag == "fp8":
                        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
                        runtime["fp8-context-forwards"] += int(fp8_enabled)
                        runtime["fp8-recipe-forwards"] += int(
                            fp8_enabled and FP8GlobalStateManager.get_fp8_recipe() is not None
                        )
                        if hasattr(_module, "will_execute_quantized"):
                            runtime["fp8-quantized-forwards"] += int(
                                _module.will_execute_quantized(fp8_enabled)
                            )
                    elif _tag == "parallel":
                        name = type(_module).__name__
                        weight = getattr(_module, "weight", None)
                        group = getattr(_module, "tp_group", None)
                        size = torch.distributed.get_world_size(group) if group is not None else 1
                        if weight is not None and "ColumnParallelLinear" in name:
                            output_size = getattr(
                                _module, "output_size", getattr(_module, "out_features", None)
                            )
                            runtime["tp-column-partition-forwards"] += int(
                                size > 1
                                and output_size is not None
                                and weight.shape[0] * size == output_size
                            )
                        elif weight is not None and "RowParallelLinear" in name:
                            input_size = getattr(
                                _module, "input_size", getattr(_module, "in_features", None)
                            )
                            runtime["tp-row-partition-forwards"] += int(
                                size > 1
                                and input_size is not None
                                and weight.shape[1] * size == input_size
                            )

                module.register_forward_pre_hook(record_module)


def _as_float_list(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values.tolist()
    return list(values)


def _recorded(runtime, name, index=slice(1, None)):
    """Return part of every positive tuple counter recorded under `name`."""
    return {
        key[index]
        for key, count in runtime.items()
        if isinstance(key, tuple) and key[0] == name and count > 0
    }


def _snapshot_requests(requests):
    snapshots = []
    for request in requests:
        snapshots.append(
            {
                "request_id": request.request_id,
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
        assert request.request_id == reference["request_id"]
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


def _assert_fixed_history_logit_parity(actual, expected, atol):
    assert actual.keys() == expected.keys()
    failures = []
    for key in sorted(expected):
        input_token, replayed, logits, last_kv = actual[key]
        expected_input, _, expected_logits, expected_kv = expected[key]
        assert input_token == expected_input
        assert replayed, f"fixed-history comparison at {key} did not replay a CUDA graph"
        assert torch.isfinite(logits).all() and torch.isfinite(expected_logits).all()
        assert torch.isfinite(last_kv).all() and torch.isfinite(expected_kv).all()
        top2 = expected_logits.topk(2)
        actual_top2 = logits.topk(2)
        margin = float((top2.values[0] - top2.values[1]).item())
        max_diff = float((logits - expected_logits).abs().max().item())
        kv_diff = float((last_kv - expected_kv).abs().max().item())
        print(
            f"request/position {key}: max_abs={max_diff:.8g}, eager_margin={margin:.8g}, "
            f"replay_margin={float((actual_top2.values[0] - actual_top2.values[1]).item()):.8g}, "
            f"last_kv_max_abs={kv_diff:.8g}"
        )
        actual_top1, expected_top1 = logits.argmax().item(), expected_logits.argmax().item()
        if actual_top1 != expected_top1 or max_diff > atol or kv_diff > atol:
            failures.append((key, max_diff, kv_diff, margin, actual_top1, expected_top1))
    assert not failures, failures


class _AsyncPairwiseHarness(_DynamicInferenceEngineTestBase):
    """Shared real-engine runner for the async pair owners."""

    @pytest.fixture(autouse=True)
    def _reset_token_rounder(self):
        """FP8 runs raise TOKEN_ROUNDER to the recipe's alignment; restore it after each test."""
        yield
        _reset_rounder()

    @classmethod
    def _run_scenario_mode(
        cls,
        scenario,
        mode,
        stop_tokens=None,
        termination_token=None,
        teacher_tokens=None,
        force_graph_eager=False,
    ):
        RouterReplay.clear_global_router_replay_instances()
        config = dict(_BASE_PAIR_CONFIG)
        config.update(scenario.config)
        config["async_sched_mode"] = mode
        env = cls._build_test_env(_DynamicEngineTestConfig(**config))
        # New FP8 cases round eager steps to the recipe's alignment; main's own FP8 case
        # keeps the upstream test rounder.
        model_config = env.engine.controller.inference_wrapped_model.model.config
        if model_config.fp8 and scenario.name != "fp8-transformer-engine":
            from megatron.core.fp8_utils import get_fp8_align_size

            DynamicInferenceContext.TOKEN_ROUNDER = get_fp8_align_size(model_config.fp8_recipe)
        env.requests = _make_scenario_requests(
            env, scenario, stop_tokens=stop_tokens, termination_token=termination_token
        )
        model = env.engine.controller.inference_wrapped_model.model
        runtime = Counter()
        if mode == AsyncScheduleMode.ASYNC:
            _instrument_scenario_runtime(env, scenario, runtime, teacher_tokens)
        if env.config.defer_cuda_graph_creation:
            env.engine.create_cuda_graphs()
            capture_evidence = {
                key: count
                for key, count in runtime.items()
                if key
                in {
                    "full-attention-kernel-calls",
                    "fused-rope-kernel",
                    "sink-correction-calls",
                    "softmax-offset-calls",
                    "softmax-offset-produced",
                    "swa-kernel-calls",
                }
                or (
                    isinstance(key, tuple)
                    and key[0] in ("cuda-graph-warmup-steps", "flash-attention-version")
                )
                or (isinstance(key, str) and key.startswith(("nccl-", "fp8-")))
            }
            runtime.clear()
            runtime.update(capture_evidence)
        if force_graph_eager:
            for module in model.modules():
                module._should_call_local_cudagraph = lambda *_args, **_kwargs: False
            if hasattr(model, "_mtp_cudagraph_manager"):
                mtp_forward = model.compute_mtp_single_step
                model.compute_mtp_single_step = lambda *args, **kwargs: mtp_forward(
                    *args, **(kwargs | {"eager": True})
                )
        if "disagg-handoff" in scenario.signals:
            env.engine._kv_transfer_role = "decode"
            env.engine.context.memory_buffer.zero_()

        def step():
            context = env.engine.context
            runtime["max_active"] = max(
                runtime["max_active"], context.total_request_count - context.paused_request_count
            )
            runtime["chunked-prefill-steps"] += int(context.chunked_prefill_request_id >= 0)
            result = env.engine.step_modern()
            runtime["steps"] += 1
            runtime["cuda_graph_steps"] += int(context.using_cuda_graph_this_step())
            runtime["max_paused"] = max(runtime["max_paused"], context.paused_request_count)
            for request in result["finished_requests"]:
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
                if "disagg-handoff" in scenario.signals:
                    from megatron.core.inference.disaggregation.decode_admission import (
                        can_admit_prefilled_decode,
                    )

                    local_blocks = env.engine.context.kv_block_allocator.allocate_memory_blocks(1)
                    assert local_blocks is not None
                    resume_token = 50 + request_id
                    pending = PendingKvImport(
                        request_id=request_id,
                        prompt=request.prompt_tokens.tolist(),
                        sampling_params=request.sampling_params,
                        local_blocks=[int(local_blocks[0])],
                        hashes=[],
                        cached_prefix_block_count=0,
                        handle=None,
                        future=env.engine._loop.create_future(),
                        resume_tokens=[resume_token],
                    )
                    env.engine._pending_kv_imports.append(pending)
                    runtime["max_pending_kv_imports"] = max(
                        runtime["max_pending_kv_imports"], env.engine.pending_kv_import_count
                    )
                    pending.terminal_state_reported = True
                    env.engine._record_handoff_completion_notification(
                        request_id, failed=False, source_safe=True
                    )
                    runtime["disagg-admission-deferrals"] += int(
                        not can_admit_prefilled_decode(env.engine.context, 1)
                    )
                elif isinstance(request, DynamicVLMInferenceRequest):
                    env.engine.context.add_vlm_request_data(
                        request_id,
                        image_embeddings=request.image_embeddings,
                        image_token_mask=request.image_token_mask,
                    )
                    env.engine._add_request(request)
                else:
                    prompt = request.prompt if request.prompt is not None else request.prompt_tokens
                    env.engine.add_request(request_id, prompt, request.sampling_params)
                if "disagg-handoff" not in scenario.signals:
                    env.requests[request_id] = env.engine.get_request(request_id)
                    env.requests[request_id].state = "pending"
            runtime["waves"] += 1
            runtime["max_waiting"] = max(
                runtime["max_waiting"], len(env.engine.waiting_request_ids)
            )
            step()
            if "suspend-resume" in scenario.signals and (
                wave_idx == 0 or "repeat-suspend-resume" in scenario.signals
            ):
                memory_ptr = env.engine.context.memory_buffer.data_ptr()
                weight_epoch = env.engine._weight_epoch
                pending_before_suspend = env.engine.controller._async_sched_forward.is_valid
                runners_before = [
                    runner
                    for manager in model.modules()
                    if isinstance(manager, CudaGraphManager)
                    for runner in manager.cudagraph_runners
                    if runner.fwd_graph_recorded
                ]
                graph_ids_before = [id(runner.fwd_graph) for runner in runners_before]
                runtime["pending-before-suspend"] += int(pending_before_suspend)
                from megatron.core.inference.engines import dynamic_engine as dynamic_engine_module

                original_delete_cuda_graphs = dynamic_engine_module.delete_cuda_graphs

                def traced_delete_cuda_graphs():
                    result = original_delete_cuda_graphs()
                    runtime["cuda-graph-delete-calls"] += 1
                    return result

                with mock.patch.object(
                    dynamic_engine_module, "delete_cuda_graphs", new=traced_delete_cuda_graphs
                ):
                    env.engine.suspend()
                if "vlm-stale-media" in scenario.signals:
                    vlm_requests = [
                        entry.record[-1]
                        for entry in env.engine.requests.values()
                        if isinstance(entry.record[-1], DynamicVLMInferenceRequest)
                    ]
                    runtime["vlm-stale-media-preserved"] += int(
                        bool(vlm_requests)
                        and all(request.image_embeddings is not None for request in vlm_requests)
                    )
                if "weight-refit" in scenario.signals:
                    model = env.engine.controller.inference_wrapped_model.model
                    parameter = next(
                        parameter
                        for parameter in model.parameters()
                        if parameter.is_floating_point()
                    )
                    before_refit = parameter.detach().flatten()[0].clone()
                    parameter.add_(0.03125)
                    runtime["weight-refit-mutations"] += 1
                    runtime["weight-refit-values-changed"] += int(
                        not torch.equal(before_refit, parameter.detach().flatten()[0])
                    )
                runtime["pending-after-suspend"] += int(
                    env.engine.controller._async_sched_forward.is_valid
                )
                env.engine.resume()
                all_runners_after = [
                    runner
                    for manager in model.modules()
                    if isinstance(manager, CudaGraphManager)
                    for runner in manager.cudagraph_runners
                ]
                runners_after = [
                    runner for runner in all_runners_after if runner.fwd_graph_recorded
                ]
                runtime["cuda-graph-runner-objects-preserved"] += int(
                    all(runner in all_runners_after for runner in runners_before)
                )
                runtime["cuda-graph-objects-preserved"] += int(
                    bool(graph_ids_before)
                    and len(runners_after) == len(graph_ids_before)
                    and all(
                        id(runner.fwd_graph) == graph_id
                        for runner, graph_id in zip(runners_after, graph_ids_before)
                    )
                )
                _instrument_cuda_graph_replays(
                    env.engine.controller.inference_wrapped_model.model, env.engine.context, runtime
                )
                runtime["cuda-graph-replays-at-resume"] = runtime["cuda-graph-runner-replays"]
                runtime["weight-epoch-increments"] += env.engine._weight_epoch - weight_epoch
                for request_id in env.engine.resume_request_ids:
                    env.requests[request_id] = env.engine.get_request(request_id)
                runtime["pending-after-resume"] += int(
                    env.engine.controller._async_sched_forward.is_valid
                )
                runtime["suspend-resume"] += 1
                runtime["static-pointer-preserved"] += int(
                    env.engine.context.memory_buffer.data_ptr() == memory_ptr
                )

        while env.engine.has_unfinished_requests() or env.engine.pending_kv_import_count > 0:
            step()
            assert runtime["steps"] < 400, f"{scenario.name} did not converge"

        if "context-reset" in scenario.signals:
            for _ in range(2):
                memory_ptr = env.engine.context.memory_buffer.data_ptr()
                graph_runners = [
                    (runner, runner.fwd_graph)
                    for manager in model.modules()
                    if isinstance(manager, CudaGraphManager)
                    for runner in manager.cudagraph_runners
                    if runner.fwd_graph_recorded
                ]
                replays_before = runtime["cuda-graph-runner-replays"]
                env.engine.reset()
                runtime["context-resets"] += 1
                runtime["context-reset-pointer-preserved"] += int(
                    env.engine.context.memory_buffer.data_ptr() == memory_ptr
                )
                if env.config.num_cuda_graphs:
                    runtime["context-reset-graphs-preserved"] += int(
                        bool(graph_runners)
                        and all(
                            runner.fwd_graph_recorded and runner.fwd_graph is graph
                            for runner, graph in graph_runners
                        )
                    )

                probe_id = len(env.requests)
                prompt = torch.arange(5, dtype=torch.int64, device="cuda") + 61 + probe_id
                sampling_params = SamplingParams(
                    num_tokens_to_generate=3, termination_id=-1, top_k=1
                )
                env.engine.add_request(probe_id, prompt, sampling_params)
                env.requests.append(env.engine.get_request(probe_id))
                env.requests[probe_id].state = "pending"
                while env.engine.has_unfinished_requests():
                    step()
                    assert runtime["steps"] < 400, f"{scenario.name} reset probe did not converge"
                if env.config.num_cuda_graphs:
                    runtime["context-reset-replays"] += int(
                        runtime["cuda-graph-runner-replays"] > replays_before
                    )
        return env, runtime

    @staticmethod
    def _assert_runtime_signals(env, scenario, runtime, stop_tokens, termination_token):
        context = env.engine.context
        controller = env.engine.controller
        model_config = controller.inference_wrapped_model.model.config

        assert runtime["waves"] == 3
        assert len(env.requests) > context.max_requests
        if "disagg-handoff" in scenario.signals:
            assert runtime["max_pending_kv_imports"] > 0
        else:
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
        assert runtime["model-forward"] > 0
        assert runtime["sampling-kernel"] > 0
        assert runtime["metadata-consumed"] > 0
        assert runtime["metadata-compactions"] > 0
        assert not controller._async_sched_forward.is_valid
        assert context.total_request_count == context.active_token_count == 0

        signals = set(scenario.signals)
        executed_graph_dimensions = {
            key[1:]
            for key, count in runtime.items()
            if isinstance(key, tuple) and key[0] == "cuda-graph-dimension" and count > 0
        }
        replayed_module_names = _recorded(runtime, "cuda-graph-replayed-module", 1)
        if "eager" in signals:
            assert runtime["cuda_graph_steps"] == 0
            assert runtime["cuda-graph-runner-replays"] == 0
            assert not context.cuda_graph_batch_dimensions_list
        if "gpt" in signals:
            assert runtime["module-forward:gpt"] > 0
        if "vlm" in signals:
            assert runtime["vlm-image-forwards"] > 0
            assert runtime["vlm-image-token-count"] > 0
            if "cuda-graph" in signals:
                replayed_steps = _recorded(runtime, "cuda-graph-replay-step", 1)
                image_steps = _recorded(runtime, "vlm-image-step", 1)
                assert replayed_steps & image_steps
        if "vlm-stale-media" in signals:
            assert env.engine.allow_stale_multimodal_embeddings
            assert runtime["vlm-stale-media-preserved"] == runtime["suspend-resume"]
        if "disagg-handoff" in signals:
            assert isinstance(env.engine, StateHandoffDynamicInferenceEngine)
            assert runtime["disagg-admission-deferrals"] > 0
            imports = _recorded(runtime, "disagg-import")
            assert len(imports) == len(env.requests)
            assert runtime["disagg-prefill-tokens-skipped"] == sum(_PROMPT_LENGTHS)
            if "cuda-graph" in signals:
                live_inputs = _recorded(runtime, "cuda-graph-live-input")
                assert any(
                    request_id in request_ids and resume_token in input_ids
                    for _padded, _real, request_ids, input_ids in live_inputs
                    for request_id, resume_token in imports
                )
        if "torch-backend" in signals:
            assert runtime["sampling-backend:torch"] > 0
        if "persist" in signals:
            assert context.kv_cache_management_mode == KVCacheManagementMode.PERSIST
        if "offload" in signals:
            assert context.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
        if "recompute" in signals:
            assert context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE
        if "static-pointers" in signals:
            assert context.static_kv_memory_pointers
        if "tms" in signals:
            assert context._uses_torch_memory_saver
        if "suspend-resume" in signals:
            expected_cycles = len(_REQUEST_WAVES) if "repeat-suspend-resume" in signals else 1
            assert runtime["suspend-resume"] == expected_cycles
            assert runtime["pending-before-suspend"] == expected_cycles
            assert runtime["weight-epoch-increments"] == expected_cycles
            if "recompute" in signals:
                assert runtime["pending-after-suspend"] == 0
                assert runtime["pending-after-resume"] == 0
            else:
                assert runtime["pending-after-suspend"] == expected_cycles
                assert runtime["pending-after-resume"] == expected_cycles
            if "static-pointers" in signals:
                assert runtime["static-pointer-preserved"] == expected_cycles
            if "cuda-graph" in signals:
                assert (
                    runtime["cuda-graph-runner-replays"] > runtime["cuda-graph-replays-at-resume"]
                )
                if "static-pointers" in signals:
                    assert runtime["cuda-graph-delete-calls"] == 0
                    assert runtime["cuda-graph-create-calls"] == 0
                    assert runtime["cuda-graph-runner-objects-preserved"] == expected_cycles
                    assert runtime["cuda-graph-objects-preserved"] == expected_cycles
                else:
                    assert runtime["cuda-graph-create-calls"] == 0
        if "context-reset" in signals:
            assert runtime["context-resets"] == 2
            assert runtime["context-reset-pointer-preserved"] == 2
            if "cuda-graph" in signals:
                assert runtime["context-reset-graphs-preserved"] == 2
                assert runtime["context-reset-replays"] == 2
        if "weight-refit" in signals:
            assert runtime["weight-refit-mutations"] == runtime["suspend-resume"]
            assert runtime["weight-refit-values-changed"] == runtime["suspend-resume"]
            assert env.engine._weight_epoch == runtime["suspend-resume"]
            expected_salts = {f"w{epoch}" for epoch in range(1, runtime["suspend-resume"] + 1)}
            assert {request.block_hash_salt for request in env.requests} == expected_salts
        if "dp-offset" in signals:
            assert context.config.offset_sampling_seed_by_dp_rank
        if "last-logits" in signals:
            assert context.config.materialize_only_last_token_logits
            assert runtime["last-logits-forward"] == runtime["model-forward"]
            assert runtime["logits-dtype:torch.bfloat16"] > 0
        if "graph-warmup" in signals:
            assert model_config.cuda_graph_warmup_steps == 1
            assert runtime[("cuda-graph-warmup-steps", 1)] > 0
        if "events" in signals:
            for request in env.requests:
                event_types = [event.type for event in request.events]
                assert event_types.count(DynamicInferenceEventType.GENERATED_TOKEN) == len(
                    request.generated_tokens
                )
        if "chunked" in signals:
            assert context.enable_chunked_prefill
            assert any(len(request.prompt_tokens) > context.max_tokens for request in env.requests)
            assert runtime["chunked-prefill-steps"] > 0
            if "cuda-graph" in signals:
                chunked_steps = _recorded(runtime, "chunked-forward-step", 1)
                replayed_steps = _recorded(runtime, "cuda-graph-replay-step", 1)
                assert chunked_steps & replayed_steps
        if "capacity" in signals:
            assert context.max_requests == 4
            assert context.max_tokens == 8
            assert runtime["max_active"] == context.max_requests
        if "prefix-hit" in signals:
            assert env.engine._prefix_cache_hits > 0
            assert env.engine._prefill_tokens_skipped > 0
            if "cuda-graph" in signals:
                prefix_hit_steps = _recorded(runtime, "prefix-hit-forward-step", 1)
                replayed_steps = _recorded(runtime, "cuda-graph-replay-step", 1)
                assert runtime["prefix-hit-forward-skipped-tokens"] > 0
                assert prefix_hit_steps & replayed_steps
        if "ref-zero" in signals:
            assert context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
            assert runtime["prefix-blocks-deregistered"] > 0
        if "cuda-graph" in signals:
            assert context.cuda_graph_batch_dimensions_list
            assert runtime["cuda_graph_steps"] > 0
            assert runtime["cuda-graph-runner-replays"] > 0
            assert executed_graph_dimensions
            available_dimensions = {
                (dimensions.token_count, dimensions.prefill_req_count, dimensions.decode_req_count)
                for dimensions in context.cuda_graph_batch_dimensions_list
            }
            assert executed_graph_dimensions <= available_dimensions
            replayed_graph_dimensions = _recorded(runtime, "cuda-graph-replay-dimension")
            assert replayed_graph_dimensions == executed_graph_dimensions
            assert (
                runtime["cuda-graph-real-token-scalar-matches"]
                == runtime["cuda-graph-runner-replays"]
            )
            if runtime["cuda-graph-padded-replays"]:
                assert (
                    runtime["cuda-graph-zero-input-padding"] == runtime["cuda-graph-padded-replays"]
                )
            graph_output_pointers = _recorded(runtime, "cuda-graph-output-pointer", 1)
            assert graph_output_pointers
            assert controller._all_logits_cuda.data_ptr() not in graph_output_pointers
        if "graph-decode-config" in signals:
            assert not context.use_cuda_graphs_for_non_decode_steps
            assert context.config.cuda_graph_sizing_distribution == (
                CudaGraphSizingDistribution.EXPONENTIAL
            )
            assert context.config.cuda_graph_max_tokens == 16
            assert runtime["cuda-graph-scope:block"] > 0
            decode_token_counts = {
                token_count
                for token_count, prefill_count, _ in executed_graph_dimensions
                if prefill_count == 0
            }
            assert len(decode_token_counts) > 1
            assert all(
                token_count > 0 and token_count & (token_count - 1) == 0
                for token_count in decode_token_counts
            )
            assert all(prefill_count == 0 for _, prefill_count, _ in executed_graph_dimensions)
            assert runtime["cuda-graph-decode-forwards"] > 0
        if "graph-mixed-config" in signals:
            assert context.use_cuda_graphs_for_non_decode_steps
            assert context.config.cuda_graph_mixed_prefill_count == 2
            assert (
                context.config.cuda_graph_sizing_distribution == CudaGraphSizingDistribution.LINEAR
            )
            assert context.config.cuda_graph_all_prefills
            assert context.config.cuda_graph_max_tokens == 24
            assert runtime["cuda-graph-scope:layer"] > 0
            mixed_token_counts = {
                token_count
                for token_count, prefill_count, _ in executed_graph_dimensions
                if prefill_count > 0
            }
            assert len(mixed_token_counts) > 1
            linear_step = context.max_tokens // scenario.config["num_cuda_graphs"]
            assert linear_step == 8
            assert context.max_tokens in mixed_token_counts
            configured_mixed_token_counts = sorted(
                {
                    dimensions.token_count
                    for dimensions in context.cuda_graph_batch_dimensions_list
                    if dimensions.prefill_req_count > 0 and dimensions.decode_req_count > 0
                }
            )
            assert mixed_token_counts.issubset(configured_mixed_token_counts)
            assert any(
                right - left == linear_step
                for left, right in zip(
                    configured_mixed_token_counts, configured_mixed_token_counts[1:]
                )
            )
            assert any(
                prefill_count == context.config.cuda_graph_mixed_prefill_count and decode_count > 0
                for _, prefill_count, decode_count in executed_graph_dimensions
            )
            assert runtime["cuda-graph-mixed-forwards"] > 0
        if "graph-all-prefill" in signals:
            assert runtime["cuda-graph-all-prefill-forwards"] > 0
        if "graph-bounded-config" in signals:
            assert context.use_cuda_graphs_for_non_decode_steps
            assert not context.config.cuda_graph_all_prefills
            assert context.config.cuda_graph_max_tokens == 16
            assert max(token_count for token_count, _, _ in executed_graph_dimensions) == 16
            assert runtime["cuda-graph-prefill-forwards"] > 0
            oversized_steps = {
                (step, token_count, prefill_count, decode_count)
                for step, token_count, prefill_count, decode_count in _recorded(
                    runtime, "model-step-dimension"
                )
                if token_count > context.config.cuda_graph_max_tokens and prefill_count > 0
            }
            assert oversized_steps
            replayed_steps = _recorded(runtime, "cuda-graph-replay-step", 1)
            assert all(step not in replayed_steps for step, *_ in oversized_steps)
        if "sampled" in signals:
            assert all(request.generated_tokens for request in env.requests)
        if "request-seed" in signals:
            assert {request.sampling_params.seed for request in env.requests} == {17, 29}
            assert runtime["request-seed-kernel"] > 0
        for sampling_filter in ("temperature-filter", "top-k-filter", "top-p-filter"):
            if sampling_filter in signals:
                assert runtime[sampling_filter] > 0
        if {
            "temperature-filter",
            "top-k-filter",
            "top-p-filter",
        } & signals and "flashinfer" not in signals:
            assert runtime["sampling-filter-rows"] > 0
            assert runtime["sampling-filter-nonempty-rows"] == runtime["sampling-filter-rows"]
            assert runtime["temperature-scale-exact-rows"] == runtime["sampling-filter-rows"]
        if "top-k-filter" in signals and "flashinfer" not in signals:
            assert runtime["top-k-support-exact-rows"] == runtime["sampling-filter-rows"]
        if "top-p-filter" in signals and "flashinfer" not in signals:
            assert runtime["top-p-support-exact-rows"] == runtime["sampling-filter-rows"]
        if "logprobs" in signals:
            assert runtime["log-probs-calculations"] > 0
            for request in env.requests:
                if request.sampling_params.return_log_probs:
                    assert request.generated_log_probs is not None
                    assert len(request.generated_log_probs) == len(request.generated_tokens)
                else:
                    assert request.prompt_log_probs is None
                    assert request.generated_log_probs is None
        if "full-logits" in signals:
            assert not context.config.materialize_only_last_token_logits
            assert runtime["full-logits-forward"] == runtime["model-forward"]
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
            assert runtime["log-probs-mode:processed_logprobs"] > 0
            assert runtime["log-probs-kernel"] > 0
        if "flashinfer" in signals:
            assert runtime["sampling-backend:flashinfer"] > 0
            assert (
                sum(
                    count
                    for key, count in runtime.items()
                    if isinstance(key, str) and key.startswith("flashinfer-kernel:")
                )
                > 0
            )
            assert runtime["flashinfer-valid-support-rows"] == runtime["flashinfer-support-rows"]
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
            proposed = int(env.engine._spec_tokens_proposed_per_pos.sum())
            accepted = int(env.engine._spec_tokens_accepted_per_pos.sum())
            assert proposed > 0
            if "cuda-graph" in signals:
                owners = _recorded(runtime, "cuda-graph-replay-owner", 1)
                assert "compute_mtp_single_step" in owners
            if "mtp-accept-all" in signals:
                assert accepted == proposed
                assert context.enable_mtp_kv_cache
                assert model_config.mtp_use_repeated_layer
        if "metadata-compaction" in signals:
            assert runtime["non-prefix-survivor-compactions"] > 0
        if "hybrid" in signals:
            assert context.is_hybrid_model
            assert hasattr(context, "mamba_conv_states_shape")
            assert runtime["mamba-state-commits"] > 0
            assert runtime["mamba-state-compactions"] > 0
            expected_replay_module = {"gdn": "GatedDeltaNet", "gdp": "GatedDeltaProductMixer"}.get(
                scenario.config.get("ssm_mixer"), "MambaMixer"
            )
            assert (
                runtime["module-forward:hybrid"] > 0
                or expected_replay_module in replayed_module_names
            )
        if "transformer-engine" in signals:
            assert model_config.transformer_impl == "transformer_engine"
            assert runtime["module-forward:transformer-engine"] > 0 or any(
                name.startswith("TE") for name in replayed_module_names
            )
        if "inference-optimized" in signals:
            assert model_config.transformer_impl == "inference_optimized"
            assert runtime["module-forward:inference-optimized"] > 0 or any(
                name.startswith("Inference") for name in replayed_module_names
            )
        if "fp8" in signals:
            assert model_config.fp8 is not None
            if "cuda-graph" in signals:
                assert runtime["cuda-graph-fp8-replays"] > 0
            else:
                assert runtime["module-forward:fp8"] > 0
            assert runtime["fp8-context-forwards"] > 0
            assert runtime["fp8-quantized-forwards"] > 0
            assert runtime["fp8-recipe-forwards"] > 0
        if "mxfp8" in signals:
            from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

            experts = [
                module
                for module in controller.inference_wrapped_model.model.modules()
                if type(module).__name__ == "InferenceGroupedMLP"
            ]
            assert experts
            assert runtime["module-forward:moe"] > 0
            assert all(module._uses_mxfp8_weights for module in experts)
            assert all(isinstance(module._fc1_weight, MXFP8Tensor) for module in experts)
            assert all(isinstance(module._fc2_weight, MXFP8Tensor) for module in experts)
        if "routing-replay" in signals:
            assert model_config.moe_enable_routing_replay
            assert context.moe_routing_metadata is not None
            owners = _recorded(runtime, "cuda-graph-replay-owner")
            replayed_steps = {step for owner, step in owners if owner == "forward"}
            recorded_steps = _recorded(runtime, "routing-record-step", 1)
            assert replayed_steps & recorded_steps
            for request in env.requests:
                routing = request.routing_indices
                assert routing is not None
                assert routing.shape[0] == (
                    len(request.prompt_tokens) + len(request.generated_tokens) - 1
                )
                assert routing.shape[2] == model_config.moe_router_topk
                assert ((routing >= 0) & (routing < model_config.num_moe_experts)).all()
        if "fused-rope" in signals:
            assert context.use_flashinfer_fused_rope
            assert model_config.hidden_size // model_config.num_attention_heads == 64
            model = controller.inference_wrapped_model.model
            assert model.rotary_pos_emb.inv_freq.is_cuda
            assert model.rotary_pos_emb_cache[context.max_sequence_length].is_cuda
            assert runtime["fused-rope-kernel"] > 0
        if "flash-attention-version" in signals:
            version = model_config.flash_attention_version
            assert version in (2, 3, 4)
            assert runtime[("flash-attention-version", version)] > 0
            if "cuda-graph" in signals:
                assert "SelfAttention" in replayed_module_names
        if "swa-all" in signals:
            assert model_config.window_size == scenario.config["window_size"]
            assert runtime["swa-kernel-calls"] > 0
            assert runtime["full-attention-kernel-calls"] == 0
            assert "cuda-graph" not in signals or "SelfAttention" in replayed_module_names
        if "swa-alternating" in signals:
            assert model_config.window_size == scenario.config["window_size"]
            assert runtime["swa-kernel-calls"] > 0
            assert runtime["full-attention-kernel-calls"] > 0
            if "cuda-graph" in signals:
                assert "SelfAttention" in replayed_module_names
                attention_layers = [
                    module
                    for module in controller.inference_wrapped_model.model.modules()
                    if type(module).__name__ == "SelfAttention"
                ]
                window_layers = [
                    is_layer_window_attention(
                        module.config.window_size,
                        module.config.window_attn_skip_freq,
                        module.layer_number,
                    )
                    for module in attention_layers
                ]
                assert any(window_layers) and not all(window_layers)
        if "softmax-sink" in signals:
            assert model_config.softmax_type == scenario.config["softmax_type"]
            assert runtime["softmax-offset-calls"] > 0
            assert runtime["softmax-offset-produced"] == runtime["softmax-offset-calls"]
            assert runtime["sink-correction-calls"] > 0
            assert "cuda-graph" not in signals or "SelfAttention" in replayed_module_names
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
            seed = torch.tensor(expected_seed, dtype=torch.int64, device="cuda")
            gathered_seeds = [
                torch.empty_like(seed)
                for _ in range(torch.distributed.get_world_size(controller.dp_group))
            ]
            torch.distributed.all_gather(gathered_seeds, seed, group=controller.dp_group)
            assert [peer.item() for peer in gathered_seeds] == [
                model_config.inference_sampling_seed + rank
                for rank in range(torch.distributed.get_world_size(controller.dp_group))
            ]
            generated = torch.tensor(
                [token for request in env.requests for token in request.generated_tokens],
                dtype=torch.int64,
                device="cuda",
            )
            gathered_tokens = [
                torch.empty_like(generated)
                for _ in range(torch.distributed.get_world_size(controller.dp_group))
            ]
            torch.distributed.all_gather(gathered_tokens, generated, group=controller.dp_group)
            assert any(not torch.equal(peer, generated) for peer in gathered_tokens)
        if "parallel" in signals:
            assert (
                model_config.tensor_model_parallel_size > 1
                or model_config.pipeline_model_parallel_size > 1
                or model_config.expert_model_parallel_size > 1
            )
            if "cuda-graph" in signals:
                assert any("ParallelLinear" in name for name in replayed_module_names)
            else:
                assert runtime["module-forward:parallel"] > 0
            if model_config.tensor_model_parallel_size > 1:
                if "cuda-graph" in signals:
                    assert runtime["cuda-graph-runner-replays"] > 0
                elif model_config.transformer_impl == "inference_optimized":
                    assert runtime["tp-collective:optimized-all-gather"] > 0
                    assert runtime["tp-collective:optimized-reduce-scatter"] > 0
                    assert runtime["tp-sp-gather-dimensions"] > 0
                    assert runtime["tp-sp-reduce-scatter-dimensions"] > 0
                else:
                    assert runtime["tp-column-partition-forwards"] > 0
                    assert runtime["tp-row-partition-forwards"] > 0
                if (
                    model_config.sequence_parallel
                    and model_config.transformer_impl != "inference_optimized"
                ):
                    if "cuda-graph" not in signals:
                        assert (
                            runtime["tp-collective:reduce_scatter_to_sequence_parallel_region"] > 0
                        )
                        assert runtime["tp-sp-reduce-scatter-dimensions"] > 0
                elif not model_config.sequence_parallel:
                    assert runtime["tp-collective:reduce_from_tensor_model_parallel_region"] > 0
            if model_config.pipeline_model_parallel_size > 1:
                assert runtime["pipeline-logits-broadcasts"] > 0
        if "etp" in signals:
            assert model_config.expert_tensor_parallel_size == 2
            assert parallel_state.get_expert_tensor_parallel_world_size() == 2
            dispatchers = {
                type(module.token_dispatcher).__name__
                for module in controller.inference_wrapped_model.model.modules()
                if type(module).__name__ == "MoELayer"
            }
            assert dispatchers == {"MoEAllGatherTokenDispatcher"}
        if "moe" in signals:
            assert model_config.num_moe_experts is not None
            if "etp" in signals:
                assert runtime["module-forward:moe"] > 0
            elif model_config.expert_model_parallel_size > 1 and "nvls-dispatch" not in signals:
                assert context._nccl_ep_dispatcher
                # Inference MoE may dispatch through split execution helpers that
                # bypass nn.Module forward hooks. The live dispatcher calls below
                # are the feature-owning runtime boundary.
                assert runtime["nccl-token-dispatches"] > 0
            elif "cuda-graph" in signals:
                assert "InferenceGroupedMLP" in replayed_module_names
            else:
                assert runtime["module-forward:moe"] > 0
        if "nccl-dispatch" in signals:
            assert model_config.expert_model_parallel_size > 1
            assert model_config.inference_moe_token_dispatcher_type == "nccl"
            assert runtime["nccl-dispatchers-installed"] > 0
            assert runtime["nccl-token-dispatches"] > 0
            assert runtime["nccl-token-dispatches"] == runtime["nccl-token-combines"]
            assert runtime["nccl-combine-before-dispatch"] == 0
            assert runtime["nccl-dispatch-inflight"] == 0
            if "cuda-graph" in signals:
                assert "MoELayer" in replayed_module_names
                nondecode_steps = {
                    step
                    for step, _, prefill_count, _ in _recorded(runtime, "model-step-dimension")
                    if prefill_count > 0
                }
                replayed_steps = {
                    step
                    for owner, step in _recorded(runtime, "cuda-graph-replay-owner")
                    if owner == "forward"
                }
                assert nondecode_steps
                assert nondecode_steps.isdisjoint(replayed_steps)
        if "nvls-dispatch" in signals:
            assert model_config.expert_model_parallel_size > 1
            assert model_config.inference_moe_token_dispatcher_type == "nvls"
            assert context._nvls_dispatcher
            assert not context._nccl_ep_dispatcher
            if "cuda-graph" in signals:
                assert context.use_cuda_graphs_for_non_decode_steps
                assert runtime["cuda-graph-prefill-forwards"] > 0
                replayed_steps = {
                    step
                    for owner, step in _recorded(runtime, "cuda-graph-replay-owner")
                    if owner == "forward"
                }
                nondecode_steps = {
                    step
                    for step, _, prefill_count, _ in _recorded(runtime, "model-step-dimension")
                    if prefill_count > 0
                }
                assert replayed_steps & nondecode_steps

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
        if scenario.parity == "reproducible" or not scenario.exact_top_n:
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
        _reset_rounder()
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

    @pytest.mark.parametrize(
        "scenario", _ASYNC_SUSPEND_RESUME_SCENARIOS, ids=lambda case: case.name
    )
    @torch.inference_mode()
    def test_async_suspend_resume_mode_matches_legacy(self, scenario):
        """Real OFFLOAD and RECOMPUTE cycles preserve output parity."""
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
class TestCudaGraphDifferential(_AsyncPairwiseHarness):
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        _reset_rounder()
        Utils.destroy_model_parallel()

    @staticmethod
    def _with_graphs(scenario):
        config = dict(scenario.config)
        config.update(
            num_cuda_graphs=4, force_build_cuda_graphs=True, defer_cuda_graph_creation=True
        )
        repeat = ("repeat-suspend-resume",) if "suspend-resume" in scenario.signals else ()
        signals = tuple(dict.fromkeys((*scenario.signals, *repeat, "cuda-graph")))
        return replace(scenario, config=config, signals=signals)

    @staticmethod
    def _release_graphs():
        gc.collect()
        delete_cuda_graphs()
        torch.cuda.empty_cache()

    @staticmethod
    def _weight_snapshot(env):
        model = env.engine.controller.inference_wrapped_model.model
        return {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }

    @classmethod
    def _assert_graph_pair(cls, scenario):
        graph_scenario = cls._with_graphs(scenario)
        stop_tokens = None
        termination_token = None
        try:
            if scenario.request_profile.startswith("stop"):
                probe_env, _ = cls._run_scenario_mode(graph_scenario, AsyncScheduleMode.LEGACY)
                stop_tokens = tuple(probe_env.requests[1].generated_tokens[1:3])
                assert len(stop_tokens) == 2
                del probe_env
                cls._release_graphs()
            elif scenario.request_profile == "termination":
                probe_env, _ = cls._run_scenario_mode(graph_scenario, AsyncScheduleMode.LEGACY)
                termination_token = probe_env.requests[1].generated_tokens[0]
                assert termination_token != -1
                del probe_env
                cls._release_graphs()
            eager_env, eager_runtime = cls._run_scenario_mode(
                graph_scenario,
                AsyncScheduleMode.ASYNC,
                stop_tokens=stop_tokens,
                termination_token=termination_token,
                force_graph_eager=True,
            )
            expected = _snapshot_requests(eager_env.requests)
            expected_weights = cls._weight_snapshot(eager_env)
            eager_logits = eager_runtime.get("fixed-history-logit-snapshots", {})
            teacher_tokens = {request["request_id"]: request["tokens"] for request in expected}
            if "fixed-history-logits" in scenario.signals:
                expected_logit_keys = {
                    (request.request_id, len(request.prompt_tokens) + offset - 1)
                    for request in eager_env.requests
                    for offset in range(len(request.generated_tokens))
                }
                assert expected_logit_keys <= eager_logits.keys()
                discarded_logit_keys = eager_logits.keys() - expected_logit_keys
                print(
                    f"fixed-history: completed={len(expected_logit_keys)}, discarded={sorted(discarded_logit_keys)}"
                )
            assert eager_runtime["cuda-graph-runner-replays"] == 0
            del eager_env
            cls._release_graphs()
            graph_env, graph_runtime = cls._run_scenario_mode(
                graph_scenario,
                AsyncScheduleMode.ASYNC,
                stop_tokens=stop_tokens,
                termination_token=termination_token,
                teacher_tokens=teacher_tokens,
            )
            actual_weights = cls._weight_snapshot(graph_env)
            assert expected_weights.keys() == actual_weights.keys()
            assert all(
                torch.equal(expected_weights[name], actual_weights[name])
                for name in expected_weights
            )
            graph_parity = scenario.parity
            graph_logits = {}
            if "fixed-history-logits" in scenario.signals:
                graph_logits = graph_runtime.get("fixed-history-logit-snapshots", {})
                assert eager_logits and graph_logits
                assert graph_logits.keys() == expected_logit_keys | discarded_logit_keys
            _assert_request_parity(
                graph_env.requests,
                expected,
                scenario.atol,
                exact_numerics=graph_parity == "exact" and not eager_logits,
                exact_top_n=graph_parity == "exact" and scenario.exact_top_n and not eager_logits,
            )
            cls._assert_runtime_signals(
                graph_env, graph_scenario, graph_runtime, stop_tokens, termination_token
            )
            live_inputs = _recorded(graph_runtime, "cuda-graph-live-input")
            assert live_inputs
            inputs_by_shape_and_requests = {}
            for padded, _real, request_ids, input_ids in live_inputs:
                inputs_by_shape_and_requests.setdefault((padded, request_ids), set()).add(input_ids)
            assert any(len(inputs) > 1 for inputs in inputs_by_shape_and_requests.values())
            if graph_parity != "exact" or eager_logits:
                graph_expected = _snapshot_requests(graph_env.requests)
                graph_requests = graph_env.requests
                del graph_env
                cls._release_graphs()
                repeat_env, repeat_runtime = cls._run_scenario_mode(
                    graph_scenario,
                    AsyncScheduleMode.ASYNC,
                    stop_tokens=stop_tokens,
                    termination_token=termination_token,
                    teacher_tokens=teacher_tokens,
                )
                repeat_exact = "fixed-history-logits" not in scenario.signals
                if not repeat_exact:
                    repeat_logits = repeat_runtime.get("fixed-history-logit-snapshots", {})
                    _assert_fixed_history_logit_parity(repeat_logits, graph_logits, 0.0)
                    _assert_fixed_history_logit_parity(repeat_logits, eager_logits, scenario.atol)
                    _assert_fixed_history_logit_parity(graph_logits, eager_logits, scenario.atol)
                    _assert_request_parity(
                        graph_requests,
                        expected,
                        scenario.atol,
                        exact_numerics=True,
                        exact_top_n=graph_parity == "exact" and scenario.exact_top_n,
                    )
                _assert_request_parity(
                    repeat_env.requests,
                    expected if eager_logits else graph_expected,
                    scenario.atol,
                    exact_numerics=repeat_exact or bool(eager_logits),
                    exact_top_n=repeat_exact or bool(eager_logits),
                )
                cls._assert_runtime_signals(
                    repeat_env, graph_scenario, repeat_runtime, stop_tokens, termination_token
                )
        finally:
            cls._release_graphs()

    @pytest.mark.parametrize(
        "scenario", _CUDA_GRAPH_DIFFERENTIAL_SCENARIOS, ids=lambda case: case.name
    )
    @torch.inference_mode()
    def test_eager_matches_actual_replay_for_owned_pair(self, scenario):
        _check_scenario_prerequisite(scenario)
        self._assert_graph_pair(scenario)


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
            expert_tensor_parallel_size=config.get("expert_tensor_parallel_size", 1),
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
            _reset_rounder()
            Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "scenario", _ASYNC_PARALLEL_GRAPH_SCENARIOS, ids=lambda case: case.name
    )
    @torch.inference_mode()
    def test_eager_matches_actual_replay_for_parallel_pair(self, scenario):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 4:
            pytest.skip("parallel CUDA-graph pair owners require at least four GPUs")
        config = scenario.config
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=config.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=config.get("pipeline_model_parallel_size", 1),
            expert_model_parallel_size=config.get("expert_model_parallel_size", 1),
            expert_tensor_parallel_size=config.get("expert_tensor_parallel_size", 1),
        )
        try:
            TestCudaGraphDifferential._assert_graph_pair(scenario)
        finally:
            TestCudaGraphDifferential._release_graphs()
            _reset_rounder()
            Utils.destroy_model_parallel()


def _controller_with_pending_logits():
    controller = TextGenerationController.__new__(TextGenerationController)
    controller._async_sched_forward = AsyncScheduleForwardState()
    controller._async_sched_forward.set_pending(4, torch.tensor([0, 1, 2]))
    return controller


def test_async_reset_clears_pending_logits():
    """Engine reset cannot expose logits produced for the previous request batch."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(reset=mock.Mock())
    engine.controller = _controller_with_pending_logits()
    engine.num_speculative_tokens = 1
    engine._loop = None
    engine._vision_embedding_cache = {}
    engine._vision_embedding_cache_bytes = 0

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

    assert not engine.controller._async_sched_forward.is_valid
    assert engine.controller._async_sched_forward.token_row_indices is None


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
    engine._vision_embedding_cache = {}
    engine._vision_embedding_cache_bytes = 0

    with (
        mock.patch.object(DynamicInferenceEngine, "suspend_resume_ctx", return_value=nullcontext()),
        mock.patch("megatron.core.inference.engines.dynamic_engine.InferenceMode.unset_active"),
    ):
        engine.suspend()

    assert engine.controller._async_sched_forward.is_valid is preserve_pending
    if preserve_pending:
        assert torch.equal(
            engine.controller._async_sched_forward.token_row_indices, torch.tensor([0, 1, 2])
        )
    else:
        assert engine.controller._async_sched_forward.token_row_indices is None


@pytest.mark.parametrize(
    ("survivor_idxs", "expected_token_rows"),
    [
        pytest.param([0, 2], [0, 1, 4, 5], id="monotonic-with-gap"),
        pytest.param([3, 1, 2], [6, 7, 2, 3, 4, 5], id="nonmonotonic"),
    ],
)
def test_async_compaction_preserves_all_request_metadata(survivor_idxs, expected_token_rows):
    """Middle-row completion compacts every metadata field into survivor order."""
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.num_speculative_tokens = 1
    controller._enable_cuda_graph = False
    original_logits = torch.arange(32).reshape(1, 8, 4)
    controller._all_logits_cuda = original_logits.clone()
    controller._async_sched_forward = AsyncScheduleForwardState()
    controller._async_sched_forward.set_pending(4, torch.arange(8))

    metadata = {
        "temperature": torch.tensor([0.5, 0.7, 0.9, 1.1]),
        "top_k": torch.tensor([2, 4, 8, 16]),
        "top_p": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "termination_id": torch.tensor([90, 91, 92, 93]),
        "return_log_probs": torch.tensor([True, False, True, False]),
        "skip_prompt_log_probs": torch.tensor([False, True, False, True]),
        "top_n_logprobs": torch.tensor([2, 0, 5, 1]),
        "custom_metadata": torch.tensor([101, 202, 303, 404]),
    }
    gpu_view = SimpleNamespace(
        temperature=metadata["temperature"].clone(),
        top_k=metadata["top_k"].clone(),
        top_p=metadata["top_p"].clone(),
    )
    context = SimpleNamespace(active_request_metadata=metadata, gpu_view=gpu_view)
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    expected = {label: values[survivor_idxs].clone() for label, values in metadata.items()}

    controller._compact_async_sched_forward(torch.tensor(survivor_idxs))

    for label, values in expected.items():
        assert torch.equal(
            context.active_request_metadata[label][: len(survivor_idxs)], values
        ), label
    assert torch.equal(gpu_view.temperature[: len(survivor_idxs)], expected["temperature"])
    assert torch.equal(gpu_view.top_k[: len(survivor_idxs)], expected["top_k"])
    assert torch.equal(gpu_view.top_p[: len(survivor_idxs)], expected["top_p"])
    assert torch.equal(controller._all_logits_cuda, original_logits[:, expected_token_rows])
    assert torch.equal(
        controller._async_sched_forward.token_row_indices, torch.tensor(expected_token_rows)
    )


def test_post_process_enforces_per_request_logprob_policy():
    """Post-processing must honor opt-outs and require requested logprobs."""
    requests = []
    for request_id, return_log_probs in enumerate((True, False)):
        request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.tensor([1, 2], dtype=torch.int64),
            sampling_params=SamplingParams(
                num_tokens_to_generate=2,
                termination_id=-1,
                return_log_probs=return_log_probs,
                skip_prompt_log_probs=True,
            ),
        )
        request.add_event_add_engine()
        requests.append(request)

    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(kv_block_allocator=SimpleNamespace())
    engine.requests = {
        request.request_id: SimpleNamespace(record=[request]) for request in requests
    }
    engine.finished_request_count = 0
    engine.evicted_request_count = 0
    engine.track_generated_token_events = False
    engine.num_speculative_tokens = 0
    engine.stop_word_being_finished_ids = set()
    engine.stop_word_finished_request_ids = set()

    active_request_ids, finished_records = engine.post_process_requests(
        request_ids=torch.tensor([0, 1], dtype=torch.int64),
        finished_request_ids=torch.empty(0, dtype=torch.int64),
        evict_request_ids=None,
        step_time=0.0,
        sample=torch.tensor([11, 22], dtype=torch.int64),
        accepted_tokens=None,
        log_probs=[[-1.0], [-2.0]],
        consumed_chunked_prefill_request_id=-1,
    )

    assert active_request_ids == [0, 1]
    assert finished_records == []
    assert requests[0].generated_log_probs == [-1.0]
    assert requests[1].prompt_log_probs is None
    assert requests[1].generated_log_probs is None

    with pytest.raises(AssertionError, match="requested log probs, but none were produced"):
        engine.post_process_requests(
            request_ids=torch.tensor([0], dtype=torch.int64),
            finished_request_ids=torch.empty(0, dtype=torch.int64),
            evict_request_ids=None,
            step_time=0.0,
            sample=torch.tensor([33], dtype=torch.int64),
            accepted_tokens=None,
            log_probs=None,
            consumed_chunked_prefill_request_id=-1,
        )


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


def test_base_engine_rejects_kv_handoff_commands():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)

    assert InferenceStateHandoffMixin not in DynamicInferenceEngine.mro()
    assert engine.pending_kv_import_count == 0
    with pytest.raises(RuntimeError, match="SUBMIT_REQUEST_WITH_KV"):
        engine.add_request_with_kv_handoff(1, [], SamplingParams(), {}, [])
    with pytest.raises(RuntimeError, match="RELEASE_KV"):
        engine.release_handoff_blocks(1)


def test_disagg_engine_resolves_handoff_methods_from_mixin():
    for engine_cls in (StateHandoffDynamicInferenceEngine, DisaggDynamicInferenceEngine):
        assert (
            engine_cls.add_request_with_kv_handoff
            is InferenceStateHandoffMixin.add_request_with_kv_handoff
        )
