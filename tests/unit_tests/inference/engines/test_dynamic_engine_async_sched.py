# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import os
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass, field
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
from megatron.core.inference.contexts.dynamic_context import DynamoHelper
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
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
from megatron.core.transformer.utils import is_layer_window_attention
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicInferenceEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
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
            "inference_config_overrides": {
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.LRU
            },
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
        "hybrid-mamba",
        "model:hybrid-mamba",
        "interaction:mamba-state-compaction",
        config={"model_provider": "hybrid"},
        signals=("hybrid", "metadata-compaction"),
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
            "inference_moe_token_dispatcher_type": "nccl",
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "moe", "nccl-dispatch", "parallel"),
    ),
    _pair_scenario(
        "optimized-tp2-sp",
        "interaction:optimized-tp-sp",
        "seed:shared-across-dp",
        config={
            "tensor_model_parallel_size": 2,
            "sequence_parallel": True,
            "transformer_impl": "inference_optimized",
            "inference_config_overrides": {"offset_sampling_seed_by_dp_rank": False},
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


def _instrument_scenario_runtime(env, scenario, runtime):
    """Record production calls and state transitions used as pairwise evidence."""
    controller = env.engine.controller
    context = env.engine.context
    _instrument_async_phases(controller, runtime)

    original_forward = controller._dynamic_step_forward_logits

    def traced_forward(*args, **kwargs):
        if context.using_cuda_graph_this_step():
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
            from megatron.core.inference.sampling.torch_sampling import TorchSampling

            original_filter_logits = TorchSampling.filter_logits

            def traced_filter_logits(logits, temperature, top_k, top_p, **filter_kwargs):
                runtime["temperature-filter"] += int(temperature != 1.0)
                runtime["top-k-filter"] += int(top_k >= 1)
                runtime["top-p-filter"] += int(top_p > 0.0)
                return original_filter_logits(logits, temperature, top_k, top_p, **filter_kwargs)

            with mock.patch.object(
                TorchSampling, "filter_logits", new=staticmethod(traced_filter_logits)
            ):
                return original_sample_kernel(*args, **kwargs)

        if sampling_class == "FlashInferSampling":
            from megatron.core.inference.sampling import flashinfer_sampling

            flashinfer_kernels = {
                "sampling_from_probs": (),
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
                    return original(*kernel_args, **kernel_kwargs)

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

    original_compact = controller._compact_async_sched_logits

    def traced_compact(survivor_idxs):
        if survivor_idxs.numel() > 0:
            runtime["metadata-compactions"] += 1
            runtime["non-prefix-survivor-compactions"] += int(
                not torch.equal(
                    survivor_idxs, torch.arange(survivor_idxs.numel(), device=survivor_idxs.device)
                )
            )
        return original_compact(survivor_idxs)

    controller._compact_async_sched_logits = traced_compact

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
        "transformer-engine": lambda name, module: name.startswith("TE")
        or "transformer_engine" in module,
        "inference-optimized": lambda name, module: name.startswith("Inference")
        and "inference_layers" in module,
        "fp8": lambda name, module: name.startswith("TE") or "transformer_engine" in module,
        "hybrid": lambda name, module: name == "MambaMixer",
        "moe": lambda name, module: name in {"MoELayer", "InferenceTopKRouter"},
        "parallel": lambda name, module: "ParallelLinear" in name
        or name in {"ColumnParallelLinear", "RowParallelLinear"},
    }
    active_tags = set(scenario.signals) & watched_modules.keys()
    model = controller.inference_wrapped_model.model
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
            _instrument_scenario_runtime(env, scenario, runtime)

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
                pending_before_suspend = env.engine.controller._async_sched_logits.is_valid
                runtime["pending-before-suspend"] += int(pending_before_suspend)
                env.engine.suspend()
                runtime["pending-after-suspend"] += int(
                    env.engine.controller._async_sched_logits.is_valid
                )
                env.engine.resume()
                runtime["pending-after-resume"] += int(
                    env.engine.controller._async_sched_logits.is_valid
                )
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
        assert runtime["model-forward"] > 0
        assert runtime["sampling-kernel"] > 0
        assert runtime["metadata-consumed"] > 0
        assert runtime["metadata-compactions"] > 0
        assert not controller._async_sched_logits.is_valid
        assert context.total_request_count == context.active_token_count == 0

        signals = set(scenario.signals)
        executed_graph_dimensions = {
            key[1:]
            for key, count in runtime.items()
            if isinstance(key, tuple) and key[0] == "cuda-graph-dimension" and count > 0
        }
        if "eager" in signals:
            assert runtime["cuda_graph_steps"] == 0
            assert not context.cuda_graph_batch_dimensions_list
        if "gpt" in signals:
            assert runtime["module-forward:gpt"] > 0
        if "torch-backend" in signals:
            assert runtime["sampling-backend:torch"] > 0
        if "persist" in signals:
            assert context.kv_cache_management_mode == KVCacheManagementMode.PERSIST
        if "static-pointers" in signals:
            assert context.static_kv_memory_pointers
        if "suspend-resume" in signals:
            assert runtime["suspend-resume"] == runtime["static-pointer-preserved"] == 1
            assert (
                runtime["pending-before-suspend"]
                == runtime["pending-after-suspend"]
                == runtime["pending-after-resume"]
                == 1
            )
        if "dp-offset" in signals:
            assert context.config.offset_sampling_seed_by_dp_rank
        if "last-logits" in signals:
            assert context.config.materialize_only_last_token_logits
            assert runtime["last-logits-forward"] == runtime["model-forward"]
            assert runtime["logits-dtype:torch.bfloat16"] > 0
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
        if "capacity" in signals:
            assert context.max_requests == 4
            assert context.max_tokens == 8
            assert runtime["max_active"] == context.max_requests
        if "prefix-hit" in signals:
            assert env.engine._prefix_cache_hits > 0
            assert env.engine._prefill_tokens_skipped > 0
        if "ref-zero" in signals:
            assert context.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
            assert runtime["prefix-blocks-deregistered"] > 0
        if "cuda-graph" in signals:
            assert context.cuda_graph_batch_dimensions_list
            assert runtime["cuda_graph_steps"] > 0
            assert executed_graph_dimensions
            available_dimensions = {
                (dimensions.token_count, dimensions.prefill_req_count, dimensions.decode_req_count)
                for dimensions in context.cuda_graph_batch_dimensions_list
            }
            assert executed_graph_dimensions <= available_dimensions
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
        if "graph-bounded-config" in signals:
            assert context.use_cuda_graphs_for_non_decode_steps
            assert not context.config.cuda_graph_all_prefills
            assert context.config.cuda_graph_max_tokens == 16
            assert max(token_count for token_count, _, _ in executed_graph_dimensions) == 16
            assert runtime["cuda-graph-prefill-forwards"] > 0
        if "sampled" in signals:
            assert all(request.generated_tokens for request in env.requests)
        for sampling_filter in ("temperature-filter", "top-k-filter", "top-p-filter"):
            if sampling_filter in signals:
                assert runtime[sampling_filter] > 0
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
        if "metadata-compaction" in signals:
            assert runtime["non-prefix-survivor-compactions"] > 0
        if "hybrid" in signals:
            assert context.is_hybrid_model
            assert hasattr(context, "mamba_conv_states_shape")
            assert runtime["mamba-state-commits"] > 0
            assert runtime["mamba-state-compactions"] > 0
            assert runtime["module-forward:hybrid"] > 0
        if "transformer-engine" in signals:
            assert model_config.transformer_impl == "transformer_engine"
            assert runtime["module-forward:transformer-engine"] > 0
        if "inference-optimized" in signals:
            assert model_config.transformer_impl == "inference_optimized"
            assert runtime["module-forward:inference-optimized"] > 0
        if "fp8" in signals:
            assert model_config.fp8 is not None
            assert runtime["module-forward:fp8"] > 0
            assert runtime["fp8-context-forwards"] > 0
            assert runtime["fp8-quantized-forwards"] > 0
            assert runtime["fp8-recipe-forwards"] > 0
        if "fused-rope" in signals:
            assert context.use_flashinfer_fused_rope
            assert model_config.hidden_size // model_config.num_attention_heads == 16
            model = controller.inference_wrapped_model.model
            assert model.rotary_pos_emb.inv_freq.is_cuda
            assert model.rotary_pos_emb_cache[context.max_sequence_length].is_cuda
            assert runtime["fused-rope-kernel"] > 0
        if "swa-all" in signals:
            assert model_config.window_size == scenario.config["window_size"]
            assert runtime["swa-kernel-calls"] > 0
            assert runtime["full-attention-kernel-calls"] == 0
        if "swa-alternating" in signals:
            assert model_config.window_size == scenario.config["window_size"]
            assert runtime["swa-kernel-calls"] > 0
            assert runtime["full-attention-kernel-calls"] > 0
        if "softmax-sink" in signals:
            assert model_config.softmax_type == scenario.config["softmax_type"]
            assert runtime["softmax-offset-calls"] > 0
            assert runtime["softmax-offset-produced"] == runtime["softmax-offset-calls"]
            assert runtime["sink-correction-calls"] > 0
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
            assert runtime["module-forward:parallel"] > 0
            if model_config.tensor_model_parallel_size > 1:
                if model_config.transformer_impl == "inference_optimized":
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
                    assert runtime["tp-collective:reduce_scatter_to_sequence_parallel_region"] > 0
                    assert runtime["tp-sp-reduce-scatter-dimensions"] > 0
                elif not model_config.sequence_parallel:
                    assert runtime["tp-collective:reduce_from_tensor_model_parallel_region"] > 0
            if model_config.pipeline_model_parallel_size > 1:
                assert runtime["pipeline-logits-broadcasts"] > 0
        if "moe" in signals:
            assert model_config.num_moe_experts is not None
            assert context._nccl_ep_dispatcher
            # Inference MoE may dispatch through split execution helpers that
            # bypass nn.Module forward hooks. The live dispatcher calls below
            # are the feature-owning runtime boundary.
            assert runtime["nccl-token-dispatches"] > 0
        if "nccl-dispatch" in signals:
            assert model_config.expert_model_parallel_size > 1
            assert model_config.inference_moe_token_dispatcher_type == "nccl"
            assert runtime["nccl-dispatchers-installed"] > 0
            assert runtime["nccl-token-dispatches"] > 0
            assert runtime["nccl-token-dispatches"] == runtime["nccl-token-combines"]
            assert runtime["nccl-combine-before-dispatch"] == 0
            assert runtime["nccl-dispatch-inflight"] == 0

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
