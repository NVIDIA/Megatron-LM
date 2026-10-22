# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pairwise coverage for chunked prefill and dynamic-inference companions.

The feature oracle is an otherwise-identical one-shot prefill run.  A treatment
only receives credit when the real context admits the long request in at least
two non-final chunks and one of those admissions shares a step with decoding.
The observer deliberately wraps ``DynamicInferenceContext.add_request`` rather
than restating the scheduler's chunk arithmetic in the test.

Prompt-score sidecars and raw visible top-N parity are owned by NVIDIA/Megatron-LM
#7063, the base of this campaign, and are intentionally not repeated here.
Gated Delta Net is classified as unsupported rather than counted: its production
prefill path explicitly rejects chunked execution.
"""

import gc
import os
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Optional
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, PrefixCachingEvictionPolicy
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    Status,
)
from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer.attention import HAVE_FA3, HAVE_FA4
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    is_batch_invariant_mode_enabled,
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import (
    TestPrefixCacheRealEngineMatrix as _PrefixEngineHarness,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig as _DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import reset_rounder as _reset_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    skip_if_mamba_sequence_packing_not_available,
)
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _ASYNC_PAIR_SCENARIOS,
    _ASYNC_PARALLEL_SCENARIOS,
    _ASYNC_SUSPEND_RESUME_SCENARIOS,
    _assert_request_parity,
    _AsyncPairScenario,
    _AsyncPairwiseHarness,
    _check_scenario_prerequisite,
    _instrument_scenario_runtime,
    _snapshot_requests,
)
from tests.unit_tests.test_utilities import Utils

_BLOCK_SIZE = 256
_CHUNK_BUDGET = 128
_FULL_BUDGET = 1024
_PROMPT_LENGTH = 3 * _CHUNK_BUDGET + 17
_PREFIX_LENGTH = 2 * _BLOCK_SIZE
_PREFIX_FOLLOWER_LENGTH = _PREFIX_LENGTH + 300
_MAX_SEQUENCE_LENGTH = 896
_BATCH_INVARIANT_FA_VERSION = 4 if HAVE_FA4 else (3 if HAVE_FA3 else None)
_TARGET_ID = 1
_DONOR_ID = 10


def _inherited_scenario(name: str) -> _AsyncPairScenario:
    scenarios = (
        *_ASYNC_PAIR_SCENARIOS,
        *_ASYNC_SUSPEND_RESUME_SCENARIOS,
        *_ASYNC_PARALLEL_SCENARIOS,
    )
    return next(scenario for scenario in scenarios if scenario.name == name)


def _owned_scenario(
    source: str,
    name: str,
    *,
    config: Optional[dict[str, object]] = None,
    sampling: Optional[tuple[dict[str, object], ...]] = None,
    signals: tuple[str, ...] = (),
    prerequisite: Optional[str] = None,
    parity: Optional[str] = None,
    atol: Optional[float] = None,
) -> _AsyncPairScenario:
    """Reuse an async-campaign scenario while changing only this campaign's axis."""
    inherited = _inherited_scenario(source)
    return replace(
        inherited,
        name=name,
        config={**inherited.config, **(config or {})},
        sampling=inherited.sampling if sampling is None else sampling,
        signals=signals,
        prerequisite=prerequisite,
        parity=inherited.parity if parity is None else parity,
        # Chunking changes BF16 batch shapes; allow a few rounding units in scores.
        atol=max(inherited.atol, 5.0e-3) if atol is None else atol,
    )


@dataclass(frozen=True)
class _ChunkCase:
    """One chunked-prefill companion row and its test-only runtime oracle."""

    scenario: _AsyncPairScenario
    mode: AsyncScheduleMode = AsyncScheduleMode.ASYNC
    prefix_flow: Optional[str] = None
    suspend_mode: Optional[str] = None
    stop_and_events: bool = False
    repeated_treatment: bool = False
    congestion: bool = False
    chunk_budget: int = _CHUNK_BUDGET
    full_budget: int = _FULL_BUDGET
    chunk_counters: tuple[str, ...] = ("model-forward",)
    consumed_counters: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.scenario.name


_CASES = (
    _ChunkCase(
        _owned_scenario("chunked-capacity", "legacy-congestion", signals=("chunked", "gpt")),
        mode=AsyncScheduleMode.LEGACY,
        congestion=True,
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario("chunked-capacity", "async-overlap-congestion", signals=("chunked", "gpt")),
        congestion=True,
        chunk_counters=("module-forward:gpt", "_run_async_sched_step_no_overlap"),
    ),
    _ChunkCase(
        _owned_scenario(
            "prefix-ref-zero",
            "prefix-ref-zero-multichunk",
            signals=("chunked", "gpt", "prefix-hit", "ref-zero"),
        ),
        prefix_flow="ref-zero",
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "prefix-lru-chunked", "prefix-lru-multichunk", signals=("chunked", "gpt", "prefix-hit")
        ),
        prefix_flow="lru",
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "graph-mixed-layer-linear",
            "cuda-graph-mixed-phase",
            config={
                "cuda_graph_all_prefills": True,
                "cuda_graph_max_tokens": 2 * _BLOCK_SIZE,
                "cuda_graph_mixed_prefill_count": 1,
            },
            signals=("chunked", "cuda-graph", "gpt"),
        ),
        # all-prefills captures to the context ceiling; 512 fits the one-shot prompt.
        full_budget=2 * _BLOCK_SIZE,
        chunk_counters=("cuda-graph-mixed-forwards",),
    ),
    _ChunkCase(
        _owned_scenario(
            "hybrid-mamba",
            "mamba-batch-invariant",
            config={
                "batch_invariant_mode": True,
                "batch_invariant_backend": "triton",
                "flash_attention_version": _BATCH_INVARIANT_FA_VERSION,
            },
            signals=("chunked", "hybrid"),
            prerequisite="mamba",
        ),
        # A mixed 129-token row pads to the production 64-token boundary.
        chunk_budget=_CHUNK_BUDGET + 64,
        chunk_counters=("mamba-batch-invariant-prefills", "mamba-fp32-prefills"),
    ),
    _ChunkCase(
        _owned_scenario(
            "hybrid-mamba",
            "gdp-prefix-state",
            config={
                "enable_prefix_caching": True,
                "model_provider": "hybrid",
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.LRU,
                "ssm_mixer": "gdp",
            },
            signals=("chunked", "gdp", "hybrid", "prefix-hit"),
            # Use the prefix-cache oracle's absolute bound for recurrent cache scores.
            atol=5.0e-2,
        ),
        prefix_flow="lru",
        chunk_counters=("gdp-prefills", "mamba-restores"),
    ),
    _ChunkCase(
        _owned_scenario(
            "hybrid-mamba",
            "mamba-multichunk",
            signals=("chunked", "hybrid", "mamba"),
            prerequisite="mamba",
            parity="reproducible",
            atol=5.0e-3,
        ),
        repeated_treatment=True,
        chunk_counters=("mamba-prefills",),
    ),
    _ChunkCase(
        _owned_scenario(
            "mtp-depth-one",
            "mtp2-rejection",
            config={"num_speculative_tokens": 2},
            signals=("chunked", "gpt", "mtp"),
        ),
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "processed-skip-prompt-logprobs",
            "processed-hidden-flashinfer",
            config={"sampling_backend": "flashinfer", "materialize_only_last_token_logits": True},
            sampling=(
                {
                    "return_log_probs": True,
                    "skip_prompt_log_probs": True,
                    "temperature": 0.8,
                    "top_k": 12,
                    "top_p": 0.9,
                    "top_n_logprobs": 3,
                },
            ),
            signals=("chunked", "flashinfer", "gpt", "logprobs", "processed-logprobs"),
            prerequisite="flashinfer",
            parity="reproducible",
        ),
        repeated_treatment=True,
        chunk_counters=("module-forward:gpt",),
        consumed_counters=(
            "log-probs-calculations",
            "log-probs-mode:processed_logprobs",
            "log-probs-kernel",
            "sampling-backend:flashinfer",
            "temperature-filter",
            "top-k-filter",
            "top-p-filter",
        ),
    ),
    _ChunkCase(
        _owned_scenario(
            "stop-sequence-keep",
            "stop-and-events",
            config={"track_generated_token_events": True},
            signals=("chunked", "events", "gpt"),
        ),
        stop_and_events=True,
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "dense-eager-events",
            "suspend-persist",
            config={
                "kv_cache_management_mode": "persist",
                "static_kv_memory_pointers": True,
                "track_generated_token_events": False,
            },
            signals=("chunked", "gpt", "persist"),
        ),
        suspend_mode="persist",
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "offload-suspend-resume", "suspend-offload", signals=("chunked", "gpt", "offload")
        ),
        suspend_mode="offload",
        chunk_counters=("module-forward:gpt",),
    ),
    _ChunkCase(
        _owned_scenario(
            "fp8-transformer-engine",
            "te-fp8-fused-rope",
            config={
                "hidden_size": 256,
                "position_embedding_type": "rope",
                "transformer_impl": "transformer_engine",
                "use_flashinfer_fused_rope": True,
            },
            signals=("chunked", "fp8", "fused-rope", "transformer-engine"),
            prerequisite="fp8",
            parity="reproducible",
            atol=5.0e-3,
        ),
        repeated_treatment=True,
        chunk_counters=(
            "fp8-context-forwards",
            "fp8-quantized-forwards",
            "fp8-recipe-forwards",
            "fused-rope-kernel",
        ),
    ),
    _ChunkCase(
        _owned_scenario(
            "alternating-swa-learnable-sink",
            "swa-sink",
            config={"flash_attention_version": 3},
            signals=("chunked", "gpt", "softmax-sink", "swa-alternating"),
        ),
        chunk_counters=("swa-kernel-calls", "full-attention-kernel-calls", "sink-correction-calls"),
    ),
    _ChunkCase(
        _owned_scenario(
            "tp2-pp2-sp-dp2",
            "tp2-pp2-sp-dp2",
            config={"offset_sampling_seed_by_dp_rank": False},
            signals=("chunked", "parallel", "sampled"),
            parity="reproducible",
        ),
        repeated_treatment=True,
        chunk_counters=(
            "pipeline-logits-broadcasts",
            "tp-column-partition-forwards",
            "tp-row-partition-forwards",
            "tp-collective:sequence-parallel-all-gather",
            "tp-collective:reduce_scatter_to_sequence_parallel_region",
            "tp-sp-gather-dimensions",
            "tp-sp-reduce-scatter-dimensions",
        ),
        consumed_counters=("sampling-backend:torch", "temperature-filter", "top-k-filter"),
    ),
    _ChunkCase(
        _owned_scenario(
            "moe-ep2-nccl",
            "ep2-moe-optimized",
            signals=("chunked", "inference-optimized", "moe", "nccl-dispatch", "parallel"),
            parity="reproducible",
        ),
        repeated_treatment=True,
        chunk_counters=(
            "module-forward:inference-optimized",
            "nccl-token-dispatches",
            "nccl-token-combines",
        ),
    ),
)


@dataclass(frozen=True)
class _Admission:
    """Facts observed at one real ``DynamicInferenceContext.add_request`` call."""

    request_id: int
    finished_before: int
    remaining_before: int
    logical_span: int
    computed_tokens: int
    kv_offset: int
    cached_token_delta: int
    decode_requests_before: int
    partial: bool


@dataclass
class _Session:
    requests: list[DynamicInferenceRequest]
    admissions: list[_Admission]
    runtime: Counter
    prefix_hits: int
    prefill_tokens_skipped: int
    prefix_policy: PrefixCachingEvictionPolicy
    suspended: bool = False
    suspend_cursor_before: Optional[tuple[int, int]] = None
    suspend_cursor_after: Optional[tuple[int, int]] = None
    suspend_target_admission_count: Optional[int] = None
    suspend_chunked_id_before: Optional[int] = None
    suspend_chunked_id_after: Optional[int] = None
    suspend_storage_bytes: Optional[tuple[int, int, int]] = None
    suspend_storage_pointers: Optional[tuple[int, int, int]] = None


def _generated_event_count(request: DynamicInferenceRequest) -> int:
    return sum(event.type is DynamicInferenceEventType.GENERATED_TOKEN for event in request.events)


def _install_admission_observer(context, admissions: list[_Admission]) -> None:
    """Record the scheduler's chosen span and the context's executed row."""
    original = context.add_request

    def observed(request, prefill_chunk_length=None):
        finished_before = request.finished_chunk_token_count
        remaining_before = request.remaining_prompt_length
        logical_span = remaining_before if prefill_chunk_length is None else prefill_chunk_length
        cached_before = request.num_cached_tokens
        decode_before = context.num_decode_requests
        result = original(request, prefill_chunk_length=prefill_chunk_length)

        indexes = (
            context.request_ids[: context.total_request_count] == request.request_id
        ).nonzero()
        assert indexes.numel() == 1, f"request {request.request_id} has no unique context row"
        request_idx = int(indexes[0].item())
        admissions.append(
            _Admission(
                request_id=request.request_id,
                finished_before=finished_before,
                remaining_before=remaining_before,
                logical_span=logical_span,
                computed_tokens=int(context.request_query_lengths[request_idx].item()),
                kv_offset=int(context.request_kv_length_offsets[request_idx].item()),
                cached_token_delta=request.num_cached_tokens - cached_before,
                decode_requests_before=decode_before,
                partial=(
                    prefill_chunk_length is not None and prefill_chunk_length < remaining_before
                ),
            )
        )
        return result

    context.add_request = observed


def _install_chunk_forward_observer(env, partial_forward_ids: list[int]) -> None:
    """Identify the request whose non-final chunk entered each real forward."""
    context = env.engine.context
    controller = env.engine.controller
    original = controller._dynamic_step_forward_logits

    def observed(*args, **kwargs):
        request_id = context.chunked_prefill_request_id
        if request_id >= 0:
            partial_forward_ids.append(request_id)
        return original(*args, **kwargs)

    controller._dynamic_step_forward_logits = observed


def _install_consumed_chunk_observer(engine, consumed_chunk_ids: list[int]) -> None:
    """Record the chunk ID paired with each result at the bookkeeping boundary."""
    original = engine.post_process_requests

    def observed(*args, **kwargs):
        consumed_chunk_ids.append(kwargs["consumed_chunked_prefill_request_id"])
        return original(*args, **kwargs)

    engine.post_process_requests = observed


def _install_fa3_kvcache_witness(env, runtime: Counter) -> None:
    """Witness FA3's target decode and any trailing metadata it returns."""
    context = env.engine.context
    original_kvcache = attention_module.flash_attn3_with_kvcache

    def traced_kvcache(*args, **kwargs):
        active_ids = context.request_ids[context.paused_request_count : context.total_request_count]
        target_decode = context.is_decode_only() and _TARGET_ID in active_ids.tolist()
        result = original_kvcache(*args, **kwargs)
        if target_decode:
            assert kwargs.get("return_softmax_lse") is True
            assert isinstance(result, tuple) and len(result) >= 2
            runtime["fa3-target-decode-kvcache-calls"] += 1
            if len(result) > 2:
                runtime["fa3-target-decode-kvcache-metadata-calls"] += 1
        return result

    model = env.engine.controller.inference_wrapped_model.model
    for module in model.modules():
        if type(module).__name__ != "SelfAttention":
            continue
        original_flash = module.flash_decode_and_prefill

        def traced_flash(*args, _original=original_flash, **kwargs):
            with mock.patch.object(
                attention_module, "flash_attn3_with_kvcache", new=traced_kvcache
            ):
                return _original(*args, **kwargs)

        module.flash_decode_and_prefill = traced_flash


def _install_sequence_parallel_gather_witness(env, runtime: Counter) -> None:
    """Observe the SP all-gather in the grad-accumulation linear path."""
    from megatron.core.tensor_parallel import layers as tp_layers

    model = env.engine.controller.inference_wrapped_model.model
    for module in model.modules():
        if type(module).__name__ != "ColumnParallelLinear" or not module.sequence_parallel:
            continue
        tp_size = torch.distributed.get_world_size(module.tp_group)
        original_forward = module.forward

        def observed(*args, _original=original_forward, _tp_size=tp_size, **kwargs):
            original_collective = tp_layers.dist_all_gather_func

            def traced_collective(output, input_, *collective_args, **collective_kwargs):
                result = original_collective(output, input_, *collective_args, **collective_kwargs)
                runtime["tp-collective:sequence-parallel-all-gather"] += 1
                runtime["tp-sp-gather-dimensions"] += int(
                    output.shape[0] == input_.shape[0] * _tp_size
                )
                return result

            tp_layers.dist_all_gather_func = traced_collective
            try:
                return _original(*args, **kwargs)
            finally:
                tp_layers.dist_all_gather_func = original_collective

        module.forward = observed


def _install_recurrent_witnesses(env, runtime: Counter) -> None:
    """Count real Mamba/GDP/GDN prefill calls and recurrent-state restorations."""
    context = env.engine.context
    model = env.engine.controller.inference_wrapped_model.model
    for module in model.modules():
        class_name = type(module).__name__
        if class_name not in {"GatedDeltaProductMixer", "GatedDeltaNet", "MambaMixer"}:
            continue
        counter = {
            "GatedDeltaProductMixer": "gdp-prefills",
            "GatedDeltaNet": "gdn-prefills",
            "MambaMixer": "mamba-prefills",
        }[class_name]
        original = module.ssm_prefill

        def observed(*args, _class_name=class_name, _counter=counter, _original=original, **kwargs):
            runtime[_counter] += 1
            if _class_name == "MambaMixer":
                runtime["mamba-fp32-prefills"] += int(
                    context.mamba_ssm_states_dtype == torch.float32
                )
                runtime["mamba-batch-invariant-prefills"] += int(
                    context.batch_invariant_mode and is_batch_invariant_mode_enabled()
                )
            return _original(*args, **kwargs)

        module.ssm_prefill = observed

    allocator = context.mamba_slot_allocator
    if allocator is not None:
        original_restore = allocator.restore_to_live

        def observed_restore(*args, **kwargs):
            restored = original_restore(*args, **kwargs)
            runtime["mamba-restores"] += int(restored)
            return restored

        allocator.restore_to_live = observed_restore


def _request(
    env,
    case: _ChunkCase,
    request_id: int,
    prompt_tokens: torch.Tensor,
    *,
    output_length: int,
    stop_tokens: Optional[tuple[int, ...]] = None,
) -> DynamicInferenceRequest:
    config = env.config
    sampling_kwargs = {
        "num_tokens_to_generate": output_length,
        "termination_id": -1,
        "temperature": config.temperature,
        "top_k": config.top_k or 1,
        "top_p": config.top_p,
        "return_log_probs": config.return_log_probs,
        "skip_prompt_log_probs": config.skip_prompt_log_probs,
    }
    if case.scenario.sampling:
        sampling_kwargs.update(case.scenario.sampling[request_id % len(case.scenario.sampling)])
    if stop_tokens is not None and request_id == _TARGET_ID:
        sampling_kwargs.update(
            stop_words=[" ".join(str(token) for token in stop_tokens)],
            detokenize_stop_sequence=True,
        )
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(**sampling_kwargs),
        block_size_tokens=env.engine.context.block_size_tokens,
        enable_prefix_caching=env.engine.context.enable_prefix_caching,
    )


def _tokens(length: int, offset: int = 0) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.int64, device="cuda") + offset) % 97


def _config(case: _ChunkCase, chunked: bool) -> _DynamicEngineTestConfig:
    config = {
        "return_log_probs": True,
        "flash_attention_version": 3,
        "materialize_only_last_token_logits": False,
        **case.scenario.config,
    }
    config.update(
        num_requests=0,
        min_prompt_length=1,
        max_prompt_length=_PREFIX_FOLLOWER_LENGTH if case.prefix_flow else _PROMPT_LENGTH,
        num_tokens_to_generate=8,
        max_sequence_length=_MAX_SEQUENCE_LENGTH,
        context_buffer_size_gb=0.05,
        context_block_size_tokens=_BLOCK_SIZE,
        context_max_requests=4,
        context_max_tokens=case.chunk_budget if chunked else case.full_budget,
        enable_chunked_prefill=chunked,
        async_sched_mode=case.mode,
        num_gap_steps=0,
        top_k=1,
    )
    return _DynamicEngineTestConfig(**config)


def _build_env(case: _ChunkCase, chunked: bool):
    config = _config(case, chunked)
    builder = _PrefixEngineHarness if case.name == "gdp-prefix-state" else _AsyncPairwiseHarness
    env = builder._build_test_env(config)
    env.engine.controller.tokenizer.detokenize = lambda tokens, **_: (
        f"tok_{tokens[0]}" if tokens else ""
    )
    env.engine.controller.tokenizer.tokenize = lambda text: [int(token) for token in text.split()]
    env.engine.controller.tokenizer.bos = None
    if case.name == "te-fp8-fused-rope":
        # The CPU-initialized fixture needs its FlashInfer cos/sin source on CUDA.
        model = env.engine.controller.inference_wrapped_model.model
        model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.cuda()
        model.rotary_pos_emb_cache.clear()
    if case.name == "mamba-batch-invariant":
        _set_rounder(64)
        context = env.engine.context
        # Keep unused V finite; poisoned-page behavior has its own regression.
        context.memory_buffer[1].zero_()
        model_config = env.engine.controller.inference_wrapped_model.model.config
        assert context.batch_invariant_mode
        assert not context.enable_prefix_caching
        assert context.mamba_ssm_states_dtype == torch.float32
        assert model_config.flash_attention_version == _BATCH_INVARIANT_FA_VERSION
        assert model_config.attention_backend.name == "flash"
        assert model_config.batch_invariant_backend == "triton"
    return env


def _case_prerequisites(case: _ChunkCase) -> None:
    if case.name == "swa-sink" and not HAVE_FA3:
        pytest.skip("swa-sink requires FlashAttention 3")
    _check_scenario_prerequisite(case.scenario)
    if case.name == "gdp-prefix-state":
        skip_if_mamba_sequence_packing_not_available("hybrid", "gdp")
    elif case.name == "mamba-batch-invariant":
        if not te_supports_batch_invariant_attention() or _BATCH_INVARIANT_FA_VERSION is None:
            pytest.skip("batch-invariant Mamba needs TE support and FlashAttention 3 or 4")
    elif case.name == "te-fp8-fused-rope":
        pytest.importorskip("flashinfer")
        if not is_te_min_version("2.2.0"):
            pytest.skip("Transformer Engine 2.2.0 is required")
        fp8_available, reason = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason)


def _initialize_topology(case: _ChunkCase) -> None:
    config = case.scenario.config
    tp = int(config.get("tensor_model_parallel_size", 1))
    pp = int(config.get("pipeline_model_parallel_size", 1))
    ep = int(config.get("expert_model_parallel_size", 1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if case.name == "tp2-pp2-sp-dp2" and world_size != 8:
        pytest.skip("tp2-pp2-sp-dp2 requires exactly eight GPUs")
    if case.name == "ep2-moe-optimized" and world_size < 2:
        pytest.skip("ep2-moe-optimized requires at least two GPUs")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
    )
    assert parallel_state.get_tensor_model_parallel_world_size() == tp
    assert parallel_state.get_pipeline_model_parallel_world_size() == pp
    assert parallel_state.get_expert_model_parallel_world_size() == ep
    if case.name == "tp2-pp2-sp-dp2":
        assert parallel_state.get_data_parallel_world_size() == 2


def _cleanup() -> None:
    gc.collect()
    delete_cuda_graphs()
    VllmFusedMoeBuffers._delete_buffers()
    torch.cuda.empty_cache()


def _assert_contiguous_admissions(admissions: list[_Admission], expected_length: int) -> None:
    cursor = 0
    for admission in admissions:
        assert admission.finished_before == cursor
        cursor += admission.logical_span
    assert cursor == expected_length


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestChunkedPrefillPairwise(_AsyncPairwiseHarness):
    """Chunked prefill must preserve each exercised dynamic-inference contract."""

    @classmethod
    @torch.inference_mode()
    def _run_session(
        cls, case: _ChunkCase, *, chunked: bool, stop_tokens: Optional[tuple[int, ...]] = None
    ) -> _Session:
        manager = (
            set_batch_invariant_mode(True)
            if case.name == "mamba-batch-invariant"
            else nullcontext()
        )
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(
                deterministic or case.name == "mamba-batch-invariant"
            )
            with manager:
                return cls._run_session_impl(case, chunked=chunked, stop_tokens=stop_tokens)
        finally:
            torch.use_deterministic_algorithms(deterministic)

    @classmethod
    def _run_session_impl(
        cls, case: _ChunkCase, *, chunked: bool, stop_tokens: Optional[tuple[int, ...]] = None
    ) -> _Session:
        env = _build_env(case, chunked)
        engine = env.engine
        admissions: list[_Admission] = []
        runtime = Counter()
        partial_forward_ids: list[int] = []
        consumed_chunk_ids: list[int] = []
        finished: dict[int, DynamicInferenceRequest] = {}
        suspended = False
        suspend_before = None
        suspend_after = None
        suspend_target_admission_count = None
        suspend_chunked_id_before = None
        suspend_chunked_id_after = None
        suspend_storage_bytes = None
        suspend_storage_pointers = None
        target_decode_pending_from_no_overlap = False

        if chunked:
            _install_admission_observer(engine.context, admissions)
            _instrument_scenario_runtime(env, case.scenario, runtime)
            if case.name == "swa-sink":
                _install_fa3_kvcache_witness(env, runtime)
            if case.name == "tp2-pp2-sp-dp2":
                _install_sequence_parallel_gather_witness(env, runtime)
            _install_recurrent_witnesses(env, runtime)
            _install_chunk_forward_observer(env, partial_forward_ids)
            _install_consumed_chunk_observer(engine, consumed_chunk_ids)

        def live_request(request_id: int) -> Optional[DynamicInferenceRequest]:
            if request_id not in engine.requests:
                return None
            return engine.get_request(request_id)

        def add(request: DynamicInferenceRequest) -> None:
            engine._add_request(request)

        def memory_buffer_state() -> tuple[int, int]:
            memory_buffer = getattr(engine.context, "memory_buffer", None)
            if memory_buffer is None:
                return 0, 0
            return memory_buffer.untyped_storage().nbytes(), memory_buffer.data_ptr()

        def target_is_active() -> bool:
            active_ids = engine.context.request_ids[
                engine.context.paused_request_count : engine.context.total_request_count
            ]
            return _TARGET_ID in active_ids.tolist()

        def step() -> None:
            nonlocal suspended, suspend_before, suspend_after
            nonlocal suspend_target_admission_count
            nonlocal suspend_chunked_id_before, suspend_chunked_id_after
            nonlocal suspend_storage_bytes, suspend_storage_pointers
            nonlocal target_decode_pending_from_no_overlap
            runtime["max-waiting"] = max(runtime["max-waiting"], len(engine.waiting_request_ids))
            target_before = live_request(_TARGET_ID)
            generated_before = len(target_before.generated_tokens) if target_before else 0
            events_before = _generated_event_count(target_before) if target_before else 0
            target_active_before = target_before is not None and target_is_active()
            pending_handoff_before = target_decode_pending_from_no_overlap
            target_decode_pending_from_no_overlap = False
            partial_count_before = len(partial_forward_ids)
            consumed_count_before = len(consumed_chunk_ids)
            no_overlap_before = runtime["_run_async_sched_step_no_overlap"]
            overlap_before = runtime["_run_async_sched_step_overlap"]
            counter_before = {key: runtime[key] for key in case.chunk_counters}
            consumed_counter_before = {key: runtime[key] for key in case.consumed_counters}
            proposed_before = int(engine._spec_tokens_proposed_per_pos.sum())
            accepted_before = int(engine._spec_tokens_accepted_per_pos.sum())

            result = engine.step_modern()
            runtime["steps"] += 1
            new_partial_ids = partial_forward_ids[partial_count_before:]
            newly_consumed_chunk_ids = consumed_chunk_ids[consumed_count_before:]
            assert len(newly_consumed_chunk_ids) <= 1
            consumed_target_partial = _TARGET_ID in newly_consumed_chunk_ids
            if _TARGET_ID in new_partial_ids:
                target_after = live_request(_TARGET_ID)
                assert target_after is not None
                assert len(target_after.generated_tokens) == generated_before
                assert _generated_event_count(target_after) == events_before
                runtime["target-partial-no-output"] += 1
                for key in case.chunk_counters:
                    runtime[f"target-partial:{key}"] += runtime[key] - counter_before[key]

            if consumed_target_partial:
                target_after = live_request(_TARGET_ID)
                assert target_after is not None
                assert len(target_after.generated_tokens) == generated_before
                assert _generated_event_count(target_after) == events_before
                runtime["target-consumed-partial-no-output"] += 1
                runtime["target-consumed-partial-step:decode-mtp-proposed"] += (
                    int(engine._spec_tokens_proposed_per_pos.sum()) - proposed_before
                )
                runtime["target-consumed-partial-step:decode-mtp-accepted"] += (
                    int(engine._spec_tokens_accepted_per_pos.sum()) - accepted_before
                )
                for key in case.consumed_counters:
                    runtime[f"target-consumed-partial:{key}"] += (
                        runtime[key] - consumed_counter_before[key]
                    )

            target_after = live_request(_TARGET_ID)
            no_overlap_executed = runtime["_run_async_sched_step_no_overlap"] > no_overlap_before
            overlap_executed = runtime["_run_async_sched_step_overlap"] > overlap_before
            target_active_after = target_after is not None and target_is_active()
            if case.name == "async-overlap-congestion":
                if pending_handoff_before and overlap_executed:
                    assert target_active_before
                    assert target_before.remaining_prompt_length == 0
                    assert engine.decode_only.consumed is True
                    assert engine.decode_only.launched is True
                    assert target_active_after
                    assert target_after.remaining_prompt_length == 0
                    assert len(target_after.generated_tokens) > generated_before
                    runtime["target-no-overlap-to-overlap-handoff"] += 1
                if (
                    no_overlap_executed
                    and engine.decode_only.launched is True
                    and target_active_after
                    and target_after.remaining_prompt_length == 0
                    and runtime["target-partial:_run_async_sched_step_no_overlap"] > 0
                ):
                    target_decode_pending_from_no_overlap = True
                    runtime["target-decode-launched-by-no-overlap"] += 1

            if target_after is not None and len(target_after.generated_tokens) > generated_before:
                runtime["target-mtp-proposed"] += (
                    int(engine._spec_tokens_proposed_per_pos.sum()) - proposed_before
                )
                runtime["target-mtp-accepted"] += (
                    int(engine._spec_tokens_accepted_per_pos.sum()) - accepted_before
                )

            for request in result["finished_requests"]:
                finished[request.request_id] = request

            target = live_request(_TARGET_ID)
            target_partials = [
                admission
                for admission in admissions
                if admission.request_id == _TARGET_ID and admission.partial
            ]
            if (
                chunked
                and case.suspend_mode is not None
                and not suspended
                and len(target_partials) >= 2
                and target is not None
            ):
                suspend_before = (target.finished_chunk_token_count, target.remaining_prompt_length)
                suspend_target_admission_count = len(
                    [admission for admission in admissions if admission.request_id == _TARGET_ID]
                )
                suspend_chunked_id_before = engine.context.chunked_prefill_request_id
                storage_before, pointer_before = memory_buffer_state()
                engine.suspend()
                storage_suspended, pointer_suspended = memory_buffer_state()
                engine.resume()
                storage_resumed, pointer_resumed = memory_buffer_state()
                suspend_storage_bytes = (storage_before, storage_suspended, storage_resumed)
                suspend_storage_pointers = (pointer_before, pointer_suspended, pointer_resumed)
                suspend_chunked_id_after = engine.context.chunked_prefill_request_id
                target = live_request(_TARGET_ID)
                assert target is not None
                suspend_after = (target.finished_chunk_token_count, target.remaining_prompt_length)
                suspended = True

        short = _request(env, case, 0, _tokens(12, 7), output_length=12)
        if case.prefix_flow:
            shared = _tokens(_PREFIX_LENGTH, 11)
            donor_prompt = torch.cat((shared, _tokens(5, 31)))
            follower_prompt = torch.cat((shared, _tokens(300, 53)))
            donor = _request(env, case, _DONOR_ID, donor_prompt, output_length=12)
            target = _request(
                env, case, _TARGET_ID, follower_prompt, output_length=8, stop_tokens=stop_tokens
            )
            add(donor)
            if case.prefix_flow == "ref-zero":
                while live_request(_DONOR_ID).remaining_prompt_length > 0:
                    step()
            else:
                while _DONOR_ID not in finished:
                    step()
                add(short)
                step()
            add(target)
            if case.prefix_flow == "ref-zero":
                add(short)
        else:
            target = _request(
                env,
                case,
                _TARGET_ID,
                _tokens(_PROMPT_LENGTH, 41),
                output_length=8,
                stop_tokens=stop_tokens,
            )
            add(short)
            step()
            add(target)

        tail_request_count = 4 if case.congestion else 1
        for tail_index in range(tail_request_count):
            request_id = 2 + tail_index
            add(_request(env, case, request_id, _tokens(19, 73 + 7 * tail_index), output_length=6))
        while engine.has_unfinished_requests():
            step()
            assert runtime["steps"] < 256, f"{case.name} did not converge"

        assert engine.context.total_request_count == 0
        assert engine.context.active_token_count == 0
        requests = [finished[request_id] for request_id in sorted(finished)]
        for request in requests:
            # Check both runs before snapshot/parity can turn missing scores into [].
            assert request.generated_tokens
            if request.sampling_params.return_log_probs:
                assert request.generated_log_probs is not None
                assert len(request.generated_log_probs) == len(request.generated_tokens)
                if request.sampling_params.skip_prompt_log_probs:
                    assert not request.prompt_log_probs
                else:
                    assert request.prompt_log_probs is not None
                    assert len(request.prompt_log_probs) == len(request.prompt_tokens) - 1 > 0
            else:
                assert not request.prompt_log_probs and not request.generated_log_probs
        if case.name == "tp2-pp2-sp-dp2":
            assert not engine.context.config.offset_sampling_seed_by_dp_rank
            payload = torch.tensor(
                [
                    value
                    for request in requests
                    for value in (
                        request.request_id,
                        len(request.generated_tokens),
                        *request.generated_tokens,
                    )
                ],
                dtype=torch.int64,
                device="cuda",
            )
            peers = [
                torch.empty_like(payload)
                for _ in range(torch.distributed.get_world_size(engine.controller.dp_group))
            ]
            torch.distributed.all_gather(peers, payload, group=engine.controller.dp_group)
            assert all(torch.equal(peer, payload) for peer in peers)
            runtime["dp-shared-seed-equality"] += 1
        session = _Session(
            requests=requests,
            admissions=admissions,
            runtime=runtime,
            prefix_hits=engine._prefix_cache_hits,
            prefill_tokens_skipped=engine._prefill_tokens_skipped,
            prefix_policy=engine.context.prefix_caching_eviction_policy,
            suspended=suspended,
            suspend_cursor_before=suspend_before,
            suspend_cursor_after=suspend_after,
            suspend_target_admission_count=suspend_target_admission_count,
            suspend_chunked_id_before=suspend_chunked_id_before,
            suspend_chunked_id_after=suspend_chunked_id_after,
            suspend_storage_bytes=suspend_storage_bytes,
            suspend_storage_pointers=suspend_storage_pointers,
        )
        del env
        return session

    @staticmethod
    def _assert_treatment(case: _ChunkCase, session: _Session) -> None:
        target_admissions = [
            admission for admission in session.admissions if admission.request_id == _TARGET_ID
        ]
        partials = [admission for admission in target_admissions if admission.partial]
        assert len(partials) >= 2, f"{case.name} did not execute two non-final chunks"
        assert any(
            admission.decode_requests_before > 0 for admission in partials
        ), f"{case.name} never mixed a partial prefill with decode"
        assert session.runtime["target-partial-no-output"] >= 2
        assert all(
            0 < admission.computed_tokens <= admission.logical_span for admission in partials
        )
        assert all(admission.kv_offset >= admission.finished_before for admission in partials)
        for key in case.chunk_counters:
            assert (
                session.runtime[f"target-partial:{key}"] > 0
            ), f"{case.name} did not execute {key} in a target partial-prefill step"
        for key in case.consumed_counters:
            assert (
                session.runtime[f"target-consumed-partial:{key}"] > 0
            ), f"{case.name} did not consume {key} from a target partial-prefill step"
        if case.consumed_counters or case.name == "mtp2-rejection":
            assert session.runtime["target-consumed-partial-no-output"] >= 2

        if case.congestion:
            assert session.runtime["max-waiting"] > 4
        if case.name == "legacy-congestion":
            assert session.runtime["_run_async_sched_step_no_overlap"] == 0
            assert session.runtime["_run_async_sched_step_overlap"] == 0
        elif case.name == "async-overlap-congestion":
            assert session.runtime["target-decode-launched-by-no-overlap"] > 0
            assert session.runtime["target-no-overlap-to-overlap-handoff"] > 0

        if case.suspend_mode is None:
            expected_length = _PREFIX_FOLLOWER_LENGTH if case.prefix_flow else _PROMPT_LENGTH
            _assert_contiguous_admissions(target_admissions, expected_length)

        if case.prefix_flow:
            assert session.prefix_hits > 0
            assert session.prefill_tokens_skipped > 0
            assert any(admission.cached_token_delta > 0 for admission in partials)
            if case.prefix_flow == "ref-zero":
                assert session.prefix_policy == PrefixCachingEvictionPolicy.REF_ZERO
                assert session.runtime["prefix-blocks-deregistered"] > 0
            else:
                assert session.prefix_policy == PrefixCachingEvictionPolicy.LRU

        if case.name == "mtp2-rejection":
            assert session.runtime["target-mtp-proposed"] > 0
            assert session.runtime["target-mtp-accepted"] < session.runtime["target-mtp-proposed"]
            assert session.runtime["target-consumed-partial-step:decode-mtp-proposed"] > 0
            assert (
                session.runtime["target-consumed-partial-step:decode-mtp-accepted"]
                < session.runtime["target-consumed-partial-step:decode-mtp-proposed"]
            )

        if case.name == "processed-hidden-flashinfer":
            for request in session.requests:
                assert not request.prompt_log_probs
                assert not request.prompt_top_n_logprobs
                assert request.generated_log_probs is not None
                assert len(request.generated_log_probs) == len(request.generated_tokens)
                assert request.generated_top_n_logprobs is not None
                assert len(request.generated_top_n_logprobs) == len(request.generated_tokens)

        if case.name == "tp2-pp2-sp-dp2":
            assert session.runtime["dp-shared-seed-equality"] == 1
        if case.name == "swa-sink":
            assert session.runtime["fa3-target-decode-kvcache-calls"] > 0

        if case.name == "ep2-moe-optimized":
            assert (
                session.runtime["nccl-token-dispatches"] == session.runtime["nccl-token-combines"]
            )
            assert session.runtime["nccl-combine-before-dispatch"] == 0
            assert session.runtime["nccl-dispatch-inflight"] == 0

        if case.suspend_mode:
            assert session.suspended
            assert session.suspend_cursor_before is not None
            assert session.suspend_cursor_after is not None
            assert session.suspend_target_admission_count is not None
            assert session.suspend_chunked_id_before == _TARGET_ID
            assert session.suspend_storage_bytes is not None
            assert session.suspend_storage_pointers is not None
            storage_before, storage_suspended, storage_resumed = session.suspend_storage_bytes
            pointer_before, pointer_suspended, pointer_resumed = session.suspend_storage_pointers
            assert storage_before > 0 and storage_resumed == storage_before
            if case.suspend_mode == "recompute":
                assert storage_suspended == 0 and pointer_suspended == 0
                assert session.suspend_cursor_after == (0, _PROMPT_LENGTH)
                assert session.suspend_chunked_id_after == -1
                _assert_contiguous_admissions(
                    target_admissions[: session.suspend_target_admission_count],
                    session.suspend_cursor_before[0],
                )
                _assert_contiguous_admissions(
                    target_admissions[session.suspend_target_admission_count :], _PROMPT_LENGTH
                )
            elif case.suspend_mode == "offload":
                assert storage_suspended == 0 and pointer_suspended == 0
                assert pointer_before != 0 and pointer_resumed != 0
                assert session.suspend_cursor_after == session.suspend_cursor_before
                assert session.suspend_chunked_id_after == session.suspend_chunked_id_before
                _assert_contiguous_admissions(target_admissions, _PROMPT_LENGTH)
            else:
                assert storage_suspended == storage_before
                assert (pointer_before, pointer_suspended, pointer_resumed) == (
                    pointer_before,
                    pointer_before,
                    pointer_before,
                )
                assert session.suspend_cursor_after == session.suspend_cursor_before
                assert session.suspend_chunked_id_after == session.suspend_chunked_id_before
                _assert_contiguous_admissions(target_admissions, _PROMPT_LENGTH)

        for request in session.requests:
            assert request.status == Status.COMPLETED
            generated_event_count = _generated_event_count(request)
            if case.stop_and_events:
                assert generated_event_count == len(request.generated_tokens)
            else:
                assert generated_event_count == 0

    @staticmethod
    def _assert_stop_oracle(baseline, treatment, stop_tokens: tuple[int, ...]) -> None:
        baseline_by_id = {
            request_id: request for request_id, request in zip(baseline[0], baseline[1])
        }
        treatment_by_id = {request.request_id: request for request in treatment.requests}
        for request_id, reference in baseline_by_id.items():
            actual = treatment_by_id[request_id]
            if request_id != _TARGET_ID:
                assert actual.generated_tokens == reference["tokens"]
                continue
            tokens = reference["tokens"]
            stop_end = next(
                index + len(stop_tokens)
                for index in range(len(tokens) - len(stop_tokens) + 1)
                if tuple(tokens[index : index + len(stop_tokens)]) == stop_tokens
            )
            assert actual.generated_tokens == tokens[:stop_end]
            event_types = [event.type for event in actual.events]
            assert event_types.count(DynamicInferenceEventType.GENERATED_TOKEN) == stop_end
            assert event_types.count(DynamicInferenceEventType.FINISH) == 1
            assert [
                event_type
                for event_type in event_types
                if event_type
                in (DynamicInferenceEventType.GENERATED_TOKEN, DynamicInferenceEventType.FINISH)
            ] == [DynamicInferenceEventType.GENERATED_TOKEN] * stop_end + [
                DynamicInferenceEventType.FINISH
            ]

    @pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
    @torch.inference_mode()
    def test_chunked_prefill_matches_one_shot(self, case: _ChunkCase) -> None:
        _case_prerequisites(case)
        _initialize_topology(case)
        try:
            baseline = self._run_session(case, chunked=False)
            baseline_ids = [request.request_id for request in baseline.requests]
            baseline_snapshot = _snapshot_requests(baseline.requests)
            stop_tokens = None
            if case.stop_and_events:
                target = baseline_snapshot[baseline_ids.index(_TARGET_ID)]
                assert len(target["tokens"]) >= 4
                stop_tokens = tuple(target["tokens"][2:4])
            del baseline
            _cleanup()

            treatment = self._run_session(case, chunked=True, stop_tokens=stop_tokens)
            self._assert_treatment(case, treatment)
            treatment_ids = [request.request_id for request in treatment.requests]
            assert treatment_ids == baseline_ids

            if case.stop_and_events:
                self._assert_stop_oracle((baseline_ids, baseline_snapshot), treatment, stop_tokens)
            else:
                _assert_request_parity(
                    treatment.requests,
                    baseline_snapshot,
                    case.scenario.atol,
                    exact_numerics=case.scenario.parity == "exact",
                    exact_top_n=False,
                )

            if (
                case.name == "swa-sink"
                and treatment.runtime["fa3-target-decode-kvcache-metadata-calls"] == 0
            ):
                pytest.skip("installed FlashAttention 3 returns no trailing KV-cache metadata")

            if case.repeated_treatment:
                expected = _snapshot_requests(treatment.requests)
                del treatment
                _cleanup()
                repeat = self._run_session(case, chunked=True, stop_tokens=stop_tokens)
                self._assert_treatment(case, repeat)
                _assert_request_parity(
                    repeat.requests,
                    expected,
                    case.scenario.atol,
                    exact_numerics=True,
                    exact_top_n=True,
                )
        finally:
            _cleanup()
            _reset_rounder()
            Utils.destroy_model_parallel()

    @pytest.mark.parametrize("later_length", [128, 384], ids=["bypass", "defer"])
    @torch.inference_mode()
    def test_prefix_conflict_bypass_preserves_outputs(self, later_length):
        case = replace(
            next(case for case in _CASES if case.name == "prefix-lru-multichunk"), chunk_budget=640
        )
        _initialize_topology(case)
        try:
            for chunked in (False, True):
                env = _build_env(case, chunked)
                admissions = []
                _install_admission_observer(env.engine.context, admissions)
                for request_id, length in enumerate((512, 512, later_length), start=1):
                    offset = 73 if request_id == 3 else 41
                    env.engine._add_request(
                        _request(env, case, request_id, _tokens(length, offset), output_length=4)
                    )
                completed = []
                for step in range(32):
                    result = env.engine.step_modern()
                    completed.extend(result["finished_requests"])
                    if chunked and step == 0:
                        assert [a.request_id for a in admissions] == (
                            [1, 3] if later_length == 128 else [1]
                        )
                        assert list(env.engine.waiting_request_ids) == (
                            [2] if later_length == 128 else [2, 3]
                        )
                        assert env.engine.context.chunked_prefill_request_id == -1
                    if not env.engine.has_unfinished_requests():
                        break
                assert len(completed) == 3
                completed.sort(key=lambda request: request.request_id)
                for request in completed:
                    assert len(request.generated_tokens) == len(request.generated_log_probs) == 4
                    assert len(request.prompt_log_probs) == len(request.prompt_tokens) - 1
                if chunked:
                    _assert_request_parity(
                        completed, reference, case.scenario.atol, exact_top_n=False
                    )
                    assert admissions[1].request_id == (3 if later_length == 128 else 2)
                else:
                    reference = _snapshot_requests(completed)
                del env
                _cleanup()
        finally:
            _cleanup()
            _reset_rounder()
            Utils.destroy_model_parallel()
