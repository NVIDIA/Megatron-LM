# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pairwise coverage for MTP speculative decoding in dynamic inference.

The core oracle compares ordinary decoding with MTP decoding from models that
have identical weights and the same dormant MTP modules.  Deterministic rows
still call the real base model and real MTP layer before replacing only their
logits, which makes acceptance and rejection exact without turning the model
path into a mock.  Runtime witnesses observe proposal generation, verification,
request-local acceptance, KV rewind, and (for hybrid models) the exact recurrent
state selected by the rewind.

Async scheduling, chunked prefill, CUDA graphs, and the prefix-cache REF_ZERO
policy already have runnable MTP rows in their owning campaigns.  This module
does not repeat them; its cache row adds the missing LRU pressure cell.
"""

import gc
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch
from transformer_engine.pytorch import RMSNorm as TERMSNorm
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.mtp_metadata import MTPForwardMode
from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tensor_parallel import InferenceColumnParallelLinear
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
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase as _DynamicEngineTestBase,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    skip_if_mamba_sequence_packing_not_available,
)
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _assert_request_parity,
    _AsyncPairScenario,
    _instrument_attention_runtime,
    _instrument_scenario_runtime,
    _snapshot_requests,
)
from tests.unit_tests.test_utilities import Utils

_VOCAB_SIZE = 100
_BLOCK_SIZE = 256
_BASE_TOKEN = 7
_REJECT_TOKEN = 8
_BATCH_INVARIANT_FA_VERSION = 4 if HAVE_FA4 else (3 if HAVE_FA3 else None)


@dataclass(frozen=True)
class _Case:
    """One MTP interaction and the exact treatment pattern used as its oracle."""

    name: str
    depth: int = 2
    pattern: str = "accept"
    repeated: bool = False
    prompt_lengths: tuple[int, ...] = (5, 9)
    output_lengths: tuple[int, ...] = (9, 8)
    config: dict[str, object] = field(default_factory=dict)
    signals: tuple[str, ...] = ("gpt",)
    prerequisite: Optional[str] = None
    exact_top_n: bool = True
    atol: float = 1.0e-3


@dataclass
class _MTPWitness:
    """Request-local evidence collected from the real speculative path."""

    accepted_by_request: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    acceptance_batches: list[dict[int, int]] = field(default_factory=list)
    proposal_request_ids: list[tuple[int, ...]] = field(default_factory=list)
    layer_request_ids: list[tuple[int, ...]] = field(default_factory=list)
    raw_depths: list[Optional[int]] = field(default_factory=list)
    logical_depths: list[int] = field(default_factory=list)
    rewinds: dict[int, list[tuple[int, int, int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class _Session:
    requests: list
    runtime: Counter
    witness: _MTPWitness
    proposed: tuple[int, ...]
    accepted: tuple[int, ...]
    suspend_state: dict[str, object] = field(default_factory=dict)


def _scenario(case: _Case) -> _AsyncPairScenario:
    """Adapt a row to the established dynamic-engine runtime instrumentation."""
    return _AsyncPairScenario(
        name=case.name,
        pairs=(f"speculation:{case.name}",),
        config=case.config,
        signals=case.signals,
        prerequisite=case.prerequisite,
        atol=case.atol,
    )


def _config(case: _Case, *, mtp_active: bool) -> _DynamicEngineTestConfig:
    """Build equal-model controls while varying only active speculation."""
    values = {
        "num_requests": 0,
        "min_prompt_length": min(case.prompt_lengths),
        "max_prompt_length": max(case.prompt_lengths),
        "num_tokens_to_generate": max(case.output_lengths),
        "max_sequence_length": max(case.prompt_lengths) + max(case.output_lengths) + case.depth + 4,
        "context_buffer_size_gb": 0.04,
        "context_block_size_tokens": _BLOCK_SIZE,
        "context_max_requests": max(4, len(case.prompt_lengths) + 1),
        "context_max_tokens": max(64, sum(case.prompt_lengths) + len(case.prompt_lengths) * 8),
        "num_speculative_tokens": case.depth if mtp_active else 0,
        "mtp_num_layers": 1 if case.repeated else case.depth,
        "mtp_use_repeated_layer": case.repeated,
        "materialize_only_last_token_logits": False,
        "position_embedding_type": "rope",
        "use_flashinfer_fused_rope": False,
        "async_sched_mode": AsyncScheduleMode.LEGACY,
        "sampling_backend": "torch",
        "flash_attention_version": 3,
        "top_k": 1,
    }
    values.update(case.config)
    if values.get("model_provider", "gpt") == "gpt":
        # Cached repeated heads require no positions; separate heads still exercise RoPE.
        values["position_embedding_type"] = "none" if case.repeated else "rope"
        values["use_flashinfer_fused_rope"] = False
    return _DynamicEngineTestConfig(**values)


def _prompt(length: int, request_id: int) -> torch.Tensor:
    return (
        torch.arange(length, dtype=torch.int64, device=torch.cuda.current_device())
        + request_id * 17
    ) % (_VOCAB_SIZE - 1)


def _sampling(case: _Case, output_length: int) -> SamplingParams:
    kwargs = {
        "num_tokens_to_generate": output_length,
        "termination_id": -1,
        "return_log_probs": True,
        "skip_prompt_log_probs": True,
        "top_n_logprobs": 3,
        "temperature": float(case.config.get("temperature", 1.0)),
        "top_k": int(case.config.get("top_k", 1)),
        "top_p": float(case.config.get("top_p", 0.0)),
    }
    return SamplingParams(**kwargs)


def _set_logits(logits: torch.Tensor, tokens, *, distribution: bool = False) -> None:
    """Replace logits after a real layer call with an exact test distribution."""
    logits.fill_(-torch.inf)
    if distribution:
        logits[..., _BASE_TOKEN] = 3.0
        logits[..., _REJECT_TOKEN] = 2.5
        logits[..., _REJECT_TOKEN + 1] = 2.0
        return
    if isinstance(tokens, int):
        logits[..., tokens] = 100.0
        logits[..., (tokens + 1) % _VOCAB_SIZE] = 90.0
        logits[..., (tokens + 2) % _VOCAB_SIZE] = 80.0
        return
    for row, token in enumerate(tokens):
        logits[row, ..., token] = 100.0
        logits[row, ..., (token + 1) % _VOCAB_SIZE] = 90.0
        logits[row, ..., (token + 2) % _VOCAB_SIZE] = 80.0


def _active_request_ids(context) -> list[int]:
    active = slice(context.paused_request_count, context.total_request_count)
    return [int(value) for value in context.request_ids[active].tolist()]


def _install_mtp_witnesses(env, case: _Case, runtime: Counter) -> _MTPWitness:
    """Wrap real production calls and retain request-local speculative evidence."""
    witness = _MTPWitness()
    engine = env.engine
    context = engine.context
    controller = engine.controller
    model = controller.inference_wrapped_model.model

    _instrument_scenario_runtime(env, _scenario(case), runtime)
    if case.config.get("window_size") is not None:
        for module in model.modules():
            if type(module).__name__ == "SelfAttention":
                _instrument_attention_runtime(module, runtime)

    if hasattr(model, "mtp"):
        for layer in model.mtp.layers:
            real_layer_forward = layer.forward_single_position

            def observed_layer_forward(*args, _real=real_layer_forward, **kwargs):
                request_ids = tuple(_active_request_ids(context))
                mtp = context.mtp_metadata
                committing = (
                    context.enable_mtp_kv_cache and mtp.forward_mode is MTPForwardMode.COMMIT
                )
                result = _real(*args, **kwargs)
                if committing:
                    assert kwargs["inference_context"] is context and request_ids
                    runtime["real-mtp-kv-commit"] += 1
                elif context.enable_mtp_kv_cache and not runtime["mtp-proposal-inflight"]:
                    assert kwargs["inference_context"] is context
                    assert mtp.forward_mode is MTPForwardMode.DRAFT
                    assert (mtp.active_block_table == mtp.dummy_block_idx).all()
                    runtime["mtp-dummy-commit"] += 1
                else:
                    witness.layer_request_ids.append(request_ids)
                    runtime["real-mtp-layer-forward"] += 1
                return result

            layer.forward_single_position = observed_layer_forward

        for module in model.mtp.modules():
            class_name = type(module).__name__
            module_name = type(module).__module__
            if case.config.get("window_size") is not None and class_name == "SelfAttention":
                real_mtp_attention = module.core_attention.forward

                def observed_mtp_attention(
                    *args, _module=module, _real=real_mtp_attention, **kwargs
                ):
                    if _module.core_attention.scale_mask_softmax.window_size is not None:
                        runtime["mtp-swa-attention-forwards"] += 1
                    else:
                        runtime["mtp-full-attention-forwards"] += 1
                    return _real(*args, **kwargs)

                module.core_attention.forward = observed_mtp_attention
                runtime["mtp-attention-modules-installed"] += 1

            is_te_module = class_name.startswith("TE") or "transformer_engine" in module_name
            if "fp8" in case.signals and is_te_module:

                def observed_mtp_fp8(_module, _inputs):
                    fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
                    runtime["mtp-fp8-module-forwards"] += 1
                    runtime["mtp-fp8-context-forwards"] += int(fp8_enabled)
                    runtime["mtp-fp8-recipe-forwards"] += int(
                        fp8_enabled and FP8GlobalStateManager.get_fp8_recipe() is not None
                    )

                module.register_forward_pre_hook(observed_mtp_fp8)
                runtime["mtp-fp8-modules-installed"] += 1

            if "nccl-dispatch" in case.signals and class_name == "MoELayer":
                dispatcher = getattr(module, "_inference_token_dispatcher", None)
                if (
                    dispatcher is not None
                    and type(dispatcher).__name__ == "NCCLAllGatherDispatcher"
                ):
                    real_dispatch = dispatcher.token_dispatch
                    real_combine = dispatcher.token_combine

                    def observed_mtp_dispatch(*args, _real=real_dispatch, **kwargs):
                        result = _real(*args, **kwargs)
                        runtime["mtp-nccl-token-dispatches"] += 1
                        runtime["mtp-nccl-dispatch-inflight"] += 1
                        return result

                    def observed_mtp_combine(*args, _real=real_combine, **kwargs):
                        runtime["mtp-nccl-combine-before-dispatch"] += int(
                            runtime["mtp-nccl-dispatch-inflight"] <= 0
                        )
                        result = _real(*args, **kwargs)
                        runtime["mtp-nccl-token-combines"] += 1
                        runtime["mtp-nccl-dispatch-inflight"] -= 1
                        return result

                    dispatcher.token_dispatch = observed_mtp_dispatch
                    dispatcher.token_combine = observed_mtp_combine
                    runtime["mtp-nccl-dispatchers-installed"] += 1

            tp_size = int(case.config.get("tensor_model_parallel_size", 1))
            if tp_size > 1 and class_name == "ColumnParallelLinear":
                weight = getattr(module, "weight", None)
                output_size = getattr(module, "output_size", getattr(module, "out_features", None))
                assert weight is not None and weight.shape[0] * tp_size == output_size
                assert module.sequence_parallel
                real_parallel_forward = module.forward

                def observed_mtp_column(*args, _real=real_parallel_forward, **kwargs):
                    input_ = args[0] if args else kwargs["input_"]
                    output = _real(*args, **kwargs)
                    runtime["mtp-tp-sp-gather-dimensions"] += int(
                        output[0].shape[0] == input_.shape[0] * tp_size
                    )
                    runtime["mtp-tp-column-partition-forwards"] += 1
                    return output

                module.forward = observed_mtp_column
                runtime["mtp-tp-column-partitions-installed"] += 1

            if tp_size > 1 and class_name == "RowParallelLinear":
                weight = getattr(module, "weight", None)
                input_size = getattr(module, "input_size", getattr(module, "in_features", None))
                assert weight is not None and weight.shape[1] * tp_size == input_size
                assert module.sequence_parallel
                real_parallel_forward = module.forward

                def observed_mtp_row(*args, _real=real_parallel_forward, **kwargs):
                    input_ = args[0] if args else kwargs["input_"]
                    output = _real(*args, **kwargs)
                    runtime["mtp-tp-sp-reduce-scatter-dimensions"] += int(
                        output[0].shape[0] * tp_size == input_.shape[0]
                    )
                    runtime["mtp-tp-row-partition-forwards"] += 1
                    return output

                module.forward = observed_mtp_row
                runtime["mtp-tp-row-partitions-installed"] += 1

    real_serial_mtp = controller._compute_serial_mtp_and_sample

    def observed_serial_mtp(*args, **kwargs):
        sampling_keys = (
            "sampling-kernel",
            "temperature-filter",
            "top-k-filter",
            "top-p-filter",
            f"sampling-backend:{controller._sampling_backend}",
        )
        before = {key: runtime[key] for key in sampling_keys}
        mtp_module = None
        real_broadcast = None
        if controller.model_is_pipeline_parallel:
            from megatron.core.inference.text_generation_controllers import mtp_controller_mixin

            mtp_module = mtp_controller_mixin
            real_broadcast = mtp_module.broadcast_from_last_pipeline_stage

            def observed_mtp_broadcast(*broadcast_args, **broadcast_kwargs):
                result = real_broadcast(*broadcast_args, **broadcast_kwargs)
                runtime["mtp-pipeline-logits-broadcasts"] += 1
                return result

            mtp_module.broadcast_from_last_pipeline_stage = observed_mtp_broadcast
        try:
            result = real_serial_mtp(*args, **kwargs)
        finally:
            if mtp_module is not None:
                mtp_module.broadcast_from_last_pipeline_stage = real_broadcast
        runtime["mtp-serial-steps"] += 1
        for key in sampling_keys:
            runtime[f"mtp-local:{key}"] += runtime[key] - before[key]
        return result

    controller._compute_serial_mtp_and_sample = observed_serial_mtp

    real_forward = model.forward

    def observed_forward(*args, **kwargs):
        result = real_forward(*args, **kwargs)
        runtime["real-base-forward"] += 1
        runtime["base-forward-with-batch-invariant"] += int(
            context.batch_invariant_mode and is_batch_invariant_mode_enabled()
        )
        if result is None or not parallel_state.is_pipeline_last_stage():
            return result
        if case.pattern == "natural":
            return result
        if case.pattern == "distribution":
            _set_logits(result, _BASE_TOKEN, distribution=True)
        else:
            _set_logits(result, _BASE_TOKEN)
        return result

    model.forward = observed_forward

    if hasattr(model, "mtp"):
        real_mtp = model.compute_mtp_single_step
        repeated_call_index = 0

        def observed_mtp(
            hidden_states,
            next_token_ids,
            position_ids,
            depth=None,
            eager=False,
            cache_key=None,
            mtp_inference_context=None,
        ):
            nonlocal repeated_call_index
            layer_calls_before = len(witness.layer_request_ids)
            runtime["mtp-proposal-inflight"] += 1
            hidden_states, logits = real_mtp(
                hidden_states,
                next_token_ids,
                position_ids,
                depth,
                eager=eager,
                cache_key=cache_key,
                mtp_inference_context=mtp_inference_context,
            )
            runtime["mtp-proposal-inflight"] -= 1
            logical_depth = int(depth) if depth is not None else repeated_call_index % case.depth
            repeated_call_index += 1
            request_ids = _active_request_ids(context)
            assert len(witness.layer_request_ids) == layer_calls_before + 1
            assert witness.layer_request_ids[-1] == tuple(request_ids)
            runtime["real-mtp-forward"] += 1
            runtime["mtp-forward-with-batch-invariant"] += int(
                context.batch_invariant_mode and is_batch_invariant_mode_enabled()
            )
            runtime["mtp-position-id-forwards"] += int(
                position_ids.dtype == torch.int64
                and bool((position_ids[:, : len(request_ids)] >= 0).all())
            )
            witness.raw_depths.append(depth)
            witness.logical_depths.append(logical_depth)
            witness.proposal_request_ids.append(tuple(request_ids))

            if case.pattern == "natural":
                return hidden_states, logits
            if case.pattern == "distribution":
                _set_logits(logits, _BASE_TOKEN, distribution=True)
                return hidden_states, logits
            if case.pattern == "accept":
                tokens = [_BASE_TOKEN] * logits.shape[0]
            elif case.pattern == "reject":
                tokens = [_REJECT_TOKEN] * logits.shape[0]
            elif case.pattern == "partial":
                tokens = [_BASE_TOKEN if logical_depth == 0 else _REJECT_TOKEN] * logits.shape[0]
            else:
                assert case.pattern == "heterogeneous"
                acceptance_caps = {0: case.depth, 1: 1, 2: 0}
                tokens = [
                    (
                        _BASE_TOKEN
                        if logical_depth < acceptance_caps.get(request_id, 0)
                        else _REJECT_TOKEN
                    )
                    for request_id in request_ids
                ]
                tokens.extend([_REJECT_TOKEN] * (logits.shape[0] - len(tokens)))
            _set_logits(logits, tokens)
            return hidden_states, logits

        model.compute_mtp_single_step = observed_mtp

    real_verify = controller._verify_speculative_tokens

    def observed_verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        runtime["mtp-verifier-calls"] += 1
        runtime["mtp-verified-slots"] += int(result[1].numel())
        return result

    controller._verify_speculative_tokens = observed_verify

    real_prepare = controller._prepare_speculative_tokens_for_next_forward_pass

    def observed_prepare(num_decode_requests, *args, **kwargs):
        result = real_prepare(num_decode_requests, *args, **kwargs)
        request_ids = _active_request_ids(context)[:num_decode_requests]
        counts = controller._accepted_token_counts_per_request[:num_decode_requests].tolist()
        batch = {request_id: int(count) for request_id, count in zip(request_ids, counts)}
        witness.acceptance_batches.append(batch)
        for request_id, count in batch.items():
            witness.accepted_by_request[request_id].append(count)
            runtime[("accepted-count", request_id, count)] += 1
        runtime["mtp-prepare-calls"] += 1
        return result

    controller._prepare_speculative_tokens_for_next_forward_pass = observed_prepare

    real_rewind = controller._rewind_kv_cache

    def observed_rewind(*args, **kwargs):
        request_ids = _active_request_ids(context)
        active_slice = slice(context.paused_request_count, context.total_request_count)
        before_offsets = context.request_last_kv_block_offset[active_slice].clone()
        before_blocks = context.request_kv_block_counts[active_slice].clone()
        prefill = context.request_in_prefill_status_tensor[active_slice].clone()
        accepted_counts = controller._accepted_token_counts_per_request[: len(request_ids)].clone()

        mamba_expected = []
        if context.is_hybrid_model:
            state_ids = context.mamba_metadata.request_to_mamba_state_idx[active_slice].tolist()
            for idx, (is_prefill, state_id, accepted_count) in enumerate(
                zip(prefill.tolist(), state_ids, accepted_counts.tolist())
            ):
                if not is_prefill:
                    mamba_expected.append(
                        (
                            idx,
                            state_id,
                            context.mamba_intermediate_conv_states[
                                :, state_id, accepted_count
                            ].clone(),
                            context.mamba_intermediate_ssm_states[
                                :, state_id, accepted_count
                            ].clone(),
                        )
                    )

        blocks_to_release, remove_mask = real_rewind(*args, **kwargs)
        after_offsets = context.request_last_kv_block_offset[active_slice]
        after_blocks = context.request_kv_block_counts[active_slice]
        for idx, request_id in enumerate(request_ids):
            witness.rewinds[request_id].append(
                (
                    int(before_offsets[idx]),
                    int(after_offsets[idx]),
                    int(before_blocks[idx]),
                    int(after_blocks[idx]),
                )
            )
        runtime["mtp-rewind-calls"] += 1
        runtime["rewind-released-blocks"] += int(remove_mask.sum())

        for _, state_id, expected_conv, expected_ssm in mamba_expected:
            assert torch.equal(context.mamba_conv_states[:, state_id], expected_conv)
            assert torch.equal(context.mamba_ssm_states[:, state_id], expected_ssm)
            runtime["mamba-selective-rewind-checks"] += 1
        return blocks_to_release, remove_mask

    controller._rewind_kv_cache = observed_rewind
    return witness


def _same_model_weights(ordinary_env, mtp_env) -> None:
    ordinary = ordinary_env.engine.controller.inference_wrapped_model.model.state_dict()
    treatment = mtp_env.engine.controller.inference_wrapped_model.model.state_dict()
    assert ordinary.keys() == treatment.keys()
    for name in ordinary:
        ordinary_value = ordinary[name]
        treatment_value = treatment[name]
        if ordinary_value is None or treatment_value is None:
            assert name.endswith("_extra_state"), name
            assert ordinary_value is None and treatment_value is None, name
            continue
        assert isinstance(ordinary_value, torch.Tensor), name
        assert isinstance(treatment_value, torch.Tensor), name
        assert ordinary_value.dtype == treatment_value.dtype, name
        assert ordinary_value.layout == treatment_value.layout, name
        assert ordinary_value.shape == treatment_value.shape, name
        assert torch.equal(ordinary_value, treatment_value), name


def _staged_request_tokens(engine) -> dict[int, tuple[int, ...]]:
    """Return the live base-plus-draft input row owned by each active request."""
    context = engine.context
    controller = engine.controller
    request_ids = _active_request_ids(context)
    tokens_per_request = context.num_speculative_tokens + 1
    expected_token_count = len(request_ids) * tokens_per_request
    assert context.active_token_count == expected_token_count
    grouped = context.token_to_input_ids[:expected_token_count].view(
        len(request_ids), tokens_per_request
    )

    sampled = torch.cat(
        (
            controller._sampled_tokens_cuda[: len(request_ids)].unsqueeze(0),
            controller._sampled_mtp_tokens_cuda[:, : len(request_ids)],
        ),
        dim=0,
    ).T.cpu()
    assert torch.equal(grouped, sampled)
    return {
        request_id: tuple(int(token) for token in row.tolist())
        for request_id, row in zip(request_ids, grouped)
    }


def _logical_generated_tokens(engine, request_ids) -> dict[int, tuple[int, ...]]:
    """Snapshot request output across recompute checkpoints, keyed by request ID."""
    return {
        request_id: tuple(engine.requests[request_id].record.merge().generated_tokens)
        for request_id in request_ids
    }


def _build_env(case: _Case, *, mtp_active: bool):
    env = _DynamicEngineTestBase._build_test_env(_config(case, mtp_active=mtp_active))
    model = env.engine.controller.inference_wrapped_model.model
    if case.config.get("model_provider", "gpt") == "gpt":
        assert model.position_embedding_type == ("none" if case.repeated else "rope")
        assert env.engine.context.enable_mtp_kv_cache is (case.repeated and mtp_active)
        assert not env.engine.context.use_flashinfer_fused_rope
    tokenizer = env.engine.controller.tokenizer
    tokenizer.bos = None
    tokenizer.tokenize = lambda text: [int(token) for token in text.split()]
    tokenizer.detokenize = lambda tokens, **_: " ".join(str(token) for token in tokens)
    if case.config.get("batch_invariant_mode"):
        # _build_test_env uses the small general-test rounder; BI kernels require 64.
        _set_rounder(64)
    return env


def _run_session(
    env, case: _Case, *, suspend_once: bool = False, request_order: Optional[tuple[int, ...]] = None
) -> _Session:
    """Run one real engine to completion, optionally suspending staged drafts."""
    runtime = Counter()
    witness = _install_mtp_witnesses(env, case, runtime)
    engine = env.engine
    order = request_order or tuple(range(len(case.prompt_lengths)))
    for request_id in order:
        engine.add_request(
            request_id=request_id,
            prompt=_prompt(case.prompt_lengths[request_id], request_id),
            sampling_params=_sampling(case, case.output_lengths[request_id]),
        )

    finished = {}
    suspended = False
    suspend_state = {}
    for step in range(256):
        result = engine.step_modern()
        runtime["engine-steps"] += 1
        for request in result["finished_requests"]:
            finished[request.request_id] = request
        if (
            suspended
            and "regenerated_staged" not in suspend_state
            and runtime["real-mtp-forward"] > suspend_state["mtp_calls_before_suspend"]
        ):
            regenerated_staged = _staged_request_tokens(engine)
            if regenerated_staged:
                suspend_state["regenerated_ids"] = tuple(regenerated_staged)
                suspend_state["regenerated_staged"] = regenerated_staged
        if suspend_once and not suspended and runtime["real-mtp-forward"] > 0:
            before_staged = _staged_request_tokens(engine)
            before_ids = tuple(before_staged)
            assert before_ids
            suspend_state["before_ids"] = before_ids
            suspend_state["before_staged"] = before_staged
            suspend_state["before_generated"] = _logical_generated_tokens(engine, before_ids)
            suspend_state["before_acceptance"] = {
                request_id: tuple(witness.accepted_by_request[request_id])
                for request_id in before_ids
            }
            suspend_state["mtp_calls_before_suspend"] = runtime["real-mtp-forward"]
            memory_buffer = getattr(engine.context, "memory_buffer", None)
            suspend_state["before_bytes"] = (
                memory_buffer.untyped_storage().nbytes() if memory_buffer is not None else 0
            )
            suspend_state["before_pointer"] = (
                memory_buffer.data_ptr() if memory_buffer is not None else 0
            )
            engine.suspend()
            suspended = True
            memory_buffer = getattr(engine.context, "memory_buffer", None)
            suspend_state["suspended_bytes"] = (
                memory_buffer.untyped_storage().nbytes() if memory_buffer is not None else 0
            )
            engine.resume()
            memory_buffer = getattr(engine.context, "memory_buffer", None)
            suspend_state["resumed_bytes"] = (
                memory_buffer.untyped_storage().nbytes() if memory_buffer is not None else 0
            )
            suspend_state["resumed_pointer"] = (
                memory_buffer.data_ptr() if memory_buffer is not None else 0
            )
            after_ids = tuple(_active_request_ids(engine.context))
            suspend_state["after_resume_ids"] = after_ids
            suspend_state["after_resume_staged"] = (
                _staged_request_tokens(engine) if after_ids else {}
            )
            suspend_state["after_resume_waiting_ids"] = tuple(engine.waiting_request_ids)
            suspend_state["after_resume_generated"] = _logical_generated_tokens(engine, before_ids)
            runtime["suspend-resume-cycles"] += 1
        if not engine.has_unfinished_requests():
            break
    else:
        pytest.fail(f"{case.name} did not converge")

    assert finished.keys() == set(range(len(case.prompt_lengths)))
    requests = [finished[request_id] for request_id in sorted(finished)]
    assert engine.context.total_request_count == 0
    assert engine.context.active_token_count == 0
    if suspend_once:
        suspend_state["final_acceptance"] = {
            request_id: tuple(witness.accepted_by_request[request_id])
            for request_id in range(len(case.prompt_lengths))
        }
    return _Session(
        requests=requests,
        runtime=runtime,
        witness=witness,
        proposed=tuple(int(value) for value in engine._spec_tokens_proposed_per_pos.tolist()),
        accepted=tuple(int(value) for value in engine._spec_tokens_accepted_per_pos.tolist()),
        suspend_state=suspend_state,
    )


def _assert_complete(session: _Session, case: _Case) -> None:
    for request, output_length in zip(session.requests, case.output_lengths):
        assert request.status.name == "COMPLETED"
        expected_length = output_length
        assert len(request.generated_tokens) == expected_length
        assert request.generated_log_probs is not None
        assert len(request.generated_log_probs) == expected_length
        assert torch.isfinite(torch.tensor(request.generated_log_probs)).all()
        assert request.generated_top_n_logprobs is not None
        assert len(request.generated_top_n_logprobs) == expected_length
        assert all(
            str(token) in top_n
            for token, top_n in zip(request.generated_tokens, request.generated_top_n_logprobs)
        )
        selected_top_n = [
            top_n[str(token)]
            for token, top_n in zip(request.generated_tokens, request.generated_top_n_logprobs)
        ]
        assert selected_top_n == pytest.approx(request.generated_log_probs, rel=0, abs=case.atol)
        if case.pattern not in {"natural", "distribution"}:
            assert all(token == _BASE_TOKEN for token in request.generated_tokens)


def _assert_mtp_active(session: _Session, case: _Case) -> None:
    runtime = session.runtime
    assert runtime["real-base-forward"] > 0
    if parallel_state.is_pipeline_last_stage():
        assert runtime["real-mtp-forward"] > 0
        assert bool(runtime["real-mtp-kv-commit"]) is case.repeated
        if case.repeated:
            assert (
                runtime["real-mtp-kv-commit"] + runtime["mtp-dummy-commit"]
                == runtime["mtp-serial-steps"]
            )
        assert runtime["real-mtp-layer-forward"] == runtime["real-mtp-forward"]
        assert runtime["mtp-position-id-forwards"] == runtime["real-mtp-forward"]
        assert session.witness.layer_request_ids == session.witness.proposal_request_ids
    else:
        assert runtime["real-mtp-forward"] == 0
    assert runtime["mtp-verifier-calls"] > 0
    assert runtime["mtp-prepare-calls"] == runtime["mtp-verifier-calls"]
    assert runtime["mtp-rewind-calls"] > 0
    assert sum(session.proposed) > 0
    assert len(session.proposed) == len(session.accepted) == case.depth
    assert all(
        0 <= accepted <= proposed for accepted, proposed in zip(session.accepted, session.proposed)
    )
    for request_id in range(len(case.prompt_lengths)):
        assert session.witness.accepted_by_request[request_id]
        assert session.witness.rewinds[request_id]

    expected = {"accept": case.depth, "reject": 0, "partial": 1}.get(case.pattern)
    if expected is not None:
        assert all(
            count == expected
            for counts in session.witness.accepted_by_request.values()
            for count in counts
        )


def _run_mtp_pair(case: _Case) -> tuple[_Session, _Session]:
    """Compare ordinary and speculative decoding with byte-identical weights."""
    ordinary_env = _build_env(case, mtp_active=False)
    mtp_env = _build_env(case, mtp_active=True)
    _same_model_weights(ordinary_env, mtp_env)
    ordinary_model = ordinary_env.engine.controller.inference_wrapped_model.model
    mtp_model = mtp_env.engine.controller.inference_wrapped_model.model
    if case.config.get("model_provider", "gpt") == "gpt":
        assert ordinary_model.position_embedding_type == ("none" if case.repeated else "rope")
        assert mtp_model.position_embedding_type == ordinary_model.position_embedding_type
        assert not ordinary_env.engine.context.use_flashinfer_fused_rope
        assert not mtp_env.engine.context.use_flashinfer_fused_rope
    assert mtp_env.engine.controller.num_mtp_depths == case.depth
    assert bool(getattr(mtp_model, "mtp_process", False)) == parallel_state.is_pipeline_last_stage()
    if mtp_model.mtp_process:
        assert hasattr(mtp_model, "mtp")
        assert mtp_model.mtp.mtp_use_repeated_layer is case.repeated
        assert len(mtp_model.mtp.layers) == (1 if case.repeated else case.depth)
        if case.config.get("transformer_impl") == "inference_optimized":
            for layer in mtp_model.mtp.layers:
                assert isinstance(layer.eh_proj, InferenceColumnParallelLinear), type(layer.eh_proj)
                for norm_name in ("enorm", "hnorm", "final_layernorm"):
                    norm = getattr(layer, norm_name)
                    assert isinstance(norm, TERMSNorm), (norm_name, type(norm))
    else:
        assert not hasattr(mtp_model, "mtp")

    ordinary = _run_session(ordinary_env, case)
    treatment = _run_session(mtp_env, case)
    _assert_complete(ordinary, case)
    _assert_complete(treatment, case)
    _assert_request_parity(
        treatment.requests,
        _snapshot_requests(ordinary.requests),
        case.atol,
        exact_top_n=case.exact_top_n,
    )
    assert ordinary.runtime["real-base-forward"] > 0
    assert ordinary.runtime["real-mtp-forward"] == 0
    assert not ordinary.proposed
    _assert_mtp_active(treatment, case)
    del ordinary_env, mtp_env
    return ordinary, treatment


def _cleanup() -> None:
    gc.collect()
    delete_cuda_graphs()
    VllmFusedMoeBuffers._delete_buffers()
    torch.cuda.empty_cache()


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestMTPPairwise(_DynamicEngineTestBase):
    """Single-topology MTP pair owners omitted from inherited campaigns."""

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
        _cleanup()
        _set_rounder(64)
        Utils.destroy_model_parallel()

    def teardown_method(self):
        _cleanup()

    @torch.inference_mode()
    def test_natural_depth_one_matches_ordinary_same_weight_rope(self):
        """An untouched real MTP head preserves greedy tokens and scores."""
        _, treatment = _run_mtp_pair(
            _Case(
                name="natural-depth1-rope",
                depth=1,
                pattern="natural",
                prompt_lengths=(5, 11),
                output_lengths=(8, 7),
                exact_top_n=False,
            )
        )
        assert treatment.witness.raw_depths and set(treatment.witness.raw_depths) == {0}

    @torch.inference_mode()
    def test_separate_depth_three_all_accept(self):
        """Each distinct real MTP layer contributes a fully accepted proposal."""
        case = _Case(name="separate-depth3-accept", depth=3, pattern="accept")
        _, treatment = _run_mtp_pair(case)
        assert set(treatment.witness.raw_depths) == {0, 1, 2}
        assert set(treatment.witness.logical_depths) == {0, 1, 2}
        assert all(value > 0 for value in treatment.accepted)

    @pytest.mark.parametrize("repeated", [False, True], ids=["separate", "cached-repeated"])
    @torch.inference_mode()
    def test_depth_three_request_local_acceptance_and_block_rewind(self, repeated):
        """Reject across a block boundary, retaining only cached lookahead ownership."""
        case = _Case(
            name="repeated-depth3-heterogeneous",
            depth=3,
            pattern="heterogeneous",
            repeated=repeated,
            prompt_lengths=(254, 254, 254),
            output_lengths=(9, 9, 9),
            config={"context_max_tokens": 1024},
        )
        _, treatment = _run_mtp_pair(case)
        assert set(treatment.witness.raw_depths) == ({None} if repeated else {0, 1, 2})
        assert set(treatment.witness.logical_depths) == {0, 1, 2}
        assert {0: 3, 1: 1, 2: 0} in treatment.witness.acceptance_batches
        assert all(count == 3 for count in treatment.witness.accepted_by_request[0])
        assert all(count == 1 for count in treatment.witness.accepted_by_request[1])
        assert all(count == 0 for count in treatment.witness.accepted_by_request[2])
        rewinds = treatment.witness.rewinds[2]
        assert (treatment.runtime["rewind-released-blocks"] == 0) is repeated
        assert any(before < after for before, after, _, _ in rewinds)
        if repeated:
            assert all(before == after for _, _, before, after in rewinds)
        else:
            assert any(before > after for _, _, before, after in rewinds)

    @torch.inference_mode()
    def test_mtp_lru_prefix_cache_pressure_matches_cache_off(self):
        """A hit-bearing MTP request survives real LRU exhaustion and eviction."""
        owner = _PrefixEngineHarness()
        owner._run_engine_case(
            {"name": "mtp2-lru", "feature": "mtp", "policy": PrefixCachingEvictionPolicy.LRU}
        )

    @pytest.mark.parametrize("mode", ["persist", "offload", "recompute"])
    @torch.inference_mode()
    def test_suspend_resume_preserves_or_rebuilds_request_owned_mtp_state(self, mode):
        """Each KV residency mode preserves output and stages the same drafts after resume."""
        case = _Case(
            name=f"suspend-{mode}",
            depth=2,
            pattern="partial",
            prompt_lengths=(5, 8),
            output_lengths=(12, 11),
            config={
                "kv_cache_management_mode": mode,
                "static_kv_memory_pointers": mode == "persist",
            },
        )
        uninterrupted_env = _build_env(case, mtp_active=True)
        suspended_env = _build_env(case, mtp_active=True)
        _same_model_weights(uninterrupted_env, suspended_env)
        uninterrupted = _run_session(uninterrupted_env, case)
        suspended = _run_session(suspended_env, case, suspend_once=True)
        _assert_request_parity(
            suspended.requests,
            _snapshot_requests(uninterrupted.requests),
            case.atol,
            exact_top_n=True,
        )
        _assert_mtp_active(uninterrupted, case)
        _assert_mtp_active(suspended, case)
        assert suspended.runtime["suspend-resume-cycles"] == 1
        state = suspended.suspend_state
        request_ids = tuple(range(len(case.prompt_lengths)))
        expected_staged = {
            request_id: (_BASE_TOKEN, _BASE_TOKEN, _REJECT_TOKEN) for request_id in request_ids
        }
        assert state["before_ids"] == request_ids
        assert state["before_staged"] == expected_staged
        assert state["before_generated"] == state["after_resume_generated"]
        assert set(state["regenerated_ids"]) == set(request_ids)
        assert state["regenerated_staged"] == expected_staged
        for request_id in request_ids:
            before_acceptance = state["before_acceptance"][request_id]
            final_acceptance = state["final_acceptance"][request_id]
            assert final_acceptance[: len(before_acceptance)] == before_acceptance
            assert len(final_acceptance) > len(before_acceptance)
        assert state["before_bytes"] == state["resumed_bytes"] > 0
        if mode == "persist":
            assert state["after_resume_ids"] == request_ids
            assert state["after_resume_staged"] == expected_staged
            assert not state["after_resume_waiting_ids"]
            assert state["suspended_bytes"] == state["before_bytes"]
            assert state["resumed_pointer"] == state["before_pointer"]
        elif mode == "offload":
            assert state["after_resume_ids"] == request_ids
            assert state["after_resume_staged"] == expected_staged
            assert not state["after_resume_waiting_ids"]
            assert state["suspended_bytes"] == 0
        else:
            assert state["after_resume_ids"] == ()
            assert state["after_resume_staged"] == {}
            assert set(state["after_resume_waiting_ids"]) == set(request_ids)
            assert state["suspended_bytes"] == 0

    @pytest.mark.parametrize("backend", ["torch", "flashinfer"])
    @torch.inference_mode()
    def test_sampling_filters_are_reproducible_with_mtp(self, backend):
        """MTP proposals use the selected sampler and its temperature/k/p filters."""
        if backend == "flashinfer":
            pytest.importorskip("flashinfer")
        active_filter = "top-k" if backend == "torch" else "top-p"
        inactive_filter = "top-p" if backend == "torch" else "top-k"
        case = _Case(
            name=f"sampling-{backend}",
            depth=2,
            pattern="distribution",
            prompt_lengths=(5, 7, 9),
            output_lengths=(12, 11, 10),
            config={
                "sampling_backend": backend,
                "temperature": 0.8,
                "top_k": 3 if backend == "torch" else 0,
                "top_p": 0.95 if backend == "flashinfer" else 0.0,
            },
        )
        first_env = _build_env(case, mtp_active=True)
        repeat_env = _build_env(case, mtp_active=True)
        _same_model_weights(first_env, repeat_env)
        first = _run_session(first_env, case)
        repeat = _run_session(repeat_env, case)
        _assert_request_parity(
            repeat.requests, _snapshot_requests(first.requests), case.atol, exact_top_n=True
        )
        for session in (first, repeat):
            _assert_mtp_active(session, case)
            assert session.runtime[f"sampling-backend:{backend}"] > 0
            assert session.runtime["temperature-filter"] > 0
            assert session.runtime[f"{active_filter}-filter"] > 0
            assert session.runtime[f"{inactive_filter}-filter"] == 0
            assert session.runtime["mtp-serial-steps"] > 0
            assert session.runtime[f"mtp-local:sampling-backend:{backend}"] > 0
            assert session.runtime["mtp-local:temperature-filter"] > 0
            assert session.runtime[f"mtp-local:{active_filter}-filter"] > 0
            assert session.runtime[f"mtp-local:{inactive_filter}-filter"] == 0
            assert all(
                token in {_BASE_TOKEN, _REJECT_TOKEN, _REJECT_TOKEN + 1}
                for request in session.requests
                for token in request.generated_tokens
            )

    @pytest.mark.parametrize("output_length", [5, 6])
    @torch.inference_mode()
    def test_length_remainder_trims_tokens_and_scores(self, output_length):
        """Non-step-aligned output lengths retain one score per visible token."""
        _run_mtp_pair(
            _Case(
                name=f"length-remainder-{output_length}",
                depth=2,
                pattern="accept",
                prompt_lengths=(5,),
                output_lengths=(output_length,),
            )
        )

    @torch.inference_mode()
    def test_hybrid_mamba_partial_acceptance_selects_exact_intermediate_state(self):
        """Hybrid rewind commits the recurrent state at each accepted depth."""
        skip_if_mamba_sequence_packing_not_available("mamba")
        case = _Case(
            name="hybrid-mamba-partial",
            depth=2,
            pattern="partial",
            prompt_lengths=(5, 8),
            output_lengths=(9, 8),
            config={"model_provider": "hybrid"},
            signals=("hybrid",),
            atol=5.0e-3,
        )
        _, treatment = _run_mtp_pair(case)
        assert treatment.runtime["module-forward:hybrid"] > 0
        assert treatment.runtime["mamba-selective-rewind-checks"] > 0

    @torch.inference_mode()
    def test_transformer_engine_fp8_executes_mtp_layers(self):
        """MTP and base TE modules both execute inside an enabled FP8 context."""
        if not is_te_min_version("2.2.0"):
            pytest.skip("Transformer Engine 2.2.0 is required")
        available, reason = check_fp8_support()
        if not available:
            pytest.skip(reason)
        case = _Case(
            name="te-fp8",
            depth=2,
            pattern="accept",
            config={"fp8": True, "hidden_size": 128},
            signals=("fp8", "transformer-engine"),
            atol=5.0e-3,
        )
        ordinary, treatment = _run_mtp_pair(case)
        assert ordinary.runtime["fp8-context-forwards"] > 0
        assert treatment.runtime["fp8-context-forwards"] > 0
        assert treatment.runtime["fp8-recipe-forwards"] > 0
        assert treatment.runtime["module-forward:transformer-engine"] > 0
        assert treatment.runtime["mtp-fp8-modules-installed"] > 0
        assert treatment.runtime["mtp-fp8-module-forwards"] > 0
        assert treatment.runtime["mtp-fp8-context-forwards"] > 0
        assert treatment.runtime["mtp-fp8-recipe-forwards"] > 0

    @torch.inference_mode()
    def test_dense_batch_invariant_mtp_is_exact_across_batch_composition(self):
        """A natural MTP request is bitwise invariant to neighbors and row order."""
        if _BATCH_INVARIANT_FA_VERSION is None or not te_supports_batch_invariant_attention():
            pytest.skip("batch-invariant attention needs TE support and FlashAttention 3 or 4")
        case_alone = _Case(
            name="dense-batch-invariant-alone",
            depth=2,
            pattern="natural",
            prompt_lengths=(9,),
            output_lengths=(8,),
            config={
                "batch_invariant_mode": True,
                "batch_invariant_backend": "triton",
                "flash_attention_version": _BATCH_INVARIANT_FA_VERSION,
                "context_max_tokens": 256,
            },
        )
        case_batch = _Case(
            name="dense-batch-invariant-neighbors",
            depth=2,
            pattern="natural",
            prompt_lengths=(9, 5, 13),
            output_lengths=(8, 7, 9),
            config=case_alone.config,
        )
        with set_batch_invariant_mode(True):
            alone_env = _build_env(case_alone, mtp_active=True)
            target_first_env = _build_env(case_batch, mtp_active=True)
            target_last_env = _build_env(case_batch, mtp_active=True)
            _same_model_weights(alone_env, target_first_env)
            _same_model_weights(alone_env, target_last_env)
            alone = _run_session(alone_env, case_alone)
            target_first = _run_session(target_first_env, case_batch)
            target_last = _run_session(target_last_env, case_batch, request_order=(1, 2, 0))
        for session in (alone, target_first, target_last):
            _assert_mtp_active(session, case_alone if session is alone else case_batch)
            assert session.runtime["base-forward-with-batch-invariant"] > 0
            assert session.runtime["mtp-forward-with-batch-invariant"] > 0
        reference = _snapshot_requests([alone.requests[0]])
        _assert_request_parity([target_first.requests[0]], reference, 0.0, exact_top_n=True)
        _assert_request_parity([target_last.requests[0]], reference, 0.0, exact_top_n=True)
        assert target_first.witness.accepted_by_request[0] == alone.witness.accepted_by_request[0]
        assert target_last.witness.accepted_by_request[0] == alone.witness.accepted_by_request[0]

    @torch.inference_mode()
    def test_alternating_swa_executes_with_mtp(self):
        """Both local-window and full-attention layers run in an MTP session."""
        case = _Case(
            name="alternating-swa",
            depth=2,
            pattern="accept",
            prompt_lengths=(9, 13),
            output_lengths=(9, 8),
            config={"window_size": (4, 0), "window_attn_skip_freq": 2},
        )
        ordinary, treatment = _run_mtp_pair(case)
        assert ordinary.runtime["swa-kernel-calls"] > 0
        assert ordinary.runtime["full-attention-kernel-calls"] > 0
        assert treatment.runtime["swa-kernel-calls"] > 0
        assert treatment.runtime["full-attention-kernel-calls"] > 0
        assert treatment.runtime["mtp-attention-modules-installed"] == 2
        assert treatment.runtime["mtp-swa-attention-forwards"] > 0
        assert treatment.runtime["mtp-full-attention-forwards"] > 0


_PARALLEL_CASES = (
    _Case(
        name="pp2",
        depth=2,
        pattern="accept",
        config={"pipeline_model_parallel_size": 2},
        signals=("gpt",),
    ),
    _Case(
        name="ep2-moe-optimized",
        depth=2,
        pattern="accept",
        config={
            "expert_model_parallel_size": 2,
            "use_moe_layer_spec": True,
            "inference_moe_token_dispatcher_type": "nccl",
            "transformer_impl": "inference_optimized",
        },
        signals=("inference-optimized", "moe", "nccl-dispatch"),
    ),
    _Case(
        name="tp2-pp2-sp-dp2",
        depth=2,
        pattern="accept",
        prompt_lengths=(5, 8, 11),
        output_lengths=(9, 8, 7),
        config={
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
            "sequence_parallel": True,
            "offset_sampling_seed_by_dp_rank": False,
        },
        signals=(),
    ),
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestMTPPairwiseParallel(_DynamicEngineTestBase):
    """MTP pair owners whose production path requires multiple GPUs."""

    @pytest.mark.parametrize("case", _PARALLEL_CASES, ids=lambda case: case.name)
    @torch.inference_mode()
    def test_parallel_topology_matches_ordinary_same_weight(self, case):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp = int(case.config.get("tensor_model_parallel_size", 1))
        pp = int(case.config.get("pipeline_model_parallel_size", 1))
        ep = int(case.config.get("expert_model_parallel_size", 1))
        required = tp * pp * ep
        if case.name == "tp2-pp2-sp-dp2" and world_size != 8:
            pytest.skip("tp2-pp2-sp-dp2 requires exactly eight GPUs")
        if world_size < required or world_size % required:
            pytest.skip(f"{case.name} requires a world size divisible by {required}")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
        )
        try:
            assert parallel_state.get_tensor_model_parallel_world_size() == tp
            assert parallel_state.get_pipeline_model_parallel_world_size() == pp
            assert parallel_state.get_expert_model_parallel_world_size() == ep
            ordinary, treatment = _run_mtp_pair(case)
            runtime = treatment.runtime
            if case.name == "pp2":
                assert runtime["pipeline-logits-broadcasts"] > 0
                assert runtime["mtp-pipeline-logits-broadcasts"] > 0
            elif case.name == "ep2-moe-optimized":
                assert runtime["module-forward:inference-optimized"] > 0
                assert runtime["module-forward:moe"] > 0
                assert runtime["nccl-dispatchers-installed"] > 0
                assert runtime["nccl-token-dispatches"] > 0
                assert runtime["nccl-token-dispatches"] == runtime["nccl-token-combines"]
                assert runtime["nccl-combine-before-dispatch"] == 0
                assert runtime["mtp-nccl-dispatchers-installed"] > 0
                assert runtime["mtp-nccl-token-dispatches"] > 0
                assert runtime["mtp-nccl-token-dispatches"] == runtime["mtp-nccl-token-combines"]
                assert runtime["mtp-nccl-combine-before-dispatch"] == 0
            else:
                assert parallel_state.get_data_parallel_world_size() == 2
                outputs = [request.generated_tokens for request in treatment.requests]
                acceptance = dict(treatment.witness.accepted_by_request)
                dp_group = parallel_state.get_data_parallel_group_gloo()
                gathered = [None] * torch.distributed.get_world_size(group=dp_group)
                torch.distributed.all_gather_object(gathered, (outputs, acceptance), group=dp_group)
                assert all(rank_result == (outputs, acceptance) for rank_result in gathered)
                assert runtime["pipeline-logits-broadcasts"] > 0
                assert runtime["mtp-pipeline-logits-broadcasts"] > 0
                if parallel_state.is_pipeline_last_stage():
                    assert runtime["mtp-tp-column-partitions-installed"] > 0
                    assert runtime["mtp-tp-row-partitions-installed"] > 0
                    assert runtime["mtp-tp-column-partition-forwards"] > 0
                    assert runtime["mtp-tp-row-partition-forwards"] > 0
                    assert runtime["mtp-tp-sp-gather-dimensions"] > 0
                    assert runtime["mtp-tp-sp-reduce-scatter-dimensions"] > 0
                else:
                    assert runtime["mtp-tp-column-partitions-installed"] == 0
                    assert runtime["mtp-tp-row-partitions-installed"] == 0
            assert ordinary.requests and treatment.requests
        finally:
            _cleanup()
            _set_rounder(64)
            Utils.destroy_model_parallel()
