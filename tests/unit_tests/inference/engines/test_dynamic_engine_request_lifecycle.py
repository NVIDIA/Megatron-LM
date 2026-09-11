# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import fields

import pytest
import torch

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.request_lifecycle_test_utils import (
    RequestLifecyclePairwiseBase,
    _feature_keys,
    _request_feature_key,
)
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import _AsyncPairScenario

REQUEST_FIELD_POLICY = {
    "request_id": "checkpoint:preserve / merge:first",
    "prompt": "checkpoint:reset / merge:first",
    "sampling_params": "checkpoint:copy-and-reduce-budget / merge:first",
    "inference_parameters": "checkpoint:drop-deprecated-alias / merge:drop",
    "prompt_tokens": "checkpoint:append-output / merge:first / wire:opt-in",
    "prompt_length": "checkpoint:reset / merge:reset / wire:derive",
    "arrival_time": "checkpoint:reset / merge:reset",
    "status": "checkpoint:preserve / merge:last",
    "encoder_prompt": "checkpoint:drop / merge:drop",
    "generated_text": "checkpoint:reset / merge:reset / finalize:decode-concatenated-tokens",
    "segments": "checkpoint:drop / merge:drop-unsupported",
    "generated_segments": "checkpoint:drop / merge:drop-unsupported",
    "generated_sequence_lengths": "checkpoint:drop / merge:drop-unsupported",
    "generated_tokens": "checkpoint:reset / merge:concatenate",
    "prompt_log_probs": "checkpoint:reset / merge:most-complete-original-prompt",
    "generated_log_probs": "checkpoint:reset / merge:concatenate",
    "prompt_top_n_logprobs": "checkpoint:reset / merge:most-complete-original-prompt",
    "generated_top_n_logprobs": "checkpoint:reset / merge:concatenate",
    "generated_length": "checkpoint:reset / merge:derive-from-output",
    "tpot": "checkpoint:reset / merge:concatenate",
    "remaining_prompt_tokens": "checkpoint:cumulative-prompt / merge:reinitialize",
    "policy_epoch": "checkpoint:deep-copy / merge:last-deep-copy",
    "kv_cache_epoch": "checkpoint:reset-for-recompute / merge:last-deep-copy",
    "latency": "checkpoint:record-owned / merge:record",
    "routing_indices": "checkpoint:reset / merge:concatenate",
    "finished_chunk_token_count": "checkpoint:reset / merge:reset",
    "stop_word_ids": "checkpoint:deep-copy / merge:drop-terminal-only",
    "cg_wait_iters": "checkpoint:reset-admission-local / merge:reset",
    "block_size_tokens": "checkpoint:preserve / merge:first",
    "enable_prefix_caching": "checkpoint:preserve / merge:first",
    "num_cached_tokens": "checkpoint:reset / merge:first-observation",
    "precomputed_block_hashes": "checkpoint:recompute / merge:first",
    "ttft": "checkpoint:reset / merge:first-populated",
    "events": "checkpoint:new-segment / merge:concatenate",
    "event_add_engine": "checkpoint:preserve-original / merge:drop / wire:drop",
}

SAMPLING_FIELD_POLICY = {
    "temperature": "preserve",
    "top_k": "preserve",
    "top_p": "preserve",
    "return_log_probs": "preserve",
    "skip_prompt_log_probs": "preserve",
    "return_segments": "preserve",
    "num_tokens_to_generate": "subtract-generated",
    "num_tokens_total": "clear-after-budget-conversion",
    "termination_id": "preserve",
    "top_n_logprobs": "preserve",
    "return_prompt_top_n_logprobs": "preserve-derived-policy",
    "add_BOS": "preserve",
    "stop_words": "deep-copy-preserve",
    "detokenize_stop_sequence": "preserve",
    "return_prompt_tokens": "preserve",
    "streaming": "preserve",
    "streaming_interval": "preserve",
}


def test_checkpoint_field_policy_tables_are_exhaustive():
    request_fields = {field.name for field in fields(DynamicInferenceRequest)}
    sampling_fields = {field.name for field in fields(SamplingParams)}
    assert request_fields == set(REQUEST_FIELD_POLICY), (
        f"unclassified={sorted(request_fields - REQUEST_FIELD_POLICY)}, "
        f"stale={sorted(REQUEST_FIELD_POLICY - request_fields)}"
    )
    assert sampling_fields == set(SAMPLING_FIELD_POLICY), (
        f"unclassified={sorted(sampling_fields - SAMPLING_FIELD_POLICY)}, "
        f"stale={sorted(SAMPLING_FIELD_POLICY - sampling_fields)}"
    )


REQUEST_LIFECYCLE_MATRIX = {
    "persist-te-swa-stochastic": "test_persist_te_swa_stochastic",
    "offload-dynamic-fp8": "test_offload_dynamic_fp8",
    "uvm-offload-static-managed-capacity": "test_uvm_offload_static_managed_capacity",
}


def test_matrix_manifest_has_one_runtime_owner_per_row():
    assert len(REQUEST_LIFECYCLE_MATRIX) == 3
    assert all(
        hasattr(TestRequestLifecycleCorePairwise, owner)
        for owner in REQUEST_LIFECYCLE_MATRIX.values()
    )


_PERSIST_TE_SWA = _AsyncPairScenario(
    name="persist-te-swa-stochastic",
    pairs=("kv:persist", "implementation:transformer-engine", "attention:swa-sink"),
    config={
        "kv_cache_management_mode": "persist",
        "static_kv_memory_pointers": True,
        "transformer_impl": "transformer_engine",
        "window_size": (4, 0),
        "window_attn_skip_freq": 2,
        "softmax_type": "learnable",
    },
    sampling=({"temperature": 0.8, "top_k": 8},),
    signals=(
        "transformer-engine",
        "swa-alternating",
        "softmax-sink",
        "sampled",
        "temperature-filter",
        "top-k-filter",
    ),
    prerequisite="transformer-engine",
    parity="reproducible",
)

_OFFLOAD_FP8 = _AsyncPairScenario(
    name="offload-dynamic-fp8",
    pairs=("kv:offload", "precision:fp8"),
    config={"kv_cache_management_mode": "offload", "static_kv_memory_pointers": False, "fp8": True},
    signals=("fp8",),
    prerequisite="fp8",
    atol=5.0e-3,
)

_UVM_OFFLOAD = _AsyncPairScenario(
    name="uvm-offload-static-managed-capacity",
    pairs=("kv:offload", "memory:uvm-capacity"),
    config={
        "kv_cache_management_mode": "offload",
        "static_kv_memory_pointers": True,
        "unified_memory_level": 1,
        "context_buffer_size_gb": 0.002,
        "context_paused_buffer_size_gb": 0.002,
    },
    signals=("gpt",),
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecycleCorePairwise(RequestLifecyclePairwiseBase):
    @torch.inference_mode()
    def test_persist_te_swa_stochastic(self):
        result = self._assert_pair(_PERSIST_TE_SWA)
        assert result.witness["pointer_before"] > 0
        for key in _feature_keys(_PERSIST_TE_SWA):
            assert (
                result.runtime[_request_feature_key(result.witness["request_id"], key)]
                > result.witness["feature_counts_before"][key]
                > 0
            )

    @torch.inference_mode()
    def test_offload_dynamic_fp8(self):
        result = self._assert_pair(_OFFLOAD_FP8)
        assert result.witness["storage_bytes_before"] > 0
        for key in _feature_keys(_OFFLOAD_FP8):
            assert (
                result.runtime[_request_feature_key(result.witness["request_id"], key)]
                > result.witness["feature_counts_before"][key]
                > 0
            )

    @torch.inference_mode()
    def test_uvm_offload_static_managed_capacity(self):
        result = self._assert_pair(_UVM_OFFLOAD)
        assert result.witness["prefetch_succeeded"]
        assert max(result.witness["block_ids"]) >= result.witness["cuda_only_usable_blocks"]
        key = _request_feature_key(result.witness["request_id"], "module-forward:gpt")
        assert (
            result.runtime[key] > result.witness["feature_counts_before"]["module-forward:gpt"] > 0
        )
