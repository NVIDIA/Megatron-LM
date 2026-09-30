# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused request-identity coverage for pause, resume, and eviction."""

import os
from collections import Counter

import pytest
import torch

from megatron.core.inference.inference_request import DynamicInferenceEventType
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.request_lifecycle_test_utils import (
    RequestLifecyclePairwiseBase,
    _active_request_row,
    _allocate_leaving,
    _assert_engine_drained,
    _collect_finished,
    _event_count,
    _install_incrementing_logits,
    _install_nccl_request_witnesses,
    _make_manual_request,
    _release_filler,
    _run_treatment_pair,
    _RunResult,
    _track_checkpoint_calls,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder as _set_rounder
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _ASYNC_PARALLEL_SCENARIOS,
    _instrument_scenario_runtime,
)
from tests.unit_tests.test_utilities import Utils

_EP2_NCCL = next(case for case in _ASYNC_PARALLEL_SCENARIOS if case.name == "moe-ep2-nccl")


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecyclePairwise(RequestLifecyclePairwiseBase):
    @classmethod
    def _run_chunked_companion_evict(cls, *, treatment):
        config, env = cls._manual_env(
            max_sequence_length=544,
            context_max_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
            materialize_only_last_token_logits=True,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        runtime = Counter()
        _install_incrementing_logits(env, runtime, config.vocab_size)
        companion_id, chunk_id = 41, 73
        companion = _make_manual_request(
            context, companion_id, torch.arange(255, device="cuda", dtype=torch.int64) % 97, 6
        )
        chunk = _make_manual_request(
            context, chunk_id, (torch.arange(513, device="cuda", dtype=torch.int64) + 17) % 97, 4
        )
        for request in (companion, chunk):
            request.sampling_params.add_attributes({"top_n_logprobs": 0})
        head_filler = _allocate_leaving(allocator, 4) if treatment else None
        futures = [engine._add_request(companion)]
        checkpoint_counts, completed = Counter(), {}
        _track_checkpoint_calls(engine, checkpoint_counts, companion_id)
        _collect_finished(engine, engine.step_modern(), completed)
        assert (
            companion.generated_tokens
            and context.request_last_kv_block_offset[
                _active_request_row(context, companion_id)
            ].item()
            == 255
        )
        target_forwards_before = runtime[("model-forward", 0, companion_id)]

        futures.append(engine._add_request(chunk))
        _track_checkpoint_calls(engine, checkpoint_counts, chunk_id)
        engine.schedule_waiting_requests()
        assert context.chunked_prefill_request_id == chunk_id
        assert chunk.finished_chunk_token_count == 255
        assert len(chunk.remaining_prompt_tokens) == 258
        tail_filler = _allocate_leaving(allocator, 0) if treatment else None
        evictions_before = engine.evicted_request_count
        _collect_finished(engine, engine.step_modern(), completed)
        witness = None
        if treatment:
            assert engine.evicted_request_count == evictions_before + 1
            assert checkpoint_counts[companion_id] == 1
            assert context.chunked_prefill_request_id == chunk_id
            assert chunk.finished_chunk_token_count == 255
            assert _event_count(chunk, DynamicInferenceEventType.EVICT) == 0
            assert runtime[("model-forward", 0, companion_id)] > target_forwards_before
            assert runtime[("model-forward", 0, chunk_id)] > 0
            witness = {
                "request_id": companion_id,
                "chunk_id": chunk_id,
                "partial_chunk_tokens": chunk.finished_chunk_token_count,
            }
        _release_filler(allocator, tail_filler)
        _release_filler(allocator, head_filler)

        for _ in range(128):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine, engine.step_modern(), completed)
        else:
            pytest.fail("chunked companion eviction did not drain")
        _assert_engine_drained(engine, futures, completed, (companion_id, chunk_id))
        requests = [completed[request_id] for request_id in (companion_id, chunk_id)]
        assert not requests[0].prompt_log_probs
        assert len(requests[0].generated_tokens) == len(requests[0].generated_log_probs)
        assert not requests[0].prompt_top_n_logprobs
        assert not requests[0].generated_top_n_logprobs
        if treatment:
            assert checkpoint_counts[companion_id] == 1 and not checkpoint_counts[chunk_id]
            assert _event_count(requests[0], DynamicInferenceEventType.PAUSE) == 1
            assert _event_count(requests[0], DynamicInferenceEventType.EVICT) == 1
        return _RunResult(requests, checkpoint_counts, runtime, witness)

    @torch.inference_mode()
    def test_chunked_companion_zero_budget_evict(self):
        try:
            treatment = _run_treatment_pair(self._run_chunked_companion_evict, self._cleanup)
            assert treatment.witness["partial_chunk_tokens"] == 255
        finally:
            self._cleanup()


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecyclePairwiseEP2(RequestLifecyclePairwiseBase):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    @classmethod
    def _run_retained_pause_ep2(cls, *, treatment):
        config, env = TestRequestLifecyclePairwise._manual_env.__func__(
            cls,
            **_EP2_NCCL.config,
            max_sequence_length=320,
            context_max_tokens=320,
            context_paused_buffer_size_gb=0.002,
            materialize_only_last_token_logits=True,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        assert allocator.paused_limit >= 1
        runtime = Counter()
        _instrument_scenario_runtime(env, _EP2_NCCL, runtime)
        _install_incrementing_logits(env, runtime, config.vocab_size)
        _install_nccl_request_witnesses(env, runtime)
        target_id, companion_id = 61, 62
        target = _make_manual_request(
            context, target_id, torch.arange(255, device="cuda", dtype=torch.int64) % 97, 8
        )
        companion = _make_manual_request(
            context, companion_id, torch.arange(8, device="cuda", dtype=torch.int64) + 13, 12
        )
        filler = _allocate_leaving(allocator, 2) if treatment else None
        futures = [engine._add_request(target), engine._add_request(companion)]
        checkpoint_counts, completed = Counter(), {}
        _track_checkpoint_calls(engine, checkpoint_counts, target_id, companion_id)
        _collect_finished(engine, engine.step_modern(), completed)
        _collect_finished(engine, engine.step_modern(), completed)
        witness = None
        dispatch_at_pause = None
        if treatment:
            assert context.paused_request_count == 1
            assert context.request_ids[0].item() == target_id
            assert allocator.get_paused_used() == 1
            assert engine.evicted_request_count == 0
            assert checkpoint_counts[target_id] == 0
            paused = engine.get_request(target_id)
            assert _event_count(paused, DynamicInferenceEventType.PAUSE) == 1
            assert _event_count(paused, DynamicInferenceEventType.EVICT) == 0
            dispatch_at_pause = runtime[("nccl-dispatch", target_id)]
            assert dispatch_at_pause > 0
            _release_filler(allocator, filler)
            filler = None

        for _ in range(32):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine, engine.step_modern(), completed)
        else:
            pytest.fail("EP2 retained-pause requests did not drain")
        _release_filler(allocator, filler)
        _assert_engine_drained(engine, futures, completed, (target_id, companion_id))
        assert runtime["nccl-token-dispatches"] == runtime["nccl-token-combines"] > 0
        assert runtime["nccl-combine-before-dispatch"] == runtime["nccl-dispatch-inflight"] == 0
        if treatment:
            assert runtime[("nccl-dispatch", target_id)] > dispatch_at_pause
            assert engine.evicted_request_count == 0
            assert not checkpoint_counts
            witness = {
                "request_id": target_id,
                "dispatches_at_pause": dispatch_at_pause,
                "dispatches_total": runtime[("nccl-dispatch", target_id)],
            }
        requests = [completed[request_id] for request_id in (target_id, companion_id)]
        return _RunResult(requests, checkpoint_counts, runtime, witness)

    @torch.inference_mode()
    def test_retained_pause_ep2(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            pytest.skip("the retained-pause EP2 owner requires exactly two GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        )
        try:
            treatment = _run_treatment_pair(self._run_retained_pause_ep2, self._cleanup)
            assert treatment.witness["dispatches_total"] > treatment.witness["dispatches_at_pause"]
        finally:
            self._cleanup()
            _set_rounder(64)
            Utils.destroy_model_parallel()
