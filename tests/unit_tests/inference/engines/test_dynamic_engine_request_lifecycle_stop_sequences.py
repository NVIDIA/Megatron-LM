# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused coverage for stop sequences that cross request checkpoints."""

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
    _install_mamba_slot_witnesses,
    _install_real_mtp_logits,
    _make_manual_request,
    _release_filler,
    _run_treatment_pair,
    _RunResult,
    _track_checkpoint_calls,
)
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _AsyncPairScenario,
    _check_scenario_prerequisite,
)

_HYBRID_MAMBA_STOP_KEEP = _AsyncPairScenario(
    name="single-evict-hybrid-mamba-stop-keep",
    pairs=("lifecycle:evict", "model:hybrid-mamba", "termination:stop-keep"),
    config={"model_provider": "hybrid"},
    signals=("hybrid",),
    prerequisite="mamba",
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRequestLifecycleStopSequences(RequestLifecyclePairwiseBase):
    @classmethod
    def _run_hybrid_mamba_stop_keep(cls, *, treatment):
        config, env = cls._manual_env(
            model_provider="hybrid",
            max_sequence_length=320,
            context_max_tokens=320,
            materialize_only_last_token_logits=False,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        assert context.mamba_slot_allocator is None
        runtime = Counter()
        _install_incrementing_logits(env, runtime, config.vocab_size)
        _install_mamba_slot_witnesses(env, runtime)
        request_id = 61
        prompt = torch.zeros(255, device="cuda", dtype=torch.int64)
        prompt[-1] = 10
        request = _make_manual_request(
            context, request_id, prompt, 8, full_logprobs=True, stop_words=["12 13"], keep_stop=True
        )
        filler = _allocate_leaving(allocator, 1) if treatment else None
        future = engine._add_request(request)
        checkpoint_counts, completed = Counter(), {}
        _track_checkpoint_calls(engine, checkpoint_counts, request_id)
        _collect_finished(engine, engine.step_modern(), completed)
        assert request.generated_tokens == [11]
        row = _active_request_row(context, request_id)
        slot_before = int(context.mamba_metadata.request_to_mamba_state_idx[row].item())
        _collect_finished(engine, engine.step_modern(), completed)
        witness = None
        if treatment:
            assert (engine.evicted_request_count, checkpoint_counts[request_id]) == (1, 1)
            checkpointed_tokens = engine.get_request(request_id).prompt_tokens[len(prompt) :]
            assert checkpointed_tokens.tolist() == [11, 12]
            assert context.total_request_count == 0
            assert context.mamba_metadata.request_to_mamba_state_idx[0].item() == -1
            assert context.mamba_metadata.mamba_state_free_slot_count == context.max_requests
            assert runtime[("mamba-slot-free", request_id)] == 1
            _release_filler(allocator, filler)
            filler = None
            engine.schedule_waiting_requests()
            row = _active_request_row(context, request_id)
            slot_after = int(context.mamba_metadata.request_to_mamba_state_idx[row].item())
            assert context.mamba_metadata.mamba_state_free_slot_count == context.max_requests - 1
            assert runtime[("mamba-slot-acquire", request_id)] == 2
            witness = {
                "request_id": request_id,
                "slot_before": slot_before,
                "slot_after": slot_after,
            }

        for _ in range(16):
            if not engine.has_unfinished_requests():
                break
            _collect_finished(engine, engine.step_modern(), completed)
        else:
            pytest.fail("hybrid Mamba stop-keep request did not drain")
        _release_filler(allocator, filler)
        _assert_engine_drained(engine, [future], completed, (request_id,))
        merged = completed[request_id]
        assert merged.generated_tokens == [11, 12, 13]
        score_count = len(merged.generated_tokens)
        assert (
            len(merged.generated_log_probs) == len(merged.generated_top_n_logprobs) == score_count
        )
        if treatment:
            assert _event_count(merged, DynamicInferenceEventType.EVICT) == 1
            assert all(runtime[("mamba-forward", phase, request_id)] > 0 for phase in range(2))
        return _RunResult([merged], checkpoint_counts, runtime, witness)

    @classmethod
    def _run_repeated_mtp_stop_strip(cls, *, treatment):
        config, env = cls._manual_env(
            max_sequence_length=600,
            context_max_tokens=640,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            position_embedding_type="none",
            vocab_size=512,
        )
        engine = env.engine
        context = engine.context
        allocator = context.kv_block_allocator
        runtime = Counter()
        _install_real_mtp_logits(env, runtime, config.vocab_size)
        request_id = 87
        prompt = torch.remainder(torch.arange(255, device="cuda") + 268, config.vocab_size)
        request = _make_manual_request(
            context,
            request_id,
            prompt,
            300,
            full_logprobs=False,
            stop_words=["267 268"],
            keep_stop=False,
        )
        first_filler = _allocate_leaving(allocator, 1) if treatment else None
        future = engine._add_request(request)
        checkpoint_counts, completed = Counter(), {}
        _track_checkpoint_calls(engine, checkpoint_counts, request_id)
        _collect_finished(engine, engine.step_modern(), completed)
        if treatment:
            assert (engine.evicted_request_count, checkpoint_counts[request_id]) == (1, 1)
            assert engine.get_request(request_id).prompt_tokens[len(prompt) :].tolist() == [11]
            _release_filler(allocator, first_filler)
            first_filler = None

        second_filler = None
        second_boundary = None
        for _ in range(160):
            if not engine.has_unfinished_requests():
                break
            live_ids = context.request_ids[: context.total_request_count].tolist()
            if (
                treatment
                and engine.evicted_request_count == 1
                and second_filler is None
                and request_id in live_ids
                and checkpoint_counts[request_id] == 1
            ):
                row = _active_request_row(context, request_id)
                block_count = int(context.request_kv_block_counts[row].item())
                offset = int(context.request_last_kv_block_offset[row].item())
                if block_count == 2 and offset >= 253:
                    second_filler = _allocate_leaving(allocator, 0)
                    assert second_filler is not None
            _collect_finished(engine, engine.step_modern(), completed)
            if treatment and engine.evicted_request_count == 2 and second_boundary is None:
                checkpointed_tokens = engine.get_request(request_id).prompt_tokens[len(prompt) :]
                second_boundary = (len(checkpointed_tokens), int(checkpointed_tokens[-1]))
                _release_filler(allocator, second_filler)
                second_filler = None
        else:
            pytest.fail("repeated MTP eviction request did not drain")
        _release_filler(allocator, first_filler)
        _release_filler(allocator, second_filler)
        _assert_engine_drained(engine, [future], completed, (request_id,))
        merged = completed[request_id]
        assert merged.generated_tokens == list(range(11, 267))
        score_count = len(merged.generated_tokens)
        assert (
            len(merged.generated_log_probs) == len(merged.generated_top_n_logprobs) == score_count
        )
        assert _event_count(merged, DynamicInferenceEventType.GENERATED_TOKEN) == len(
            merged.generated_tokens
        )
        witness = None
        if treatment:
            assert second_boundary == (257, 267)
            assert engine.evicted_request_count == 2
            assert checkpoint_counts[request_id] == 2
            assert _event_count(merged, DynamicInferenceEventType.PAUSE) == 2
            assert _event_count(merged, DynamicInferenceEventType.EVICT) == 2
            for phase in range(3):
                for depth in range(2):
                    assert runtime[("mtp-depth", phase, request_id, depth)] > 0
            assert all(int(count) > 0 for count in engine._spec_tokens_proposed_per_pos)
            witness = {
                "request_id": request_id,
                "evictions": engine.evicted_request_count,
                "second_boundary": second_boundary,
            }
        return _RunResult([merged], checkpoint_counts, runtime, witness)

    @torch.inference_mode()
    def test_single_evict_hybrid_mamba_stop_keep(self):
        _check_scenario_prerequisite(_HYBRID_MAMBA_STOP_KEEP)
        try:
            treatment = _run_treatment_pair(self._run_hybrid_mamba_stop_keep, self._cleanup)
        finally:
            self._cleanup()

    @torch.inference_mode()
    def test_repeated_evict_mtp2_stop_strip(self):
        try:
            treatment = _run_treatment_pair(self._run_repeated_mtp_stop_strip, self._cleanup)
            assert treatment.witness["evictions"] == 2
        finally:
            self._cleanup()
