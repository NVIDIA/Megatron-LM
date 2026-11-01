# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import Counter

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.inference_request import (
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.engines.request_lifecycle_test_utils import (
    RequestLifecyclePairwiseBase,
    _active_request_ids,
    _assert_engine_drained,
    _collect_finished,
)


class _ReorderedMetadataRequest(DynamicInferenceRequest):
    """Exercise a supported non-default schema without changing field semantics."""

    @staticmethod
    def get_metadata_types():
        return list(reversed(DynamicInferenceRequest.get_metadata_types()))


def _requests(context, request_type=DynamicInferenceRequest):
    policies = (
        SamplingParams(
            num_tokens_to_generate=6,
            termination_id=-1,
            top_k=1,
            return_log_probs=True,
            skip_prompt_log_probs=False,
            top_n_logprobs=3,
            # Both sequences end on token 13; the longest simultaneous match wins.
            stop_words=["12 13", "11 12 13"],
            detokenize_stop_sequence=False,
        ),
        SamplingParams(
            num_tokens_to_generate=2,
            termination_id=-1,
            stop_token_ids=[22, 55],
            top_k=1,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            top_n_logprobs=1,
        ),
        SamplingParams(
            num_tokens_to_generate=4,
            termination_id=-1,
            top_k=2,
            return_log_probs=False,
            top_n_logprobs=0,
        ),
    )
    prompts = ([8, 9, 10], [18, 19, 20], [28, 29, 30])
    return [
        request_type(
            request_id=request_id,
            prompt_tokens=torch.tensor(prompt, dtype=torch.int64, device="cuda"),
            sampling_params=policies[request_id],
            block_size_tokens=context.block_size_tokens,
            enable_prefix_caching=context.enable_prefix_caching,
        )
        for request_id, prompt in enumerate(prompts)
    ]


def _install_step_witnesses(env, runtime, step_index, vocab_size):
    engine = env.engine
    context = engine.context
    model = engine.controller.inference_wrapped_model.model
    original_forward = model.forward

    def deterministic_forward(*args, **kwargs):
        request_ids = tuple(_active_request_ids(context))
        logits = original_forward(*args, **kwargs)
        tokens = kwargs.get("tokens", kwargs.get("input_ids", args[0] if args else None))
        assert tokens is not None
        if logits.shape[1] != tokens.shape[1]:
            tokens = tokens[:, context.active_logit_idxs[: logits.shape[1]]]
        next_tokens = torch.remainder(tokens + 1, vocab_size)
        logits.copy_(-torch.arange(vocab_size, device=logits.device, dtype=logits.dtype))
        # A 40-logit gap makes the selected token effectively deterministic while
        # retaining a normal-range second probability for the top-k=2 support oracle.
        logits.scatter_(-1, next_tokens.unsqueeze(-1), 40.0)
        runtime[("model", step_index[0], request_ids)] += 1
        runtime[("metadata", step_index[0], request_ids)] = {
            label: tuple(context.active_request_metadata[label][: len(request_ids)].tolist())
            for label, _ in context.request_metadata_types
        }
        return logits

    model.forward = deterministic_forward

    sampler = engine.controller._sampling
    original_sample = sampler.sample_kernel

    def traced_sample(*args, **kwargs):
        request_ids = tuple(_active_request_ids(context))
        result = original_sample(*args, **kwargs)
        runtime[("sample", step_index[0], request_ids)] += 1
        return result

    sampler.sample_kernel = traced_sample

    original_scores = sampler.log_probs_kernel

    def traced_scores(*args, **kwargs):
        request_ids = tuple(_active_request_ids(context))
        result = original_scores(*args, **kwargs)
        runtime[("scores", step_index[0], request_ids)] += 1
        runtime[("score-support", step_index[0], request_ids)] = tuple(
            torch.isfinite(result).sum(dim=-1).tolist()
        )
        return result

    sampler.log_probs_kernel = traced_scores

    original_compact = engine.controller._compact_async_sched_forward

    def traced_compact(survivor_idxs):
        runtime[("compact", step_index[0], tuple(survivor_idxs.tolist()))] += 1
        return original_compact(survivor_idxs)

    engine.controller._compact_async_sched_forward = traced_compact


def _assert_scores(request):
    if request.request_id == 2:
        assert request.prompt_log_probs is None
        assert request.generated_log_probs is None
        assert request.prompt_top_n_logprobs is None
        assert request.generated_top_n_logprobs is None
        return

    expected_prompt_scores = 2 if request.request_id == 0 else 0
    expected_top_n = 3 if request.request_id == 0 else 1
    assert len(request.prompt_log_probs or []) == expected_prompt_scores
    assert len(request.generated_log_probs) == len(request.generated_tokens)
    assert len(request.prompt_top_n_logprobs or []) == expected_prompt_scores
    assert len(request.generated_top_n_logprobs) == len(request.generated_tokens)

    for score in list(request.prompt_log_probs or []) + request.generated_log_probs:
        assert torch.isfinite(torch.tensor(score))
        assert score == pytest.approx(0.0, abs=1.0e-4)
    for token, scores in zip(request.generated_tokens, request.generated_top_n_logprobs):
        assert len(scores) == expected_top_n
        assert f"<{token}>" in scores
        assert scores[f"<{token}>"] == pytest.approx(0.0, abs=1.0e-4)
    if request.request_id == 0:
        for token, scores in zip(request.prompt_tokens[1:].tolist(), request.prompt_top_n_logprobs):
            assert len(scores) == expected_top_n
            assert f"<{token}>" in scores


class TestSamplingOutputMatrix(RequestLifecyclePairwiseBase):
    @classmethod
    def _run_mode(cls, mode, sampling_backend, request_type=DynamicInferenceRequest):
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        request_metadata_types = request_type.get_metadata_types()
        configured_metadata_types = (
            None if request_type is DynamicInferenceRequest else request_metadata_types
        )
        config, env = cls._manual_env(
            async_sched_mode=mode,
            sampling_backend=sampling_backend,
            logprobs_mode="processed_logprobs",
            materialize_only_last_token_logits=False,
            request_metadata_types=configured_metadata_types,
            vocab_size=64,
        )
        engine = env.engine
        assert engine.controller._sampling_backend == sampling_backend
        assert engine.context.request_metadata_types == request_metadata_types
        requests = _requests(engine.context, request_type)
        futures = [engine._add_request(request) for request in requests]
        runtime = Counter()
        step_index = [-1]
        _install_step_witnesses(env, runtime, step_index, config.vocab_size)
        completed = {}

        for step in range(16):
            step_index[0] = step
            result = engine.step_modern()
            _collect_finished(engine, result, completed)
            if 1 in completed and any(request_id in engine.requests for request_id in (0, 2)):
                runtime["middle-finished-with-survivors"] += 1
            if not engine.has_unfinished_requests():
                break
        else:
            pytest.fail(f"{mode.value} engine did not drain")

        _assert_engine_drained(engine, futures, completed, range(3))
        finished = [completed[request_id] for request_id in range(3)]
        assert [request.generated_tokens for request in finished] == [
            [],
            [21, 22],
            [31, 32, 33, 34],
        ]
        assert [request.finish_reason for request in finished] == ["stop", "stop", "length"]
        assert all(request.status == Status.COMPLETED for request in finished)
        assert finished[0].sampling_params.stop_words == ["12 13", "11 12 13"]
        assert finished[1].sampling_params.stop_token_ids == [22, 55]
        assert finished[1].generated_length == finished[1].sampling_params.num_tokens_to_generate
        assert runtime["middle-finished-with-survivors"] > 0

        for request in finished:
            generated_events = [
                event.payload["token_id"]
                for event in request.events
                if event.type == DynamicInferenceEventType.GENERATED_TOKEN
            ]
            assert generated_events == request.generated_tokens
            assert request.generated_length == len(request.generated_tokens)
            assert request.events[-1].type == DynamicInferenceEventType.FINISH
            _assert_scores(request)

        model_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "model" and set(key[2]) == {0, 1, 2}
        }
        sample_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "sample" and set(key[2]) == {0, 1, 2}
        }
        score_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "scores" and set(key[2]) == {0, 1, 2}
        }
        assert model_steps & sample_steps & score_steps
        survivor_model_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "model" and key[2] == (0, 2)
        }
        survivor_sample_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "sample" and key[2] == (0, 2)
        }
        survivor_score_steps = {
            key[1]
            for key in runtime
            if isinstance(key, tuple) and key[0] == "scores" and key[2] == (0, 2)
        }
        assert survivor_model_steps & survivor_sample_steps & survivor_score_steps
        survivor_score_support = {
            value
            for key, value in runtime.items()
            if isinstance(key, tuple)
            and key[0] == "score-support"
            and key[2] == (0, 2)
            and len(value) == 2
        }
        assert (1, 2) in survivor_score_support
        survivor_metadata = [
            values
            for key, values in runtime.items()
            if isinstance(key, tuple) and key[0] == "metadata" and key[2] == (0, 2)
        ]
        expected_survivor_metadata = {
            "temperature": (1.0, 1.0),
            "top_k": (1, 2),
            # SamplingParams canonicalizes the disabled public value 1.0 to 0.0.
            "top_p": (0.0, 0.0),
            "termination_id": (-1, -1),
            "return_log_probs": (True, False),
            "skip_prompt_log_probs": (False, False),
            "top_n_logprobs": (3, 0),
            "seed": (-1, -1),
        }
        assert expected_survivor_metadata in survivor_metadata
        if mode == AsyncScheduleMode.ASYNC:
            compact_steps = {
                key[1]
                for key in runtime
                if isinstance(key, tuple) and key[0] == "compact" and key[2] == (0, 2)
            }
            assert compact_steps

        snapshot = {
            request.request_id: {
                "tokens": list(request.generated_tokens),
                "finish_reason": request.finish_reason,
                "prompt_scores": list(request.prompt_log_probs or []),
                "generated_scores": list(request.generated_log_probs or []),
                "prompt_top_n": request.prompt_top_n_logprobs,
                "generated_top_n": request.generated_top_n_logprobs,
            }
            for request in finished
        }
        del completed, engine, env, finished, futures, requests
        cls._cleanup()
        return snapshot

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="real dynamic engine requires CUDA")
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @torch.inference_mode()
    def test_heterogeneous_scores_stops_and_middle_compaction(self, sampling_backend):
        """Legacy and async engines preserve exact tokens, scores, stops, and events."""
        legacy = self._run_mode(AsyncScheduleMode.LEGACY, sampling_backend)
        asynchronous = self._run_mode(AsyncScheduleMode.ASYNC, sampling_backend)

        assert asynchronous.keys() == legacy.keys()
        for request_id in legacy:
            actual = asynchronous[request_id]
            expected = legacy[request_id]
            assert actual["tokens"] == expected["tokens"]
            assert actual["finish_reason"] == expected["finish_reason"]
            assert actual["prompt_scores"] == pytest.approx(expected["prompt_scores"], abs=1.0e-4)
            assert actual["generated_scores"] == pytest.approx(
                expected["generated_scores"], abs=1.0e-4
            )
            # Top-k=1 maps all non-selected alternatives to the same -inf score, so
            # their token identities are not part of the contract. Compare row sizes
            # and score multisets; _assert_scores pins selected-token membership.
            for field in ("prompt_top_n", "generated_top_n"):
                actual_rows = actual[field] or []
                expected_rows = expected[field] or []
                assert len(actual_rows) == len(expected_rows)
                for actual_row, expected_row in zip(actual_rows, expected_rows):
                    assert len(actual_row) == len(expected_row)
                    assert sorted(actual_row.values()) == pytest.approx(
                        sorted(expected_row.values()), abs=1.0e-4
                    )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="real dynamic engine requires CUDA")
    @torch.inference_mode()
    def test_custom_metadata_schema_survives_gpu_compaction(self):
        """A configured non-default schema stays aligned through live GPU compaction."""
        legacy = self._run_mode(
            AsyncScheduleMode.LEGACY, "torch", request_type=_ReorderedMetadataRequest
        )
        asynchronous = self._run_mode(
            AsyncScheduleMode.ASYNC, "torch", request_type=_ReorderedMetadataRequest
        )
        assert asynchronous == legacy
