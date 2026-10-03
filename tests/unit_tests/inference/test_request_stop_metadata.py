# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit token stops and engine termination metadata survive transport."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, RequestEntry
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


def _engine():
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.extra_eos_token_id_set = controller._build_extra_eos_token_id_set(
        SimpleNamespace(eod=11, generation_config={"eos_token_id": [2, 11]})
    )
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = controller
    engine.stop_word_being_finished_ids = set()
    return engine


def _request(tokens, termination_id=11, stop_ids=None):
    return DynamicInferenceRequest(
        request_id=0,
        prompt="prompt",
        prompt_tokens=torch.tensor([8]),
        generated_tokens=tokens,
        generated_log_probs=[-1.0] * len(tokens),
        sampling_params=SamplingParams(
            termination_id=termination_id, stop_token_ids=stop_ids, num_tokens_to_generate=3
        ),
        status=Status.COMPLETED,
    )


@pytest.mark.parametrize("termination_id", [11, -1])
@pytest.mark.parametrize("tokens", [[7], [5, 7, 9], [7, 9, 5]])
def test_explicit_stop_keeps_stop_token_and_trims_following_scores(termination_id, tokens):
    engine = _engine()
    request = _request([], termination_id, [7])
    scores = [-1.0, -2.0, -3.0][: len(tokens)]
    top_n = {0: scores.copy()}
    actual, actual_scores, hit = engine._truncate_at_mid_block_eos(
        request, tokens, scores, top_n, 0
    )
    keep = tokens.index(7) + 1
    assert hit
    assert actual == tokens[:keep]
    assert actual_scores == scores[:keep]
    assert top_n[0] == scores[:keep]


@pytest.mark.parametrize(
    "tokens,termination_id,stop_ids,expected",
    [
        ([4, 5, 6], 11, None, "length"),
        ([4, 5, 11], 11, None, "stop"),
        ([4, 5, 2], 11, None, "stop"),
        ([4, 5, 7], 11, [7], "stop"),
        ([4, 5, 2], -1, None, "length"),
        ([4, 5, 7], -1, [7], "stop"),
        ([], 11, None, "length"),
    ],
)
def test_finish_reason_and_serialization(tokens, termination_id, stop_ids, expected):
    request = _request(tokens, termination_id, stop_ids)
    _engine()._set_finish_reason(request)
    assert request.finish_reason == expected
    restored = DynamicInferenceRequest.deserialize(request.serialize())
    assert restored.finish_reason == expected
    assert restored.sampling_params.stop_token_ids == stop_ids
    merged = DynamicInferenceRequestRecord.from_request(request).merge()
    assert merged.finish_reason == expected


def test_stripped_string_stop_reason_is_preserved():
    request = _request([4, 5, 6])
    request.finish_reason = "stop"
    _engine()._set_finish_reason(request)
    assert request.finish_reason == "stop"


@pytest.mark.parametrize("invalid", [[-1], [True], [1.2], "7"])
def test_invalid_stop_ids_rejected(invalid):
    with pytest.raises(ValueError, match="nonnegative integers"):
        SamplingParams(stop_token_ids=invalid)


def test_stop_ids_round_trip_and_normalize_without_mutating_input():
    ids = [9, 7, 9]
    params = SamplingParams(stop_token_ids=ids)
    assert ids == [9, 7, 9]
    assert SamplingParams.deserialize(params.serialize()).stop_token_ids == [7, 9]


@pytest.mark.parametrize(
    "tokens,budget,expected,kept",
    [
        ([7], 3, "stop", [7]),
        ([5, 7, 9], 3, "stop", [5, 7]),
        ([5, 7, 9], 1, "length", [5]),
        ([7], 1, "stop", [7]),
        ([11], 1, "stop", [11]),
        ([2], 1, "stop", [2]),
        ([5], 1, "length", [5]),
    ],
)
def test_postprocess_completion_and_deferred_stop(tokens, budget, expected, kept):
    async def run():
        engine = _engine()
        engine.finished_request_count = 0
        engine.evicted_request_count = 0
        engine.track_generated_token_events = False
        engine.num_speculative_tokens = len(tokens) - 1
        engine._spec_steps = 0
        engine.stop_word_finished_request_ids = set()
        engine.context = SimpleNamespace(
            kv_block_allocator=object(), remove_vlm_request_data=lambda _: None
        )
        request = _request([], stop_ids=[7])
        request.status = Status.ACTIVE_AND_GENERATING_TOKENS
        request.sampling_params.num_tokens_to_generate = budget
        request.sampling_params.return_log_probs = True
        request.sampling_params.skip_prompt_log_probs = False
        request.add_event_add_engine()
        future = asyncio.get_running_loop().create_future()
        engine.requests = {
            0: RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(request), future=future
            )
        }
        # Model EOS / length are detected by controller bookkeeping; custom
        # stop ids are discovered by engine postprocessing and finish next step.
        finished = [0] if len(tokens) >= budget or tokens[-1] in [2, 11] else []
        active, completed = engine.post_process_requests(
            request_ids=torch.tensor([0]),
            finished_request_ids=torch.tensor(finished),
            evict_request_ids=None,
            step_time=0.0,
            sample=torch.tensor([tokens[-1]]),
            accepted_tokens=torch.tensor([tokens[:-1]]) if len(tokens) > 1 else None,
            log_probs=[[-1.0] * len(tokens)],
            consumed_chunked_prefill_request_id=-1,
        )
        if active:
            assert not future.done()
            assert engine._get_and_clear_stop_word_finished_ids(active) == {0}
            # A speculative next step must not append its tokens or overwrite stop.
            engine.num_speculative_tokens = 0
            active, completed = engine.post_process_requests(
                request_ids=torch.tensor([0]),
                finished_request_ids=torch.tensor([0]),
                evict_request_ids=None,
                step_time=0.0,
                sample=torch.tensor([9]),
                accepted_tokens=None,
                log_probs=[[-2.0]],
                consumed_chunked_prefill_request_id=-1,
            )
        assert not active
        assert len(completed) == 1
        actual = future.result()
        assert actual is completed[0]
        assert actual.finish_reason == expected
        assert actual.generated_tokens == kept
        assert actual.generated_log_probs == [-1.0] * len(kept)
        assert actual.generated_length == len(kept)
        assert not engine.requests

    asyncio.run(run())
