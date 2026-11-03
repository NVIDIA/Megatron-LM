# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for token stops, stop words, and dynamic inference finish metadata."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock
from unittest.mock import MagicMock, patch

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


class MockDynamicInferenceRequest:
    """Mock class for DynamicInferenceRequest to test stop word detection."""

    def __init__(
        self,
        request_id: int,
        generated_tokens: Optional[List[int]] = None,
        stop_word_ids: Optional[List[List[int]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.request_id = request_id
        self.generated_tokens = generated_tokens if generated_tokens is not None else []
        self.stop_word_ids = stop_word_ids
        self.sampling_params = sampling_params or SamplingParams()


class TestStopWordDetection:
    """Test stop word detection logic."""

    def _check_stop_words_for_request_post_append(
        self, request: MockDynamicInferenceRequest, num_speculative_tokens: int = 0
    ) -> tuple:
        """
        Check if a request should stop due to stop words (after token is appended).

        This mirrors the logic in DynamicInferenceEngine._check_stop_words_for_request_post_append.
        Returns (stop_word_hit, num_tokens_trimmed).
        """
        if request.stop_word_ids is None or len(request.stop_word_ids) == 0:
            return False, 0

        generated_tokens = request.generated_tokens

        for stop_word_ids in request.stop_word_ids:
            stop_len = len(stop_word_ids)
            if len(generated_tokens) >= stop_len:
                for i in range(num_speculative_tokens + 1):
                    end_idx = -i if i > 0 else None
                    if list(generated_tokens[-stop_len - i : end_idx]) == stop_word_ids:
                        if i > 0:
                            request.generated_tokens = request.generated_tokens[:-i]
                        return True, i

        return False, 0

    def test_no_stop_words_configured(self):
        """Test that requests without stop words configured don't trigger stop."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=None
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False
        assert trim == 0

    def test_empty_stop_words_list(self):
        """Test that empty stop words list doesn't trigger stop."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_single_token_stop_word_match(self):
        """Test detection of single-token stop word."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is True
        assert trim == 0
        assert request.generated_tokens == [100, 200, 300]

    def test_single_token_stop_word_no_match(self):
        """Test no detection when single-token stop word doesn't match."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[400]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_multi_token_stop_word_match(self):
        """Test detection of multi-token stop word."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[200, 300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is True
        assert trim == 0

    def test_multi_token_stop_word_no_match_partial(self):
        """Test no detection when only partial stop word matches."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200], stop_word_ids=[[200, 300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_multi_token_stop_word_no_match_wrong_order(self):
        """Test no detection when tokens are present but in wrong order."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 300, 200], stop_word_ids=[[200, 300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_multiple_stop_words_first_matches(self):
        """Test with multiple stop words where first one matches."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[300], [400], [500]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is True

    def test_multiple_stop_words_second_matches(self):
        """Test with multiple stop words where second one matches."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 400], stop_word_ids=[[300], [400], [500]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is True

    def test_multiple_stop_words_none_match(self):
        """Test with multiple stop words where none match."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 600], stop_word_ids=[[300], [400], [500]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_stop_word_longer_than_generated(self):
        """Test that stop word longer than generated tokens doesn't crash."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[1, 2, 3, 4, 5]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_stop_word_exact_length_match(self):
        """Test stop word that matches entire generated sequence."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[100, 200, 300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is True

    def test_empty_generated_tokens(self):
        """Test with no generated tokens."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[], stop_word_ids=[[300]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False

    def test_stop_word_in_middle_not_end(self):
        """Test that stop word in middle of sequence doesn't trigger (only end matters)."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[200]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(request)
        assert hit is False


class TestStopWordSpeculativeDecoding:
    """Test stop word detection and truncation with speculative decoding."""

    def _check_stop_words_for_request_post_append(
        self, request: MockDynamicInferenceRequest, num_speculative_tokens: int = 0
    ) -> tuple:
        """Mirror of DynamicInferenceEngine._check_stop_words_for_request_post_append."""
        if request.stop_word_ids is None or len(request.stop_word_ids) == 0:
            return False, 0

        generated_tokens = request.generated_tokens

        for stop_word_ids in request.stop_word_ids:
            stop_len = len(stop_word_ids)
            if len(generated_tokens) >= stop_len:
                for i in range(num_speculative_tokens + 1):
                    end_idx = -i if i > 0 else None
                    if list(generated_tokens[-stop_len - i : end_idx]) == stop_word_ids:
                        if i > 0:
                            request.generated_tokens = request.generated_tokens[:-i]
                        return True, i

        return False, 0

    def test_stop_word_at_end_no_trim(self):
        """Stop word is the last token — no trimming needed."""
        # Speculative tokens: [tok1, STOP, tok3] appended, stop word at end of accepted
        # But here STOP is at the very end after all tokens
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 20, 42], stop_word_ids=[[42]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 0
        assert request.generated_tokens == [10, 20, 42]

    def test_stop_word_with_one_extra_token(self):
        """Stop word is second-to-last — one extra token should be trimmed."""
        # Speculative appended [tok1, STOP, tok3], STOP=42 at position -2
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 20, 42, 99], stop_word_ids=[[42]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 1
        assert request.generated_tokens == [10, 20, 42]

    def test_stop_word_with_two_extra_tokens(self):
        """Stop word is third-to-last — two extra tokens should be trimmed."""
        # Speculative appended [STOP, tok2, tok3], STOP=42 at position -3
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 42, 77, 88], stop_word_ids=[[42]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 2
        assert request.generated_tokens == [10, 42]

    def test_multi_token_stop_word_with_extra_tokens(self):
        """Multi-token stop word found mid-speculative-batch."""
        # Speculative appended [tok1, STOP_A, STOP_B, tok4], stop word is [STOP_A, STOP_B]
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 20, 42, 43, 99], stop_word_ids=[[42, 43]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 1
        assert request.generated_tokens == [10, 20, 42, 43]

    def test_multi_token_stop_word_with_two_extra(self):
        """Multi-token stop word with two extra tokens after."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 42, 43, 77, 88], stop_word_ids=[[42, 43]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 2
        assert request.generated_tokens == [10, 42, 43]

    def test_no_stop_word_speculative(self):
        """No stop word in speculative batch — nothing happens."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 20, 30, 40], stop_word_ids=[[42]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is False
        assert trim == 0
        assert request.generated_tokens == [10, 20, 30, 40]

    def test_stop_word_outside_speculative_window(self):
        """Stop word exists but is outside the speculative search window."""
        # Stop word [42] is at position -4, but num_speculative_tokens=2
        # so we only check positions -1, -2, -3 (i=0,1,2)
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[42, 10, 20, 30], stop_word_ids=[[42]]
        )
        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is False
        assert trim == 0

    def test_log_probs_trimming_scenario(self):
        """Verify that the trim count can be used to trim log probs correctly."""
        # Simulate: speculative batch appended [tok1, STOP, tok3]
        # Log probs: [lp1, lp2, lp3]
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[10, 20, 42, 99], stop_word_ids=[[42]]
        )
        log_probs = [-1.5, -0.3, -2.1]

        hit, trim = self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        )
        assert hit is True
        assert trim == 1

        # Trim log probs the same way the engine does
        if trim > 0:
            log_probs = log_probs[:-trim]

        assert log_probs == [-1.5, -0.3]
        assert request.generated_tokens == [10, 20, 42]

    def test_speculative_stop_word_at_end(self):
        """Test stop word at end of speculative tokens (no truncation needed)."""
        # Speculative tokens appended: [200, 300], stop word is [300]
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[300]]
        )
        assert self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        ) == (True, 0)
        assert request.generated_tokens == [100, 200, 300]

    def test_speculative_stop_word_in_middle_truncates(self):
        """Test that stop word in middle of speculative tokens truncates trailing tokens."""
        # Speculative tokens appended: [200, 300, 400], stop word is [200]
        # Token 200 is at position -3, so tokens [300, 400] should be truncated
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300, 400], stop_word_ids=[[200]]
        )
        assert self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=3
        ) == (True, 2)
        assert request.generated_tokens == [100, 200]

    def test_speculative_multi_token_stop_word_in_middle_truncates(self):
        """Test multi-token stop word in middle of speculative tokens truncates."""
        # Generated: [100, 200, 300, 400, 500], stop word is [200, 300]
        # Stop word ends at -2, so tokens [400, 500] should be truncated
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300, 400, 500], stop_word_ids=[[200, 300]]
        )
        assert self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=4
        ) == (True, 2)
        assert request.generated_tokens == [100, 200, 300]

    def test_speculative_stop_word_not_found(self):
        """Test no stop word found even with speculative scanning."""
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300, 400], stop_word_ids=[[999]]
        )
        assert self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=3
        ) == (False, 0)
        assert request.generated_tokens == [100, 200, 300, 400]

    def test_speculative_stop_word_one_trailing_token(self):
        """Test stop word with exactly one trailing token to truncate."""
        # Generated: [100, 200, 300], stop word is [200], one trailing token [300]
        request = MockDynamicInferenceRequest(
            request_id=1, generated_tokens=[100, 200, 300], stop_word_ids=[[200]]
        )
        assert self._check_stop_words_for_request_post_append(
            request, num_speculative_tokens=2
        ) == (True, 1)
        assert request.generated_tokens == [100, 200]


class TestStopWordTrackingFlow:
    """Test the stop word tracking flow between steps."""

    def test_stop_word_finished_ids_tracking(self):
        """Test that stop_word_finished_request_ids correctly tracks requests."""
        stop_word_finished_request_ids = set()
        stop_word_being_finished_ids = set()

        # Simulate detecting stop word in post_process_requests
        request_id = 42
        stop_word_finished_request_ids.add(request_id)

        assert request_id in stop_word_finished_request_ids
        assert len(stop_word_finished_request_ids) == 1

        # Simulate callback being called
        active_request_ids = [42, 43, 44]
        result = stop_word_finished_request_ids & set(active_request_ids)
        stop_word_being_finished_ids = result
        stop_word_finished_request_ids -= result

        assert request_id in stop_word_being_finished_ids
        assert request_id not in stop_word_finished_request_ids

    def test_skip_extra_token_for_stop_word_requests(self):
        """Test that extra token is skipped for stop word finished requests."""
        stop_word_being_finished_ids = {42}
        generated_tokens = {
            42: [100, 200, 300],  # Already has tokens from previous step
            43: [100, 200],
        }

        new_tokens = {42: 999, 43: 301}  # New tokens to potentially append

        for request_id, token in new_tokens.items():
            if request_id not in stop_word_being_finished_ids:
                generated_tokens[request_id].append(token)

        # Request 42 should NOT have the extra token
        assert generated_tokens[42] == [100, 200, 300]
        # Request 43 should have the new token
        assert generated_tokens[43] == [100, 200, 301]


class TestSamplingParamsStopWords:
    """Test SamplingParams stop words field."""

    def test_stop_words_default_none(self):
        """Test that stop_words defaults to None."""
        params = SamplingParams()
        assert params.stop_words is None

    def test_stop_words_can_be_set(self):
        """Test that stop_words can be set."""
        params = SamplingParams(stop_words=["STOP", "END"])
        assert params.stop_words == ["STOP", "END"]

    def test_stop_words_empty_list(self):
        """Test that stop_words can be empty list."""
        params = SamplingParams(stop_words=[])
        assert params.stop_words == []


class TestMidSpeculativeBlockEos:
    """EOS inside a speculative block, the token-id analogue of a mid-block stop word.

    A speculative step emits `accepted_tokens + [sampled_token]`, but the controller's
    termination check only inspects the last of those. An EOS on an accepted draft
    position is a verified target token (the draft matched), not a rolled-back draft,
    so it must end the request -- otherwise the rest of the block, and every later
    step, is emitted after EOS. `_find_mid_block_eos` scans for it and
    `_truncate_at_mid_block_eos` drops what follows, keeping scores aligned.
    """

    def _engine(self, eod=2, eos_token_id=None):
        """An engine with only the state `_find_mid_block_eos` reads (no model, no GPU)."""
        tokenizer = SimpleNamespace(eod=eod)
        if eos_token_id is not None:
            tokenizer.generation_config = {"eos_token_id": eos_token_id}

        controller = TextGenerationController.__new__(TextGenerationController)
        controller.extra_eos_token_id_set = controller._build_extra_eos_token_id_set(tokenizer)

        engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
        engine.controller = controller
        engine.stop_word_being_finished_ids = set()
        return engine

    def _request(self, termination_id=2, request_id=0):
        return MockDynamicInferenceRequest(
            request_id=request_id, sampling_params=SamplingParams(termination_id=termination_id)
        )

    def test_eos_on_accepted_draft_token_is_found(self):
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(), [50, 51, 2, 52, 53]) == 2

    def test_second_declared_eos_is_found(self):
        # <|im_end|> (11) is not the request's termination_id but the model declares it.
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(), [50, 11, 52]) == 1

    def test_first_eos_wins(self):
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(), [50, 11, 2, 52]) == 1

    def test_block_without_eos_returns_none(self):
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(), [50, 51, 52]) is None

    def test_single_token_step_is_skipped(self):
        # Non-speculative steps were already checked by the controller; re-checking
        # would double-handle a request the controller is finishing this same step.
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(), [2]) is None

    def test_ignore_eos_block_is_skipped(self):
        engine = self._engine(eos_token_id=[2, 11])
        assert engine._find_mid_block_eos(self._request(termination_id=-1), [50, 2, 52]) is None

    def test_request_already_being_finished_is_skipped(self):
        # Its tokens are not appended this step, so there is nothing to truncate.
        engine = self._engine(eos_token_id=[2, 11])
        engine.stop_word_being_finished_ids = {7}
        assert engine._find_mid_block_eos(self._request(request_id=7), [50, 2, 52]) is None

    def test_single_eos_model_still_truncates_mid_block(self):
        # The mid-block gap is independent of multi-EOS: a model declaring one eos
        # also over-generated when that eos landed on an accepted draft position.
        engine = self._engine()
        assert engine._find_mid_block_eos(self._request(), [50, 2, 52]) == 1

    def test_truncate_drops_tokens_and_scores_after_eos(self):
        # Scores must stay aligned with tokens: 5 tokens -> 3, so 5 scores -> 3.
        engine = self._engine(eos_token_id=[2, 11])
        top_n = {0: ["a", "b", "c", "d", "e"]}
        tokens, log_probs, hit = engine._truncate_at_mid_block_eos(
            self._request(), [50, 51, 2, 52, 53], [0.1, 0.2, 0.3, 0.4, 0.5], top_n, 0
        )
        assert hit is True
        assert tokens == [50, 51, 2]
        assert log_probs == [0.1, 0.2, 0.3]
        assert top_n[0] == ["a", "b", "c"]

    def test_truncate_without_eos_returns_inputs_untouched(self):
        engine = self._engine(eos_token_id=[2, 11])
        top_n = {0: ["a", "b", "c"]}
        tokens, log_probs, hit = engine._truncate_at_mid_block_eos(
            self._request(), [50, 51, 52], [0.1, 0.2, 0.3], top_n, 0
        )
        assert hit is False
        assert tokens == [50, 51, 52]
        assert log_probs == [0.1, 0.2, 0.3]
        assert top_n[0] == ["a", "b", "c"]

    def test_truncate_on_trailing_eos_trims_nothing_but_still_finishes(self):
        # EOS already last: nothing to drop, but the request must still be finished.
        engine = self._engine(eos_token_id=[2, 11])
        top_n = {0: ["a", "b", "c"]}
        tokens, log_probs, hit = engine._truncate_at_mid_block_eos(
            self._request(), [50, 51, 2], [0.1, 0.2, 0.3], top_n, 0
        )
        assert hit is True
        assert tokens == [50, 51, 2]
        assert log_probs == [0.1, 0.2, 0.3]
        assert top_n[0] == ["a", "b", "c"]

    def test_truncate_handles_absent_scores(self):
        # Most requests ask for neither log probs nor top-n.
        engine = self._engine(eos_token_id=[2, 11])
        tokens, log_probs, hit = engine._truncate_at_mid_block_eos(
            self._request(), [50, 2, 52], None, None, 0
        )
        assert (tokens, log_probs, hit) == ([50, 2], None, True)

    def test_truncate_leaves_other_requests_top_n_alone(self):
        engine = self._engine(eos_token_id=[2, 11])
        top_n = {0: ["a", "b", "c"], 1: ["x", "y", "z"]}
        engine._truncate_at_mid_block_eos(self._request(), [50, 2, 52], None, top_n, 1)
        assert top_n[0] == ["a", "b", "c"]
        assert top_n[1] == ["x", "y"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _stop_metadata_engine():
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.extra_eos_token_id_set = controller._build_extra_eos_token_id_set(
        SimpleNamespace(eod=11, generation_config={"eos_token_id": [2, 11]})
    )
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = controller
    engine.stop_word_being_finished_ids = set()
    return engine


def _stop_metadata_request(tokens, termination_id=11, stop_ids=None):
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
    engine = _stop_metadata_engine()
    request = _stop_metadata_request([], termination_id, [7])
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
    request = _stop_metadata_request(tokens, termination_id, stop_ids)
    _stop_metadata_engine()._set_finish_reason(request)
    assert request.finish_reason == expected
    restored = DynamicInferenceRequest.deserialize(request.serialize())
    assert restored.finish_reason == expected
    assert restored.sampling_params.stop_token_ids == stop_ids
    merged = DynamicInferenceRequestRecord.from_request(request).merge()
    assert merged.finish_reason == expected


def test_stripped_string_stop_reason_is_preserved():
    request = _stop_metadata_request([4, 5, 6])
    request.finish_reason = "stop"
    _stop_metadata_engine()._set_finish_reason(request)
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
    "tokens,budget,expected,kept,resume",
    [
        ([7], 3, "stop", [7], False),
        ([5, 7, 9], 3, "stop", [5, 7], False),
        pytest.param([5, 7, 9], 1, "length", [5], False, id="length-before-token-stop"),
        pytest.param([7], 0, "length", [], False, id="zero-budget-discards-token-stop"),
        pytest.param([7], 1, "stop", [7], False, id="token-stop-at-length-boundary"),
        ([11], 1, "stop", [11], False),
        ([2], 1, "stop", [2], False),
        ([5], 1, "length", [5], False),
        pytest.param([8], 3, "stop", [], True, id="checkpoint-chunked-stop"),
        pytest.param([7], 1, None, [7], False, id="handoff-token-stop"),
        pytest.param([8], 1, None, [], False, id="handoff-stripped-stop"),
    ],
)
def test_postprocess_completion_and_deferred_stop(tokens, budget, expected, kept, resume):
    async def run():
        engine = _stop_metadata_engine()
        engine.finished_request_count = 0
        engine.evicted_request_count = 0
        engine.track_generated_token_events = False
        engine.num_speculative_tokens = len(tokens) - 1
        engine._spec_steps = 0
        engine.stop_word_finished_request_ids = set()
        engine.context = SimpleNamespace(
            kv_block_allocator=object(),
            remove_vlm_request_data=lambda _: None,
            chunked_prefill_request_id=-1,
        )
        request = _stop_metadata_request([], stop_ids=[7])
        request.stop_word_ids = [[8]]
        if expected is None:
            request.sampling_params.do_kv_handoff = True
            engine._prepare_handoff_metadata_batch = mock.Mock(return_value={})
            engine._capture_handoff_meta = mock.Mock()
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
            if resume:
                # Eviction and recompute-suspend both checkpoint before re-admission.
                engine.requests[0].record.checkpoint()
                assert engine.get_request(0).generated_tokens == []
                assert engine.get_request(0).finish_reason is None
                engine.context.chunked_prefill_request_id = 0
                for _ in range(2):
                    assert engine._get_and_clear_stop_word_finished_ids(active) == set()
                    assert engine.stop_word_finished_request_ids == {0}
                engine.context.chunked_prefill_request_id = -1
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
