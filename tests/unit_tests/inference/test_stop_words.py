# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for stop word functionality in dynamic inference."""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from megatron.core.inference.sampling_params import SamplingParams


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
