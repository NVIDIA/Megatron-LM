# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight unit tests for `dynamic_engine.py`.

The existing `test_dynamic_engine.py` exercises the engine end-to-end on a
real model + KV cache + ProcessGroup (heavy). This file complements it with
mock-based unit tests for the standalone helpers and engine methods that
operate on internal state, so they run quickly with no GPU / no distributed
setup and can boost coverage of the file's pure-Python surface.
"""

from collections import deque
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.engines.dynamic_engine import (
    DynamicInferenceEngine,
    EngineState,
    EngineSuspendedError,
    RequestEntry,
    format_mem_bytes,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams


def _make_request(
    request_id=1,
    prompt_tokens=None,
    generated_tokens=None,
    stop_word_ids=None,
    precomputed_block_hashes=None,
    sampling_params=None,
):
    if prompt_tokens is None:
        prompt_tokens = torch.tensor([1, 2, 3])
    if sampling_params is None:
        sampling_params = SamplingParams(num_tokens_to_generate=5, termination_id=0)
    req = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        sampling_params=sampling_params,
        generated_tokens=list(generated_tokens or []),
    )
    req.stop_word_ids = stop_word_ids
    if precomputed_block_hashes is not None:
        req.precomputed_block_hashes = precomputed_block_hashes
    return req


def _make_engine_skeleton(
    waiting_request_ids=None,
    requests=None,
    stop_word_finished=None,
    num_speculative_tokens=0,
    prefix_coordination_waits=0,
    mamba_hash_to_block_id=None,
    context_has_unfinished=False,
):
    """Build a DynamicInferenceEngine via __new__ + injected attributes.

    The real __init__ asserts isinstance() on a TextGenerationController and a
    DynamicInferenceContext (heavy). Bypassing __init__ lets us exercise the
    methods that only touch self.* attributes.
    """
    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng.waiting_request_ids = deque(waiting_request_ids or [])
    eng.requests = requests if requests is not None else {}
    eng.stop_word_finished_request_ids = set(stop_word_finished or [])
    eng.stop_word_being_finished_ids = set()
    eng.num_speculative_tokens = num_speculative_tokens
    eng._prefix_coordination_waits = prefix_coordination_waits
    eng.context = MagicMock()
    eng.context.has_unfinished_requests.return_value = context_has_unfinished
    eng.context.mamba_slot_allocator = MagicMock()
    eng.context.mamba_slot_allocator.hash_to_block_id = mamba_hash_to_block_id or {}
    return eng


class TestEngineState:

    def test_engine_state_members_distinct(self):
        """All EngineState members have unique values."""
        values = {s.value for s in EngineState}
        assert len(values) == len(list(EngineState))

    def test_engine_state_contains_protocol_members(self):
        """The enum exposes every state used by the protocol."""
        names = {s.name for s in EngineState}
        for required in {
            "RUNNING",
            "PAUSING",
            "PAUSED",
            "UNPAUSING",
            "SUSPENDING",
            "SUSPENDED",
            "RESUMING",
            "RESUMED",
            "STOPPING",
            "STOPPED",
        }:
            assert required in names

    def test_state_events_subset(self):
        """`_STATE_EVENTS` contains only stable states."""
        stable = set(DynamicInferenceEngine._STATE_EVENTS)
        assert stable.issubset(set(EngineState))
        # Pausing / suspending / etc. are transitional, not stable.
        assert EngineState.PAUSING not in stable
        assert EngineState.RUNNING in stable
        assert EngineState.STOPPED in stable


class TestEngineSuspendedError:

    def test_inherits_from_exception(self):
        """EngineSuspendedError is an Exception subclass."""
        assert issubclass(EngineSuspendedError, Exception)

    def test_can_be_raised_and_caught(self):
        """The error can be raised and caught."""
        with pytest.raises(EngineSuspendedError):
            raise EngineSuspendedError("engine paused")


class TestFormatMemBytes:

    def test_zero_bytes_uses_fallthrough(self):
        """0 bytes hits the trailing fallthrough format ('%d bytes')."""
        # The for-loop only matches mem_bytes >= 1**0 = 1, so 0 falls through.
        assert format_mem_bytes(0) == "0 bytes"

    def test_sub_kib_uses_bytes_with_decimal(self):
        """Values in [1, 1024) render via the loop's bytes branch with one decimal."""
        # 512 >= 1**0 → "%.1f %s" % (512/1, "bytes") = "512.0 bytes".
        assert format_mem_bytes(512) == "512.0 bytes"

    def test_kib(self):
        """Values in [1 KiB, 1 MiB) render in kb."""
        out = format_mem_bytes(1024)
        assert out.endswith("kb")
        assert out.startswith("1.0")

    def test_mib(self):
        """Values in [1 MiB, 1 GiB) render in mb."""
        out = format_mem_bytes(2 * 1024**2)
        assert out.endswith("mb")
        assert out.startswith("2.0")

    def test_gib(self):
        """Values in [1 GiB, 1 TiB) render in gb."""
        out = format_mem_bytes(3 * 1024**3)
        assert out.endswith("gb")
        assert out.startswith("3.0")

    def test_tib(self):
        """Values >= 1 TiB render in tb."""
        out = format_mem_bytes(5 * 1024**4)
        assert out.endswith("tb")
        assert out.startswith("5.0")


class TestRequestEntry:

    def test_constructs_with_record_and_future(self):
        """RequestEntry holds a record + an asyncio.Future."""
        import asyncio

        req = _make_request()
        record = DynamicInferenceRequestRecord.from_request(req)
        loop = asyncio.new_event_loop()
        try:
            fut = loop.create_future()
            entry = RequestEntry(record=record, future=fut)
            assert entry.record is record
            assert entry.future is fut
        finally:
            loop.close()

    def test_kw_only_construction(self):
        """RequestEntry rejects positional args (kw_only dataclass)."""
        with pytest.raises(TypeError):
            RequestEntry(MagicMock(), MagicMock())  # positional → TypeError


class TestHasUnfinishedRequests:

    def test_returns_true_when_context_has_unfinished(self):
        """has_unfinished_requests defers to the context when waiting queue is empty."""
        eng = _make_engine_skeleton(context_has_unfinished=True)
        assert eng.has_unfinished_requests() is True

    def test_returns_true_when_waiting_queue_nonempty(self):
        """has_unfinished_requests is True when only the waiting queue has entries."""
        eng = _make_engine_skeleton(context_has_unfinished=False, waiting_request_ids=[1, 2])
        assert eng.has_unfinished_requests() is True

    def test_returns_false_when_both_empty(self):
        """has_unfinished_requests is False when both context and waiting queue are empty."""
        eng = _make_engine_skeleton(context_has_unfinished=False)
        assert eng.has_unfinished_requests() is False


class TestGetRequest:

    def test_returns_latest_record_entry(self):
        """get_request(id) returns the most recent request in the record."""
        req = _make_request(request_id=42)
        record = DynamicInferenceRequestRecord.from_request(req)
        entry = RequestEntry(record=record, future=MagicMock())
        eng = _make_engine_skeleton(requests={42: entry})
        assert eng.get_request(42) is req

    def test_raises_keyerror_for_unknown_id(self):
        """Unknown request id raises KeyError."""
        eng = _make_engine_skeleton(requests={})
        with pytest.raises(KeyError):
            eng.get_request(999)


class TestGetPrefixCoordinationMetrics:

    def test_returns_waits_dict(self):
        """get_prefix_coordination_metrics surfaces the internal counter."""
        eng = _make_engine_skeleton(prefix_coordination_waits=7)
        assert eng.get_prefix_coordination_metrics() == {"waits": 7}


class TestGetAndClearStopWordFinishedIds:

    def test_empty_set_short_circuits(self):
        """Empty stop-word set returns an empty set without inspecting active_request_ids."""
        eng = _make_engine_skeleton(stop_word_finished=set())
        assert eng._get_and_clear_stop_word_finished_ids([1, 2, 3]) == set()
        assert eng.stop_word_being_finished_ids == set()

    def test_intersects_with_active_and_clears(self):
        """Returns the intersection with active_request_ids and clears those ids."""
        eng = _make_engine_skeleton(stop_word_finished={1, 2, 3})
        result = eng._get_and_clear_stop_word_finished_ids([2, 3, 4])
        assert result == {2, 3}
        # Cleared ids removed from the pending set.
        assert eng.stop_word_finished_request_ids == {1}
        # being_finished mirrors the result.
        assert eng.stop_word_being_finished_ids == {2, 3}

    def test_disjoint_active_returns_empty(self):
        """When no stop-word-finished id is currently active, returns empty set."""
        eng = _make_engine_skeleton(stop_word_finished={1, 2})
        result = eng._get_and_clear_stop_word_finished_ids([10, 20])
        assert result == set()
        # Pending set is unchanged (nothing intersected).
        assert eng.stop_word_finished_request_ids == {1, 2}


class TestCheckStopWordsForRequestPostAppend:

    def test_no_stop_words_returns_no_hit(self):
        """A request without stop_word_ids never reports a hit."""
        eng = _make_engine_skeleton(num_speculative_tokens=0)
        req = _make_request(stop_word_ids=None)
        assert eng._check_stop_words_for_request_post_append(req) == (False, 0)

    def test_empty_stop_words_returns_no_hit(self):
        """Empty stop_word_ids list returns no hit."""
        eng = _make_engine_skeleton(num_speculative_tokens=0)
        req = _make_request(stop_word_ids=[])
        assert eng._check_stop_words_for_request_post_append(req) == (False, 0)

    def test_too_few_tokens_returns_no_hit(self):
        """If generated_tokens is shorter than the stop word, no hit."""
        eng = _make_engine_skeleton(num_speculative_tokens=0)
        req = _make_request(generated_tokens=[1], stop_word_ids=[[1, 2, 3]])
        assert eng._check_stop_words_for_request_post_append(req) == (False, 0)

    def test_stop_word_at_end_no_speculative(self):
        """Stop word ending exactly at the last token: hit, trim removes the stop sequence."""
        eng = _make_engine_skeleton(num_speculative_tokens=0)
        # detokenize_stop_sequence=False (default) trims the stop word out.
        req = _make_request(generated_tokens=[7, 8, 9, 99, 100], stop_word_ids=[[99, 100]])
        hit, trimmed = eng._check_stop_words_for_request_post_append(req)
        assert hit is True
        assert trimmed == 2  # i=0 + stop_len=2 since detokenize_stop_sequence=False
        assert req.generated_tokens == [7, 8, 9]

    def test_stop_word_at_end_keeps_when_detokenize_stop_sequence(self):
        """detokenize_stop_sequence=True keeps the stop sequence in generated_tokens."""
        sp = SamplingParams(
            num_tokens_to_generate=5, termination_id=0, detokenize_stop_sequence=True
        )
        eng = _make_engine_skeleton(num_speculative_tokens=0)
        req = _make_request(
            generated_tokens=[1, 2, 99, 100], stop_word_ids=[[99, 100]], sampling_params=sp
        )
        hit, trimmed = eng._check_stop_words_for_request_post_append(req)
        assert hit is True
        assert trimmed == 0
        assert req.generated_tokens == [1, 2, 99, 100]

    def test_stop_word_inside_speculative_tokens_trims_trailing(self):
        """With speculative decoding, a stop word followed by extra tokens trims the trailing slice."""
        eng = _make_engine_skeleton(num_speculative_tokens=2)
        # Tokens [..., 99, 100, 200, 201]. Stop word [99, 100] sits 2 positions from the end.
        # detokenize_stop_sequence=False → trim = i + stop_len = 2 + 2 = 4 → generated_tokens cut.
        req = _make_request(generated_tokens=[1, 2, 99, 100, 200, 201], stop_word_ids=[[99, 100]])
        hit, trimmed = eng._check_stop_words_for_request_post_append(req)
        assert hit is True
        # Stop word ends at index -3; with i=2, end_idx=-2 picks generated_tokens[-4:-2] = [99,100].
        assert trimmed == 4
        assert req.generated_tokens == [1, 2]

    def test_no_match_returns_no_hit(self):
        """Tokens that don't end with any stop word return (False, 0)."""
        eng = _make_engine_skeleton(num_speculative_tokens=2)
        req = _make_request(generated_tokens=[1, 2, 3, 4], stop_word_ids=[[99, 100], [200]])
        hit, trimmed = eng._check_stop_words_for_request_post_append(req)
        assert hit is False
        assert trimmed == 0


class TestFindMambaMatchCount:

    def test_no_precomputed_hashes_returns_zero(self):
        """A request with no precomputed_block_hashes returns 0."""
        eng = _make_engine_skeleton(mamba_hash_to_block_id={})
        req = _make_request(precomputed_block_hashes=[])
        assert eng._find_mamba_match_count(req) == 0

    def test_returns_index_of_farthest_match_plus_one(self):
        """Returns i+1 of the farthest hash present in the mamba map."""
        eng = _make_engine_skeleton(mamba_hash_to_block_id={111: 0, 333: 2})
        req = _make_request(precomputed_block_hashes=[111, 222, 333, 444])
        # Iterate from the end: 444 ✗, 333 ✓ at i=2 → return i+1 = 3.
        assert eng._find_mamba_match_count(req) == 3

    def test_no_match_returns_zero(self):
        """If no hash is in the mamba map, returns 0."""
        eng = _make_engine_skeleton(mamba_hash_to_block_id={1: 0})
        req = _make_request(precomputed_block_hashes=[5, 6, 7])
        assert eng._find_mamba_match_count(req) == 0

    def test_only_first_block_matches(self):
        """A match at index 0 still produces a positive count."""
        eng = _make_engine_skeleton(mamba_hash_to_block_id={42: 0})
        req = _make_request(precomputed_block_hashes=[42, 7, 8])
        # End-to-start: 8 ✗, 7 ✗, 42 ✓ at i=0 → return 1.
        assert eng._find_mamba_match_count(req) == 1
