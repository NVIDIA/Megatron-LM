# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.raw_completions import (
    _get_non_none,
    _parse_stop_token_sequences,
    _parse_top_n_logprobs,
    _single_token_stops,
    _trim_at_stop_sequences,
    _truncate_top_logprobs,
)


def test_parse_stop_token_sequences_accepts_token_zero():
    assert _parse_stop_token_sequences(
        {"stop_token_ids": [0, 3], "stop_token_id_sequences": [[4, 5]]}
    ) == [[0], [3], [4, 5]]


def test_single_token_stops_collects_every_single_token_sequence():
    assert _single_token_stops([[0], [4, 5], [3]]) == [0, 3]
    assert _single_token_stops([[4, 5]]) is None
    assert _single_token_stops([]) is None


def test_get_non_none_treats_null_as_omitted():
    assert _get_non_none({"temperature": None}, "temperature", 0.6) == 0.6
    assert _get_non_none({}, "temperature", 0.6) == 0.6
    assert _get_non_none({"temperature": 0.0}, "temperature", 0.6) == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {"stop_token_ids": [False]},
        {"stop_token_id_sequences": [[1, True]]},
        {"stop_token_id_sequences": [[]]},
    ],
)
def test_parse_stop_token_sequences_rejects_invalid_tokens(payload):
    with pytest.raises(ValueError):
        _parse_stop_token_sequences(payload)


def test_trim_at_stop_sequences_uses_earliest_match_and_trims_logprobs():
    tokens, logprobs, hit = _trim_at_stop_sequences(
        [10, 11, 12, 13, 14], [-1.0, -1.1, -1.2, -1.3, -1.4], [[13], [11, 12]]
    )

    assert hit
    assert tokens == [10]
    assert logprobs == [-1.0]


def test_trim_at_stop_sequences_trims_token_metadata():
    tokens, top_logprobs, hit = _trim_at_stop_sequences(
        [10, 11, 12], [{"10": -1.0}, {"11": -1.1}, {"12": -1.2}], [[12]]
    )

    assert hit
    assert tokens == [10, 11]
    assert top_logprobs == [{"10": -1.0}, {"11": -1.1}]


def test_trim_at_stop_sequences_no_match_is_noop():
    tokens, logprobs, hit = _trim_at_stop_sequences([10, 11], [-1.0, -1.1], [[12]])

    assert not hit
    assert tokens == [10, 11]
    assert logprobs == [-1.0, -1.1]


@pytest.mark.parametrize(
    ("payload", "prompt_logprobs", "logprobs", "expected"),
    [
        ({"top_prompt_logprobs": 1}, True, False, (1, 0, 1)),
        ({"top_logprobs": 2}, False, True, (0, 2, 2)),
        ({"top_prompt_logprobs": 2, "top_logprobs": 1}, True, True, (2, 1, 2)),
        ({}, False, False, (0, 0, 0)),
    ],
)
def test_parse_top_n_logprobs_accepts_valid_modes(payload, prompt_logprobs, logprobs, expected):
    assert _parse_top_n_logprobs(payload, prompt_logprobs, logprobs) == expected


@pytest.mark.parametrize(
    ("payload", "prompt_logprobs", "logprobs"),
    [
        ({"top_prompt_logprobs": 1}, False, False),
        ({"top_logprobs": 1}, False, False),
        ({"top_prompt_logprobs": -1}, True, False),
    ],
)
def test_parse_top_n_logprobs_rejects_invalid_modes(payload, prompt_logprobs, logprobs):
    with pytest.raises(ValueError):
        _parse_top_n_logprobs(payload, prompt_logprobs, logprobs)


def test_truncate_top_logprobs_preserves_requested_candidates():
    top_logprobs = [{"10": -2.0, "11": -0.5, "12": -1.0}, {"20": -0.2, "21": -1.2, "22": -2.2}]

    assert _truncate_top_logprobs(top_logprobs, 2) == [
        {"11": -0.5, "12": -1.0},
        {"20": -0.2, "21": -1.2},
    ]
    assert _truncate_top_logprobs(top_logprobs, 0) is None


def test_truncate_top_logprobs_passes_through_missing_positions():
    assert _truncate_top_logprobs([None, {"10": -2.0, "11": -0.5}], 1) == [None, {"11": -0.5}]
