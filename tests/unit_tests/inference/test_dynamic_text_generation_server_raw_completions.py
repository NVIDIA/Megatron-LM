# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.raw_completions import (
    _first_single_token_stop,
    _parse_stop_token_sequences,
    _parse_top_n_logprobs,
    _trim_at_stop_sequences,
    _truncate_top_logprobs,
)


def test_parse_stop_token_sequences_accepts_token_zero():
    assert _parse_stop_token_sequences(
        {"stop_token_ids": [0, 3], "stop_token_id_sequences": [[4, 5]]}
    ) == [[0], [3], [4, 5]]
    assert _first_single_token_stop([[0]]) == 0


@pytest.mark.parametrize(
    "request",
    [
        {"stop_token_ids": [False]},
        {"stop_token_id_sequences": [[1, True]]},
        {"stop_token_id_sequences": [[]]},
    ],
)
def test_parse_stop_token_sequences_rejects_invalid_tokens(request):
    with pytest.raises(ValueError):
        _parse_stop_token_sequences(request)


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
    ("request", "prompt_logprobs", "logprobs", "expected"),
    [
        ({"top_prompt_logprobs": 1}, True, False, (1, 0, 1)),
        ({"top_logprobs": 2}, False, True, (0, 2, 2)),
        ({"top_prompt_logprobs": 2, "top_logprobs": 1}, True, True, (2, 1, 2)),
        ({}, False, False, (0, 0, 0)),
    ],
)
def test_parse_top_n_logprobs_accepts_valid_modes(request, prompt_logprobs, logprobs, expected):
    assert _parse_top_n_logprobs(request, prompt_logprobs, logprobs) == expected


@pytest.mark.parametrize(
    ("request", "prompt_logprobs", "logprobs"),
    [
        ({"top_prompt_logprobs": 1}, False, False),
        ({"top_logprobs": 1}, False, False),
        ({"top_prompt_logprobs": -1}, True, False),
    ],
)
def test_parse_top_n_logprobs_rejects_invalid_modes(request, prompt_logprobs, logprobs):
    with pytest.raises(ValueError):
        _parse_top_n_logprobs(request, prompt_logprobs, logprobs)


def test_truncate_top_logprobs_preserves_requested_candidates():
    top_logprobs = [{"10": -2.0, "11": -0.5, "12": -1.0}, {"20": -0.2, "21": -1.2, "22": -2.2}]

    assert _truncate_top_logprobs(top_logprobs, 2) == [
        {"11": -0.5, "12": -1.0},
        {"20": -0.2, "21": -1.2},
    ]
    assert _truncate_top_logprobs(top_logprobs, 0) is None
