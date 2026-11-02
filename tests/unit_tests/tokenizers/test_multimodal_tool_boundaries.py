# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.tokenizers.vision.libraries import multimodal_tokenizer
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
    _build_nemotron6_moe_native_tool_boundary_mask,
    _find_nemotron6_moe_assistant_indices,
)

USER = [10, 3263, 1010, 999, 11, 1010]
ASSISTANT = [10, 1503, 19464, 1010, 200, 11, 1010]
TOOL_RESPONSE = [10, 3263, 1010, 16, 300, 11, 1010]


def _assistant_indices(tokens: np.ndarray) -> np.ndarray:
    return np.where((tokens[:-2] == 10) & (tokens[1:-1] == 1503) & (tokens[2:] == 19464))[0] + 1


@pytest.mark.parametrize(
    ("has_nonempty_thinking_trace", "tool_response_as_turn_boundary", "expected_assistant_turns"),
    [(True, False, [0, 1, 2]), (True, True, [2]), (False, False, [0, 1, 2]), (False, True, [2])],
)
def test_nemotron6_tool_response_turn_boundary(
    has_nonempty_thinking_trace, tool_response_as_turn_boundary, expected_assistant_turns
):
    tokens = np.asarray(
        USER + ASSISTANT + TOOL_RESPONSE + ASSISTANT + TOOL_RESPONSE + ASSISTANT, dtype=np.int64
    )

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=has_nonempty_thinking_trace,
        tool_response_as_turn_boundary=tool_response_as_turn_boundary,
        native_tool_boundary_mask=np.asarray([False, True, True]),
    )

    assert (
        selected_indices.tolist() == _assistant_indices(tokens)[expected_assistant_turns].tolist()
    )


def test_nemotron6_tool_boundary_does_not_change_text_sample():
    tokens = np.asarray(USER + ASSISTANT + USER + ASSISTANT, dtype=np.int64)

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=False,
        tool_response_as_turn_boundary=True,
        native_tool_boundary_mask=np.asarray([False, False]),
    )

    assert len(selected_indices) == 2


@pytest.mark.parametrize("tool_response_as_turn_boundary", [False, True])
def test_nemotron6_tool_boundary_preserves_last_user_boundary(tool_response_as_turn_boundary):
    tokens = np.asarray(USER + ASSISTANT + USER + ASSISTANT, dtype=np.int64)

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=True,
        tool_response_as_turn_boundary=tool_response_as_turn_boundary,
        native_tool_boundary_mask=np.asarray([False, False]),
    )

    assert len(selected_indices) == 1


def test_nemotron6_tool_boundary_ignores_legacy_markup():
    tokens = np.asarray(USER + ASSISTANT + TOOL_RESPONSE + ASSISTANT, dtype=np.int64)

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=False,
        tool_response_as_turn_boundary=True,
        native_tool_boundary_mask=np.asarray([False, False]),
    )

    assert len(selected_indices) == 2


def test_nemotron6_no_thinking_uses_latest_tool_not_latest_user():
    tokens = np.asarray(
        USER + ASSISTANT + TOOL_RESPONSE + ASSISTANT + USER + ASSISTANT, dtype=np.int64
    )

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=False,
        tool_response_as_turn_boundary=True,
        native_tool_boundary_mask=np.asarray([False, True, False]),
    )

    assert len(selected_indices) == 2


def test_nemotron6_coalesces_consecutive_native_tool_boundaries():
    conversation = [
        {"role": "user"},
        {"role": "assistant"},
        {"role": "tool"},
        {"role": "tool"},
        {"role": "assistant"},
    ]
    native_tool_boundary_mask = _build_nemotron6_moe_native_tool_boundary_mask(conversation)
    tokens = np.asarray(USER + ASSISTANT + TOOL_RESPONSE + ASSISTANT, dtype=np.int64)

    selected_indices = _find_nemotron6_moe_assistant_indices(
        tokens,
        prompt_format="nemotron6-moe",
        train_only_on_last_assistant_turn=True,
        has_nonempty_thinking_trace=False,
        tool_response_as_turn_boundary=True,
        native_tool_boundary_mask=native_tool_boundary_mask,
    )

    assert native_tool_boundary_mask.tolist() == [False, True]
    assert len(selected_indices) == 1


def test_nemotron6_assistant_masking_rejects_other_prompt_format():
    with pytest.raises(ValueError, match="only supports prompt_format='nemotron6-moe'"):
        _find_nemotron6_moe_assistant_indices(
            np.asarray([], dtype=np.int64),
            prompt_format="chatml",
            train_only_on_last_assistant_turn=True,
            has_nonempty_thinking_trace=False,
            tool_response_as_turn_boundary=False,
            native_tool_boundary_mask=np.asarray([], dtype=bool),
        )


def test_nemotron6_tool_boundary_rejects_role_render_mismatch():
    tokens = np.asarray(USER + ASSISTANT, dtype=np.int64)

    with pytest.raises(ValueError, match="roles do not match rendered user-turn"):
        _find_nemotron6_moe_assistant_indices(
            tokens,
            prompt_format="nemotron6-moe",
            train_only_on_last_assistant_turn=True,
            has_nonempty_thinking_trace=False,
            tool_response_as_turn_boundary=True,
            native_tool_boundary_mask=np.asarray([False, True]),
        )


def test_nemotron6_tool_boundary_requires_rendered_user_boundary():
    tokens = np.asarray(ASSISTANT, dtype=np.int64)

    with pytest.raises(ValueError, match="did not render as user boundaries"):
        _find_nemotron6_moe_assistant_indices(
            tokens,
            prompt_format="nemotron6-moe",
            train_only_on_last_assistant_turn=True,
            has_nonempty_thinking_trace=False,
            tool_response_as_turn_boundary=True,
            native_tool_boundary_mask=np.asarray([True]),
        )


class _Nemotron6TokenizerContractStub:
    def __init__(self, overrides=None):
        self._decodes = {
            (10,): "<|im_start|>",
            (11,): "<|im_end|>",
            (16,): "<tool_response>",
            (1010,): "\n",
            (1503, 19464): "assistant",
            (3263,): "user",
        }
        if overrides is not None:
            self._decodes.update(overrides)

    def add_tokens(self, tokens, special_tokens):
        return 0

    def __len__(self):
        return 20000

    def convert_tokens_to_ids(self, token):
        return 0

    def decode(self, token_ids, clean_up_tokenization_spaces):
        return self._decodes[tuple(token_ids)]

    def get_vocab(self):
        return {}


def _patch_contract_tokenizer(monkeypatch, tokenizer):
    auto_tokenizer = SimpleNamespace(from_pretrained=lambda **kwargs: tokenizer)
    monkeypatch.setattr(multimodal_tokenizer, "HAVE_TRANSFORMERS", True)
    monkeypatch.setattr(
        multimodal_tokenizer,
        "transformers",
        SimpleNamespace(AutoTokenizer=auto_tokenizer),
        raising=False,
    )


def test_nemotron6_tokenizer_contract_accepts_expected_ids(monkeypatch):
    _patch_contract_tokenizer(monkeypatch, _Nemotron6TokenizerContractStub())

    MegatronMultimodalTokenizer(
        path="unused", prompt_format="nemotron6-moe", special_tokens=[], image_tag_type=""
    )


def test_nemotron6_tokenizer_contract_is_validated_at_initialization(monkeypatch):
    _patch_contract_tokenizer(monkeypatch, _Nemotron6TokenizerContractStub({(10,): "unexpected"}))

    with pytest.raises(ValueError, match=r"token IDs \[10\].*expected '<\|im_start\|>'"):
        MegatronMultimodalTokenizer(
            path="unused", prompt_format="nemotron6-moe", special_tokens=[], image_tag_type=""
        )


def test_nemotron6_tool_boundary_requires_last_assistant_mode():
    tokenizer = MegatronMultimodalTokenizer.__new__(MegatronMultimodalTokenizer)

    with pytest.raises(ValueError, match="requires train_only_on_last_assistant_turn"):
        tokenizer.tokenize_conversation(
            [], return_target=True, add_generation_prompt=False, tool_response_as_turn_boundary=True
        )


def test_nemotron6_tool_boundary_rejects_raw_conversation():
    tokenizer = MegatronMultimodalTokenizer.__new__(MegatronMultimodalTokenizer)

    with pytest.raises(ValueError, match="skip_chat_template=True"):
        tokenizer.tokenize_conversation(
            [],
            return_target=True,
            add_generation_prompt=False,
            train_only_on_last_assistant_turn=True,
            skip_chat_template=True,
            tool_response_as_turn_boundary=True,
        )
