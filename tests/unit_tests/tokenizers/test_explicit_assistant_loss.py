# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.core.tokenizers.vision.libraries import multimodal_tokenizer
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)


class MockNemotron6Tokenizer:
    """Render the token boundaries used by the nemotron6 masking path."""

    _special_to_id = {
        "\ue001": 10,
        "\ue002": 1503,
        "\ue003": 19464,
        "\ue004": 1010,
        "\ue005": 11,
        "\ue006": 3263,
        "\ue007": 4000,
    }
    _id_to_special = {value: key for key, value in _special_to_id.items()}
    _text_offset = 20000
    _contract_decodes = {
        (10,): "<|im_start|>",
        (11,): "<|im_end|>",
        (16,): "<tool_response>",
        (1010,): "\n",
        (1503, 19464): "assistant",
        (3263,): "user",
    }

    def __init__(
        self, *, drop_last_assistant_header: bool = False, drop_last_assistant_end: bool = False
    ) -> None:
        self.pad_token_id = 0
        self.drop_last_assistant_header = drop_last_assistant_header
        self.drop_last_assistant_end = drop_last_assistant_end

    def __len__(self) -> int:
        return 65536

    def add_tokens(self, extra_tokens: list[str], *args, **kwargs) -> int:
        return 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [
            self._special_to_id.get(character, ord(character) + self._text_offset)
            for character in text
        ]

    def decode(self, tokens: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        del clean_up_tokenization_spaces
        contract_value = self._contract_decodes.get(tuple(tokens))
        if contract_value is not None:
            return contract_value
        return "".join(
            self._id_to_special.get(token, chr(token - self._text_offset)) for token in tokens
        )

    def get_vocab(self) -> dict[str, int]:
        return {}

    def apply_chat_template(
        self, conversation: list[dict], *, tokenize: bool, add_generation_prompt: bool, **kwargs
    ) -> str:
        del kwargs
        assert not tokenize
        rendered: list[str] = []
        assistant_count = sum(turn["role"] == "assistant" for turn in conversation)
        seen_assistants = 0
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "assistant":
                seen_assistants += 1
                is_last = seen_assistants == assistant_count
                header = (
                    "\ue001\ue007\ue004"
                    if self.drop_last_assistant_header and is_last
                    else "\ue001\ue002\ue003\ue004"
                )
                end = "\ue007\ue004" if self.drop_last_assistant_end and is_last else "\ue005\ue004"
                rendered.append(header + content + end)
            elif role == "user":
                rendered.append("\ue001\ue006\ue004" + content + "\ue005\ue004")
            else:
                rendered.append("\ue001\ue007\ue004" + content + "\ue005\ue004")
        if add_generation_prompt:
            rendered.append("\ue001\ue002\ue003\ue004")
        return "".join(rendered)


def _tokenizer(
    monkeypatch,
    *,
    keep_history_thinking: bool = True,
    drop_last_assistant_header: bool = False,
    drop_last_assistant_end: bool = False,
) -> MegatronMultimodalTokenizer:
    tokenizer = MockNemotron6Tokenizer(
        drop_last_assistant_header=drop_last_assistant_header,
        drop_last_assistant_end=drop_last_assistant_end,
    )
    auto_tokenizer = SimpleNamespace(from_pretrained=lambda **kwargs: tokenizer)
    monkeypatch.setattr(multimodal_tokenizer, "HAVE_TRANSFORMERS", True)
    monkeypatch.setattr(
        multimodal_tokenizer,
        "transformers",
        SimpleNamespace(AutoTokenizer=auto_tokenizer),
        raising=False,
    )
    return MegatronMultimodalTokenizer(
        path="unused",
        prompt_format="nemotron6-moe",
        special_tokens=[],
        image_tag_type="",
        keep_history_thinking=keep_history_thinking,
    )


def _conversation() -> list[dict]:
    return [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "V"},
        {"role": "assistant", "content": "B"},
        {"role": "tool", "content": "T"},
        {"role": "assistant", "content": "C"},
    ]


def _text_token(character: str) -> int:
    return ord(character) + MockNemotron6Tokenizer._text_offset


def _target_span_count(target: np.ndarray) -> int:
    trainable = target != IGNORE_INDEX
    return int(trainable[0]) + int(np.sum(trainable[1:] & ~trainable[:-1]))


def test_explicit_assistant_loss_selects_multiple_turns(monkeypatch) -> None:
    conversation = _conversation()
    conversation[4]["content"] = [{"type": "text", "text": "B"}]
    tokens, target = _tokenizer(monkeypatch).tokenize_conversation(
        conversation,
        return_target=True,
        add_generation_prompt=False,
        assistant_turn_loss=[False, True, True],
    )

    assert _target_span_count(target) == 2
    for character, selected in (("A", False), ("B", True), ("C", True)):
        [position] = np.where(tokens == _text_token(character))[0]
        if selected:
            assert target[position] == tokens[position]
        else:
            assert target[position] == IGNORE_INDEX

    assistant_prefixes = np.where(tokens == 1503)[0]
    assert len(assistant_prefixes) == 3
    assert np.all(target[assistant_prefixes] == IGNORE_INDEX)


def test_legacy_nemotron6_masking_remains_implicit(monkeypatch) -> None:
    _, target = _tokenizer(monkeypatch, keep_history_thinking=False).tokenize_conversation(
        _conversation(), return_target=True, add_generation_prompt=False
    )
    assert _target_span_count(target) == 3


def test_legacy_last_assistant_mode_remains_independent(monkeypatch) -> None:
    conversation = _conversation()
    for turn in conversation:
        if turn["role"] == "assistant":
            turn["content"] = f"<think>reason</think>{turn['content']}"

    tokens, target = _tokenizer(monkeypatch, keep_history_thinking=False).tokenize_conversation(
        conversation,
        return_target=True,
        add_generation_prompt=False,
        train_only_on_last_assistant_turn=True,
    )

    assert _target_span_count(target) == 2
    for character, selected in (("A", False), ("B", True), ("C", True)):
        [position] = np.where(tokens == _text_token(character))[0]
        assert bool(target[position] != IGNORE_INDEX) is selected


def test_legacy_positional_skip_chat_template_argument_is_preserved(monkeypatch) -> None:
    _, target = _tokenizer(monkeypatch, keep_history_thinking=False).tokenize_conversation(
        _conversation(), True, False, False, False
    )
    assert _target_span_count(target) == 3


@pytest.mark.parametrize(
    ("assistant_turn_loss", "error"),
    [
        ([True], "length does not match"),
        ([False, False, False], "at least one true"),
        ([False, 1, True], "values must be booleans"),
    ],
)
def test_explicit_assistant_loss_rejects_invalid_masks(
    monkeypatch, assistant_turn_loss: list[bool], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _tokenizer(monkeypatch).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            assistant_turn_loss=assistant_turn_loss,
        )


def test_explicit_assistant_loss_requires_history_thinking(monkeypatch) -> None:
    with pytest.raises(ValueError, match="tokenizer-keep-history-thinking"):
        _tokenizer(monkeypatch, keep_history_thinking=False).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_last_turn_mode(monkeypatch) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _tokenizer(monkeypatch).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            train_only_on_last_assistant_turn=True,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_tool_boundary_mode(monkeypatch) -> None:
    with pytest.raises(ValueError, match="incompatible with tool_response"):
        _tokenizer(monkeypatch).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            tool_response_as_turn_boundary=True,
            assistant_turn_loss=[False, True, True],
        )


@pytest.mark.parametrize(("return_target", "add_generation_prompt"), [(False, False), (True, True)])
def test_explicit_assistant_loss_requires_training_output_mode(
    monkeypatch, return_target: bool, add_generation_prompt: bool
) -> None:
    with pytest.raises(ValueError, match="requires return_target=True"):
        _tokenizer(monkeypatch).tokenize_conversation(
            _conversation(),
            return_target=return_target,
            add_generation_prompt=add_generation_prompt,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_raw_conversation(monkeypatch) -> None:
    with pytest.raises(ValueError, match="incompatible with skip_chat_template"):
        _tokenizer(monkeypatch).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            skip_chat_template=True,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_unsupported_prompt_format(monkeypatch) -> None:
    tokenizer = _tokenizer(monkeypatch)
    tokenizer._prompt_format = "chatml"
    with pytest.raises(ValueError, match="only supported for nemotron6-moe"):
        tokenizer.tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_rendered_boundary_mismatch(monkeypatch) -> None:
    with pytest.raises(ValueError, match="rendered assistant boundaries"):
        _tokenizer(monkeypatch, drop_last_assistant_header=True).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            assistant_turn_loss=[False, True, True],
        )


def test_explicit_assistant_loss_rejects_missing_end_boundary(monkeypatch) -> None:
    with pytest.raises(ValueError, match="missing nemotron6 assistant end boundary"):
        _tokenizer(monkeypatch, drop_last_assistant_end=True).tokenize_conversation(
            _conversation(),
            return_target=True,
            add_generation_prompt=False,
            assistant_turn_loss=[False, False, True],
        )
