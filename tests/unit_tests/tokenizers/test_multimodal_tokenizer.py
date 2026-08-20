# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)


class _FakeTokenizer:
    """Minimal character tokenizer for lightweight multimodal-tokenizer tests."""

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture
def tokenizer():
    """Build a multimodal tokenizer without loading a Hugging Face model."""
    instance = object.__new__(MegatronMultimodalTokenizer)
    instance._tokenizer = _FakeTokenizer()
    instance._image_tag = None
    return instance


def test_tokenize_uses_legacy_image_text_marker(tokenizer):
    """Image tags should wrap the existing textual image marker before tokenization."""
    tokenizer._image_tag = ("<img>", "</img>")

    tokens = tokenizer.tokenize(f"A{IMAGE_TOKEN}B")

    assert tokenizer.detokenize(tokens) == f"A<img>{IMAGE_TOKEN}</img>B"


def test_apply_image_tag_handles_conversations(tokenizer):
    """Every textual image marker in a conversation should receive the configured tags."""
    tokenizer._image_tag = ("<Image>", "</Image>")
    conversation = [{"role": "user", "content": f"{IMAGE_TOKEN}A{IMAGE_TOKEN}"}]

    assert tokenizer._apply_image_tag(conversation) == [
        {"role": "user", "content": f"<Image>{IMAGE_TOKEN}</Image>A<Image>{IMAGE_TOKEN}</Image>"}
    ]


def test_tokenize_raw_conversation_masks_non_assistant_turns(tokenizer):
    """Raw tokenization should train only on assistant content."""
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]

    tokens, target = tokenizer._tokenize_raw_conversation(turns, return_target=True)

    np.testing.assert_array_equal(tokens, [ord("h"), ord("i"), ord("o"), ord("k")])
    np.testing.assert_array_equal(target, [IGNORE_INDEX, IGNORE_INDEX, ord("o"), ord("k")])


def test_thinking_trace_detection(tokenizer):
    """Only non-empty assistant thinking traces should be detected."""
    empty_trace = [{"role": "assistant", "content": "<think> </think>answer"}]
    nonempty_trace = [{"role": "assistant", "content": "<think>reasoning</think>answer"}]

    assert not tokenizer._has_nonempty_thinking_trace(empty_trace)
    assert tokenizer._has_nonempty_thinking_trace(nonempty_trace)
