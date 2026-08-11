# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest

from megatron.core.models.multimodal.llava_model import (
    DEFAULT_IMAGE_TOKEN_INDEX,
    DEFAULT_SOUND_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN,
    SOUND_TOKEN,
)
from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)
from megatron.core.tokenizers.vision.vision_tokenizer import MegatronTokenizerVision


class _FakeTokenizer:
    """Minimal character tokenizer for testing multimodal marker replacement."""

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
    instance._image_token_index = DEFAULT_IMAGE_TOKEN_INDEX
    instance._sound_token_index = DEFAULT_SOUND_TOKEN_INDEX
    instance._sound_start_token_id = 1001
    instance._sound_end_token_id = 1002
    return instance


def test_render_and_encode_structured_parts(tokenizer):
    """Image and audio parts should be replaced by their sentinel ID spans."""
    parts = [
        {"type": "text", "text": "A"},
        {"type": "image"},
        {"type": "text", "text": "B"},
        {"type": "audio", "num_embeddings": 2},
    ]

    rendered, replacements = tokenizer._render_parts(parts)

    assert rendered == f"A{tokenizer._MM_MARKER}B{tokenizer._MM_MARKER}"
    assert replacements == [
        [DEFAULT_IMAGE_TOKEN_INDEX],
        [1001, DEFAULT_SOUND_TOKEN_INDEX, DEFAULT_SOUND_TOKEN_INDEX, 1002],
    ]
    assert tokenizer._encode_with_markers(rendered, replacements) == [
        ord("A"),
        DEFAULT_IMAGE_TOKEN_INDEX,
        ord("B"),
        1001,
        DEFAULT_SOUND_TOKEN_INDEX,
        DEFAULT_SOUND_TOKEN_INDEX,
        1002,
    ]


def test_render_structured_image_with_prompt_tags(tokenizer):
    """Image prompt tags should wrap the sentinel without changing its ID."""
    tokenizer._image_tag = ("<Image>", "</Image>")

    rendered, replacements = tokenizer._render_parts([{"type": "image"}])

    assert rendered == f"<Image>{tokenizer._MM_MARKER}</Image>"
    assert replacements == [[DEFAULT_IMAGE_TOKEN_INDEX]]


def test_vision_wrapper_exposes_image_sentinel(tokenizer):
    """The public vision wrapper should expose the inner tokenizer's image sentinel."""
    wrapper = object.__new__(MegatronTokenizerVision)
    wrapper._tokenizer = tokenizer

    assert wrapper.image_token_index == DEFAULT_IMAGE_TOKEN_INDEX


def test_render_parts_validates_reserved_marker_and_audio_length(tokenizer):
    """Malformed structured parts should fail before reaching the base tokenizer."""
    with pytest.raises(ValueError, match="reserved multimodal marker"):
        tokenizer._render_parts([{"type": "text", "text": tokenizer._MM_MARKER}])

    with pytest.raises(ValueError, match="Invalid num_embeddings"):
        tokenizer._render_parts([{"type": "audio", "num_embeddings": -1}])


def test_tokenize_raw_conversation_masks_non_assistant_turns(tokenizer):
    """Raw tokenization should train only on assistant content."""
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]

    tokens, target = tokenizer._tokenize_raw_conversation(turns, [[], []], return_target=True)

    np.testing.assert_array_equal(tokens, [ord("h"), ord("i"), ord("o"), ord("k")])
    np.testing.assert_array_equal(target, [IGNORE_INDEX, IGNORE_INDEX, ord("o"), ord("k")])


def test_thinking_trace_detection_supports_structured_content(tokenizer):
    """Only non-empty assistant thinking traces should be detected."""
    empty_trace = [{"role": "assistant", "content": "<think> </think>answer"}]
    structured_trace = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>reasoning</think>answer"},
                {"type": "image"},
            ],
        }
    ]

    assert not tokenizer._has_nonempty_thinking_trace(empty_trace)
    assert tokenizer._has_nonempty_thinking_trace(structured_trace)


def test_detokenize_surfaces_multimodal_sentinels(tokenizer):
    """Negative sentinel IDs should be rendered as readable multimodal tokens."""
    tokens = np.asarray(
        [ord("A"), DEFAULT_IMAGE_TOKEN_INDEX, ord("B"), DEFAULT_SOUND_TOKEN_INDEX, ord("C")]
    )

    assert tokenizer.detokenize(tokens) == f"A{IMAGE_TOKEN}B{SOUND_TOKEN}C"
