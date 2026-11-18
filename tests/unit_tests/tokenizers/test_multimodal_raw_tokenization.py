# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest

from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)


class _OffsetTokenizer:
    def __init__(self, mode="valid") -> None:
        self.mode = mode
        self.encode_calls = []

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        self.encode_calls.append(text)
        return [ord(character) for character in text]

    def __call__(self, text, **kwargs):
        if self.mode == "unsupported":
            raise NotImplementedError("Offsets unavailable")
        ids = [ord(character) for character in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if self.mode == "missing":
            return {"input_ids": ids}
        if self.mode == "mismatched":
            offsets = offsets[:-1]
        elif self.mode == "inside_token":
            offsets[1] = (1, 3)
        elif self.mode == "zero_width":
            offsets[0] = (0, 0)
        return {"input_ids": ids, "offset_mapping": offsets}


def _tokenizer(mode="valid"):
    tokenizer = MegatronMultimodalTokenizer.__new__(MegatronMultimodalTokenizer)
    tokenizer._tokenizer = _OffsetTokenizer(mode)
    tokenizer.tokenizer = tokenizer._tokenizer
    tokenizer.use_gigatoken = False
    return tokenizer


@pytest.mark.parametrize(
    "mode", ["unsupported", "missing", "mismatched", "inside_token", "zero_width"]
)
def test_raw_offset_failure_matches_prefix_fallback(mode):
    fast = _tokenizer()
    slow = _tokenizer(mode)
    assert slow._tokenize_text_with_offsets("abcd", [2, 4]) is None
    turns = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]
    fast_tokens, fast_targets = fast._tokenize_raw_conversation(turns, [[], []], True)
    slow_tokens, slow_targets = slow._tokenize_raw_conversation(turns, [[], []], True)
    np.testing.assert_array_equal(slow_tokens, fast_tokens)
    np.testing.assert_array_equal(slow_targets, fast_targets)
    np.testing.assert_array_equal(slow_targets, [-100, -100, ord("c"), ord("d")])


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("mode", ["valid", "unsupported"])
def test_raw_inference_matches_target_path(with_image, mode):
    tokenizer = _tokenizer(mode)
    turns = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]
    replacements = [[], []]
    if with_image:
        turns[0]["content"] += tokenizer._MM_MARKER
        replacements[0] = [[-200]]
    expected, _ = tokenizer._tokenize_raw_conversation(turns, replacements, True)
    tokenizer._tokenizer.encode_calls.clear()
    actual = tokenizer._tokenize_raw_conversation(turns, replacements, False)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    assert tokenizer._tokenizer.encode_calls == (["ab", "cd"] if with_image else ["abcd"])
