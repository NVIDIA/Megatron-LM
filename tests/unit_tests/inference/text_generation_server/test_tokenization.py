# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.text_generation_server.tokenization import _tokenize_prompts_and_batch


class _FakeTokenizerEod:
    eod = 99

    def tokenize(self, text):
        # Produce deterministic, length-varying token sequences.
        return list(range(1, len(text) + 1))


class _FakeTokenizerEosId:
    eos_id = 7

    def tokenize(self, text):
        return [42] * len(text)


class _FakeTokenizerNoEod:
    """A tokenizer that exposes neither `eod` nor `eos_id`."""

    def tokenize(self, text):
        return [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="builds CUDA tensors")
class TestTokenizePromptsAndBatch:

    def test_pads_to_max_prompt_plus_tokens_to_generate(self):
        """All prompts are padded to max(prompt_len) + tokens_to_generate."""
        prompts = ["abc", "abcdef"]  # tokens [1,2,3] and [1,2,3,4,5,6]
        tokens, lengths = _tokenize_prompts_and_batch(
            _FakeTokenizerEod(), prompts, tokens_to_generate=2, add_BOS=False
        )
        # max_prompt_len = 6, samples_length = 6 + 2 = 8.
        assert tokens.shape == (2, 8)
        assert lengths.tolist() == [3, 6]
        # Padded with eod_token (99).
        assert tokens[0, 3:].tolist() == [99] * 5
        assert tokens[1, 6:].tolist() == [99] * 2

    def test_add_bos_prepends_eod_to_each_prompt(self):
        """add_BOS=True prepends the eod token to every prompt."""
        prompts = ["a"]
        tokens, lengths = _tokenize_prompts_and_batch(
            _FakeTokenizerEod(), prompts, tokens_to_generate=1, add_BOS=True
        )
        # tokens for "a" → [1]; with BOS → [99, 1]; pad to len 3.
        assert tokens[0, 0].item() == 99
        assert tokens[0, 1].item() == 1
        assert lengths.tolist() == [2]

    def test_eos_id_attribute_used_when_eod_missing(self):
        """If the tokenizer has eos_id but not eod, eos_id is used as the pad/BOS token."""
        prompts = ["xy"]
        tokens, lengths = _tokenize_prompts_and_batch(
            _FakeTokenizerEosId(), prompts, tokens_to_generate=1, add_BOS=True
        )
        # BOS = eos_id = 7, then [42, 42] = tokenize("xy"); pad with 7.
        assert tokens[0, 0].item() == 7
        assert tokens[0, 1].item() == 42

    def test_missing_eod_attribute_raises(self):
        """A tokenizer with neither eod nor eos_id triggers AttributeError."""
        with pytest.raises(AttributeError, match="No eod token"):
            _tokenize_prompts_and_batch(
                _FakeTokenizerNoEod(), ["x"], tokens_to_generate=1, add_BOS=False
            )

    def test_tensors_are_cuda_long(self):
        """The returned tensors live on CUDA with dtype long."""
        prompts = ["a"]
        tokens, lengths = _tokenize_prompts_and_batch(
            _FakeTokenizerEod(), prompts, tokens_to_generate=2, add_BOS=False
        )
        assert tokens.is_cuda
        assert tokens.dtype == torch.long
        assert lengths.is_cuda
        assert lengths.dtype == torch.long
