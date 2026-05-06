# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.tokenization import (
    _tokenize_prompts_and_batch,
)


class _FakeTokenizer:
    eod = 0

    def tokenize(self, text):
        return list(range(1, len(text) + 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="builds CUDA tensors")
class TestDynamicServerTokenization:

    def test_basic_padding(self):
        """All prompts are padded to max(prompt_len) + tokens_to_generate using eod_token."""
        tokens, lengths = _tokenize_prompts_and_batch(
            _FakeTokenizer(), ["ab", "abcd"], tokens_to_generate=1, add_BOS=False
        )
        # max len = 4, samples_length = 5.
        assert tokens.shape == (2, 5)
        assert lengths.tolist() == [2, 4]
        # Pad with eod_token=0.
        assert tokens[0, 2:].tolist() == [0, 0, 0]
        assert tokens[1, 4:].tolist() == [0]

    def test_add_bos_prepends_eod(self):
        """add_BOS=True prepends the eod token to each prompt."""
        tokens, _ = _tokenize_prompts_and_batch(
            _FakeTokenizer(), ["x"], tokens_to_generate=2, add_BOS=True
        )
        assert tokens[0, 0].item() == 0  # BOS == eod_token
