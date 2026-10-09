# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest

from megatron.core.tokenizers.text import MegatronTokenizerText
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer

HAS_SUPPORT = "megatron.core.tokenizers.utils.has_gigatoken_support"


def _hf_tokenizer(use_gigatoken: bool) -> HuggingFaceTokenizer:
    """Builds a HuggingFaceTokenizer without loading any files."""
    tok = object.__new__(HuggingFaceTokenizer)
    tok.use_gigatoken = use_gigatoken
    tok.tokenizer = MagicMock()
    return tok


def _text_tokenizer(library: str) -> MegatronTokenizerText:
    tok = object.__new__(MegatronTokenizerText)
    tok.library = library
    tok._tokenizer = MagicMock()
    return tok


class TestHuggingFaceEncodeBatch:
    def test_gigatoken_delegates_with_parallel(self):
        tok = _hf_tokenizer(use_gigatoken=True)
        expected = object()
        tok.tokenizer.tokenizer.encode_batch.return_value = expected

        with patch(HAS_SUPPORT, return_value=True):
            result = tok.encode_batch(["a", "b"])

        assert result is expected
        tok.tokenizer.tokenizer.encode_batch.assert_called_once_with(["a", "b"], parallel=True)

    def test_gigatoken_not_installed(self):
        tok = _hf_tokenizer(use_gigatoken=True)
        with patch(HAS_SUPPORT, return_value=False):
            with pytest.raises(ModuleNotFoundError, match="gigatoken"):
                tok.encode_batch(["a"])
        tok.tokenizer.tokenizer.encode_batch.assert_not_called()

    def test_requires_gigatoken(self):
        tok = _hf_tokenizer(use_gigatoken=False)
        with pytest.raises(NotImplementedError, match="use_gigatoken=True"):
            tok.encode_batch(["a"])


class TestTokenizeBatch:
    @pytest.mark.parametrize("library", ["huggingface", "megatron"])
    def test_supported_libraries_delegate(self, library):
        tok = _text_tokenizer(library)
        expected = object()
        tok._tokenizer.encode_batch.return_value = expected

        assert tok.tokenize_batch(["x", "y"]) is expected
        tok._tokenizer.encode_batch.assert_called_once_with(["x", "y"])

    @pytest.mark.parametrize("library", ["sentencepiece", "tiktoken", "null-text", "byte-level"])
    def test_unsupported_libraries_raise(self, library):
        tok = _text_tokenizer(library)
        with pytest.raises(NotImplementedError, match="huggingface"):
            tok.tokenize_batch(["x"])
        tok._tokenizer.encode_batch.assert_not_called()
