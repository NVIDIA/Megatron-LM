# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import builtins
import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.incremental_detokenizer import (
    HuggingFaceFastIncrementalDetokenizer,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.openai_streaming import (
    openai_stream,
)


class _Tokenizer:
    def detokenize(self, tokens):
        return "".join(chr(ord("a") + token - 1) for token in tokens)


class _IncrementalDetokenizer:
    def __init__(self):
        self._text = ""

    def update(self, tokens):
        delta = "".join(chr(ord("a") + token - 1) for token in tokens)
        self._text += delta
        return delta

    @property
    def text(self):
        return self._text

    @property
    def text_length(self):
        return len(self._text)


def _make_byte_level_fast_tokenizer():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    huggingface_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    return SimpleNamespace(
        _tokenizer=SimpleNamespace(tokenizer=huggingface_tokenizer, include_special_tokens=True)
    )


@pytest.mark.asyncio
async def test_openai_stream_emits_delta_chunks_and_terminal_metadata():
    stream = AsyncStream(request_id=1, cancel=lambda: None)
    stream.put({"partial": {"request_id": 1, "new_tokens": [1, 2], "new_log_probs": [-0.1, -0.2]}})
    # Token 3 models a token completed before the engine's final reply and
    # therefore absent from its last partial frame.
    stream.put(
        {
            "final": {
                "prompt_tokens": [9, 9],
                "generated_tokens": [1, 2, 3],
                "generated_log_probs": [-0.1, -0.2, -0.3],
                "num_cached_tokens": 2,
                "sampling_params": {"num_tokens_to_generate": 3},
            }
        }
    )
    stream.finish()

    records = [
        record
        async for record in openai_stream(
            [stream],
            _Tokenizer(),
            [_IncrementalDetokenizer()],
            chat=False,
            return_log_probs=True,
            include_usage=True,
        )
    ]
    payloads = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]

    first, reconciled, finished, usage = payloads
    assert first["choices"][0]["text"] == "ab"
    assert "generation_token_ids" not in first["choices"][0]
    assert "generation_log_probs" not in first["choices"][0]
    assert "generated_text" not in first["choices"][0]
    assert "generated_length" not in first["choices"][0]
    assert first["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, -0.2]
    assert first["choices"][0]["logprobs"]["text_offset"] == [0, 1]
    assert reconciled["choices"][0]["text"] == "c"
    assert "generation_token_ids" not in reconciled["choices"][0]
    assert "generation_log_probs" not in reconciled["choices"][0]
    assert "generated_text" not in reconciled["choices"][0]
    assert "generated_length" not in reconciled["choices"][0]
    assert finished["choices"][0]["finish_reason"] == "length"
    assert finished["choices"][0]["generation_token_ids"] == [1, 2, 3]
    assert finished["choices"][0]["generation_log_probs"] == [-0.1, -0.2, -0.3]
    assert finished["choices"][0]["generated_text"] == "abc"
    assert finished["choices"][0]["generated_length"] == 3
    assert usage["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert records[-1] == "data: [DONE]\n\n"


def test_huggingface_fast_incremental_detokenizer_requires_optional_dependencies(monkeypatch):
    original_import = builtins.__import__

    def import_without_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("transformers is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_transformers)

    with pytest.raises(ImportError, match="requires the tokenizers and transformers packages"):
        HuggingFaceFastIncrementalDetokenizer(_Tokenizer(), [])


def test_huggingface_fast_incremental_detokenizer_preserves_utf8_boundaries():
    tokenizer = _make_byte_level_fast_tokenizer()
    huggingface_tokenizer = tokenizer._tokenizer.tokenizer
    token_ids = huggingface_tokenizer.encode("😀 café", add_special_tokens=False)
    detokenizer = HuggingFaceFastIncrementalDetokenizer(tokenizer, [])

    streamed_text = "".join(detokenizer.update([token_id]) for token_id in token_ids)

    assert streamed_text == huggingface_tokenizer.decode(token_ids, skip_special_tokens=False)
    assert detokenizer.text == streamed_text
    assert detokenizer.text_length == len(streamed_text)


def test_huggingface_fast_incremental_detokenizer_uses_prompt_context():
    tokenizer = _make_byte_level_fast_tokenizer()
    huggingface_tokenizer = tokenizer._tokenizer.tokenizer
    prompt_token_ids = huggingface_tokenizer.encode("hello", add_special_tokens=False)
    generated_token_ids = huggingface_tokenizer.encode(" world", add_special_tokens=False)
    detokenizer = HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_token_ids)

    streamed_text = "".join(detokenizer.update([token_id]) for token_id in generated_token_ids)
    full_text = huggingface_tokenizer.decode(
        prompt_token_ids + generated_token_ids, skip_special_tokens=False
    )
    prompt_text = huggingface_tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

    assert full_text.startswith(prompt_text)
    assert streamed_text == full_text[len(prompt_text) :]


def test_incremental_detokenizer_rejects_unsupported_tokenizer():
    with pytest.raises(
        ValueError, match="Streaming is currently supported only for Hugging Face fast tokenizers"
    ):
        HuggingFaceFastIncrementalDetokenizer(_Tokenizer(), [])
