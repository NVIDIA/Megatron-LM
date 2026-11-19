# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 The vLLM authors.

"""Incremental detokenization for OpenAI-compatible streaming responses."""

import logging
from typing import Any

from megatron.core.inference.utils import model_eos_token_ids

logger = logging.getLogger(__name__)

_INVALID_PREFIX_ERROR = "Invalid prefix encountered"


class HuggingFaceFastIncrementalDetokenizer:
    """Incrementally decode generated tokens with a Hugging Face fast tokenizer.

    This implementation follows vLLM's prompt-prefilled ``DecodeStream`` design.
    Streaming support is intentionally limited to Hugging Face fast tokenizers
    until equivalent incremental decoders exist for the other tokenizer backends.
    """

    def __init__(self, tokenizer: Any, prompt_token_ids: list[int]) -> None:
        try:
            import tokenizers
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise ImportError(
                "Incremental detokenization requires the tokenizers and transformers packages."
            ) from exc

        self._tokenizers = tokenizers
        tokenizer_wrapper = getattr(tokenizer, "_tokenizer", None)
        huggingface_tokenizer = getattr(tokenizer_wrapper, "tokenizer", None)
        if not isinstance(huggingface_tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "Streaming is currently supported only for Hugging Face fast tokenizers."
            )
        if not hasattr(self._tokenizers.decoders, "DecodeStream"):
            raise ValueError(
                "Streaming with Hugging Face fast tokenizers requires tokenizers>=0.22.0."
            )

        self._native_tokenizer = huggingface_tokenizer._tokenizer
        # Streaming must drop exactly what the non-streaming response drops, or its
        # concatenated deltas differ from `generated_text`. That is the tokens in the
        # tokenizer's special-tokens map (`all_special_ids`) and the trailing EOS ids
        # (`remove_EOD`), not every added token flagged `special`: chat models flag
        # their reasoning and tool-call markers (`<think>`, `</think>`,
        # `<tool_call>`) as special, and the reasoning and tool parsers need them.
        # `DecodeStream(skip_special_tokens=True)` would drop those markers too.
        self._skipped_token_ids = frozenset(huggingface_tokenizer.all_special_ids) | (
            model_eos_token_ids(tokenizer)
        )
        self._decode_stream = self._new_decode_stream(prompt_token_ids)
        self._text_fragments: list[str] = []
        self._text_length = 0

    def _new_decode_stream(self, prompt_token_ids: list[int] | None = None):
        kwargs = {"skip_special_tokens": False}
        if prompt_token_ids is not None:
            kwargs["ids"] = self._kept(prompt_token_ids)
        return self._tokenizers.decoders.DecodeStream(**kwargs)

    def _kept(self, token_ids: list[int]) -> list[int]:
        return [token_id for token_id in token_ids if token_id not in self._skipped_token_ids]

    def update(self, token_ids: list[int]) -> str:
        """Decode token IDs and return only newly stable text."""
        fragments = []
        for token_id in self._kept(token_ids):
            fragment = self._decode_next(token_id)
            if fragment:
                fragments.append(fragment)

        delta = "".join(fragments)
        if delta:
            self._text_fragments.append(delta)
            self._text_length += len(delta)
        return delta

    def _decode_next(self, token_id: int) -> str:
        try:
            fragment = self._decode_stream.step(self._native_tokenizer, token_id)
        except (OverflowError, TypeError):
            logger.exception("Encountered invalid token ID during streaming: %r", token_id)
            return ""
        except Exception as exc:
            if not str(exc).startswith(_INVALID_PREFIX_ERROR):
                raise
            logger.warning(
                "Resetting the incremental decoder after an invalid prefix for token ID %r.",
                token_id,
            )
            self._decode_stream = self._new_decode_stream()
            fragment = self._decode_stream.step(self._native_tokenizer, token_id)
        return fragment or ""

    @property
    def text(self) -> str:
        """Return all text emitted by the incremental decoder."""
        return "".join(self._text_fragments)

    @property
    def text_length(self) -> int:
        """Return the number of emitted characters."""
        return self._text_length
