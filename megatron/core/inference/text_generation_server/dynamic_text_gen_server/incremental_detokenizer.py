# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 The vLLM authors.

"""Incremental detokenization for OpenAI-compatible streaming responses."""

import logging
from typing import Any

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
        self._skip_special_tokens = not getattr(tokenizer_wrapper, "include_special_tokens", True)
        self._decode_stream = self._new_decode_stream(prompt_token_ids)
        self._text_fragments: list[str] = []
        self._text_length = 0

    def _new_decode_stream(self, prompt_token_ids: list[int] | None = None):
        kwargs = {"skip_special_tokens": self._skip_special_tokens}
        if prompt_token_ids is not None:
            kwargs["ids"] = list(prompt_token_ids)
        return self._tokenizers.decoders.DecodeStream(**kwargs)

    def update(self, token_ids: list[int]) -> str:
        """Decode token IDs and return only newly stable text."""
        fragments = []
        for token_id in token_ids:
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
