# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 The vLLM authors.

"""Incremental detokenization for OpenAI-compatible streaming responses."""

import logging
from typing import Any

from megatron.core.inference.utils import detokenize_tokens, model_eos_token_ids

logger = logging.getLogger(__name__)

_INVALID_PREFIX_ERROR = "Invalid prefix encountered"


class HuggingFaceFastIncrementalDetokenizer:
    """Incrementally decode generated tokens with a Hugging Face fast tokenizer.

    This implementation follows vLLM's prompt-prefilled ``DecodeStream`` design.
    Streaming support is intentionally limited to Hugging Face fast tokenizers
    until equivalent incremental decoders exist for the other tokenizer backends.
    """

    def __init__(
        self, tokenizer: Any, prompt_token_ids: list[int], *, remove_eos: bool = True
    ) -> None:
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
        self._remove_eos = remove_eos
        self._eos_token_ids = model_eos_token_ids(tokenizer)
        # Mapped special tokens remain filtered when the caller asks to retain
        # the stop sequence. EOS ids are managed separately so only their
        # trailing-position policy changes with ``remove_eos``.
        self._skipped_token_ids = (
            frozenset(huggingface_tokenizer.all_special_ids) - self._eos_token_ids
        )
        self._pending_eos_token_ids: list[int] = []
        self._decode_stream = self._new_decode_stream(prompt_token_ids)
        self._text_fragments: list[str] = []
        self._text_length = 0
        self._finished = False

    def _new_decode_stream(self, prompt_token_ids: list[int] | None = None):
        kwargs = {"skip_special_tokens": False}
        if prompt_token_ids is not None:
            kwargs["ids"] = self._kept(prompt_token_ids)
        return self._tokenizers.decoders.DecodeStream(**kwargs)

    def _kept(self, token_ids: list[int]) -> list[int]:
        return [token_id for token_id in token_ids if token_id not in self._skipped_token_ids]

    def update(self, token_ids: list[int]) -> str:
        """Decode token IDs and return only newly stable text."""
        decode_ids = []
        for token_id in token_ids:
            if token_id in self._skipped_token_ids:
                continue
            if token_id in self._eos_token_ids:
                self._pending_eos_token_ids.append(token_id)
                continue
            decode_ids.extend(self._pending_eos_token_ids)
            self._pending_eos_token_ids.clear()
            decode_ids.append(token_id)
        return self._decode(decode_ids)

    def finish(self) -> str:
        """Resolve buffered EOS tokens once their trailing position is known."""
        if self._finished:
            return ""
        self._finished = True
        decode_ids = [] if self._remove_eos else self._pending_eos_token_ids
        self._pending_eos_token_ids = []
        return self._decode(decode_ids)

    def _decode(self, token_ids: list[int]) -> str:
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


def detokenize_generated_tokens(
    tokenizer: Any,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    *,
    remove_eos: bool = True,
) -> str:
    """Decode a completed generation with the same prompt context used for streaming.

    Tokenizers may insert or withhold text at the prompt/generation boundary. A
    prompt-prefilled decoder therefore produces the authoritative generated suffix for
    Hugging Face fast tokenizers. Non-streaming-only tokenizer backends retain the
    established generated-token decoding behavior.
    """
    try:
        detokenizer = HuggingFaceFastIncrementalDetokenizer(
            tokenizer, prompt_token_ids, remove_eos=remove_eos
        )
    except (ImportError, ValueError):
        return detokenize_tokens(
            tokenizer, generated_token_ids, remove_EOD=remove_eos, skip_special_tokens=True
        )
    return detokenizer.update(generated_token_ids) + detokenizer.finish()
