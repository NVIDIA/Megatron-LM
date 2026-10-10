# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Callable
from typing import Any

from .byte_token_offsets import get_byte_token_offsets


def get_huggingface_token_offsets(
    tokenizer: Any, ids: list[int], text: str, decode: Callable[[list[int]], str]
) -> list[int]:
    """Return character offsets for IDs in their detokenized text.

    ``decode`` must use the special-token policy that produced ``text``.
    Prefix decoding takes quadratic time when no usable mapping is available.
    """
    if not ids:
        return []

    # The inference controller can remove trailing EOD tokens before decoding.
    end = len(ids)
    while end and ids[end - 1] == getattr(tokenizer, 'eos_token_id', None):
        end -= 1
    if end < len(ids) and decode(ids[:end]) == text:
        return get_huggingface_token_offsets(tokenizer, ids[:end], text, decode) + [len(text)] * (
            len(ids) - end
        )

    # Fast mappings can trim spaces that belong to byte-level tokens.
    byte_offsets = get_byte_token_offsets(tokenizer, ids, text, decode)
    if byte_offsets is not None:
        return byte_offsets

    if getattr(tokenizer, 'is_fast', False):
        try:
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        except NotImplementedError:
            encoding = {}
        mapping = encoding.get('offset_mapping')
        # Generated IDs can differ from re-encoded IDs even when counts match.
        if mapping is not None and len(mapping) == len(ids) and encoding.get('input_ids') == ids:
            return [start for start, _ in mapping]

    offsets = [0]
    for index in range(1, len(ids)):
        # Decode prefixes to preserve subword context.
        prefix = decode(ids[:index])
        offset = 0
        for prefix_char in prefix:
            if offset == len(text):
                break
            if prefix_char == text[offset]:
                offset += 1
            elif not prefix_char.isspace():
                break
            # Allow whitespace removed by full-sequence cleanup, e.g. "i ' m" -> "i'm".
        offsets.append(offset)
    return offsets
