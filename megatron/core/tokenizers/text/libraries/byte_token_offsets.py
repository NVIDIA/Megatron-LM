# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Callable
from typing import Any


def _get_fallback_byte(piece: str) -> int | None:
    """Parse a byte-fallback token."""
    if (
        len(piece) == 6
        and piece.startswith('<0x')
        and piece.endswith('>')
        and all(char in '0123456789abcdefABCDEF' for char in piece[3:5])
    ):
        return int(piece[3:5], 16)
    return None


def get_byte_token_offsets(
    tokenizer: Any, ids: list[int], text: str, decode: Callable[[list[int]], str]
) -> list[int] | None:
    """Return offsets from token bytes, or None if they do not reproduce text."""
    byte_decoder = getattr(tokenizer, 'byte_decoder', None)
    if not isinstance(byte_decoder, dict):
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        byte_decoder = None
        if backend is not None:
            try:
                from tokenizers.decoders import ByteLevel
            except ImportError:
                return None
            if isinstance(getattr(backend, 'decoder', None), ByteLevel):
                # Invert the bytes_to_unicode alphabet used by GPT-2 and ByteLevel.
                visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
                byte_decoder = {chr(byte): byte for byte in visible}
                hidden = [byte for byte in range(256) if byte not in byte_decoder.values()]
                byte_decoder.update({chr(256 + index): byte for index, byte in enumerate(hidden)})

    special_ids = set(getattr(tokenizer, 'all_special_ids', []))
    convert_ids_to_tokens = getattr(tokenizer, 'convert_ids_to_tokens', None)
    if convert_ids_to_tokens is None:
        return None
    pieces = convert_ids_to_tokens(ids)
    fallback_bytes = [
        _get_fallback_byte(piece) if piece is not None and token_id not in special_ids else None
        for token_id, piece in zip(ids, pieces)
    ]
    if byte_decoder is None and all(byte is None for byte in fallback_bytes):
        return None

    token_bytes = []
    for token_id, piece, fallback_byte in zip(ids, pieces, fallback_bytes):
        if token_id in special_ids:
            token_bytes.append(decode([token_id]).encode('utf-8'))
        else:
            try:
                if byte_decoder is not None:
                    token_bytes.append(bytes(byte_decoder[char] for char in piece))
                elif fallback_byte is not None:
                    token_bytes.append(bytes([fallback_byte]))
                else:
                    token_bytes.append(piece.replace('▁', ' ').encode('utf-8'))
            except (KeyError, TypeError, AttributeError):
                return None

    if byte_decoder is None:
        start = 0
        while start < len(ids):
            if fallback_bytes[start] is None:
                start += 1
                continue
            end = start + 1
            while end < len(ids) and (fallback_bytes[end] is not None or not token_bytes[end]):
                end += 1
            try:
                b''.join(token_bytes[start:end]).decode('utf-8')
            except UnicodeDecodeError:
                # ByteFallback replaces every byte in an invalid run.
                token_bytes[start:end] = [
                    b'\xef\xbf\xbd' if fallback_bytes[index] is not None else b''
                    for index in range(start, end)
                ]
            start = end

    encoded = bytearray()
    positions = []
    for piece_bytes in token_bytes:
        positions.append(len(encoded))
        encoded.extend(piece_bytes)
    if encoded.decode('utf-8', errors='replace') != text:
        if byte_decoder is not None or not encoded.startswith(b' '):
            return None
        if encoded[1:].decode('utf-8', errors='replace') != text:
            return None
        encoded = encoded[1:]
        positions = [max(0, position - 1) for position in positions]

    byte_offsets = []
    position = 0
    character = 0
    while position < len(encoded):
        # UnicodeDecodeError.end identifies bytes replaced by one U+FFFD.
        first = encoded[position]
        width = 1 + (first >= 0xC2) + (first >= 0xE0) + (first >= 0xF0)
        try:
            encoded[position : position + width].decode('utf-8')
        except UnicodeDecodeError as error:
            width = error.end
        byte_offsets.extend([character] * width)
        position += width
        character += 1
    byte_offsets.append(character)
    return [byte_offsets[position] for position in positions]
