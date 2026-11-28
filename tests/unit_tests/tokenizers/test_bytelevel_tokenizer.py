# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.tokenizers.text.libraries.bytelevel_tokenizer import ByteLevelTokenizer


@pytest.mark.parametrize("special_name", ["bos", "eos", "pad"])
@pytest.mark.parametrize("special_id", [256, 257, 511])
@pytest.mark.parametrize("method", ["ids_to_text", "tokens_to_text"])
def test_configured_nonbyte_special_tokens(special_name, special_id, method):
    """Configured special IDs outside the byte range are removed during decoding."""
    tokenizer = ByteLevelTokenizer(**{f"_{special_name}_id": special_id})
    text = "Hello, 世界 🌍"
    ids = tokenizer.text_to_ids(text)
    decode = getattr(tokenizer, method)

    assert decode([special_id, *ids[:9], special_id, *ids[9:], special_id]) == text
    assert decode([special_id, special_id]) == ""


def test_configured_and_additional_special_tokens():
    """Configured special IDs and appended special tokens can appear together."""
    tokenizer = ByteLevelTokenizer(
        _bos_id=256, _eos_id=257, _pad_id=511, special_tokens=["<extra>"]
    )
    ids = [
        tokenizer.bos_id,
        65,
        tokenizer.token_to_id("<extra>"),
        tokenizer.eos_id,
        tokenizer.pad_id,
    ]

    assert tokenizer.ids_to_text(ids) == "A"


@pytest.mark.parametrize("text", ["", "Hello", "世界 🌍", "\x00\x01A", "\x03A"])
def test_byte_range_round_trip(text):
    """Existing byte-range decoding remains unchanged, including configured byte IDs."""
    tokenizer = ByteLevelTokenizer(_bos_id=3)

    assert tokenizer.ids_to_text(tokenizer.text_to_ids(text)) == text


@pytest.mark.parametrize("token_id", [-1, 256, 511])
def test_unregistered_nonbyte_ids_remain_invalid(token_id):
    """An unknown non-byte ID is not silently discarded as a special token."""
    tokenizer = ByteLevelTokenizer()

    with pytest.raises(ValueError, match="bytes must be in range"):
        tokenizer.ids_to_text([65, token_id])
