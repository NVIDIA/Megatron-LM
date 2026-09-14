# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the token id -> compressed id map used by DeepSeek-V4.1 Engram hashing.

Tokens whose decoded text normalises to the same string (NFKC, accents stripped, lower
case, whitespace runs collapsed, outer whitespace stripped) share one compressed id, so the
n-gram hash of " The", "the" and "THE" is identical. The number of compressed ids must equal
``engram_compressed_vocab_size`` of the released configuration (99,092); the script fails
otherwise because every hash multiplier is derived from that number.

Usage (inside the training container, which ships ``tokenizers``)::

    python tools/dsv41/build_engram_token_map.py \
        --tokenizer-dir /path/to/DeepSeek-V4.1-Flash \
        --expected-size 99092 \
        --output /path/to/engram_token_map.pt

The normalisation rules follow ``inference/engram.py`` of the released model.
"""

import argparse
import json
import os

import torch


def build_compressed_token_map(tokenizer_dir: str):
    """Return ``(token_map: list[int], compressed_vocab_size: int)``."""
    from tokenizers import Regex, Tokenizer, normalizers

    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)

    # A single space must survive Strip(); swap it for a private-use sentinel first.
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    key_to_id = {}
    token_map = [0] * vocab_size
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # incomplete UTF-8 byte token: keep its raw form as the key
            key = tokenizer.id_to_token(token_id)
        else:
            normalised = normalizer.normalize_str(text)
            key = normalised if normalised else text
        compressed = key_to_id.get(key)
        if compressed is None:
            compressed = len(key_to_id)
            key_to_id[key] = compressed
        token_map[token_id] = compressed
    return token_map, len(key_to_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True, help="destination .pt file")
    parser.add_argument(
        "--expected-size",
        type=int,
        default=None,
        help="engram_compressed_vocab_size to verify against (99092 for V4.1-Flash)",
    )
    args = parser.parse_args()

    token_map, size = build_compressed_token_map(args.tokenizer_dir)
    if args.expected_size is not None and size != args.expected_size:
        raise SystemExit(
            f"compressed vocabulary has {size} ids, expected {args.expected_size}; the "
            "tokenizer files do not match the released model"
        )
    payload = {
        "token_map": torch.tensor(token_map, dtype=torch.int64),
        "compressed_vocab_size": size,
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    summary = {"vocab_size": len(token_map), "compressed_vocab_size": size, "output": args.output}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
