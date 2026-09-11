# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate a deterministic offline tokenizer compression artifact for Engram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer

# Keep the utility runnable in a tokenizer-only environment. Model code validates these stable
# wire-format constants independently and does not need to be imported here.
TOKENIZER_MAP_FORMAT = "megatron-engram-token-map"
TOKENIZER_MAP_VERSION = 1


def build_normalizer():
    """Build the exact official Engram compressed-token normalizer."""
    sentinel = "\ue000"
    return normalizers.Sequence(
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


def build_remap(tokenizer) -> tuple[list[int], int]:
    """Build the official first-occurrence canonical-ID projection."""
    normalizer = build_normalizer()
    key_to_new: dict[str, int] = {}
    remap = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tokenizer.convert_ids_to_tokens(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        remap.append(key_to_new[key])
    return remap, len(key_to_new)


def build_layer_multipliers(
    compressed_vocab_size: int, layer_ids: list[int], max_ngram_order: int, hash_seed: int
) -> dict[str, list[int]]:
    """Generate the official NumPy-PCG64 odd multipliers."""
    max_long = np.iinfo(np.int64).max
    half_bound = max(1, int(max_long // compressed_vocab_size) // 2)
    result = {}
    for layer_id in layer_ids:
        generator = np.random.default_rng(int(hash_seed + 10007 * layer_id))
        values = generator.integers(low=0, high=half_bound, size=(max_ngram_order,), dtype=np.int64)
        result[str(layer_id)] = [int(value) for value in values * 2 + 1]
    return result


def parse_args() -> argparse.Namespace:
    """Parse tokenizer-map generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer-ids", required=True, type=int, nargs="+")
    parser.add_argument("--max-ngram-order", type=int, default=3)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--pad-token-id", type=int, required=True)
    parser.add_argument(
        "--padded-vocab-size",
        type=int,
        default=None,
        help=(
            "Optional Megatron padded vocabulary size. Added dummy IDs map to the compressed "
            "pad ID and therefore do not change the official hash multipliers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the artifact without importing tokenizer code in model execution."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    remap, compressed_vocab_size = build_remap(tokenizer)
    if not 0 <= args.pad_token_id < len(remap):
        raise ValueError(f"pad-token-id must be in [0, {len(remap)}), got {args.pad_token_id}.")
    tokenizer_vocab_size = len(remap)
    padded_vocab_size = args.padded_vocab_size or tokenizer_vocab_size
    if padded_vocab_size < tokenizer_vocab_size:
        raise ValueError(
            "padded-vocab-size cannot be smaller than the tokenizer vocabulary; "
            f"got {padded_vocab_size} and {tokenizer_vocab_size}."
        )
    remap.extend([remap[args.pad_token_id]] * (padded_vocab_size - tokenizer_vocab_size))
    artifact = {
        "format": TOKENIZER_MAP_FORMAT,
        "version": TOKENIZER_MAP_VERSION,
        "tokenizer_name_or_path": args.tokenizer,
        "source_vocab_size": padded_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "compressed_vocab_size": compressed_vocab_size,
        "normalization": "NFKC+NFD+StripAccents+Lowercase+WhitespaceCollapse",
        "pad_token_id": args.pad_token_id,
        "compressed_pad_token_id": remap[args.pad_token_id],
        "max_ngram_order": args.max_ngram_order,
        "hash_seed": args.hash_seed,
        "layer_ids": args.layer_ids,
        "layer_multipliers": build_layer_multipliers(
            compressed_vocab_size, args.layer_ids, args.max_ngram_order, args.hash_seed
        ),
        "remap": remap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(artifact, output_file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output_file.write("\n")


if __name__ == "__main__":
    main()
