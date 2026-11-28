# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Write a common Engram JSON artifact from a released V4.1 tensor map and config."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from megatron.core.models.deepseek_v41.engram import released_engram_config
from megatron.core.models.engram.config import TOKENIZER_MAP_FORMAT, TOKENIZER_MAP_VERSION


def convert(options: dict, tensor_map: Path, output: Path, padded_vocab_size: int | None) -> None:
    """Keep released token IDs and hash constants; key the artifact by physical V layer."""
    if tensor_map.suffix not in (".pt", ".pth"):
        raise ValueError("Expected a released torch token map (.pt or .pth)")
    if tensor_map.resolve() == output.resolve():
        raise ValueError("Choose an output distinct from the released token map")
    config = released_engram_config(options, str(tensor_map))
    tokenizer_size = config.tokenizer_remap.numel()
    if padded_vocab_size is not None:
        config.validate_vocabulary(padded_vocab_size)
    data = dict(
        format=TOKENIZER_MAP_FORMAT,
        version=TOKENIZER_MAP_VERSION,
        tokenizer_vocab_size=tokenizer_size,
        source_vocab_size=config.tokenizer_remap.numel(),
        compressed_vocab_size=options["compressed_vocab_size"],
        pad_token_id=config.boundary_token_id,
        compressed_pad_token_id=config.hash_boundary_token_id,
        max_ngram_order=config.max_ngram_order,
        hash_seed=config.hash_seed,
        layer_ids=config.layer_ids,
        layer_multipliers={str(i): config.multipliers(i) for i in config.layer_ids},
        remap=config.tokenizer_remap.tolist(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        tmp = Path(directory) / "converted.json"
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        released_engram_config(options, str(tmp))
        os.replace(tmp, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Released Engram config JSON")
    parser.add_argument("--tensor-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padded-vocab-size", type=int)
    args = parser.parse_args()
    convert(
        json.loads(args.config.read_text()), args.tensor_map, args.output, args.padded_vocab_size
    )
