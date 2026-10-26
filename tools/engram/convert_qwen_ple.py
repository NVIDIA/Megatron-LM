# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert Qwen PLE weights between the HF layout and Megatron Engram modules.

HF layout (per PLE layer ``layers.{idx}.ple.``):
    ple_embedding.ngram_embedding.weight  -- one fused table over all heads, rows laid out
        head-major with per-head prime row counts, padded to a multiple of
        ``make_ngram_vocab_size_divisible_by`` (pad rows are never indexed and are dropped);
        checkpoints may store it whole or split into dim-0 shards.
    key_proj.weight / value_proj.weight   -- fused bias-free projections.
    norm_key.weight / norm_query.weight / norm_conv.weight -- zero-centered group RMSNorms.
    conv1d.weight                          -- depthwise dilated causal convolution.

Megatron layout: one canonical bundle per Engram layer with per-head tables of exactly prime
row counts; ``load_bundle_into_engram`` copies each EP rank's row slice into an initialized
Engram module. The dense weights map 1:1 because the Megatron module uses the same fused
projections and group norms.
"""

from __future__ import annotations

import argparse
import json
import re

# Allow running as a script from the repo root.
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from megatron.core.models.engram.config import allocate_qwen_table_sizes  # noqa: E402

_DENSE_KEY_MAP = {
    "key_proj.weight": "key_projection.weight",
    "value_proj.weight": "value_projection.weight",
    "norm_key.weight": "key_norm.weight",
    "norm_query.weight": "query_norm.weight",
    "norm_conv.weight": "conv_norm.weight",
    "conv1d.weight": "short_conv.weight",
}


def _open_hf_state(checkpoint_dir: Path):
    """Return a key -> tensor loader over the safetensors files of an HF checkpoint."""
    from safetensors import safe_open

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = sorted(checkpoint_dir.glob("*.safetensors"))
        if not single:
            raise FileNotFoundError(f"No safetensors files under {checkpoint_dir}.")
        weight_map = {}
        for shard in single:
            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    weight_map[key] = shard.name

    handles: dict[str, object] = {}

    def load(key: str) -> torch.Tensor:
        shard_name = weight_map[key]
        if shard_name not in handles:
            handles[shard_name] = safe_open(checkpoint_dir / shard_name, framework="pt")
        return handles[shard_name].get_tensor(key)

    return weight_map, load


def _load_full_table(weight_map, load, table_prefix: str) -> torch.Tensor:
    """Load the fused n-gram table, concatenating dim-0 checkpoint shards when present."""
    whole_key = f"{table_prefix}.weight"
    if whole_key in weight_map:
        return load(whole_key)
    shard_keys = sorted(
        (key for key in weight_map if key.startswith(f"{table_prefix}.weight")),
        key=lambda key: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key)],
    )
    if not shard_keys:
        raise KeyError(f"No table weights found under {table_prefix} in the checkpoint index.")
    return torch.cat([load(key) for key in shard_keys], dim=0)


def convert_hf_ple_layer(
    weight_map, load, hf_layer_index: int, ple_layer_index: int, *, table_sizes: tuple[int, ...]
) -> dict:
    """Extract one HF PLE layer into the canonical Megatron Engram bundle."""
    prefix = f"model.layers.{hf_layer_index}.ple."
    if f"{prefix}key_proj.weight" not in weight_map:
        # Text-only checkpoints may omit the leading "model." or use a different stem.
        candidates = {key.split("ple.")[0] for key in weight_map if ".ple." in key}
        for candidate in candidates:
            if candidate.endswith(f"layers.{hf_layer_index}."):
                prefix = f"{candidate}ple."
                break
        else:
            raise KeyError(f"No PLE weights found for HF layer {hf_layer_index}.")

    bundle = {"hf_layer_index": hf_layer_index, "ple_layer_index": ple_layer_index}
    for hf_name, megatron_name in _DENSE_KEY_MAP.items():
        bundle[megatron_name] = load(f"{prefix}{hf_name}").clone()

    full_table = _load_full_table(weight_map, load, f"{prefix}ple_embedding.ngram_embedding")

    tables, offset = [], 0
    for prime in table_sizes:
        tables.append(full_table[offset : offset + prime].clone())
        offset += prime
    # Rows beyond the per-head primes are divisibility padding and must never be indexed.
    if offset > full_table.shape[0]:
        raise ValueError(
            f"HF table has {full_table.shape[0]} rows but the head primes need {offset}; "
            "check ngram_vocab_size_base / heads / order."
        )
    bundle["tables"] = tables
    bundle["table_sizes"] = list(table_sizes)
    return bundle


@torch.no_grad()
def load_bundle_into_engram(module, bundle: dict) -> None:
    """Copy a canonical bundle into an initialized Engram module (EP-local row slices)."""
    for name in _DENSE_KEY_MAP.values():
        owner, attribute = name.rsplit(".", 1)
        parameter = getattr(getattr(module, owner.split(".")[0]), attribute)
        source = bundle[name]
        if parameter.shape != source.shape:
            raise ValueError(f"{name}: module {tuple(parameter.shape)} vs HF {tuple(source.shape)}")
        parameter.copy_(source.to(parameter.dtype))
    if len(module.embedding.tables) != len(bundle["tables"]):
        raise ValueError(
            f"Table count mismatch: module {len(module.embedding.tables)} vs "
            f"bundle {len(bundle['tables'])}."
        )
    for table, source in zip(module.embedding.tables, bundle["tables"]):
        if table.global_num_embeddings != source.shape[0]:
            raise ValueError(
                f"Table rows mismatch: module {table.global_num_embeddings} vs "
                f"HF {source.shape[0]} - prime allocation differs."
            )
        table.weight.copy_(source[table.row_start : table.row_end].to(table.weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--hf-layer-indices",
        required=True,
        type=int,
        nargs="+",
        help="0-based HF decoder layer indices carrying PLE (config.ple_layer_ids).",
    )
    parser.add_argument("--ngram-vocab-size-base", type=int, default=20_000_000)
    parser.add_argument("--max-ngram-order", type=int, default=3)
    parser.add_argument("--num-hash-heads", type=int, default=8)
    args = parser.parse_args()

    weight_map, load = _open_hf_state(args.hf_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    # One incremental prime scan covers every PLE layer (order-major heads, layers chained).
    all_table_sizes = allocate_qwen_table_sizes(
        args.ngram_vocab_size_base,
        tuple(range(1, len(args.hf_layer_indices) + 1)),
        args.max_ngram_order,
        args.num_hash_heads,
    )
    for ple_layer_index, hf_layer_index in enumerate(args.hf_layer_indices):
        bundle = convert_hf_ple_layer(
            weight_map,
            load,
            hf_layer_index,
            ple_layer_index,
            table_sizes=all_table_sizes[ple_layer_index + 1],
        )
        destination = args.output / f"ple_layer_{ple_layer_index}_hf{hf_layer_index}.pt"
        torch.save(bundle, destination)
        rows = sum(bundle["table_sizes"])
        print(f"Wrote {destination} (16-head rows={rows}, dense keys={len(_DENSE_KEY_MAP)})")


if __name__ == "__main__":
    main()
