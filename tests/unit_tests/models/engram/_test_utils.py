# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures for Engram unit and distributed tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from megatron.core.models.engram.config import TOKENIZER_MAP_FORMAT, TOKENIZER_MAP_VERSION
from megatron.core.process_groups_config import ProcessGroupCollection


def write_tokenizer_map(
    path: Path,
    *,
    vocab_size: int = 32,
    layer_ids: tuple[int, ...] = (1,),
    max_ngram_order: int = 3,
    hash_seed: int = 0,
    pad_token_id: int = 0,
    remap: list[int] | None = None,
    multipliers: tuple[int, ...] = (13, 17, 19),
) -> Path:
    """Write a small valid tokenizer map without tokenizer dependencies."""
    remap = list(range(vocab_size)) if remap is None else remap
    compressed_vocab_size = max(remap) + 1
    artifact = {
        "format": TOKENIZER_MAP_FORMAT,
        "version": TOKENIZER_MAP_VERSION,
        "source_vocab_size": len(remap),
        "compressed_vocab_size": compressed_vocab_size,
        "pad_token_id": pad_token_id,
        "compressed_pad_token_id": remap[pad_token_id],
        "max_ngram_order": max_ngram_order,
        "hash_seed": hash_seed,
        "layer_ids": list(layer_ids),
        "layer_multipliers": {
            str(layer_id): list(multipliers[:max_ngram_order]) for layer_id in layer_ids
        },
        "remap": remap,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def make_module_config(
    *,
    num_streams: int = 1,
    sequence_parallel: bool = False,
    deterministic_mode: bool = False,
    dtype=torch.float64,
):
    """Return the minimal TransformerConfig interface consumed by Engram."""
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=dtype,
        perform_initialization=True,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
        hidden_size=8,
        enable_hyper_connections=num_streams > 1,
        num_residual_streams=num_streams,
        layernorm_epsilon=1e-5,
        sequence_parallel=sequence_parallel,
        deterministic_mode=deterministic_mode,
    )


def make_pg_collection(ep=None, tp=None, expt_dp=None) -> ProcessGroupCollection:
    """Build only the process groups used by the Engram module."""
    return ProcessGroupCollection(ep=ep, tp=tp, expt_dp=expt_dp)
