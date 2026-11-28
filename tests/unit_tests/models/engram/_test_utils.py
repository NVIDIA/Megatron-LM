# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures for Engram unit and distributed tests."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.engram import symmetric_lookup
from megatron.core.models.engram.config import TOKENIZER_MAP_FORMAT, TOKENIZER_MAP_VERSION
from megatron.core.packed_seq_params import PackedSeqParams
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
        enable_mhc_connections=num_streams > 1,
        mhc_num_residual_streams=num_streams,
        layernorm_epsilon=1e-5,
        sequence_parallel=sequence_parallel,
        deterministic_mode=deterministic_mode,
    )


def packed_seq_params(*, cp_partition_mode: str | None = None, **fields) -> PackedSeqParams:
    """Build PackedSeqParams, tagged with the CP partition mode of its row when one is given.

    PackedSeqParams records that mode only on branches whose data path sets it. The memory
    reads it when present, so the tag is set as an attribute, which works with or without the
    field.
    """
    params = PackedSeqParams(**fields)
    if cp_partition_mode is not None:
        params.cp_partition_mode = cp_partition_mode
    return params


def make_pg_collection(ep=None, tp=None, expt_dp=None, cp=None) -> ProcessGroupCollection:
    """Build only the process groups used by the Engram module."""
    return ProcessGroupCollection(ep=ep, tp=tp, expt_dp=expt_dp, cp=cp)


def _world_device() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def symmetric_transport_or_skip() -> None:
    """Skip, on every rank alike, unless every rank can run the Engram symmetric transport.

    It needs CUDA, expandable segments in the launch environment's allocator configuration
    (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, read as --engram-communication-backend
    reads it; the tables become NCCL windows, and a test cannot switch the caching allocator
    over), and the libraries that ``require_symmetric_transport`` checks: nvidia-cutlass-dsl >=
    4.8.1 and nccl4py >= 0.6.0 with a matching NCCL runtime and its device bitcode. The ranks
    agree on the outcome, so none is left in a collective that another one skipped.
    """
    reason = None
    if not torch.cuda.is_available():
        reason = "the Engram symmetric transport needs CUDA"
    elif not symmetric_lookup.expandable_segments_enabled():
        reason = "Engram table windows need PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    else:
        try:
            symmetric_lookup.require_symmetric_transport()
        except ImportError as exc:
            reason = f"Engram symmetric transport unavailable: {exc}"
    if dist.is_initialized():
        missing = torch.tensor([int(reason is not None)], device=_world_device())
        dist.all_reduce(missing)
        if missing.item() and reason is None:
            reason = "Engram symmetric transport unavailable on another rank"
    if reason is not None:
        pytest.skip(reason)


@contextmanager
def closing_symmetric_transports():
    """Close the Engram symmetric transports that the enclosed code starts, on every rank alike.

    A clean close waits in a device barrier for every rank of a row group, so it runs only when
    the code passed on every rank; otherwise every rank aborts its transports.
    """
    passed = False
    try:
        yield
        passed = True
    finally:
        clean = passed
        if dist.is_initialized():
            flag = torch.tensor([int(passed)], device=_world_device())
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            clean = bool(flag.item())
        symmetric_lookup.close_engram_transports(clean=clean)
