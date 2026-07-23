# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""JSON Lines logging for high-cardinality training statistics."""

import json
import os
from collections.abc import Iterable
from contextlib import nullcontext

import torch


def _nvtx_range(message: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(message)
    return nullcontext()


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def append_training_stat(log_dir: str, stat_name: str, record: dict, rank: int | None = None):
    """Append one JSONL record for a training statistic.

    Each rank writes to its own file under ``{log_dir}/training_stats/{stat_name}/``.
    Callers decide which ranks should write; this function only handles the file layout.
    """
    rank = _get_rank() if rank is None else rank
    stat_dir = os.path.join(log_dir, "training_stats", stat_name)
    os.makedirs(stat_dir, exist_ok=True)
    filepath = os.path.join(stat_dir, f"rank{rank}.jsonl")
    with _nvtx_range("training_stats.json_dumps"):
        payload = json.dumps(record) + "\n"
    with _nvtx_range("training_stats.file_write"):
        with open(filepath, "a") as f:
            f.write(payload)


def save_raw_moments_by_name(
    log_dir: str,
    stat_name: str,
    iteration: int,
    consumed_train_samples: int,
    raw_moments_by_name: Iterable[tuple[str, dict[str, float]]],
    rank: int | None = None,
) -> None:
    """Append one named raw moments record."""
    with _nvtx_range("training_stats.dictionary_construction"):
        values = {
            name: {field: float(value) for field, value in raw_moments.items()}
            for name, raw_moments in raw_moments_by_name
        }
        if not values:
            return

        record = {
            "iter": iteration,
            "consumed_train_samples": consumed_train_samples,
            "stat": stat_name,
            "values": values,
        }

    append_training_stat(log_dir, stat_name, record, rank=rank)
