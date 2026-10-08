# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared row selection for packed and unpacked context-parallel batches."""

from collections.abc import Iterable
from typing import Any

import torch

from .types import CPLayout


def get_cp_partition_indices(
    cu_seqlens: torch.Tensor | None, total_tokens: int, cp_size: int, cp_rank: int, layout: CPLayout
) -> torch.Tensor | slice:
    """Return the rank's contiguous interval or per-document zigzag indices.

    Zigzag inputs must already meet TE's per-document alignment contract. The batch
    adapter owns padding policy; this helper never changes physical sequence boundaries.
    """
    if layout == "contiguous":
        if total_tokens % cp_size != 0:
            raise RuntimeError(
                f"Contiguous CP slicing requires total_tokens={total_tokens} to be divisible by "
                f"cp_size={cp_size}."
            )
        local_rows = total_tokens // cp_size
        return slice(cp_rank * local_rows, (cp_rank + 1) * local_rows)
    if layout != "zigzag":
        raise ValueError(f"Unsupported CP partition mode: {layout}")
    if cu_seqlens is None:
        raise ValueError("Per-document zigzag partitioning requires cu_seqlens.")

    # Import lazily to avoid initializing TE for contiguous-only batch partitioning.
    from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

    return get_thd_partitioned_indices(
        cu_seqlens.to(dtype=torch.int32), total_tokens, cp_size, cp_rank
    )


def partition_batch(
    batch: dict[str, Any], keys: Iterable[str], partition: torch.Tensor | slice, seq_dim: int
) -> None:
    """Replace selected batch tensors with their rank-local sequence rows.

    The sequence dimension is explicit so scheduler ``[T]`` tensors and ordinary
    ``[B, T]`` tensors use the same selection. Missing or None-valued keys are preserved.
    Contiguous slices remain views; callers choose whether to materialize them.
    """
    for key in keys:
        tensor = batch.get(key)
        if tensor is None:
            continue
        if isinstance(partition, slice):
            batch[key] = tensor.narrow(seq_dim, partition.start, partition.stop - partition.start)
        else:
            batch[key] = tensor.index_select(seq_dim, partition)
