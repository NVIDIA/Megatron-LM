# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group / grid helpers for hetero MIMO examples."""

from __future__ import annotations

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import resolve_tensor_parallel_weight_shards
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_parallel_layout import (
    resolve_tensor_parallel_sequence_shards,
)


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1


def get_data_lane_rank(pg_collection: ProcessGroupCollection) -> int:
    """Return the independent sample rank, excluding every context partition."""
    return pg_collection.dp_gtp_remat.rank()


def get_language_batch_parallel_size(args) -> int:
    """Count language batch partitions before or after stock argument validation."""
    _, weight_size = resolve_tensor_parallel_weight_shards(
        getattr(args, "mimo_llm_tp", 1),
        getattr(args, "tensor_parallel_num_weight_shards", None),
        getattr(args, "gtp_weight_remat_size", 1),
    )
    _, num_sequence_shards = resolve_tensor_parallel_sequence_shards(
        getattr(args, "mimo_llm_tp", 1),
        getattr(args, "tensor_parallel_num_sequence_shards", None),
        weight_size,
        getattr(args, "sequence_parallel", False),
    )
    return args.mimo_llm_dp * (weight_size // num_sequence_shards)
