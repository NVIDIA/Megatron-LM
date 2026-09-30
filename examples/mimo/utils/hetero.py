# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group / grid helpers for hetero MIMO examples."""

from __future__ import annotations

from math import gcd

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import resolve_tensor_parallel_weight_shards
from megatron.core.process_groups_config import ProcessGroupCollection


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1


def get_data_lane_rank(pg_collection: ProcessGroupCollection) -> int:
    """Return the independent sample rank, excluding every context partition."""
    sample_group = getattr(pg_collection, "dp_gtp_remat", None)
    if sample_group is not None:
        return sample_group.rank()
    # Compatibility for collections with independent CP/GTP axes only.
    gtp_group = getattr(pg_collection, "gtp_remat", None)
    gtp_size = gtp_group.size() if gtp_group is not None else 1
    gtp_rank = gtp_group.rank() if gtp_group is not None else 0
    return pg_collection.dp.rank() * gtp_size + gtp_rank


def get_language_sample_parallel_size(args) -> int:
    """Count language data lanes before or after stock argument validation."""
    _, weight_size = resolve_tensor_parallel_weight_shards(
        getattr(args, "mimo_llm_tp", 1),
        getattr(args, "tensor_parallel_num_weight_shards", None),
        getattr(args, "gtp_weight_remat_size", 1),
    )
    overlap = gcd(args.mimo_llm_cp, weight_size) if getattr(args, "gtp_remat_fold_cp", False) else 1
    return args.mimo_llm_dp * (weight_size // overlap)
