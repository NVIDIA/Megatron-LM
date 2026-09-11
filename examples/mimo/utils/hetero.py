# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group / grid helpers for hetero MIMO examples."""

from __future__ import annotations

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1


def get_data_lane_rank(pg_collection: ProcessGroupCollection) -> int:
    """Return the DP x GTP data-lane rank, excluding model-parallel dimensions."""
    gtp_group = getattr(pg_collection, "gtp_remat", None)
    gtp_size = gtp_group.size() if gtp_group is not None else 1
    gtp_rank = gtp_group.rank() if gtp_group is not None else 0
    return pg_collection.dp.rank() * gtp_size + gtp_rank
