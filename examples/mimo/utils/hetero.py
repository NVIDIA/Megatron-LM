# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group / grid helpers for hetero MIMO examples."""

from __future__ import annotations

from megatron.core.hyper_comm_grid import HyperCommGrid


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1
