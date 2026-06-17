# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group / grid helpers for hetero MIMO examples."""

from __future__ import annotations

import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


def get_grid_dim_size(grid: HyperCommGrid, dim: str) -> int:
    """Return the size of ``dim`` in a HyperCommGrid, or 1 if absent."""
    try:
        return int(grid.shape[grid.dim_names.index(dim)])
    except (ValueError, AttributeError):
        return 1


def get_group_size_or(pg, fallback: int) -> int:
    """Return ``pg``'s world size when joinable, else ``fallback``."""
    if pg is None:
        return fallback
    return dist.get_world_size(group=pg)


def get_group_rank_or(pg, fallback: int = 0) -> int:
    """Return this rank's index within ``pg``, else ``fallback``."""
    if pg is None:
        return fallback
    rank = dist.get_rank(group=pg)
    return rank if rank >= 0 else fallback


def is_process_group_member(pg) -> bool:
    """Whether the current rank belongs to ``pg`` (sentinel-based, matches core)."""
    return pg is not None and dist.get_rank(group=pg) >= 0
