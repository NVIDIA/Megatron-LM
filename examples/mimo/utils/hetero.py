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


def get_pg_size(pg) -> int:
    """World size of a required process group (asserts the group is present)."""
    assert pg is not None, "required process group is missing from the collection"
    return dist.get_world_size(group=pg)


def get_pg_rank(pg) -> int:
    """This rank's index within a required process group (asserts the group is present)."""
    assert pg is not None, "required process group is missing from the collection"
    return dist.get_rank(group=pg)


def is_process_group_member(pg) -> bool:
    """Whether the current rank belongs to ``pg`` (sentinel-based, matches core)."""
    return pg is not None and dist.get_rank(group=pg) >= 0
