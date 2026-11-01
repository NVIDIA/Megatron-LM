# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure topology helpers shared by dynamic CP initialization and scheduling."""

from typing import List


def get_valid_dynamic_context_parallel_group_sizes(total_size: int) -> List[int]:
    """Return runtime CP sizes that can partition a DPxCP rank list exactly.

    Power-of-two subgroups are used when they evenly divide the rank list. The
    full DPxCP group is always valid and reuses the existing communicator.
    """
    if total_size < 1:
        raise ValueError(f"Dynamic CP group size must be positive, got {total_size}")

    group_sizes = []
    group_size = 1
    while group_size < total_size:
        if total_size % group_size == 0:
            group_sizes.append(group_size)
        group_size *= 2
    group_sizes.append(total_size)
    return group_sizes
