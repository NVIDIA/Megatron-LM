# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner assignment for MFSDP v2 distributed buffers.

Tensor shapes and shard ranges come from the buffer's layout and placements.
Owner assignments use global process ranks.
"""

import dataclasses
from collections.abc import Callable, Iterable

import torch

from .dbuffer import DBuffer
from .layout import non_leading_numel
from .range import intersect_ranges


def ns_cost_fn(num_ns_steps: int) -> Callable[[torch.Size], int]:
    """Estimate Newton-Schulz work from a tensor's full shape.

    View the tensor as `(rows, cols) = (shape[0], shape[1:].numel())`. After
    normalization and transposing so that rows <= cols, a quintic Newton-Schulz
    step with scalar coefficients `(a, b, c)` is:

        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    The matrix products cost O(numel * min(rows, cols)) per step. We use
    `numel * (min(rows, cols) * num_ns_steps + 1)` as a relative work estimate,
    omitting constant factors and adding a linear term for normalization and
    elementwise work.
    """

    def cost_fn(shape: torch.Size) -> int:
        short_dim = min(shape[0], non_leading_numel(shape))
        return shape.numel() * (short_dim * num_ns_steps + 1)

    return cost_fn


@dataclasses.dataclass(frozen=True)
class _OwnerWork:
    """A tensor's compute cost and eligible owner ranks in buffer order."""

    tensor_index: int
    cost: float
    candidates: list[int]


def assign_owner_work(
    buffer: DBuffer,
    tensor_indices: Iterable[int],
    cost_fn: Callable[[torch.Size], float] | None = None,
) -> dict[int, int]:
    """Assign each participating tensor to a global rank holding part of it.

    Tensors with a single candidate are assigned first, followed by tensors with
    multiple candidates in descending cost order. Each tensor is assigned to its
    least-loaded holder. Ties follow buffer order.

    Args:
        buffer: Distributed buffer providing the layout, mesh, and placements.
        tensor_indices: Indices of participating tensors in `buffer.layout`. The caller
            selects eligible tensors, for example those with at least two dimensions.
        cost_fn: Positive cost estimate from a tensor's full shape. Defaults to
            Newton-Schulz with five iterations.
    """
    if cost_fn is None:
        cost_fn = ns_cost_fn(num_ns_steps=5)

    layout = buffer.layout
    mesh = buffer.mesh
    rank_ranges = {
        rank: layout.get_rank_range(mesh, buffer.placements, rank)
        for rank in mesh.mesh.flatten().tolist()
    }
    # Offset order, rather than global rank order, determines load-balancing ties.
    ranks_in_buffer_order = sorted(rank_ranges, key=lambda rank: (rank_ranges[rank].start, rank))

    work_items: list[_OwnerWork] = []
    for tensor_index in tensor_indices:
        tensor_range = layout.get_tensor_range(tensor_index)
        candidates = [
            rank
            for rank in ranks_in_buffer_order
            if intersect_ranges(tensor_range, rank_ranges[rank]).numel > 0
        ]
        shape = layout.tensor_shapes[tensor_index]
        if not candidates:
            raise RuntimeError(
                f"No eligible owner for tensor {tensor_index} with shape {shape}; "
                "no rank owns a shard."
            )
        work_items.append(_OwnerWork(tensor_index, cost_fn(shape), candidates))

    work_items.sort(key=lambda item: (len(item.candidates) > 1, -item.cost))

    running_cost: dict[int, float] = {rank: 0.0 for rank in rank_ranges}
    assignments: dict[int, int] = {}
    for item in work_items:
        owner = min(item.candidates, key=lambda rank: running_cost[rank])
        assignments[item.tensor_index] = owner
        running_cost[owner] += item.cost
    return assignments
