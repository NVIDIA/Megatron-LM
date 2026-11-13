# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Joint owner assignment across MFSDP v2 distributed buffers.

Tensor shapes and shard ranges come from the buffer's layout and placements.
Owner assignments use global process ranks.
"""

import dataclasses
from collections.abc import Iterable

from .dbuffer import DBuffer
from .layout import non_leading_numel
from .range import intersect_ranges


@dataclasses.dataclass(frozen=True)
class _OwnerWork:
    """A tensor's compute cost and eligible owner ranks in buffer order."""

    buffer_index: int
    tensor_index: int
    cost: float
    candidates: list[int]


def assign_owner_work(buffers: Iterable[DBuffer]) -> list[dict[int, int]]:
    """Jointly assign tensors with at least two dimensions to global ranks holding their shards.

    Account for single-candidate tensors across all buffers first, then assign the remaining
    tensors in descending cost order across buffers. Each tensor goes to its least-loaded
    holder. Equal-cost tensors follow input buffer order and tensor index; equal-load holders
    follow buffer offset order. Inputs must be identical across ranks.

    Costs estimate five Newton-Schulz iterations from each tensor's full shape. View the
    tensor as `(rows, cols) = (shape[0], shape[1:].numel())`. After normalization and
    transposing so that rows <= cols, a quintic Newton-Schulz step with scalar coefficients
    `(a, b, c)` is:

        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    The matrix products cost O(numel * min(rows, cols)) per step. We use
    `numel * (min(rows, cols) * 5 + 1)` as a relative work estimate, omitting constant factors
    and adding a linear term for normalization and elementwise work.

    Args:
        buffers: Distributed buffers on the same DP mesh.

    Returns:
        One tensor-index-to-owner mapping per buffer, in input order. Buffers with no
        eligible tensors have an empty mapping.
    """
    buffers = list(buffers)
    if not buffers:
        return []
    mesh = buffers[0].mesh
    ranks = mesh.mesh.flatten().tolist()
    assignments: list[dict[int, int]] = [{} for _ in buffers]
    work_items: list[_OwnerWork] = []
    for buffer_index, buffer in enumerate(buffers):
        if buffer.mesh != mesh:
            raise ValueError(
                "`assign_owner_work` requires every buffer to share the same DP mesh; "
                f"buffer {buffer_index}'s mesh differs from buffer 0's. "
                "Balance buffers per mesh."
            )
        layout = buffer.layout
        rank_ranges = {rank: layout.get_rank_range(mesh, buffer.placements, rank) for rank in ranks}
        # Offset order, rather than global rank order, determines load-balancing ties.
        ranks_in_buffer_order = sorted(
            rank_ranges, key=lambda rank: (rank_ranges[rank].start, rank)
        )
        for tensor_index, shape in enumerate(layout.tensor_shapes):
            if len(shape) < 2:
                continue
            tensor_range = layout.get_tensor_range(tensor_index)
            candidates = [
                rank
                for rank in ranks_in_buffer_order
                if intersect_ranges(tensor_range, rank_ranges[rank]).numel > 0
            ]
            if not candidates:
                raise RuntimeError(
                    f"No eligible owner for tensor {tensor_index} with shape {shape}; "
                    "no rank owns a shard."
                )
            short_dim = min(shape[0], non_leading_numel(shape))
            cost = shape.numel() * (short_dim * 5 + 1)
            work_items.append(_OwnerWork(buffer_index, tensor_index, cost, candidates))

    work_items.sort(
        key=lambda item: (
            len(item.candidates) > 1,
            -item.cost,
            item.buffer_index,
            item.tensor_index,
        )
    )

    running_cost: dict[int, float] = {rank: 0.0 for rank in ranks}
    for item in work_items:
        owner = min(item.candidates, key=lambda rank: running_cost[rank])
        assignments[item.buffer_index][item.tensor_index] = owner
        running_cost[owner] += item.cost
    return assignments
