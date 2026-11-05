# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Joint owner assignment across MFSDP v2 distributed buffers.

Tensor shapes and shard ranges come from the buffer's layout and placements.
Owner assignments use global process ranks.
"""

import dataclasses
from collections.abc import Iterable

import torch.distributed as dist

from .dbuffer import DBuffer
from .gather_scatter import get_shard_ranks
from .layout import non_leading_numel
from .range import intersect_ranges

# Identifies a buffer identically on every rank that holds it: its shard ranks, and its index
# among the buffers those ranks share, which every holder lists in the same order.
_BufferKey = tuple[tuple[int, ...], int]


@dataclasses.dataclass(frozen=True)
class _OwnerWork:
    """A tensor's compute cost and eligible owner ranks in buffer order."""

    buffer_key: _BufferKey
    tensor_index: int
    cost: float
    candidates: list[int]


def assign_owner_work(buffers: Iterable[DBuffer]) -> list[dict[int, int]]:
    """Jointly assign tensors with at least two dimensions to global ranks holding their shards.

    Account for single-candidate tensors across all buffers first, then assign the remaining
    tensors in descending cost order across buffers. Each tensor goes to its least-loaded
    holder. Equal-cost tensors follow buffer key and tensor index; equal-load holders
    follow buffer offset order. Buffers on one mesh must be identical across its ranks.

    The plan is global: every rank all-gathers the tensors of all ranks, including the experts
    of other EP ranks, and runs the same assignment, so load balances across meshes. Every rank
    must call this function. Owners are chosen among the ranks that shard a buffer together, so
    under HSDP each replica gets its own owners.

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
        buffers: This rank's distributed buffers, which may be on different DP meshes.

    Returns:
        One tensor-index-to-owner mapping per buffer, in input order. Buffers with no
        eligible tensors have an empty mapping.
    """
    buffers = list(buffers)
    assignments: list[dict[int, int]] = [{} for _ in buffers]
    # Gathered items name buffers by key; map this rank's keys back to its buffer indices.
    buffer_indices: dict[_BufferKey, int] = {}
    work_items: list[_OwnerWork] = []
    for buffer_index, buffer in enumerate(buffers):
        mesh = buffer.mesh
        ranks = tuple(get_shard_ranks(buffer))
        buffer_key = (ranks, sum(key[0] == ranks for key in buffer_indices))
        buffer_indices[buffer_key] = buffer_index
        # One holder reports each buffer, rotating over its shard ranks to spread the metadata.
        if ranks[buffer_key[1] % len(ranks)] != dist.get_rank():
            continue
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
            short_dim = min(shape[0], non_leading_numel(shape))
            cost = shape.numel() * (short_dim * 5 + 1)
            work_items.append(_OwnerWork(buffer_key, tensor_index, cost, candidates))

    gathered: list[list[_OwnerWork]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, work_items)
    work_items = sorted(
        (item for items in gathered for item in items),
        key=lambda item: (len(item.candidates) > 1, -item.cost, item.buffer_key, item.tensor_index),
    )

    running_cost: dict[int, float] = {rank: 0.0 for rank in range(dist.get_world_size())}
    for item in work_items:
        # Raise after the gather so that every rank fails alike instead of some hanging.
        if not item.candidates:
            raise RuntimeError(
                f"No eligible owner for tensor {item.tensor_index}; no rank owns a shard."
            )
        owner = min(item.candidates, key=lambda rank: running_cost[rank])
        if item.buffer_key in buffer_indices:
            assignments[buffer_indices[item.buffer_key]][item.tensor_index] = owner
        running_cost[owner] += item.cost
    return assignments
