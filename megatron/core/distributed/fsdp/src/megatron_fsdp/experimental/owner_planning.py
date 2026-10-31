# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Joint owner assignment across MFSDP v2 distributed buffers.

Tensor shapes and shard ranges come from the buffer's layout and placements.
Owner assignments use global process ranks.
"""

import dataclasses
from collections.abc import Iterable

from torch.distributed.device_mesh import DeviceMesh

from .dbuffer import DBuffer
from .layout import non_leading_numel
from .range import intersect_ranges


@dataclasses.dataclass(frozen=True)
class OwnerPlan:
    """The owners of a buffer's tensors, and which tensors need communication to reach them."""

    tensor_to_owner: dict[int, int]
    # Tensors whose shards sit on several ranks. Gather and scatter post P2P only for these, so
    # a buffer without any needs no communication.
    split_tensors: frozenset[int]


@dataclasses.dataclass(frozen=True)
class _OwnerWork:
    """A tensor's compute cost and eligible owner ranks in buffer order."""

    buffer_index: int
    tensor_index: int
    cost: float
    candidates: list[int]


def assign_owner_work(buffers: Iterable[DBuffer]) -> list[OwnerPlan]:
    """Jointly assign tensors with at least two dimensions to global ranks holding their shards.

    Account for single-candidate tensors across all buffers first, then assign the remaining
    tensors in descending cost order across buffers. Each tensor goes to its least-loaded
    holder. Equal-cost tensors follow input buffer order and tensor index; equal-load holders
    follow buffer offset order. Buffers on one mesh must be identical across its ranks.

    Each mesh is planned independently, because a joint plan would depend on buffers a rank
    cannot see, such as another expert data-parallel group's experts.

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
        buffers: Distributed buffers, which may be on different DP meshes.

    Returns:
        One plan per buffer, in input order: the tensor-index-to-owner mapping and the tensors
        whose shards span ranks. Buffers with no eligible tensors have an empty plan.
    """
    buffers = list(buffers)

    # Step 1: group the buffers by mesh, remembering each buffer's position in the input.
    positions_by_mesh: dict[DeviceMesh, list[int]] = {}
    for position, buffer in enumerate(buffers):
        positions_by_mesh.setdefault(buffer.mesh, []).append(position)

    # Step 2: assign owners within each mesh, then put the results back in input order.
    plans: list[OwnerPlan] = [OwnerPlan({}, frozenset()) for _ in buffers]
    for positions in positions_by_mesh.values():
        mesh_buffers = [buffers[position] for position in positions]
        mesh_plans = _assign_owner_work_on_mesh(mesh_buffers)
        for position, plan in zip(positions, mesh_plans, strict=True):
            plans[position] = plan
    return plans


def _assign_owner_work_on_mesh(buffers: list[DBuffer]) -> list[OwnerPlan]:
    """Jointly assign the tensors of buffers that share one mesh."""
    mesh = buffers[0].mesh
    ranks = mesh.mesh.flatten().tolist()
    assignments: list[dict[int, int]] = [{} for _ in buffers]
    split_tensors: list[set[int]] = [set() for _ in buffers]
    work_items: list[_OwnerWork] = []
    for buffer_index, buffer in enumerate(buffers):
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
            if len(candidates) > 1:
                split_tensors[buffer_index].add(tensor_index)
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
    return [
        OwnerPlan(assignment, frozenset(split))
        for assignment, split in zip(assignments, split_tensors, strict=True)
    ]
