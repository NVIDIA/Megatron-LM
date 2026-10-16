# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner assignment for MFSDP v2's all-`RowAtomic` layout.

Tensor shapes and shard ranges come from the group's existing `GlobalLayout`.
Owner assignments use global process ranks.
"""

import dataclasses
from collections.abc import Callable, Iterable
from typing import Self

import torch
from torch.distributed.device_mesh import DeviceMesh

from .layout import GlobalLayout, non_leading_numel
from .parameter_group import FsdpParameterGroup
from .placement import RowAtomic
from .range import intersect_ranges


def select_ge_2d_params(param: torch.Tensor) -> bool:
    """Whether the given tensor has dimensionality ≥2."""
    return param.ndim >= 2


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
    layout: GlobalLayout,
    mesh: DeviceMesh,
    tensor_indices: Iterable[int],
    cost_fn: Callable[[torch.Size], float] | None = None,
) -> dict[int, int]:
    """Assign each participating tensor to a global rank holding part of it.

    Tensors with a single candidate are assigned first, followed by tensors with
    multiple candidates in descending cost order. Each tensor is assigned to its
    least-loaded holder. Ties follow buffer order.

    Args:
        layout: The group's DBuffer layout, including padding and ineligible tensors.
        mesh: The device mesh across which the buffer is all-RowAtomic sharded.
        tensor_indices: Indices of participating tensors in `layout`.
        cost_fn: Positive cost estimate from a tensor's full shape. Defaults to
            Newton-Schulz with five iterations.
    """
    if cost_fn is None:
        cost_fn = ns_cost_fn(num_ns_steps=5)

    placements = (RowAtomic(),) * mesh.ndim
    rank_ranges = {
        rank: layout.get_rank_range(mesh, placements, rank) for rank in mesh.mesh.flatten().tolist()
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


@dataclasses.dataclass(frozen=True)
class GroupOwnerLayout:
    """A group's shared buffer layout and owner assignments.

    Attributes:
        mesh: The device mesh over which the buffer is all-RowAtomic sharded.
        layout: The existing `DBuffer.layout`, including all tensors in the group.
        tensor_to_owner: Participating tensor index to owner global process rank. These keys
            identify the participating tensors; no filtered layout copy is stored.
    """

    mesh: DeviceMesh
    layout: GlobalLayout
    tensor_to_owner: dict[int, int]

    @classmethod
    def from_group(
        cls,
        group: FsdpParameterGroup,
        *,
        cost_fn: Callable[[torch.Size], float] | None = None,
        eligible_fn: Callable[[torch.Tensor], bool] | None = None,
    ) -> Self:
        """Select participating parameters and balance their owner assignments.

        `eligible_fn` defaults to selecting ≥2D parameters. `cost_fn` receives
        full tensor shapes; see `assign_owner_work` for the default estimate.
        """
        if eligible_fn is None:
            eligible_fn = select_ge_2d_params
        tensor_indices = (
            i for i, param in enumerate(group.fsdp_parameters) if eligible_fn(param.sharded)
        )
        layout = group.main_weight.layout
        tensor_to_owner = assign_owner_work(layout, group.mesh, tensor_indices, cost_fn)
        return cls(mesh=group.mesh, layout=layout, tensor_to_owner=tensor_to_owner)
