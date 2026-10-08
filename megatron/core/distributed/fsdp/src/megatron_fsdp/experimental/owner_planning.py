# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner assignment and P2P packing for MFSDP v2's all-`RowAtomic` layout.

Tensor shapes and shard ranges come from the group's existing `GlobalLayout`.
Owner assignments and peer dictionaries use global process ranks.
"""

import dataclasses
from collections.abc import Callable, Iterable
from typing import Self

import torch
from torch.distributed.device_mesh import DeviceMesh

from .layout import GlobalLayout, non_leading_numel
from .parameter_group import FsdpParameterGroup
from .placement import RowAtomic
from .range import Range, intersect_ranges


def select_ge_2d_params(param: torch.Tensor) -> bool:
    """Whether the given tensor has dimensionality ≥2."""
    return param.ndim >= 2


def ns_cost_fn(num_ns_steps: int) -> Callable[[torch.Size], int]:
    """Estimate Newton-Schulz work from a tensor's full shape.

    Uses `numel * (min(rows, cols) * num_steps + 1)` with the DBuffer's
    leading-dim view `(shape[0], shape[1:].numel())`.
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
        owners: Participating tensor index to owner global process rank. These keys
            identify the participating tensors; no filtered layout copy is stored.
    """

    mesh: DeviceMesh
    layout: GlobalLayout
    owners: dict[int, int]

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
        owners = assign_owner_work(layout, group.mesh, tensor_indices, cost_fn)
        return cls(mesh=group.mesh, layout=layout, owners=owners)


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of a set of parameters.

    The owner keeps its own shard locally (no self-send), so it only receives from the other
    shard-holding ranks. `reconstruct_full` reconstructs each owned tensor by concatenating the
    per-rank shards in buffer order.

    Example:

    ```
    # We are also using some pseudocode here for brevity.

    # Assume:
    torch.distributed.get_world_size() == 2
    torch.distributed.get_rank() == 0  # We're observing from rank 0.
    mesh.mesh.tolist() == [0, 1]  # Global process ranks.
    param_0: torch.Tensor
    param_1: torch.Tensor
    # Params are in this order as observed by MFSDP.
    model.param_groups == [{"params": [param_0, param_1]}]
    param_0.shape == (6, 4)  # Global shape.
    param_1.shape == (4, 4)  # Global shape.
    layout.tensor_to_offset == (0, 24)
    layout.size == 40
    # The buffer, rather than each parameter separately, is split evenly:
    # rank 0 holds [0, 20); rank 1 holds [20, 40).
    # Choose rank 1 as both owners for this example. It holds part of param_0
    # and all of param_1; these are explicit assignments, not the default balancer's output.
    owner_layout = GroupOwnerLayout(mesh=mesh, layout=layout, owners={0: 1, 1: 1})

    param_0.local_shard.shape == (5, 4)  # Rank 0 holds param_0[0:5, ...].
    param_1.local_shard.shape == (0, 4)  # Rank 0 holds none of param_1.
    param_0.local_shard.numel() == 20
    param_1.local_shard.numel() == 0

    owner_gather_plan.send_buffers == {1: tensor(20)}  # 20 elements from param_0.
    # `owner_gather_plan.send_buffers[1]` represents the following flat buffer:
    #   +--------------------+
    #   | param_0 (20 elems) |
    #   +--------------------+
    #      element order: -->

    # Rank 0 owns nothing.
    owner_gather_plan.recv_sizes == {}
    owner_gather_plan.own_shards == {}
    owner_gather_plan.recv_offsets == {}

    # ---

    # Same settings as above, now observing from rank 1 (the owner):
    torch.distributed.get_rank() == 1

    param_0.local_shard.shape == (1, 4)  # Rank 1 holds param_0[5:6, ...].
    param_1.local_shard.shape == (4, 4)  # Rank 1 holds all of param_1.

    send_buffers = {}  # Rank 1 owns everything.
    recv_sizes = {0: 20}  # 20 elements from rank 0.
    # Rank 1's own shards, flattened (views):
    own_shards = {0: param_0.local_shard.view(-1), 1: param_1.local_shard.view(-1)}
    recv_offsets = {
        (0, 0): 0,  # `param_0` (tensor index 0) from rank 0: offset 0.
        # No entry for param_1: it is fully local to rank 1.
    }
    # Reconstruct param_0 from the received 20 elements followed by its own 4.
    # Reconstruct param_1 directly from its own 16 elements, without communication.
    ```

    Attributes:
        send_buffers: Per-destination-owner flat send buffer (this rank's shards for that owner's
            params, in tensor-index order). Only owners with non-zero send size appear.
        recv_sizes: Per-source-rank element count this rank (as an owner) receives. Only sources
            with non-zero total size appear.
        own_shards: This rank's flat local shard per owned parameter, keyed by tensor index (used
            directly in reconstruction, not communicated).
        recv_offsets: Per `(tensor_index, src_rank)`, the flat offset of this param's shard inside
            the recv buffer received from `src_rank`. Only contains tuples for which `src_rank`
            holds elements.
    """

    layout: GlobalLayout
    # Global rank -> buffer offset/length, ordered by buffer offset.
    rank_ranges: dict[int, Range]
    this_rank: int
    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], int]

    @classmethod
    def pack(cls, plan: GroupOwnerLayout, local_shards: dict[int, torch.Tensor]) -> Self:
        """Pack this rank's local shards into per-owner P2P send buffers.

        Args:
            plan: The group's owner layout.
            local_shards: This rank's local shard per parameter, only required for every parameter
                it holds elements of. Shards may be passed in any shape.
        """
        mesh = plan.mesh
        this_rank = mesh.get_rank()
        layout = plan.layout
        placements = (RowAtomic(),) * mesh.ndim
        owners = plan.owners
        tensor_indices = sorted(owners)
        owned_indices = [i for i in tensor_indices if owners[i] == this_rank]

        # Query each source once and compute both its receive size and offsets.
        rank_ranges: dict[int, Range] = {}
        recv_sizes: dict[int, int] = {}
        recv_offsets: dict[tuple[int, int], int] = {}
        for src in mesh.mesh.flatten().tolist():
            buffer_range = layout.get_rank_range(mesh, placements, src)
            rank_ranges[src] = buffer_range
            if src == this_rank:
                continue
            offset = 0
            for tensor_index in owned_indices:
                numel = intersect_ranges(layout.get_tensor_range(tensor_index), buffer_range).numel
                if numel > 0:
                    recv_offsets[(tensor_index, src)] = offset
                    offset += numel
            if offset > 0:
                recv_sizes[src] = offset

        # Reconstruction concatenates shards in buffer order, including on multi-axis meshes.
        rank_ranges = dict(sorted(rank_ranges.items(), key=lambda item: item[1].start))
        local_range = rank_ranges[this_rank]
        send_sizes: dict[int, int] = {}
        for tensor_index in tensor_indices:
            owner = owners[tensor_index]
            if owner == this_rank:
                continue
            numel = intersect_ranges(layout.get_tensor_range(tensor_index), local_range).numel
            if numel > 0:
                send_sizes[owner] = send_sizes.get(owner, 0) + numel

        send_buffers: dict[int, torch.Tensor] = {}
        if send_sizes:
            first_shard = next(iter(local_shards.values()))
            for owner, size in send_sizes.items():
                send_buffers[owner] = torch.empty(
                    size, dtype=first_shard.dtype, device=first_shard.device
                )

        # Fill each owner's send buffer in tensor-index order.
        cursors: dict[int, int] = {owner: 0 for owner in send_buffers}
        own_shards: dict[int, torch.Tensor] = {}
        for tensor_index in tensor_indices:
            owner = owners[tensor_index]
            if owner == this_rank:
                own_shards[tensor_index] = local_shards[tensor_index].flatten()
                continue
            numel = intersect_ranges(layout.get_tensor_range(tensor_index), local_range).numel
            if numel == 0:
                continue
            shard = local_shards[tensor_index]
            buf = send_buffers[owner]
            buf[cursors[owner] : cursors[owner] + numel].copy_(shard.flatten())
            cursors[owner] += numel

        return cls(
            layout=layout,
            rank_ranges=rank_ranges,
            this_rank=this_rank,
            send_buffers=send_buffers,
            recv_sizes=recv_sizes,
            own_shards=own_shards,
            recv_offsets=recv_offsets,
        )

    def reconstruct_full(
        self, param_index: int, recv_buffers: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct the full flat tensor for one owned parameter from its per-rank shards.

        Concatenates the per-rank shards in buffer order. Results can be
        `view`ed into the desired shape. For a parameter only this rank holds elements of, the own
        flat shard is returned directly.

        Args:
            param_index: Tensor index of an owned parameter in the group layout.
            recv_buffers: Per-source-rank received buffer (only sources that sent).
        """
        tensor_range = self.layout.get_tensor_range(param_index)
        shards: list[torch.Tensor] = []
        for src, buffer_range in self.rank_ranges.items():
            if src == self.this_rank:
                shards.append(self.own_shards[param_index])
                continue

            numel = intersect_ranges(tensor_range, buffer_range).numel
            if numel == 0:
                continue

            offset = self.recv_offsets[(param_index, src)]
            buf = recv_buffers[src]
            shards.append(buf[offset : offset + numel])
        if len(shards) == 1:
            return shards[0]
        return torch.cat(shards)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of a set of parameters.

    The owner keeps its own result shard (applied directly), so it only sends to the other
    shard-holding ranks.

    Attributes:
        send_buffers: Per-destination-rank flat send buffer (this owner's result shards for the
            params it owns, in tensor-index order). Only destinations with non-zero send size
            appear.
        recv_sizes: Per-owner-rank element count this rank (as a destination) receives. Only owners
            with non-zero total size appear.
        recv_offsets: Per `(tensor_index, owner_rank)`, the flat offset of this param's result shard
            inside the recv buffer received from `owner_rank`. Only contains tuples for which this
            rank holds elements.
    """

    layout: GlobalLayout
    local_range: Range
    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    recv_offsets: dict[tuple[int, int], int]

    @classmethod
    def pack(cls, plan: GroupOwnerLayout, full_results: dict[int, torch.Tensor]) -> Self:
        """Pack this owner rank's full results into per-destination P2P send buffers.

        Args:
            plan: The group's owner layout.
            full_results: Full result tensor per parameter this rank owns. Tensors may be passed in
                any shape.
        """
        mesh = plan.mesh
        this_rank = mesh.get_rank()
        layout = plan.layout
        placements = (RowAtomic(),) * mesh.ndim
        local_range = layout.get_local_range(mesh, placements)
        owners = plan.owners
        tensor_indices = sorted(owners)
        flat_results = {
            i: full_results[i].flatten() for i in tensor_indices if owners[i] == this_rank
        }

        # Query each destination once and pack its shards in tensor-index order.
        send_buffers: dict[int, torch.Tensor] = {}
        for dest in mesh.mesh.flatten().tolist():
            if dest == this_rank:
                continue
            buffer_range = layout.get_rank_range(mesh, placements, dest)
            chunks: list[torch.Tensor] = []
            for tensor_index, flat in flat_results.items():
                tensor_range = layout.get_tensor_range(tensor_index)
                shard_range = intersect_ranges(tensor_range, buffer_range)
                if shard_range.numel > 0:
                    offset = shard_range.start - tensor_range.start
                    chunks.append(flat.narrow(0, offset, shard_range.numel))
            if chunks:
                send_buffers[dest] = torch.cat(chunks)

        # Compute each owner's receive size and offsets together from our local range.
        recv_sizes: dict[int, int] = {}
        recv_offsets: dict[tuple[int, int], int] = {}
        for tensor_index in tensor_indices:
            owner = owners[tensor_index]
            if owner == this_rank:
                continue
            numel = intersect_ranges(layout.get_tensor_range(tensor_index), local_range).numel
            if numel > 0:
                offset = recv_sizes.get(owner, 0)
                recv_offsets[(tensor_index, owner)] = offset
                recv_sizes[owner] = offset + numel

        return cls(
            layout=layout,
            local_range=local_range,
            send_buffers=send_buffers,
            recv_sizes=recv_sizes,
            recv_offsets=recv_offsets,
        )

    def unpack(self, recv_buffers: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Extract this rank's local flat result shards from the per-owner recv buffers.

        Args:
            recv_buffers: Per-owner-rank received buffer (only owners that sent).

        Returns:
            Mapping from tensor index to this rank's local flat result shard, for parameters this
            rank holds elements of but does NOT own.
        """
        results: dict[int, torch.Tensor] = {}
        for (tensor_index, owner), offset in self.recv_offsets.items():
            numel = intersect_ranges(
                self.layout.get_tensor_range(tensor_index), self.local_range
            ).numel
            buf = recv_buffers[owner]
            results[tensor_index] = buf[offset : offset + numel]
        return results
