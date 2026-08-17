# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure shard-planning and owner-compute packing logic for the Muon + M-FSDPv2 owner-compute P2P
algorithm.

The central data structure is `ShardPlan`, which describes how a single 2D parameter's full matrix
is split across the DP group under M-FSDPv2's all-`Flat` layout. Given shard plans,
`assign_owner_work` balances Newton-Schulz work across owner ranks, and the pack/unpack helpers
(`pack_owner_work`/`pack_update_shards`/`unpack_update_shards`) build the flat P2P send/recv
buffers. `reconstruct_full_tensor` stitches gathered shards back into the full matrix on the owner.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from .layout import non_leading_numel


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """How a single 2D parameter's full matrix is split across the DP group.

    M-FSDPv2's all-`Flat` layout shards dim-0 rows contiguously in rank order,
    so rank `r` owns the contiguous global row range
    `[start, start + count)` where `start`/`count` come from the flat
    DBuffer layout. A rank with `count == 0` holds no shard of this parameter.
    """

    full_shape: torch.Size
    rank_rows: tuple[tuple[int, int], ...]
    row_size: int

    def __post_init__(self) -> None:
        if len(self.full_shape) != 2:
            raise ValueError(f"ShardPlan requires a 2D full_shape, got {self.full_shape}.")
        if len(self.rank_rows) == 0:
            raise ValueError("ShardPlan requires at least one rank.")
        if self.row_size != non_leading_numel(self.full_shape):
            raise ValueError(
                f"ShardPlan row_size {self.row_size} != full_shape row size "
                f"{non_leading_numel(self.full_shape)}."
            )

    @property
    def world_size(self) -> int:
        """Number of ranks in the DP group for this parameter."""
        return len(self.rank_rows)

    def rank_row_count(self, rank: int) -> int:
        """Return the number of rows owned by `rank`."""
        return self.rank_rows[rank][1]

    def shard_numel(self, rank: int) -> int:
        """Return the number of elements in `rank`'s shard."""
        return self.rank_row_count(rank) * self.row_size

    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, (_, count) in enumerate(self.rank_rows) if count > 0)

    def is_boundary(self) -> bool:
        """True if more than one rank owns a non-empty shard of this parameter."""
        return sum(1 for _, count in self.rank_rows if count > 0) > 1

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()


def compute_shard_plan(
    full_shape: torch.Size, tensor_flat_offset: int, rank_flat_shard_size: int, world_size: int
) -> ShardPlan:
    """Compute the per-rank row ranges for one 2D parameter in a flat DBuffer.

    Args:
        full_shape: Global `(rows, cols)` shape of the parameter.
        tensor_flat_offset: Flat-element offset of this parameter inside the
            DBuffer's global layout.
        rank_flat_shard_size: Flat elements each DP rank owns (uniform for the
            even all-`Flat` layout: `layout.size // world_size`).
        world_size: DP group size.

    Returns:
        The `ShardPlan` describing which rows each rank owns.
    """
    if len(full_shape) != 2:
        raise ValueError(f"compute_shard_plan requires a 2D shape, got {full_shape}.")
    row_size = non_leading_numel(full_shape)
    if row_size <= 0:
        raise ValueError(f"compute_shard_plan requires non-empty rows, got shape {full_shape}.")
    tensor_end = tensor_flat_offset + full_shape.numel()

    rank_rows: list[tuple[int, int]] = []
    for rank in range(world_size):
        rank_start = rank * rank_flat_shard_size
        rank_end = rank_start + rank_flat_shard_size
        overlap_start = max(tensor_flat_offset, rank_start)
        overlap_end = min(tensor_end, rank_end)
        if overlap_start >= overlap_end:
            rank_rows.append((0, 0))
            continue
        if (overlap_start - tensor_flat_offset) % row_size != 0:
            raise RuntimeError(
                f"Flat shard boundary is not row-aligned for shape {full_shape}: "
                f"overlap_start={overlap_start}, tensor_flat_offset={tensor_flat_offset}."
            )
        overlap_numel = overlap_end - overlap_start
        if overlap_numel % row_size != 0:
            raise RuntimeError(
                f"Flat shard overlap is not row-aligned for shape {full_shape}: "
                f"overlap_numel={overlap_numel}, row_size={row_size}."
            )
        row_start = (overlap_start - tensor_flat_offset) // row_size
        row_count = overlap_numel // row_size
        rank_rows.append((row_start, row_count))
    return ShardPlan(
        full_shape=torch.Size(full_shape), rank_rows=tuple(rank_rows), row_size=row_size
    )


def assign_owner_work(plans: Sequence[ShardPlan], num_ns_steps: int) -> dict[int, int]:
    """Assign one owner rank to each boundary parameter, balanced by NS cost.

    Only ranks that own a non-empty shard of a parameter are eligible owners.
    Assignment greedily gives each parameter to its eligible rank with the
    smallest running cost total, where a parameter's cost is the Newton-Schulz
    compute estimate `numel * (min(rows, cols) * num_steps + 1)`.

    Args:
        plans: Shard plans indexed by their position in the input sequence.
        num_ns_steps: Newton-Schulz iteration count used in the cost estimate.

    Returns:
        Mapping from parameter index (in `plans`) to owner rank.
    """
    assignments: dict[int, int] = {}
    running: dict[int, float] = {r: 0.0 for r in range(plans[0].world_size)} if plans else {}
    for param_index, plan in enumerate(plans):
        candidates = plan.owner_candidates()
        if not candidates:
            raise RuntimeError(
                f"No eligible owner for parameter {param_index} with shape {plan.full_shape}; "
                "no rank owns a shard."
            )
        if not plan.is_boundary():
            # Fully local: the single owning rank is the owner with no communication.
            assignments[param_index] = candidates[0]
            continue
        rows, cols = plan.full_shape
        short_dim = min(rows, cols)
        cost = float(plan.full_numel() * (short_dim * num_ns_steps + 1))
        owner = min(candidates, key=lambda r: (running[r], r))
        assignments[param_index] = owner
        running[owner] += cost
    return assignments


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of one chunk.

    The owner keeps its own shard locally (no self-send), so it only receives
    from the other shard-holding ranks and reconstructs each owned matrix by
    concatenating shards in rank order (own shard at the owner's rank rows).

    Attributes:
        send_buffers: Per-destination-owner flat send buffer (this rank's
            pre-NS shards for that owner's params, in param order). Only owners
            other than this rank appear.
        recv_sizes: Per-source-rank element count this rank (as an owner)
            receives. Only sources other than this rank appear.
        own_shards: This rank's local shard per owned parameter (used directly
            in reconstruction, not communicated).
        recv_offsets: Per `(param_index, src_rank)` of `(offset, numel,
            row_count)` describing where this param's shard lands inside the
            recv buffer received from `src_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]


def pack_owner_work(
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    local_shards: Sequence[torch.Tensor],
    world_size: int,
    this_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OwnerGatherPlan:
    """Pack this rank's pre-NS shards into per-owner P2P send buffers.

    Args:
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        local_shards: This rank's local pre-NS shard per parameter.
        world_size: DP group size.
        this_rank: This rank's DP index.
        device: Device for the send buffers (the pre-NS device).
        dtype: Dtype for the send buffers (the pre-NS dtype).

    Returns:
        The `OwnerGatherPlan` for this rank.
    """
    send_sizes: dict[int, int] = {o: 0 for o in range(world_size) if o != this_rank}
    recv_sizes: dict[int, int] = {s: 0 for s in range(world_size) if s != this_rank}
    for param_index, plan in enumerate(plans):
        owner = owners[param_index]
        if owner != this_rank:
            send_sizes[owner] += plan.shard_numel(this_rank)
        else:
            for src in range(world_size):
                if src == this_rank:
                    continue
                recv_sizes[src] += plan.shard_numel(src)

    send_buffers: dict[int, torch.Tensor] = {}
    for owner, size in send_sizes.items():
        send_buffers[owner] = torch.empty(size, dtype=dtype, device=device)

    # Fill each owner's send buffer in param order.
    cursors: dict[int, int] = {o: 0 for o in send_buffers}
    own_shards: dict[int, torch.Tensor] = {}
    for param_index, (plan, shard) in enumerate(zip(plans, local_shards)):
        owner = owners[param_index]
        if owner == this_rank:
            own_shards[param_index] = shard
            continue
        numel = plan.shard_numel(this_rank)
        if numel == 0:
            continue
        buf = send_buffers[owner]
        buf[cursors[owner] : cursors[owner] + numel].copy_(shard.reshape(-1))
        cursors[owner] += numel

    # Per (owned param, src) recv offset within the recv buffer from src.
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    owned_indices = [i for i in range(len(plans)) if owners[i] == this_rank]
    for src in range(world_size):
        if src == this_rank:
            continue
        offset = 0
        for param_index in owned_indices:
            plan = plans[param_index]
            numel = plan.shard_numel(src)
            recv_offsets[(param_index, src)] = (offset, numel, plan.rank_row_count(src))
            offset += numel

    return OwnerGatherPlan(
        send_buffers=send_buffers,
        recv_sizes=recv_sizes,
        own_shards=own_shards,
        recv_offsets=recv_offsets,
    )


def reconstruct_full_tensor(
    param_index: int,
    plan: ShardPlan,
    gather_plan: OwnerGatherPlan,
    recv_buffers: dict[int, torch.Tensor],
    owner_rank: int,
) -> torch.Tensor:
    """Reconstruct the full 2D tensor for one owned parameter from its per-rank shards.

    Concatenates shards in rank order: the owner's own local shard at its rank rows and each
    source's received shard at that source's rank rows. The shard content lives in `gather_plan`
    (`own_shards` + `recv_offsets`), so this works for the pre-NS owner-gather path and for a weight
    gather plan alike – pass a weight gather plan built with `pack_owner_work` to reconstruct the
    full weight parameter instead.

    Args:
        param_index: Index of the parameter within the chunk.
        plan: Shard plan for this parameter.
        gather_plan: This rank's owner-gather plan (own_shards, recv_offsets).
        recv_buffers: Per-source-rank received buffer (only sources that sent).
        owner_rank: The owner rank (== the rank running reconstruction).

    Returns:
        The full `(rows, cols)` tensor.
    """
    world_size = plan.world_size
    shards: list[torch.Tensor] = []
    for src in range(world_size):
        row_count = plan.rank_row_count(src)
        if src == owner_rank:
            shards.append(gather_plan.own_shards[param_index])
        elif row_count == 0:
            continue
        else:
            offset, numel, _ = gather_plan.recv_offsets[(param_index, src)]
            buf = recv_buffers[src]
            shards.append(buf[offset : offset + numel].view(row_count, plan.row_size))
    if len(shards) == 1:
        return shards[0].contiguous()
    return torch.cat(shards, dim=0)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of one chunk.

    The owner keeps its own update shard (applied directly), so it only sends to
    the other shard-holding ranks.

    Attributes:
        send_buffers: Per-destination-rank flat send buffer (this owner's update
            shards for the params it owns, in param order). Only destinations
            other than this rank appear.
        recv_sizes: Per-owner-rank element count this rank (as a destination)
            receives. Only owners other than this rank appear.
        recv_offsets: Per `(param_index, owner_rank)` of `(offset, numel,
            row_count)` describing where this param's update shard lands inside
            the recv buffer received from `owner_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]


def pack_update_shards(
    full_updates: dict[int, torch.Tensor],
    plans: Sequence[ShardPlan],
    owners: dict[int, int],
    world_size: int,
    this_rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OwnerScatterPlan:
    """Pack this owner rank's full updates into per-destination P2P send buffers.

    Args:
        full_updates: Full update matrix per owned parameter index.
        plans: Shard plans in parameter order.
        owners: Mapping from parameter index to owner rank.
        world_size: DP group size.
        this_rank: This rank's DP index.
        device: Device for the send buffers (the update device).
        dtype: Dtype for the send buffers (the update dtype).

    Returns:
        The `OwnerScatterPlan` for this rank.
    """
    owned_indices = [i for i in range(len(plans)) if owners[i] == this_rank]
    send_sizes: dict[int, int] = {d: 0 for d in range(world_size) if d != this_rank}
    recv_sizes: dict[int, int] = {o: 0 for o in range(world_size) if o != this_rank}
    for param_index in owned_indices:
        plan = plans[param_index]
        for dest in range(world_size):
            if dest == this_rank:
                continue
            send_sizes[dest] += plan.shard_numel(dest)
    for param_index, plan in enumerate(plans):
        owner = owners[param_index]
        if owner == this_rank:
            continue
        recv_sizes[owner] += plan.shard_numel(this_rank)

    send_buffers: dict[int, torch.Tensor] = {}
    for dest, size in send_sizes.items():
        send_buffers[dest] = torch.empty(size, dtype=dtype, device=device)

    cursors: dict[int, int] = {d: 0 for d in send_buffers}
    for dest in range(world_size):
        if dest == this_rank:
            continue
        for param_index in owned_indices:
            plan = plans[param_index]
            row_start, row_count = plan.rank_rows[dest]
            numel = row_count * plan.row_size
            if numel == 0:
                continue
            full = full_updates[param_index]
            buf = send_buffers[dest]
            buf[cursors[dest] : cursors[dest] + numel].copy_(
                full[row_start : row_start + row_count].reshape(-1)
            )
            cursors[dest] += numel

    recv_offsets: dict[tuple[int, int], tuple[int, int, int]] = {}
    for owner in range(world_size):
        if owner == this_rank:
            continue
        offset = 0
        for param_index, plan in enumerate(plans):
            if owners[param_index] != owner:
                continue
            numel = plan.shard_numel(this_rank)
            recv_offsets[(param_index, owner)] = (offset, numel, plan.rank_row_count(this_rank))
            offset += numel

    return OwnerScatterPlan(
        send_buffers=send_buffers, recv_sizes=recv_sizes, recv_offsets=recv_offsets
    )


def unpack_update_shards(
    scatter_plan: OwnerScatterPlan, recv_buffers: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Extract this rank's local update shards from the per-owner recv buffers.

    Args:
        scatter_plan: This rank's owner-scatter plan (recv_offsets).
        recv_buffers: Per-owner-rank received buffer (only owners that sent).

    Returns:
        Mapping from parameter index to the local update shard `(row_count, cols)`,
        for parameters this rank does NOT own.
    """
    updates: dict[int, torch.Tensor] = {}
    for (param_index, owner), (offset, numel, row_count) in scatter_plan.recv_offsets.items():
        if numel == 0:
            continue
        buf = recv_buffers[owner]
        row_size = numel // row_count if row_count else 1
        updates[param_index] = buf[offset : offset + numel].view(row_count, row_size)
    return updates
