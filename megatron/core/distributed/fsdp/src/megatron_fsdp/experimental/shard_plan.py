# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure shard-planning and owner-compute packing logic for MFSDP v2's all-`Flat` layout.

The central data structure is `ShardPlan`, which describes how a single 2D parameter's full matrix
is split across the DP group under MFSDP v2's all-`Flat` layout. Given shard plans,
`assign_owner_work` balances owner-compute work across owner ranks using a caller-supplied cost
function. `ShardPlan.from_layout_params` builds a plan from DBuffer layout metadata,
`OwnerGatherPlan.pack`/`OwnerScatterPlan.pack` build the flat P2P send/recv buffers,
`OwnerGatherPlan.reconstruct_full` stitches gathered shards back into the full matrix on the owner,
and `OwnerScatterPlan.unpack` extracts received result shards.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence

import torch

from .layout import GlobalLayout, non_leading_numel


@dataclasses.dataclass(frozen=True)
class ShardPlan:
    """How a single 2D parameter's full matrix is split across the DP group.

    MFSDP v2's all-`Flat` layout shards dim-0 rows contiguously in rank order, so rank `r` owns the
    contiguous global row range `[start, start + count)` where `start`/`count` come from the flat
    DBuffer layout. A rank with `count == 0` holds no shard of this parameter.

    Attributes:
        full_shape: Global `(rows, cols)` shape of the parameter.
        rank_rows: Per-rank `(row_start, row_count)` tuples; `row_count == 0` means the rank holds
            no shard.
        row_size: Number of elements per row (= `full_shape[1:].numel()`).
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

    @classmethod
    def from_layout_params(
        cls,
        full_shape: torch.Size,
        tensor_flat_offset: int,
        rank_flat_shard_size: int,
        world_size: int,
    ) -> ShardPlan:
        """Compute the per-rank row ranges for one 2D parameter in a flat DBuffer.

        Args:
            full_shape: Global `(rows, cols)` shape of the parameter.
            tensor_flat_offset: Flat-element offset of this parameter inside the DBuffer's global
                layout.
            rank_flat_shard_size: Flat elements each DP rank owns (uniform for the even all-`Flat`
                layout: `layout.size // world_size`).
            world_size: DP group size.
        """
        if len(full_shape) != 2:
            raise ValueError(f"ShardPlan.from_layout_params requires a 2D shape, got {full_shape}.")
        row_size = non_leading_numel(full_shape)
        if row_size <= 0:
            raise ValueError(
                f"ShardPlan.from_layout_params requires non-empty rows, got shape {full_shape}."
            )
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
        return cls(full_shape=torch.Size(full_shape), rank_rows=tuple(rank_rows), row_size=row_size)

    @classmethod
    def from_layout(cls, layout: GlobalLayout, tensor_index: int, world_size: int) -> ShardPlan:
        """Build a shard plan for one parameter from a `GlobalLayout`.

        Extracts the parameter's shape, flat offset, and per-rank shard size from `layout` and
        delegates to `from_layout_params`.

        Args:
            layout: The DBuffer global layout that contains the parameter.
            tensor_index: Index of the parameter within `layout`.
            world_size: DP group size.
        """
        return cls.from_layout_params(
            layout.tensor_shapes[tensor_index],
            layout.tensor_to_offset[tensor_index],
            layout.size // world_size,
            world_size,
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

    @functools.lru_cache(maxsize=None)
    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, (_, count) in enumerate(self.rank_rows) if count > 0)

    @functools.lru_cache(maxsize=None)
    def is_boundary(self) -> bool:
        """True if more than one rank owns a non-empty shard of this parameter."""
        return sum(1 for _, count in self.rank_rows if count > 0) > 1

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()


def assign_owner_work(
    plans: Sequence[ShardPlan], cost_fn: Callable[[ShardPlan], float]
) -> dict[int, int]:
    """Assign one owner rank to each boundary parameter, balanced by cost.

    Only ranks that own a non-empty shard of a parameter are eligible owners. Assignment greedily
    gives each parameter to its eligible rank with the smallest running cost total.

    Args:
        plans: Shard plans indexed by their position in the input sequence.
        cost_fn: Callable that returns a positive cost estimate for a given shard plan. The greedy
            balancer minimizes the maximum running cost total across ranks, so the cost should
            reflect the relative compute weight of owning each parameter (e.g., an orthogonalization
            cost estimate).

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
        cost = cost_fn(plan)
        owner = min(candidates, key=lambda r: (running[r], r))
        assignments[param_index] = owner
        running[owner] += cost
    return assignments


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of a set of parameters.

    The owner keeps its own shard locally (no self-send), so it only receives from the other
    shard-holding ranks and reconstructs each owned matrix by concatenating shards in rank order
    (own shard at the owner's rank rows).

    Example:

    ```
    # We are also using some pseudocode here for brevity.

    # Assume:
    torch.distributed.get_world_size() == 2
    torch.distributed.get_rank() == 0  # We're observing from rank 0
    param_0: torch.Tensor
    param_1: torch.Tensor
    # Params are in this order as observed by MFSDP.
    model.param_groups == [{"params": [param_0, param_1]}]
    # Both params are owned by rank 1 (was previously determined using `ShardPlan`s).
    param_0.owner == 1
    param_1.owner == 1

    param_0.shape == (6, 4)  # Global shape.
    param_1.shape == (4, 4)  # Global shape.
    param_0.local_shard.shape == (3, 4)  # Rank 0 has shard indexed by `[0:3, ...]`.
    param_1.local_shard.shape == (2, 4)  # Rank 0 has shard indexed by `[0:2, ...]`.
    param_0.local_shard.numel == 12
    param_1.local_shard.numel == 8

    owner_gather_plan.send_buffers == {1: tensor(20)}  # 12 + 8 = 20 elements
    # `owner_gather_plan.send_buffers[1]` represents the following in its packed flat buffer:
    #   +--------------------+-------------------+
    #   | param_0 (12 elems) | param_1 (8 elems) |
    #   +--------------------+-------------------+
    #                 byte order: -->

    # Rank 0 owns nothing
    owner_gather_plan.recv_sizes == {}
    owner_gather_plan.own_shards == {}
    owner_gather_plan.recv_offsets == {}

    # ---

    # Same settings as above, now observing from rank 1 (the owner):
    torch.distributed.get_rank() == 1

    param_0.local_shard.shape == (3, 4)  # Rank 1 has shard indexed by `[3:6, ...]`.
    param_1.local_shard.shape == (2, 4)  # Rank 1 has shard indexed by `[2:4, ...]`.

    send_buffers = {}  # Rank 1 owns everything.
    recv_sizes = {0: 20}  # 12 + 8 = 20 elements from rank 0
    own_shards = {0: param_0.local_shard, 1: param_1.local_shard}  # rank 1's own shards
    recv_offsets = {
        (0, 0): (0, 12, 3),  # `param_0` from rank 0: offset 0, 12 elems, 3 rows
        (1, 0): (12, 8, 2),  # `param_1` from rank 0: offset 12, 8 elems, 2 rows
    }
    ```

    Attributes:
        send_buffers: Per-destination-owner flat send buffer (this rank's shards for that owner's
            params, in param order). Only owners other than this rank appear.
        recv_sizes: Per-source-rank element count this rank (as an owner) receives. Only sources
            other than this rank appear.
        own_shards: This rank's local shard per owned parameter (used directly in reconstruction,
            not communicated).
        recv_offsets: Per `(param_index, src_rank)` of `(offset, numel, row_count)` describing where
            this param's shard lands inside the recv buffer received from `src_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]

    @classmethod
    def pack(
        cls,
        plans: Sequence[ShardPlan],
        owners: dict[int, int],
        local_shards: Sequence[torch.Tensor],
        world_size: int,
        this_rank: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OwnerGatherPlan:
        """Pack this rank's local shards into per-owner P2P send buffers.

        Args:
            plans: Shard plans in parameter order.
            owners: Mapping from parameter index to owner rank.
            local_shards: This rank's local shard per parameter.
            world_size: DP group size.
            this_rank: This rank's DP index.
            device: Device for the send buffers.
            dtype: Dtype for the send buffers.
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

        return cls(
            send_buffers=send_buffers,
            recv_sizes=recv_sizes,
            own_shards=own_shards,
            recv_offsets=recv_offsets,
        )

    def reconstruct_full(
        self,
        param_index: int,
        plan: ShardPlan,
        recv_buffers: dict[int, torch.Tensor],
        owner_rank: int,
    ) -> torch.Tensor:
        """Reconstruct the full 2D tensor for one owned parameter from its per-rank shards.

        Concatenates shards in rank order: the owner's own local shard at its rank rows and each
        source's received shard at that source's rank rows.

        Args:
            param_index: Index of the parameter within the sequence of plans passed to `pack`.
            plan: Shard plan for this parameter.
            recv_buffers: Per-source-rank received buffer (only sources that sent).
            owner_rank: The owner rank (== the rank running reconstruction).
        """
        shards: list[torch.Tensor] = []
        for src in range(plan.world_size):
            row_count = plan.rank_row_count(src)
            if src == owner_rank:
                shards.append(self.own_shards[param_index])
            elif row_count == 0:
                continue
            else:
                offset, numel, _ = self.recv_offsets[(param_index, src)]
                buf = recv_buffers[src]
                shards.append(buf[offset : offset + numel].view(row_count, plan.row_size))
        if len(shards) == 1:
            return shards[0].contiguous()
        return torch.cat(shards, dim=0)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of a set of parameters.

    The owner keeps its own result shard (applied directly), so it only sends to the other
    shard-holding ranks.

    Attributes:
        send_buffers: Per-destination-rank flat send buffer (this owner's result shards for the
            params it owns, in param order). Only destinations other than this rank appear.
        recv_sizes: Per-owner-rank element count this rank (as a destination) receives. Only owners
            other than this rank appear.
        recv_offsets: Per `(param_index, owner_rank)` of `(offset, numel, row_count)` describing
            where this param's result shard lands inside the recv buffer received from `owner_rank`.
    """

    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    recv_offsets: dict[tuple[int, int], tuple[int, int, int]]

    @classmethod
    def pack(
        cls,
        full_results: dict[int, torch.Tensor],
        plans: Sequence[ShardPlan],
        owners: dict[int, int],
        world_size: int,
        this_rank: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OwnerScatterPlan:
        """Pack this owner rank's full results into per-destination P2P send buffers.

        Args:
            full_results: Full result tensor per owned parameter index.
            plans: Shard plans in parameter order.
            owners: Mapping from parameter index to owner rank.
            world_size: DP group size.
            this_rank: This rank's DP index.
            device: Device for the send buffers.
            dtype: Dtype for the send buffers.
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
                full = full_results[param_index]
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

        return cls(send_buffers=send_buffers, recv_sizes=recv_sizes, recv_offsets=recv_offsets)

    def unpack(self, recv_buffers: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Extract this rank's local result shards from the per-owner recv buffers.

        Args:
            recv_buffers: Per-owner-rank received buffer (only owners that sent).

        Returns:
            Mapping from parameter index to the local result shard `(row_count, cols)`, for
            parameters this rank does NOT own.
        """
        results: dict[int, torch.Tensor] = {}
        for (param_index, owner), (offset, numel, row_count) in self.recv_offsets.items():
            if numel == 0:
                continue
            buf = recv_buffers[owner]
            row_size = numel // row_count if row_count else 1
            results[param_index] = buf[offset : offset + numel].view(row_count, row_size)
        return results
