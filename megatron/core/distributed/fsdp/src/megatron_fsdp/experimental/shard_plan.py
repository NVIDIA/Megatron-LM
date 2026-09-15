# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure shard-planning and owner-compute packing logic for MFSDP v2's all-`Flat` layout.

The central data structure is `ParameterLayout`, which describes how a single 2D parameter's full
matrix is split across the DP group under MFSDP v2's all-`Flat` layout. Given parameter layouts,
`assign_owner_work` balances owner-compute work across owner ranks using a caller-supplied cost
function. `ParameterLayout.from_layout` builds a layout from DBuffer layout metadata,
`OwnerGatherPlan.pack`/`OwnerScatterPlan.pack` build the flat P2P send/recv buffers,
`OwnerGatherPlan.reconstruct_full` stitches gathered shards back into the full matrix on the owner,
and `OwnerScatterPlan.unpack` extracts received result shards.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Self

import torch

from .layout import GlobalLayout, non_leading_numel


@dataclasses.dataclass(frozen=True)
class ParameterLayout:
    """How a single 2D parameter's full matrix is split across the DP group.

    MFSDP v2's all-`Flat` layout shards dim-0 rows contiguously in rank order, so rank `r` owns the
    contiguous global row range `[start, start + count)` where `start`/`count` come from the flat
    DBuffer layout. A rank with `count == 0` holds no shard of this parameter.

    Attributes:
        full_shape: Global `(rows, cols)` shape of the parameter.
        row_counts: Per-rank row count; `0` means the rank holds no shard.
        row_size: Number of elements per row (= `full_shape[1:].numel()`).
    """

    full_shape: torch.Size
    row_counts: tuple[int, ...]
    row_size: int

    def __post_init__(self) -> None:
        if len(self.full_shape) != 2:
            raise ValueError(f"ParameterLayout requires a 2D full_shape, got {self.full_shape}.")
        if len(self.row_counts) == 0:
            raise ValueError("ParameterLayout requires at least one rank.")
        if self.row_size != non_leading_numel(self.full_shape):
            raise ValueError(
                f"ParameterLayout row_size {self.row_size} != full_shape row size "
                f"{non_leading_numel(self.full_shape)}."
            )

    @classmethod
    def from_layout(cls, layout: GlobalLayout, tensor_index: int, dp_size: int) -> Self:
        """Build a parameter layout for one parameter from a `GlobalLayout`.

        Computes the per-rank row ranges for the given 2D parameter in a flat DBuffer layout.

        Args:
            layout: The DBuffer global layout that contains the parameter.
            tensor_index: Index of the parameter within `layout`.
            dp_size: DP group size.
        """
        full_shape = layout.tensor_shapes[tensor_index]
        tensor_flat_offset = layout.tensor_to_offset[tensor_index]
        rank_flat_shard_size = layout.size // dp_size

        if len(full_shape) != 2:
            raise ValueError(f"ParameterLayout.from_layout requires a 2D shape, got {full_shape}.")
        row_size = non_leading_numel(full_shape)
        if row_size <= 0:
            raise ValueError(
                f"ParameterLayout.from_layout requires non-empty rows, got shape {full_shape}."
            )
        tensor_end = tensor_flat_offset + full_shape.numel()

        row_counts: list[int] = []
        for rank in range(dp_size):
            rank_start = rank * rank_flat_shard_size
            rank_end = rank_start + rank_flat_shard_size
            overlap_start = max(tensor_flat_offset, rank_start)
            overlap_end = min(tensor_end, rank_end)
            if overlap_start >= overlap_end:
                row_counts.append(0)
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
            row_count = overlap_numel // row_size
            row_counts.append(row_count)
        return cls(
            full_shape=torch.Size(full_shape), row_counts=tuple(row_counts), row_size=row_size
        )

    @property
    def dp_size(self) -> int:
        """Number of ranks in the DP group for this parameter."""
        return len(self.row_counts)

    def rank_row_start(self, rank: int) -> int:
        """Return the starting row of `rank`'s shard within the full tensor."""
        return sum(self.row_counts[:rank])

    def rank_row_count(self, rank: int) -> int:
        """Return the number of rows owned by `rank`."""
        return self.row_counts[rank]

    def shard_numel(self, rank: int) -> int:
        """Return the number of elements in `rank`'s shard."""
        return self.rank_row_count(rank) * self.row_size

    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, count in enumerate(self.row_counts) if count > 0)

    def is_boundary(self) -> bool:
        """True if more than one rank owns a non-empty shard of this parameter."""
        return any(0 < count < self.full_shape[0] for count in self.row_counts)

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()


def assign_owner_work(
    layouts: Sequence[ParameterLayout], cost_fn: Callable[[ParameterLayout], float]
) -> dict[int, int]:
    """Assign one owner rank to each boundary parameter, balanced by cost.

    Non-boundary parameters stay on their original rank and are skipped. Only ranks that own a
    non-empty shard of a parameter are eligible owners. Boundary parameters are processed in
    descending cost order, and each is greedily given to its eligible rank with the smallest running
    cost total.

    Args:
        layouts: Parameter layouts indexed by their position in the input sequence.
        cost_fn: Callable that returns a positive cost estimate for a given parameter layout. The
            greedy balancer minimizes the maximum running cost total across ranks, so the cost
            should reflect the relative compute weight of owning each parameter (e.g., an
            orthogonalization cost estimate).

    Returns:
        Mapping from boundary parameter index (in `layouts`) to owner rank. Non-boundary parameters
        are absent, as they stay on their original rank.
    """
    assignments: dict[int, int] = {}
    if not layouts:
        return assignments
    running: dict[int, float] = {r: 0.0 for r in range(layouts[0].dp_size)}
    # Sort boundary params by descending cost (longest processing time first).
    boundary_costs = sorted(
        (
            (param_index, cost_fn(layout))
            for param_index, layout in enumerate(layouts)
            if layout.is_boundary()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    for param_index, cost in boundary_costs:
        layout = layouts[param_index]
        candidates = layout.owner_candidates()
        if not candidates:
            raise RuntimeError(
                f"No eligible owner for parameter {param_index} with shape {layout.full_shape}; "
                "no rank owns a shard."
            )
        owner = min(candidates, key=lambda r: running[r])
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
    # Both params are owned by rank 1 (was previously determined using `ParameterLayout`s).
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
            params, in param order). Only owners with non-zero send size appear.
        recv_sizes: Per-source-rank element count this rank (as an owner) receives. Only sources
            with non-zero total size appear.
        own_shards: This rank's local shard per owned parameter (used directly in reconstruction,
            not communicated).
        recv_offsets: Per `(param_index, src_rank)`, the flat offset of this param's shard inside
            the recv buffer received from `src_rank`.
    """

    layouts: tuple[ParameterLayout, ...]
    this_rank: int
    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    own_shards: dict[int, torch.Tensor]
    recv_offsets: dict[tuple[int, int], int]

    @classmethod
    def pack(
        cls,
        layouts: Sequence[ParameterLayout],
        owners: dict[int, int],
        local_shards: Sequence[torch.Tensor],
        dp_size: int,
        this_rank: int,
    ) -> Self:
        """Pack this rank's local shards into per-owner P2P send buffers.

        Args:
            layouts: Parameter layouts in parameter order.
            owners: Mapping from parameter index to owner rank.
            local_shards: This rank's local shard per parameter.
            dp_size: DP group size.
            this_rank: This rank's DP index.
        """
        device = local_shards[0].device
        dtype = local_shards[0].dtype
        send_sizes: dict[int, int] = {}
        recv_sizes: dict[int, int] = {}
        for param_index, layout in enumerate(layouts):
            owner = owners[param_index]
            if owner != this_rank:
                send_numel = layout.shard_numel(this_rank)
                if send_numel > 0:
                    send_sizes[owner] = send_sizes.get(owner, 0) + send_numel
                continue
            for src in range(dp_size):
                if src == this_rank:
                    continue
                recv_numel = layout.shard_numel(src)
                if recv_numel > 0:
                    recv_sizes[src] = recv_sizes.get(src, 0) + recv_numel

        send_buffers: dict[int, torch.Tensor] = {}
        for owner, size in send_sizes.items():
            send_buffers[owner] = torch.empty(size, dtype=dtype, device=device)

        # Fill each owner's send buffer in param order.
        cursors: dict[int, int] = {owner: 0 for owner in send_buffers}
        own_shards: dict[int, torch.Tensor] = {}
        for param_index, (layout, shard) in enumerate(zip(layouts, local_shards)):
            owner = owners[param_index]
            if owner == this_rank:
                own_shards[param_index] = shard
                continue
            numel = layout.shard_numel(this_rank)
            if numel == 0:
                continue
            buf = send_buffers[owner]
            buf[cursors[owner] : cursors[owner] + numel].copy_(shard.flatten())
            cursors[owner] += numel

        # Per (owned param, src) recv offset within the recv buffer from src.
        recv_offsets: dict[tuple[int, int], int] = {}
        owned_indices = [i for i in range(len(layouts)) if owners[i] == this_rank]
        for src in range(dp_size):
            if src == this_rank:
                continue
            offset = 0
            for param_index in owned_indices:
                recv_offsets[(param_index, src)] = offset
                offset += layouts[param_index].shard_numel(src)

        return cls(
            layouts=tuple(layouts),
            this_rank=this_rank,
            send_buffers=send_buffers,
            recv_sizes=recv_sizes,
            own_shards=own_shards,
            recv_offsets=recv_offsets,
        )

    def reconstruct_full(
        self, param_index: int, recv_buffers: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct the full 2D tensor for one owned parameter from its per-rank shards.

        Concatenates shards in rank order: this rank's own local shard at its rank rows and each
        source's received shard at that source's rank rows.

        Args:
            param_index: Index of the parameter within the sequence of layouts passed to `pack`.
            recv_buffers: Per-source-rank received buffer (only sources that sent).
        """
        layout = self.layouts[param_index]
        shards: list[torch.Tensor] = []
        for src in range(layout.dp_size):
            if src == self.this_rank:
                shards.append(self.own_shards[param_index])
                continue

            row_count = layout.rank_row_count(src)
            if row_count == 0:
                continue

            offset = self.recv_offsets[(param_index, src)]
            numel = layout.shard_numel(src)
            buf = recv_buffers[src]
            shards.append(buf[offset : offset + numel].view(row_count, layout.row_size))
        if len(shards) == 1:
            return shards[0]
        return torch.cat(shards, dim=0)


@dataclasses.dataclass
class OwnerScatterPlan:
    """Metadata and send buffers for the owner-scatter P2P step of a set of parameters.

    The owner keeps its own result shard (applied directly), so it only sends to the other
    shard-holding ranks.

    Attributes:
        send_buffers: Per-destination-rank flat send buffer (this owner's result shards for the
            params it owns, in param order). Only destinations with non-zero send size appear.
        recv_sizes: Per-owner-rank element count this rank (as a destination) receives. Only owners
            with non-zero total size appear.
        recv_offsets: Per `(param_index, owner_rank)`, the flat offset of this param's result shard
            inside the recv buffer received from `owner_rank`.
    """

    layouts: tuple[ParameterLayout, ...]
    this_rank: int
    send_buffers: dict[int, torch.Tensor]
    recv_sizes: dict[int, int]
    recv_offsets: dict[tuple[int, int], int]

    @classmethod
    def pack(
        cls,
        full_results: dict[int, torch.Tensor],
        layouts: Sequence[ParameterLayout],
        owners: dict[int, int],
        dp_size: int,
        this_rank: int,
    ) -> Self:
        """Pack this owner rank's full results into per-destination P2P send buffers.

        Args:
            full_results: Full result tensor per owned parameter index.
            layouts: Parameter layouts in parameter order.
            owners: Mapping from parameter index to owner rank.
            dp_size: DP group size.
            this_rank: This rank's DP index.
        """
        if full_results:
            first = next(iter(full_results.values()))
            device = first.device
            dtype = first.dtype
        owned_indices = [i for i in range(len(layouts)) if owners[i] == this_rank]
        send_sizes: dict[int, int] = {}
        recv_sizes: dict[int, int] = {}
        for param_index in owned_indices:
            layout = layouts[param_index]
            for dest in range(dp_size):
                if dest == this_rank:
                    continue
                numel = layout.shard_numel(dest)
                if numel > 0:
                    send_sizes[dest] = send_sizes.get(dest, 0) + numel
        for param_index, layout in enumerate(layouts):
            owner = owners[param_index]
            if owner == this_rank:
                continue
            numel = layout.shard_numel(this_rank)
            if numel > 0:
                recv_sizes[owner] = recv_sizes.get(owner, 0) + numel

        send_buffers: dict[int, torch.Tensor] = {}
        for dest, size in send_sizes.items():
            send_buffers[dest] = torch.empty(size, dtype=dtype, device=device)

        cursors: dict[int, int] = {receiver: 0 for receiver in send_buffers}
        for dest in range(dp_size):
            if dest == this_rank:
                continue
            for param_index in owned_indices:
                layout = layouts[param_index]
                row_count = layout.rank_row_count(dest)
                row_start = layout.rank_row_start(dest)
                numel = row_count * layout.row_size
                if numel == 0:
                    continue
                full = full_results[param_index]
                buf = send_buffers[dest]
                buf[cursors[dest] : cursors[dest] + numel].copy_(
                    full[row_start : row_start + row_count].flatten()
                )
                cursors[dest] += numel

        recv_offsets: dict[tuple[int, int], int] = {}
        for owner in range(dp_size):
            if owner == this_rank:
                continue
            offset = 0
            for param_index, layout in enumerate(layouts):
                if owners[param_index] != owner:
                    continue
                recv_offsets[(param_index, owner)] = offset
                offset += layout.shard_numel(this_rank)

        return cls(
            layouts=tuple(layouts),
            this_rank=this_rank,
            send_buffers=send_buffers,
            recv_sizes=recv_sizes,
            recv_offsets=recv_offsets,
        )

    def unpack(self, recv_buffers: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Extract this rank's local result shards from the per-owner recv buffers.

        Args:
            recv_buffers: Per-owner-rank received buffer (only owners that sent).

        Returns:
            Mapping from parameter index to the local result shard `(row_count, cols)`, for
            parameters this rank does NOT own.
        """
        results: dict[int, torch.Tensor] = {}
        for (param_index, owner), offset in self.recv_offsets.items():
            layout = self.layouts[param_index]
            numel = layout.shard_numel(self.this_rank)
            if numel == 0:
                continue
            row_count = layout.rank_row_count(self.this_rank)
            buf = recv_buffers[owner]
            results[param_index] = buf[offset : offset + numel].view(row_count, layout.row_size)
        return results
