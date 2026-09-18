# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pure parameter layout and owner-compute packing logic for MFSDP v2's all-`Flat` layout.

- `ParameterLayout` describes how a single parameter's flat element range is split across the DP
  group under MFSDP v2's all-`Flat` layout.
- `ParameterLayout.from_group` builds `{tensor_index: layout}` for eligible parameters in an
  `FsdpParameterGroup`, keyed by each parameter's index within the group.
- `assign_owner_work` balances owner-compute work across owner ranks using a caller-supplied cost
  function.
- `GroupOwnerLayout.from_group` builds a data structure capturing the per-group owner layout upon
  the above.
- `OwnerGatherPlan.pack`/`OwnerScatterPlan.pack` take the `GroupOwnerLayout` plus this rank's data
  and build the flat P2P send/recv buffers,
- `OwnerGatherPlan.reconstruct_full` stitches gathered shards back into the full flat tensor on the
  owner, and
- `OwnerScatterPlan.unpack` extracts this rank's flat result shards from the received buffers.
"""

import dataclasses
from collections.abc import Callable
from typing import Self

import torch
from torch.distributed.device_mesh import DeviceMesh

from .layout import non_leading_numel
from .parameter_group import FsdpParameterGroup


def select_ge_2d_params(param: torch.Tensor) -> bool:
    """Whether the given tensor has dimensionality ≥2."""
    return param.ndim >= 2


@dataclasses.dataclass(frozen=True)
class ParameterLayout:
    """How a single parameter's flat element range splits across the DP group.

    MFSDP v2's all-`Flat` layout gives each rank one contiguous global element range per parameter,
    in rank order, so rank `r` holds `[offset, offset + count)` where `offset` is the sum of the
    previous ranks' counts. A rank with `count == 0` holds no elements of this parameter.

    Attributes:
        full_shape: The parameter's global shape.
        flat_counts: Per-rank element count; `0` means the rank holds no elements.
    """

    full_shape: torch.Size
    flat_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.flat_counts) == 0:
            raise ValueError("ParameterLayout requires at least one rank.")
        if sum(self.flat_counts) != self.full_shape.numel():
            raise ValueError(
                f"ParameterLayout flat_counts sum {sum(self.flat_counts)} != full_shape "
                f"numel {self.full_shape.numel()}."
            )

    @classmethod
    def from_group(
        cls, group: FsdpParameterGroup, *, eligible_fn: Callable[[torch.Tensor], bool] | None = None
    ) -> dict[int, Self]:
        """Build `{tensor_index: layout}` for eligible parameters in an `FsdpParameterGroup`.

        Keys are the parameters' indices within `group.fsdp_parameters` (their tensor indices in the
        DBuffer layout). Parameters not selected by `eligible_fn` are absent from the returned dict.

        Args:
            group: The FSDP parameter group whose DBuffer layout describes the parameter placements.
            eligible_fn: Predicate selecting which parameters participate in owner-compute
                orthogonalization. Takes a parameter tensor as input and return whether the
                parameter is supposed to be included. When `None`, defaults to matching ≥2D tensors
                (`param.ndim >= 2`).
        """
        if eligible_fn is None:
            eligible_fn = select_ge_2d_params

        mesh = group.mesh
        layout = group.main_weight.layout
        dp_size = mesh.size()
        rank_flat_shard_size = layout.size // dp_size

        result: dict[int, Self] = {}
        for tensor_index, fsdp_parameter in enumerate(group.fsdp_parameters):
            param = fsdp_parameter.sharded
            if not eligible_fn(param):
                continue

            full_shape = layout.tensor_shapes[tensor_index]
            tensor_flat_offset = layout.tensor_to_offset[tensor_index]
            tensor_end = tensor_flat_offset + full_shape.numel()

            flat_counts: list[int] = []
            for rank in range(dp_size):
                rank_start = rank * rank_flat_shard_size
                rank_end = rank_start + rank_flat_shard_size
                overlap_start = max(tensor_flat_offset, rank_start)
                overlap_end = min(tensor_end, rank_end)
                flat_counts.append(max(0, overlap_end - overlap_start))

            param_layout = cls(full_shape=torch.Size(full_shape), flat_counts=tuple(flat_counts))
            result[tensor_index] = param_layout
        return result

    @property
    def dp_size(self) -> int:
        """Number of ranks in the DP group for this parameter."""
        return len(self.flat_counts)

    def full_numel(self) -> int:
        """Return the total number of elements in the full (unsharded) parameter."""
        return self.full_shape.numel()

    def rank_offset(self, rank: int) -> int:
        """Return the starting offset of `rank`'s shard within the flat parameter."""
        return sum(self.flat_counts[:rank])

    def rank_numel(self, rank: int) -> int:
        """Return the number of elements `rank` holds."""
        return self.flat_counts[rank]

    def owner_candidates(self) -> tuple[int, ...]:
        """Return the ranks that hold a non-empty shard of this parameter."""
        return tuple(r for r, count in enumerate(self.flat_counts) if count > 0)

    def is_boundary(self) -> bool:
        """True if more than one rank holds a non-empty shard of this parameter."""
        return len(self.owner_candidates()) > 1


def ns_cost_fn(num_ns_steps: int) -> Callable[[ParameterLayout], int]:
    """Cost function matching the Newton-Schulz orthogonalization compute estimate for the given
    number of Newton-Schulz iterations/steps.

    `numel * (min(rows, cols) * num_steps + 1)` under the DBuffer's leading-dim view (`(shape[0],
    shape[1:].numel())`).
    """

    def cost_fn(layout: ParameterLayout) -> int:
        shape = layout.full_shape
        short_dim = min(shape[0], non_leading_numel(shape))
        return layout.full_numel() * (short_dim * num_ns_steps + 1)

    return cost_fn


def assign_owner_work(
    layouts: dict[int, ParameterLayout], cost_fn: Callable[[ParameterLayout], float] | None = None
) -> dict[int, int]:
    """Assign one owner rank to each parameter, keyed by tensor index.

    Non-boundary parameters are assigned to their original rank (the only rank holding their
    elements, so no communication is needed) and their cost counts toward that rank's running total
    cost. Boundary parameters are processed in descending cost order and each is greedily given to
    its eligible rank with the smallest running cost total.

    Args:
        layouts: Parameter layouts keyed by each parameter's tensor index in its
            `FsdpParameterGroup`.
        cost_fn: Callable that returns a positive cost estimate for a given parameter layout. The
            greedy balancer minimizes the maximum running cost total across ranks, so the cost
            should reflect the relative compute weight of owning each parameter (e.g., an
            orthogonalization cost estimate). When `None`, defaults to a compute estimate for
            orthogonalization via Newton-Schulz with 5 iterations/steps.

    Returns:
        Mapping from tensor index to owner rank.
    """
    if cost_fn is None:
        cost_fn = ns_cost_fn(num_ns_steps=5)

    assignments: dict[int, int] = {}
    if not layouts:
        return assignments
    dp_size = next(iter(layouts.values())).dp_size
    running: dict[int, float] = {r: 0.0 for r in range(dp_size)}
    # Non-boundary parameters are assigned to their sole holder; account for their cost.
    for tensor_index, layout in layouts.items():
        if layout.is_boundary():
            continue
        (holder,) = layout.owner_candidates()
        assignments[tensor_index] = holder
        running[holder] += cost_fn(layout)
    # Sort boundary params by descending cost (longest processing time first).
    boundary_costs = sorted(
        (
            (tensor_index, cost_fn(layout))
            for tensor_index, layout in layouts.items()
            if layout.is_boundary()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    for tensor_index, cost in boundary_costs:
        layout = layouts[tensor_index]
        candidates = layout.owner_candidates()
        if not candidates:
            raise RuntimeError(
                f"No eligible owner for tensor {tensor_index} with shape {layout.full_shape}; "
                "no rank owns a shard."
            )
        owner = min(candidates, key=lambda r: running[r])
        assignments[tensor_index] = owner
        running[owner] += cost
    return assignments


@dataclasses.dataclass(frozen=True)
class GroupOwnerLayout:
    """Owner layout for one `FsdpParameterGroup`: its params and their owner ranks.

    The layouts and assignments are step-independent in the general case, so the owner layout may be
    cached across optimizer steps.

    The `tensor_index` used here refers to the parameter's/tensor's index in the
    `FsdpParameterGroup`.

    Attributes:
        group: The FSDP parameter group the layouts and owners refer to.
        layouts: `{tensor_index: layout}` for the participating parameters.
        owners: `{tensor_index: owner_rank}` with an entry for every participating parameter. Ranks
            are indices into the group's mesh.
    """

    group: FsdpParameterGroup
    layouts: dict[int, ParameterLayout]
    owners: dict[int, int]

    @classmethod
    def from_group(
        cls,
        group: FsdpParameterGroup,
        *,
        cost_fn: Callable[[ParameterLayout], float] | None = None,
        eligible_fn: Callable[[torch.Tensor], bool] | None = None,
    ) -> Self:
        """Build the owner layout for one group.

        Args:
            group: The FSDP parameter group whose DBuffer layout describes the parameter placements.
            cost_fn: Cost estimate per parameter layout used to balance owner assignments across
                ranks. When `None`, defaults to a compute estimate for orthogonalization via
                Newton-Schulz with 5 iterations/steps. See also `assign_owner_work`.
            eligible_fn: Predicate selecting which parameters participate in owner-compute
                orthogonalization. When `None`, defaults to matching ≥2D tensors. See also
                `ParameterLayout.from_group`.
        """
        layouts = ParameterLayout.from_group(group, eligible_fn=eligible_fn)
        owners = assign_owner_work(layouts, cost_fn)
        return cls(group=group, layouts=layouts, owners=owners)

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the group."""
        return self.group.mesh


@dataclasses.dataclass
class OwnerGatherPlan:
    """Metadata and send buffers for the owner-gather P2P step of a set of parameters.

    The owner keeps its own shard locally (no self-send), so it only receives from the other
    shard-holding ranks. `reconstruct_full` reconstructs each owned tensor by concatenating the
    per-rank shards in rank order (i.e., global element order).


    Example:

    ```
    # We are also using some pseudocode here for brevity.

    # Assume:
    torch.distributed.get_world_size() == 2
    torch.distributed.get_rank() == 0  # We're observing from rank 0
    param_0: torch.Tensor
    param_1: torch.Tensor
    # Params are in this order as observed by MFSDP.
    model.param_groups == [{"params": [param_0, param_1]}]
    # Both params are owned by rank 1 (was previously determined using `GroupOwnerLayout`).
    param_0.owner == 1
    param_1.owner == 1

    param_0.shape == (6, 4)  # Global shape.
    param_1.shape == (4, 4)  # Global shape.
    param_0.local_shard.shape == (3, 4)  # Rank 0 has shard indexed by `[0:3, ...]`.
    param_1.local_shard.shape == (2, 4)  # Rank 0 has shard indexed by `[0:2, ...]`.
    param_0.local_shard.numel == 12
    param_1.local_shard.numel == 8

    owner_gather_plan.send_buffers == {1: tensor(20)}  # 12 + 8 = 20 elements
    # `owner_gather_plan.send_buffers[1]` represents the following in its packed flat buffer:
    #   +--------------------+-------------------+
    #   | param_0 (12 elems) | param_1 (8 elems) |
    #   +--------------------+-------------------+
    #                 byte order: -->

    # Rank 0 owns nothing
    owner_gather_plan.recv_sizes == {}
    owner_gather_plan.own_shards == {}
    owner_gather_plan.recv_offsets == {}

    # ---

    # Same settings as above, now observing from rank 1 (the owner):
    torch.distributed.get_rank() == 1

    param_0.local_shard.shape == (3, 4)  # Rank 1 has shard indexed by `[3:6, ...]`.
    param_1.local_shard.shape == (2, 4)  # Rank 1 has shard indexed by `[2:4, ...]`.

    send_buffers = {}  # Rank 1 owns everything.
    recv_sizes = {0: 20}  # 12 + 8 = 20 elements from rank 0
    # Rank 1's own shards, flattened (views):
    own_shards = {0: param_0.local_shard.view(-1), 1: param_1.local_shard.view(-1)}
    recv_offsets = {
        (0, 0): 0,  # `param_0` (tensor index 0) from rank 0: offset 0
        (1, 0): 12,  # `param_1` (tensor index 1) from rank 0: offset 12
    }
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

    layouts: dict[int, ParameterLayout]
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
        dp_size = mesh.size()
        this_rank = mesh.get_local_rank()
        layouts = plan.layouts
        owners = plan.owners
        send_sizes: dict[int, int] = {}
        recv_sizes: dict[int, int] = {}
        for tensor_index, layout in layouts.items():
            owner = owners[tensor_index]
            if owner != this_rank:
                send_numel = layout.rank_numel(this_rank)
                if send_numel > 0:
                    send_sizes[owner] = send_sizes.get(owner, 0) + send_numel
                continue
            for src in range(dp_size):
                if src == this_rank:
                    continue
                recv_numel = layout.rank_numel(src)
                if recv_numel > 0:
                    recv_sizes[src] = recv_sizes.get(src, 0) + recv_numel

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
        for tensor_index in layouts:
            owner = owners[tensor_index]
            if owner == this_rank:
                own_shards[tensor_index] = local_shards[tensor_index].flatten()
                continue
            numel = layouts[tensor_index].rank_numel(this_rank)
            if numel == 0:
                continue
            shard = local_shards[tensor_index]
            buf = send_buffers[owner]
            buf[cursors[owner] : cursors[owner] + numel].copy_(shard.flatten())
            cursors[owner] += numel

        # Per (owned param, src) recv offset within the recv buffer from src.
        recv_offsets: dict[tuple[int, int], int] = {}
        for src in range(dp_size):
            if src == this_rank:
                continue
            offset = 0
            for tensor_index, layout in layouts.items():
                if owners[tensor_index] != this_rank:
                    continue
                numel = layout.rank_numel(src)
                if numel == 0:
                    continue
                recv_offsets[(tensor_index, src)] = offset
                offset += numel

        return cls(
            layouts=dict(layouts),
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

        Concatenates the per-rank shards in rank order (i.e., global element order). Results can be
        `view`ed into the desired shape. For a parameter only this rank holds elements of, the own
        flat shard is returned directly.

        Args:
            param_index: Tensor index of the parameter (a key of the `layouts` dict passed to
                `pack`).
            recv_buffers: Per-source-rank received buffer (only sources that sent).
        """
        layout = self.layouts[param_index]
        shards: list[torch.Tensor] = []
        for src in range(layout.dp_size):
            if src == self.this_rank:
                shards.append(self.own_shards[param_index])
                continue

            numel = layout.rank_numel(src)
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

    layouts: dict[int, ParameterLayout]
    this_rank: int
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
        dp_size = mesh.size()
        this_rank = mesh.get_local_rank()
        layouts = plan.layouts
        owners = plan.owners
        send_sizes: dict[int, int] = {}
        recv_sizes: dict[int, int] = {}
        for tensor_index, layout in layouts.items():
            owner = owners[tensor_index]
            if owner == this_rank:
                for dest in range(dp_size):
                    if dest == this_rank:
                        continue
                    numel = layout.rank_numel(dest)
                    if numel > 0:
                        send_sizes[dest] = send_sizes.get(dest, 0) + numel
                continue
            numel = layout.rank_numel(this_rank)
            if numel > 0:
                recv_sizes[owner] = recv_sizes.get(owner, 0) + numel

        send_buffers: dict[int, torch.Tensor] = {}
        if send_sizes:
            first_result = next(iter(full_results.values()))
            for dest, size in send_sizes.items():
                send_buffers[dest] = torch.empty(
                    size, dtype=first_result.dtype, device=first_result.device
                )

        # Fill each destination's send buffer in tensor-index order.
        cursors: dict[int, int] = {dest: 0 for dest in send_buffers}
        for tensor_index, layout in layouts.items():
            if owners[tensor_index] != this_rank:
                continue
            flat = full_results[tensor_index].flatten()
            for dest in range(dp_size):
                if dest == this_rank:
                    continue
                numel = layout.rank_numel(dest)
                if numel == 0:
                    continue
                offset = layout.rank_offset(dest)
                buf = send_buffers[dest]
                buf[cursors[dest] : cursors[dest] + numel].copy_(flat[offset : offset + numel])
                cursors[dest] += numel

        recv_offsets: dict[tuple[int, int], int] = {}
        for owner in range(dp_size):
            if owner == this_rank:
                continue
            offset = 0
            for tensor_index, layout in layouts.items():
                if owners[tensor_index] != owner:
                    continue
                numel = layout.rank_numel(this_rank)
                if numel == 0:
                    continue
                recv_offsets[(tensor_index, owner)] = offset
                offset += numel

        return cls(
            layouts=dict(layouts),
            this_rank=this_rank,
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
            layout = self.layouts[tensor_index]
            numel = layout.rank_numel(self.this_rank)
            buf = recv_buffers[owner]
            results[tensor_index] = buf[offset : offset + numel]
        return results
