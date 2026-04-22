# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class SliceInfo:
    """Batch dimension slice information for a rank's data partition."""

    start: int
    size: int


class BridgeDirection(str, Enum):
    """Which side of the bridge scales up, if any.

    ``FAN_IN`` — src has more DP replicas than dest; forward all-gathers
    src outputs along the batch dim, backward narrows the sibling dest
    gradient down to this src rank's slot.

    ``FAN_OUT`` — dest has more DP replicas; forward narrows, backward
    all-gathers across the sibling dest DP ranks (the adjoint of narrow
    is not zero-pad-and-scatter because every dest rank consumes a
    different slice of the same src activation).

    ``EQUAL`` — matching DP; the bridge is a pure passthrough.
    """

    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    EQUAL = "equal"


class ColocatedBridgeCommunicator:
    """Bridges tensors between colocated modules with different TP/DP layouts.

    Default ``dim_mapping`` assumes 3D ``(b, s, h)``. Callers bridging
    ``MimoModel``'s pre-flattened ``(s*b, h)`` encoder output should pass
    ``dim_mapping={'b': 0, 'h': 1}``; this relies on a uniform token count per
    sample so dim 0 divides evenly by the DP scale.

    Precondition: the input must be TP-replicated across the src TP group —
    i.e. all TP ranks inside a src DP replica hold the same tensor on the
    batch dim. The bridge never gathers along TP; violating this silently
    produces wrong results.
    """

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        src_module_name: str = "src",
        dest_module_name: str = "dest",
        dim_mapping: Optional[Dict[str, int]] = None,
    ):
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.dim_mapping = dim_mapping or {'b': 0, 's': 1, 'h': 2}
        self.current_rank = dist.get_rank()

        self._validate_grids()
        self._extract_parallelism_info()
        self._build_rank_mappings()

        # At most one direction is active; fan-in and fan-out are mutually
        # exclusive (one of ``src_dp / dest_dp`` is >1, the other is 1).
        # Equal DP uses no collective at all. Unify behind a single
        # ``gather_pg`` + ``direction`` + ``scale`` rather than a fan-in
        # and fan-out pair of attributes.
        self.gather_pg: Optional[dist.ProcessGroup] = None
        self.gather_group_ranks: List[List[int]] = []

        if self.src_dp_size > self.dest_dp_size:
            self.direction = BridgeDirection.FAN_IN
            self.scale = self.src_dp_size // self.dest_dp_size
            self.gather_group_ranks = self._build_gather_groups(
                iter_size=self.dest_dp_size,
                sibling_tp_size=self.src_tp_size,
                scale=self.scale,
                rank_to_pos=self.rank_to_src_pos,
            )
            self.gather_pg, _ = dist.new_subgroups_by_enumeration(
                self.gather_group_ranks, backend='nccl'
            )
        elif self.dest_dp_size > self.src_dp_size:
            self.direction = BridgeDirection.FAN_OUT
            self.scale = self.dest_dp_size // self.src_dp_size
            self.gather_group_ranks = self._build_gather_groups(
                iter_size=self.src_dp_size,
                sibling_tp_size=self.dest_tp_size,
                scale=self.scale,
                rank_to_pos=self.rank_to_dest_pos,
            )
            self.gather_pg, _ = dist.new_subgroups_by_enumeration(
                self.gather_group_ranks, backend='nccl'
            )
        else:
            self.direction = BridgeDirection.EQUAL
            self.scale = 1

        logging.info(
            f"[Rank {self.current_rank}] ColocatedBridgeCommunicator: "
            f"{src_module_name}({self.src_tp_size}TP/{self.src_dp_size}DP) -> "
            f"{dest_module_name}({self.dest_tp_size}TP/{self.dest_dp_size}DP), "
            f"direction={self.direction.value}, scale={self.scale}"
        )

    def _validate_grids(self):
        if self.src_grid.size != self.dest_grid.size:
            raise ValueError(
                f"Grids must span same number of ranks: "
                f"src={self.src_grid.size}, dest={self.dest_grid.size}"
            )

        if self.src_grid.rank_offset != self.dest_grid.rank_offset:
            raise ValueError(
                f"Grids must have same rank offset: "
                f"src={self.src_grid.rank_offset}, dest={self.dest_grid.rank_offset}"
            )

        # Per-grid dim checks: tp/dp required; pp and cp (if present) must be 1.
        # CP>1 also corrupts dp_idx when iterating get_rank_enum(['tp']) groups.
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            for required in ('tp', 'dp'):
                if required not in grid.dim_names:
                    raise ValueError(
                        f"{name} grid must have '{required}' dimension, "
                        f"got dim_names={grid.dim_names}"
                    )
            for singleton in ('pp', 'cp'):
                if singleton in grid.dim_names:
                    size = grid.shape[grid.dim_names.index(singleton)]
                    if size != 1:
                        raise ValueError(
                            f"{name} {singleton.upper()} must be 1 for "
                            f"ColocatedBridgeCommunicator, got {size}"
                        )

        src_dp = self.src_grid.shape[self.src_grid.dim_names.index('dp')]
        dest_dp = self.dest_grid.shape[self.dest_grid.dim_names.index('dp')]
        if src_dp % dest_dp != 0 and dest_dp % src_dp != 0:
            raise ValueError(
                f"DP sizes must be evenly divisible: src_dp={src_dp}, dest_dp={dest_dp}"
            )

    def _extract_parallelism_info(self):
        self.src_tp_size = self.src_grid.shape[self.src_grid.dim_names.index('tp')]
        self.src_dp_size = self.src_grid.shape[self.src_grid.dim_names.index('dp')]
        self.dest_tp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('tp')]
        self.dest_dp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('dp')]

    def _build_rank_mappings(self):
        self.rank_to_src_pos: Dict[int, Tuple[int, int]] = {}
        self.rank_to_dest_pos: Dict[int, Tuple[int, int]] = {}

        src_tp_groups = self.src_grid.get_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(src_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_src_pos[rank] = (dp_idx, tp_idx)

        dest_tp_groups = self.dest_grid.get_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(dest_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_dest_pos[rank] = (dp_idx, tp_idx)

    @staticmethod
    def _build_gather_groups(
        iter_size: int,
        sibling_tp_size: int,
        scale: int,
        rank_to_pos: Dict[int, Tuple[int, int]],
    ) -> List[List[int]]:
        """Build ``iter_size * sibling_tp_size`` gather groups of ``scale`` ranks.

        For each slot on the "iterating" side and each TP shard on the
        sibling side, collect the ``scale`` sibling ranks whose DP indices
        map into that slot. Append order equals group-local-rank order,
        which ``all_gather_into_tensor`` uses to concatenate outputs — do
        not sort.
        """
        groups: List[List[int]] = []
        for iter_idx in range(iter_size):
            sibling_dp_indices = range(iter_idx * scale, (iter_idx + 1) * scale)
            for sibling_tp_idx in range(sibling_tp_size):
                group_ranks = []
                for sibling_dp_idx in sibling_dp_indices:
                    for rank, (dp, tp) in rank_to_pos.items():
                        if dp == sibling_dp_idx and tp == sibling_tp_idx:
                            group_ranks.append(rank)
                            break
                groups.append(group_ranks)
        return groups

    def is_fan_in(self) -> bool:
        """True if src DP > dest DP (forward all-gathers)."""
        return self.direction is BridgeDirection.FAN_IN

    def is_fan_out(self) -> bool:
        """True if src DP < dest DP (forward narrows)."""
        return self.direction is BridgeDirection.FAN_OUT

    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Compute this rank's slice of ``batch_size`` on the narrowing side.

        For FAN_OUT this is the forward narrow; for FAN_IN it is the
        backward narrow against the post-gather batch. EQUAL returns the
        identity slice.

        Raises ``ValueError`` if ``batch_size`` is not divisible by ``scale``.
        """
        if self.direction is BridgeDirection.EQUAL:
            return SliceInfo(start=0, size=batch_size)
        self._check_divisible(batch_size)
        if self.direction is BridgeDirection.FAN_OUT:
            dp_idx = self.rank_to_dest_pos[self.current_rank][0]
        else:  # FAN_IN
            dp_idx = self.rank_to_src_pos[self.current_rank][0]
        slot = dp_idx % self.scale
        slice_size = batch_size // self.scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def _check_divisible(self, batch_size: int) -> None:
        if batch_size % self.scale != 0:
            raise ValueError(
                f"ColocatedBridgeCommunicator: batch dim size {batch_size} is "
                f"not divisible by {self.direction.value} scale={self.scale}."
            )

    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform ``tensor`` from src TP/DP layout to dest TP/DP layout.

        Raises ``ValueError`` when FAN_OUT and the batch dim is not
        divisible by ``scale``; FAN_IN only slices on the backward pass
        and re-checks via ``get_slice_info`` there.
        """
        if self.direction is BridgeDirection.FAN_OUT:
            self._check_divisible(tensor.shape[self.dim_mapping['b']])
        return _ColocatedCommunicate.apply(tensor, self)

    def destroy(self) -> None:
        """Release the NCCL subgroup created by this communicator.

        NCCL caps concurrent communicators; long-lived or repeated
        construction leaks PGs without this call.
        """
        if self.gather_pg is not None:
            dist.destroy_process_group(self.gather_pg)
            self.gather_pg = None


class _ColocatedCommunicate(torch.autograd.Function):
    """Autograd function for colocated communication with correct backward pass."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, comm: ColocatedBridgeCommunicator) -> torch.Tensor:
        ctx.comm = comm
        ctx.batch_dim = comm.dim_mapping['b']

        if comm.direction is BridgeDirection.FAN_OUT:
            # Narrow this rank's slice out of the full src batch.
            slice_info = comm.get_slice_info(tensor.shape[ctx.batch_dim])
            return tensor.narrow(ctx.batch_dim, slice_info.start, slice_info.size).contiguous()

        if comm.direction is BridgeDirection.FAN_IN:
            # All-gather sibling src outputs into a single full-batch tensor.
            return _all_gather_along_batch_dim(tensor, comm.gather_pg, ctx.batch_dim)

        # EQUAL: pure passthrough.
        return tensor.contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Adjoint of forward: narrow for fan-in, all-gather for fan-out.

        Fan-out's forward is ``narrow``, whose naive adjoint is zero-pad.
        That would leave each src rank with only its own dest rank's
        slice of the gradient, missing the contributions from every
        other dest rank that consumed a different slice of the same src
        activation. Instead we all-gather across the fan-out sibling
        group, reconstructing the full src-batch gradient (symmetric
        with the fan-in forward's all-gather).
        """
        comm = ctx.comm
        batch_dim = ctx.batch_dim

        if comm.direction is BridgeDirection.FAN_OUT:
            return _all_gather_along_batch_dim(grad_output, comm.gather_pg, batch_dim), None

        if comm.direction is BridgeDirection.FAN_IN:
            slice_info = comm.get_slice_info(grad_output.shape[batch_dim])
            return (
                grad_output.narrow(batch_dim, slice_info.start, slice_info.size).contiguous(),
                None,
            )

        return grad_output.contiguous(), None


def _all_gather_along_batch_dim(
    tensor: torch.Tensor, group: dist.ProcessGroup, batch_dim: int
) -> torch.Tensor:
    """All-gather ``tensor`` along an arbitrary batch dim into a single tensor.

    ``all_gather_into_tensor`` concatenates along dim 0, so when the
    batch dim is not 0 we move it, gather, then restore.
    """
    world_size = dist.get_world_size(group)
    src = tensor.contiguous()
    if batch_dim != 0:
        src = src.movedim(batch_dim, 0).contiguous()
    out_shape = list(src.shape)
    out_shape[0] *= world_size
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(out, src, group=group)
    if batch_dim != 0:
        out = out.movedim(0, batch_dim).contiguous()
    return out
