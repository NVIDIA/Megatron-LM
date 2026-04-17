# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class SliceInfo:
    """Batch dimension slice information for a rank's data partition."""

    start: int
    size: int


class ColocatedBridgeCommunicator:
    """Handles tensor communication between colocated modules with different TP/DP layouts."""

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

        self.all_gather_pg: Optional[dist.ProcessGroup] = None
        self.all_gather_group_ranks: List[List[int]] = []
        self.fan_out_gather_pg: Optional[dist.ProcessGroup] = None
        self.fan_out_gather_group_ranks: List[List[int]] = []

        if self.dp_scale_factor > 1:
            self._build_all_gather_groups()
        elif self.dp_scale_factor < 1:
            self._build_fan_out_gather_groups()

        logging.info(
            f"[Rank {self.current_rank}] ColocatedBridgeCommunicator: "
            f"{src_module_name}({self.src_tp_size}TP/{self.src_dp_size}DP) -> "
            f"{dest_module_name}({self.dest_tp_size}TP/{self.dest_dp_size}DP), "
            f"scale_factor={self.dp_scale_factor}"
        )

    def _validate_grids(self):
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            for required in ('tp', 'dp'):
                if required not in grid.dim_names:
                    raise ValueError(
                        f"{name} grid must have '{required}' dimension, "
                        f"got dim_names={grid.dim_names}"
                    )

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

        # Source (encoder) must have PP=1. Dest (LLM) may have PP>1 —
        # the communicator only maps to dest's first PP stage.
        if 'pp' in self.src_grid.dim_names:
            pp_size = self.src_grid.shape[self.src_grid.dim_names.index('pp')]
            if pp_size != 1:
                raise ValueError(f"Source PP must be 1 for colocated, got {pp_size}")

        # CP>1 on either grid is not yet supported. See NMFW-17 task #1:
        # _build_rank_mappings uses _gen_rank_enum(['tp']) which returns
        # cp*pp*dp groups, so enumerating them directly corrupts dp_idx
        # when CP>1. Reject explicitly until the fix lands.
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            if 'cp' in grid.dim_names:
                cp_size = grid.shape[grid.dim_names.index('cp')]
                if cp_size != 1:
                    raise ValueError(
                        f"{name} CP must be 1 for ColocatedBridgeCommunicator, " f"got {cp_size}"
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
        self.dp_scale_factor = self.src_dp_size / self.dest_dp_size

    @staticmethod
    def _get_rank_dim_coord(rank, grid, dim_name):
        """Extract a rank's coordinate for a specific grid dimension."""
        dim_idx = grid.dim_names.index(dim_name)
        temp = rank - grid.rank_offset
        for i in range(dim_idx):
            temp //= grid.shape[i]
        return temp % grid.shape[dim_idx]

    def _build_rank_mappings(self):
        self.rank_to_src_pos: Dict[int, Tuple[int, int]] = {}
        self.rank_to_dest_pos: Dict[int, Tuple[int, int]] = {}

        src_tp_groups = self.src_grid._gen_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(src_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_src_pos[rank] = (dp_idx, tp_idx)

        # For dest, only map ranks at PP stage 0 (first pipeline stage).
        # When dest has PP>1, _gen_rank_enum(['tp']) returns dp*pp groups.
        # We filter to PP=0 so dp_idx correctly indexes the DP dimension only.
        dest_has_pp = 'pp' in self.dest_grid.dim_names
        dest_tp_groups = self.dest_grid._gen_rank_enum(['tp'])
        dp_idx = 0
        for tp_group in dest_tp_groups:
            if dest_has_pp:
                pp_coord = self._get_rank_dim_coord(tp_group[0], self.dest_grid, 'pp')
                if pp_coord != 0:
                    continue
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_dest_pos[rank] = (dp_idx, tp_idx)
            dp_idx += 1

    def _build_all_gather_groups(self):
        # Ranks are appended in (src_dp_idx, src_tp_idx) slot order so that the
        # position inside ``group_ranks`` equals each participant's slot in the
        # gathered output. ``new_subgroups_by_enumeration`` assigns group-local
        # ranks by list position, and ``all_gather_into_tensor`` concatenates
        # outputs in group-local-rank order — so preserving append order is
        # load-bearing. Do not sort.
        scale = int(self.dp_scale_factor)
        all_groups: List[List[int]] = []

        for dest_dp_idx in range(self.dest_dp_size):
            src_dp_start = dest_dp_idx * scale
            src_dp_indices = range(src_dp_start, src_dp_start + scale)

            for src_tp_idx in range(self.src_tp_size):
                group_ranks = []
                for src_dp_idx in src_dp_indices:
                    for rank, (dp, tp) in self.rank_to_src_pos.items():
                        if dp == src_dp_idx and tp == src_tp_idx:
                            group_ranks.append(rank)
                            break
                all_groups.append(group_ranks)

        self.all_gather_group_ranks = all_groups
        self.all_gather_pg, _ = dist.new_subgroups_by_enumeration(all_groups, backend='nccl')

        logging.debug(f"[Rank {self.current_rank}] All-gather groups: {all_groups}")

    def get_all_gather_group(self) -> Optional[dist.ProcessGroup]:
        """Return the all-gather process group for fan-in communication."""
        return self.all_gather_pg

    def get_all_gather_world_size(self) -> int:
        """Return the world size of the all-gather group."""
        if self.all_gather_pg is None:
            return 1
        return dist.get_world_size(self.all_gather_pg)

    def _build_fan_out_gather_groups(self):
        """Build all-gather groups used by the fan-out backward.

        In fan-out forward, each dest rank takes a local slice of its src rank's
        batch. The src rank's full-batch gradient is the concatenation of the
        slice gradients produced by every dest rank that consumed one of its
        slices. We build a group per (src_dp_idx, dest_tp_idx) containing the
        ``scale = dest_dp / src_dp`` dest ranks that cover consecutive
        dest_dp_idx values mapping to this src_dp_idx. Ranks are appended in
        slot order (increasing dest_dp_idx), which equals group-local-rank
        order inside ``new_subgroups_by_enumeration`` — so the subsequent
        ``all_gather_into_tensor`` reconstructs the full-batch gradient in the
        correct layout. Do not sort this list.
        """
        scale = int(1 / self.dp_scale_factor)
        all_groups: List[List[int]] = []

        for src_dp_idx in range(self.src_dp_size):
            dest_dp_start = src_dp_idx * scale
            dest_dp_indices = range(dest_dp_start, dest_dp_start + scale)

            for dest_tp_idx in range(self.dest_tp_size):
                group_ranks = []
                for dest_dp_idx in dest_dp_indices:
                    for rank, (dp, tp) in self.rank_to_dest_pos.items():
                        if dp == dest_dp_idx and tp == dest_tp_idx:
                            group_ranks.append(rank)
                            break
                all_groups.append(group_ranks)

        self.fan_out_gather_group_ranks = all_groups
        self.fan_out_gather_pg, _ = dist.new_subgroups_by_enumeration(all_groups, backend='nccl')

    def get_fan_out_gather_group(self) -> Optional[dist.ProcessGroup]:
        """Return the all-gather process group for fan-out backward."""
        return self.fan_out_gather_pg

    def get_fan_out_gather_world_size(self) -> int:
        """Return the world size of the fan-out backward all-gather group."""
        if self.fan_out_gather_pg is None:
            return 1
        return dist.get_world_size(self.fan_out_gather_pg)

    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Compute batch slice info for the current rank given the full batch size."""
        if self.dp_scale_factor < 1:
            return self._get_fan_out_slice_info(batch_size)
        elif self.dp_scale_factor > 1:
            return self._get_fan_in_slice_info(batch_size)
        else:
            return SliceInfo(start=0, size=batch_size)

    def _get_fan_out_slice_info(self, batch_size: int) -> SliceInfo:
        # For PP>1 dest, only PP stage 0 ranks are in rank_to_dest_pos.
        # PP stage 1+ ranks still call communicate() but the result is unused.
        if self.current_rank not in self.rank_to_dest_pos:
            return SliceInfo(start=0, size=batch_size)
        dest_dp_idx = self.rank_to_dest_pos[self.current_rank][0]
        scale = int(1 / self.dp_scale_factor)
        slot = dest_dp_idx % scale
        slice_size = batch_size // scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def _get_fan_in_slice_info(self, batch_size: int) -> SliceInfo:
        if self.current_rank not in self.rank_to_src_pos:
            return SliceInfo(start=0, size=batch_size)
        src_dp_idx = self.rank_to_src_pos[self.current_rank][0]
        scale = int(self.dp_scale_factor)
        slot = src_dp_idx % scale
        slice_size = batch_size // scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def is_fan_out(self) -> bool:
        """Return True if src DP < dest DP (encoder has fewer replicas)."""
        return self.src_dp_size < self.dest_dp_size

    def is_fan_in(self) -> bool:
        """Return True if src DP > dest DP (encoder has more replicas)."""
        return self.src_dp_size > self.dest_dp_size

    def is_equal_dp(self) -> bool:
        """Return True if src and dest have same DP size."""
        return self.src_dp_size == self.dest_dp_size

    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform tensor from src TP/DP layout to dest TP/DP layout."""
        return _ColocatedCommunicate.apply(tensor, self)

    def destroy(self) -> None:
        """Release NCCL process groups created by this communicator.

        NCCL enforces a hard cap on concurrent communicators (~500). Long-lived
        or repeated construction (e.g. per-test fixtures, checkpoint reloads)
        leaks PGs without this call.
        """
        for attr in ('all_gather_pg', 'fan_out_gather_pg'):
            pg = getattr(self, attr, None)
            if pg is not None:
                dist.destroy_process_group(pg)
                setattr(self, attr, None)


class _ColocatedCommunicate(torch.autograd.Function):
    """Autograd function for colocated communication with correct backward pass."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, comm: ColocatedBridgeCommunicator) -> torch.Tensor:
        """Forward: fan-out slices, fan-in all-gathers, equal copies."""
        ctx.comm = comm
        ctx.batch_dim = comm.dim_mapping['b']
        batch_size = tensor.shape[ctx.batch_dim]

        if comm.is_fan_out():
            slice_info = comm.get_slice_info(batch_size)
            return tensor.narrow(ctx.batch_dim, slice_info.start, slice_info.size).contiguous()

        elif comm.is_fan_in():
            group = comm.get_all_gather_group()
            world_size = comm.get_all_gather_world_size()
            batch_dim = ctx.batch_dim

            # Use all_gather_into_tensor to write directly into a pre-allocated
            # output buffer, avoiding the N intermediate tensors + torch.cat copy.
            # all_gather_into_tensor concatenates along dim 0, so when batch_dim
            # is not 0 we move it to dim 0 before gathering, then restore.
            input_contig = tensor.contiguous()
            if batch_dim != 0:
                input_contig = input_contig.movedim(batch_dim, 0).contiguous()

            out_shape = list(input_contig.shape)
            out_shape[0] *= world_size
            output = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(output, input_contig, group=group)

            if batch_dim != 0:
                output = output.movedim(0, batch_dim).contiguous()
            return output

        else:
            return tensor.contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward: adjoint of forward (all-gather for fan-out, slice for fan-in).

        Fan-out forward is ``narrow``, which is a local slice with no cross-rank
        communication. Its autograd adjoint on a single rank is zero-pad, but
        that would leave each src rank with only its own dest rank's slice of
        the gradient — missing the contributions from every other dest rank
        that consumed a different slice of the same src rank's activation.
        Instead we all-gather the slice gradients across the fan-out sibling
        group, which reconstructs the full src-batch gradient and hands every
        participating rank the same full gradient (symmetric with the fan-in
        forward's all_gather_into_tensor).
        """
        comm = ctx.comm
        batch_dim = ctx.batch_dim

        if comm.is_fan_out():
            group = comm.get_fan_out_gather_group()
            world_size = comm.get_fan_out_gather_world_size()

            input_contig = grad_output.contiguous()
            if batch_dim != 0:
                input_contig = input_contig.movedim(batch_dim, 0).contiguous()

            out_shape = list(input_contig.shape)
            out_shape[0] *= world_size
            grad_input = torch.empty(out_shape, dtype=grad_output.dtype, device=grad_output.device)
            dist.all_gather_into_tensor(grad_input, input_contig, group=group)

            if batch_dim != 0:
                grad_input = grad_input.movedim(0, batch_dim).contiguous()
            return grad_input, None

        elif comm.is_fan_in():
            output_batch_size = grad_output.shape[batch_dim]
            slice_info = comm.get_slice_info(output_batch_size)
            return (
                grad_output.narrow(batch_dim, slice_info.start, slice_info.size).contiguous(),
                None,
            )

        else:
            return grad_output.contiguous(), None
