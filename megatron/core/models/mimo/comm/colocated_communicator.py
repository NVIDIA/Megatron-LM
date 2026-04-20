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
    """Handles tensor communication between colocated modules with different TP/DP layouts.

    Scope: PP=1 only on both src and dest grids.

    The default ``dim_mapping`` assumes 3D ``(b, s, h)`` tensors. When wired up
    through ``MimoModel`` the encoder output has already been flattened to
    ``(total_tokens, hidden)`` (i.e. ``(s*b, h)``); in that case callers should
    pass ``dim_mapping={'b': 0, 'h': 1}`` so dim 0 is treated as the batch/sample
    dimension. This relies on a uniform token count per sample (per image in a
    sample), which lets dim 0 be sliced evenly by the DP scale factor.

    Input precondition (TP replication): ``communicate()`` assumes the input
    tensor is already TP-replicated within each src DP group — i.e. all TP
    ranks inside a given src DP replica hold the same tensor on the batch
    dimension being bridged. Standard Megatron encoders produce TP-replicated
    output at the bridge point (post-all-reduce/gather on the hidden dim), so
    callers wiring this up through ``MimoModel`` get this for free. If you
    bridge a point where the tensor is TP-sharded on the batch dim, results
    are undefined — the communicator neither gathers along TP nor validates
    this invariant on the fast path. Set ``ColocatedBridgeCommunicator
    .CHECK_TP_REPLICATION = True`` (or pass ``check_tp_replication=True``) to
    enable a collective equality check on every forward; it is slow and
    intended only for debugging.
    """

    # Debug flag: when True, every communicate() call all-gathers across the
    # src TP group and verifies bitwise equality of the input tensor. Off by
    # default because the check is collective and expensive.
    CHECK_TP_REPLICATION: bool = False

    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        src_module_name: str = "src",
        dest_module_name: str = "dest",
        dim_mapping: Optional[Dict[str, int]] = None,
        check_tp_replication: Optional[bool] = None,
    ):
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.dim_mapping = dim_mapping or {'b': 0, 's': 1, 'h': 2}
        # Per-instance override wins over the class-level default.
        self.check_tp_replication = (
            self.CHECK_TP_REPLICATION if check_tp_replication is None else check_tp_replication
        )
        self.current_rank = dist.get_rank()

        self._validate_grids()
        self._extract_parallelism_info()
        self._build_rank_mappings()

        self.all_gather_pg: Optional[dist.ProcessGroup] = None
        self.all_gather_group_ranks: List[List[int]] = []
        self.fan_out_gather_pg: Optional[dist.ProcessGroup] = None
        self.fan_out_gather_group_ranks: List[List[int]] = []

        if self.fan_in_scale > 1:
            self._build_all_gather_groups()
        elif self.fan_out_scale > 1:
            self._build_fan_out_gather_groups()

        logging.info(
            f"[Rank {self.current_rank}] ColocatedBridgeCommunicator: "
            f"{src_module_name}({self.src_tp_size}TP/{self.src_dp_size}DP) -> "
            f"{dest_module_name}({self.dest_tp_size}TP/{self.dest_dp_size}DP), "
            f"fan_in_scale={self.fan_in_scale}, fan_out_scale={self.fan_out_scale}"
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

        # PP>1 is out of scope. Both src and dest grids must have PP=1.
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            if 'pp' in grid.dim_names:
                pp_size = grid.shape[grid.dim_names.index('pp')]
                if pp_size != 1:
                    raise ValueError(
                        f"{name} PP must be 1 for ColocatedBridgeCommunicator, got {pp_size}"
                    )

        # CP>1 on either grid is not yet supported. _build_rank_mappings uses
        # get_rank_enum(['tp']) which returns cp*pp*dp groups, so enumerating
        # them directly corrupts dp_idx when CP>1.
        for name, grid in [("src", self.src_grid), ("dest", self.dest_grid)]:
            if 'cp' in grid.dim_names:
                cp_size = grid.shape[grid.dim_names.index('cp')]
                if cp_size != 1:
                    raise ValueError(
                        f"{name} CP must be 1 for ColocatedBridgeCommunicator, got {cp_size}"
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
        # Integer fan-in / fan-out scales. At most one is > 1; the other is 1.
        self.fan_in_scale = (
            self.src_dp_size // self.dest_dp_size if self.src_dp_size > self.dest_dp_size else 1
        )
        self.fan_out_scale = (
            self.dest_dp_size // self.src_dp_size if self.dest_dp_size > self.src_dp_size else 1
        )

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

    def _build_all_gather_groups(self):
        # Ranks are appended in (src_dp_idx, src_tp_idx) slot order so that the
        # position inside ``group_ranks`` equals each participant's slot in the
        # gathered output. ``new_subgroups_by_enumeration`` assigns group-local
        # ranks by list position, and ``all_gather_into_tensor`` concatenates
        # outputs in group-local-rank order — so preserving append order is
        # load-bearing. Do not sort.
        all_groups: List[List[int]] = []

        for dest_dp_idx in range(self.dest_dp_size):
            src_dp_start = dest_dp_idx * self.fan_in_scale
            src_dp_indices = range(src_dp_start, src_dp_start + self.fan_in_scale)

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

    def _build_fan_out_gather_groups(self):
        """Build all-gather groups used by the fan-out backward.

        In fan-out forward, each dest rank takes a local slice of its src rank's
        batch. The src rank's full-batch gradient is the concatenation of the
        slice gradients produced by every dest rank that consumed one of its
        slices. We build a group per (src_dp_idx, dest_tp_idx) containing the
        ``fan_out_scale = dest_dp / src_dp`` dest ranks that cover consecutive
        dest_dp_idx values mapping to this src_dp_idx. Ranks are appended in
        slot order (increasing dest_dp_idx), which equals group-local-rank
        order inside ``new_subgroups_by_enumeration`` — so the subsequent
        ``all_gather_into_tensor`` reconstructs the full-batch gradient in the
        correct layout. Do not sort this list.
        """
        all_groups: List[List[int]] = []

        for src_dp_idx in range(self.src_dp_size):
            dest_dp_start = src_dp_idx * self.fan_out_scale
            dest_dp_indices = range(dest_dp_start, dest_dp_start + self.fan_out_scale)

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

    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Compute batch slice info for the current rank given the full batch size.

        Raises:
            ValueError: if ``batch_size`` is not divisible by the active scale
                factor (``fan_out_scale`` or ``fan_in_scale``). Silent
                truncation on non-divisible batches is a correctness bug that
                produced one-off mis-slicing in early versions.
        """
        if self.fan_out_scale > 1:
            self._check_divisible(batch_size, self.fan_out_scale, "fan_out_scale")
            return self._get_fan_out_slice_info(batch_size)
        if self.fan_in_scale > 1:
            self._check_divisible(batch_size, self.fan_in_scale, "fan_in_scale")
            return self._get_fan_in_slice_info(batch_size)
        return SliceInfo(start=0, size=batch_size)

    def _check_divisible(self, batch_size: int, scale: int, scale_name: str) -> None:
        if batch_size % scale != 0:
            raise ValueError(
                f"ColocatedBridgeCommunicator: batch dim size {batch_size} is not "
                f"divisible by {scale_name}={scale}. Non-divisible batches would "
                f"silently drop samples on the narrow/all-gather path."
            )

    def _get_fan_out_slice_info(self, batch_size: int) -> SliceInfo:
        if self.current_rank not in self.rank_to_dest_pos:
            return SliceInfo(start=0, size=batch_size)
        dest_dp_idx = self.rank_to_dest_pos[self.current_rank][0]
        slot = dest_dp_idx % self.fan_out_scale
        slice_size = batch_size // self.fan_out_scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def _get_fan_in_slice_info(self, batch_size: int) -> SliceInfo:
        if self.current_rank not in self.rank_to_src_pos:
            return SliceInfo(start=0, size=batch_size)
        src_dp_idx = self.rank_to_src_pos[self.current_rank][0]
        slot = src_dp_idx % self.fan_in_scale
        slice_size = batch_size // self.fan_in_scale
        return SliceInfo(start=slot * slice_size, size=slice_size)

    def is_fan_out(self) -> bool:
        """Return True if src DP < dest DP (encoder has fewer replicas)."""
        return self.src_dp_size < self.dest_dp_size

    def is_fan_in(self) -> bool:
        """Return True if src DP > dest DP (encoder has more replicas)."""
        return self.src_dp_size > self.dest_dp_size

    def _assert_tp_replicated(self, tensor: torch.Tensor) -> None:
        """Collective debug check that ``tensor`` is identical across src TP ranks.

        Off by default (see ``CHECK_TP_REPLICATION``). When enabled, runs an
        all-gather on the src TP group and raises if any TP peer's tensor
        differs from the local one. Expensive — debug-only.
        """
        if not self.check_tp_replication:
            return
        if self.src_tp_size <= 1:
            return
        if self.current_rank not in self.rank_to_src_pos:
            return
        tp_group = self.src_grid.get_pg('tp')
        local = tensor.contiguous()
        gathered = [torch.empty_like(local) for _ in range(self.src_tp_size)]
        dist.all_gather(gathered, local, group=tp_group)
        for peer_idx, peer in enumerate(gathered):
            if not torch.equal(peer, local):
                raise RuntimeError(
                    f"ColocatedBridgeCommunicator: TP-replication precondition "
                    f"violated at rank {self.current_rank}: input tensor differs "
                    f"from src TP peer slot {peer_idx}. The bridge requires the "
                    f"encoder output to be TP-replicated on the batch dim."
                )

    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform tensor from src TP/DP layout to dest TP/DP layout.

        Precondition: ``tensor`` must be TP-replicated within each src DP
        group (see class docstring). The communicator does not gather along
        TP; feeding a TP-sharded tensor silently produces wrong results.

        Raises:
            ValueError: if the batch dim size is not divisible by the active
                fan-in or fan-out scale factor.
            RuntimeError: only when ``check_tp_replication`` is enabled and
                the precondition is violated.
        """
        # Fan-out forward narrows dim[b] by fan_out_scale; non-divisibility
        # there would silently truncate the batch. Fan-in forward all-gathers
        # (no slice), and the backward narrow runs against the post-gather
        # size which is always divisible — so only fan-out needs a
        # forward-side divisibility guard. get_slice_info() re-checks on the
        # backward path for fan-in and fan-out both.
        if self.fan_out_scale > 1:
            batch_dim = self.dim_mapping['b']
            batch_size = tensor.shape[batch_dim]
            self._check_divisible(batch_size, self.fan_out_scale, "fan_out_scale")
        self._assert_tp_replicated(tensor)
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
            group = comm.all_gather_pg
            world_size = dist.get_world_size(group)
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
            group = comm.fan_out_gather_pg
            world_size = dist.get_world_size(group)

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
