# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

try:
    import emerging_optimizers  # pylint: disable=unused-import
except ImportError as error:
    # CI's import checker recognizes "UnavailableError" as a missing optional dependency.
    raise ImportError(
        "UnavailableError: FsdpMuon requires Emerging-Optimizers. "
        "Install it with `pip install 'megatron-fsdp[emerging-optimizers]'`."
    ) from error

from emerging_optimizers import utils as eo_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from .dbuffer import DBuffer
from .gather_scatter import gather, get_sorted_rank_ranges, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .range import Range, intersect_ranges


@dataclass(frozen=True)
class _FsdpGroupAssignment:
    """An FSDP group and its cached optimizer-group and owner-rank assignments and rank ranges."""

    fsdp_group: FsdpParameterGroup
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner_rank: dict[int, int]
    # The layout is fixed, so the ranges are computed once instead of on every gather and scatter.
    rank_ranges: dict[int, Range]

    @cached_property
    def requires_communication(self) -> bool:
        """Whether any of the optimized tensors has shards on several ranks."""
        layout = self.fsdp_group.main_weight.layout
        return any(
            sum(
                intersect_ranges(layout.get_tensor_range(i), rank_range).numel > 0
                for rank_range in self.rank_ranges.values()
            )
            > 1
            for i in self.tensor_to_optimizer_group_index
        )


def _schedule(pipelines: list["_GroupPipeline"]) -> None:
    """Reorder the groups in place: spread those that need no communication over the others.

    The last of them goes at the end. Their Newton-Schulz hides the transfers in flight, which the
    other groups' own may be too short for.
    """
    free = [p for p in pipelines if not p.assignment.requires_communication]
    comm = [p for p in pipelines if p.assignment.requires_communication]
    position = {p: (k + 1) / len(free) for k, p in enumerate(free)}
    position |= {p: (i + 0.5) / len(comm) for i, p in enumerate(comm)}
    pipelines.sort(key=lambda p: (position[p], p.assignment.requires_communication))


class _PipelineStage(Enum):
    """Host-side progress; a posted stage may still be executing on the GPU."""

    IDLE = auto()
    GATHER_POSTED = auto()
    ORTHO_POSTED = auto()
    SCATTER_POSTED = auto()


class _GroupPipeline:
    """Own one assignment's transfers and scratch storage across optimizer steps.

    The scheduler selects stages; this object enqueues their dependencies and releases
    their inputs. Optimizer configuration and math remain on FsdpMuon, passed explicitly
    to the stages that need them. No optimizer reference or callback is retained.
    Gradient availability stays fixed while an optimizer step is in flight.
    """

    assignment: _FsdpGroupAssignment
    _stage: _PipelineStage
    # Gather input: sharded momentum, released by wait_for_gather().
    _momentum_inputs: DBuffer | None
    # Gather output -> orthogonalize input: full matrices on their owners, consumed by ortho.
    _full_inputs: dict[int, torch.Tensor]
    # Gather output -> wait_for_gather(): completion handles.
    _gather_works: list[dist.Work]
    # Orthogonalize output -> scatter input; also supplies owner-local shards in apply_updates().
    _full_updates: dict[int, torch.Tensor]
    # Scatter output -> apply_updates input: received shards of remotely owned matrices.
    _local_updates: DBuffer | None
    # Scatter output -> wait_for_scatter(): completion handles.
    _scatter_works: list[dist.Work]

    def __init__(self, assignment: _FsdpGroupAssignment) -> None:
        self.assignment = assignment
        self._stage = _PipelineStage.IDLE
        self._momentum_inputs = None
        self._full_inputs = {}
        self._gather_works = []
        self._full_updates = {}
        self._local_updates = None
        self._scatter_works = []

    @property
    def has_grad(self) -> bool:
        """Whether this assignment participates in the current optimizer step."""
        group = self.assignment.fsdp_group
        return any(
            group.fsdp_parameters[i].sharded.grad is not None
            for i in self.assignment.tensor_to_optimizer_group_index
        )

    def _active_owners(self) -> dict[int, int]:
        """Derive owners from the assignment and this step's gradient availability."""
        group = self.assignment.fsdp_group
        return {
            i: self.assignment.tensor_to_owner_rank[i]
            for i in self.assignment.tensor_to_optimizer_group_index
            if group.fsdp_parameters[i].sharded.grad is not None
        }

    def gather(self, optimizer: "FsdpMuon") -> None:
        """Build sharded momentum from gradients; gather it into owners' full inputs.

        A group without communication has nothing to gather: the owner of a tensor is the only
        rank holding it, so its momentum input already is the full tensor.
        """
        assert self._stage is _PipelineStage.IDLE
        group = self.assignment.fsdp_group
        owners = self._active_owners()
        assert owners
        active = {i: self.assignment.tensor_to_optimizer_group_index[i] for i in owners}
        self._momentum_inputs = optimizer._compute_momentum_inputs(group, active)
        buffer = group.main_weight
        owned = [i for i, owner in owners.items() if owner == group.mesh.get_rank()]
        if self.assignment.requires_communication:
            self._full_inputs = {
                i: torch.empty(
                    buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device
                )
                for i in owned
            }
            self._gather_works = gather(
                self._momentum_inputs,
                self._full_inputs,
                tensor_to_owner=owners,
                rank_ranges=self.assignment.rank_ranges,
            )
        else:
            self._full_inputs = {i: self._momentum_inputs.get_tensor_view(i) for i in owned}
        self._stage = _PipelineStage.GATHER_POSTED

    def wait_for_gather(self) -> None:
        """Make the caller's stream wait for gather, then release its source buffer."""
        assert self._stage is _PipelineStage.GATHER_POSTED
        for work in self._gather_works:
            work.wait()
        self._gather_works.clear()
        self._momentum_inputs = None

    def orthogonalize(self, optimizer: "FsdpMuon") -> None:
        """Consume gathered full inputs and produce orthogonalized full updates."""
        self.wait_for_gather()
        self._full_updates = optimizer._orthogonalize_updates(self.assignment, self._full_inputs)
        self._stage = _PipelineStage.ORTHO_POSTED

    def scatter(self) -> None:
        """Scatter full updates into local update shards; retain owner-local sources.

        A group without communication has nothing to scatter: its owners hold all its updates.
        """
        assert self._stage is _PipelineStage.ORTHO_POSTED
        if self.assignment.requires_communication:
            buffer = self.assignment.fsdp_group.main_weight
            self._local_updates = DBuffer(
                buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
            )
            self._scatter_works = scatter(
                self._full_updates,
                self._local_updates,
                tensor_to_owner=self._active_owners(),
                rank_ranges=self.assignment.rank_ranges,
            )
        self._stage = _PipelineStage.SCATTER_POSTED

    def wait_for_scatter(self) -> None:
        """Make the caller's stream wait until received update shards are ready."""
        assert self._stage is _PipelineStage.SCATTER_POSTED
        for work in self._scatter_works:
            work.wait()
        self._scatter_works.clear()

    def apply_updates(self, optimizer: "FsdpMuon") -> None:
        """Apply received shards and owner-local full updates, then release both."""
        self.wait_for_scatter()
        optimizer._apply_local_updates(self.assignment, self._full_updates, self._local_updates)
        self._full_updates.clear()
        self._local_updates = None
        self._stage = _PipelineStage.IDLE


# Design choice: integrating Emerging Optimizers (EO) into FsdpMuon.
#
# 1. Inherit from EO's Muon (chosen).
#    Reuse its constructor, configuration, orthogonalize(), and update hooks.
#    Keep state, parameter groups, and checkpointing on one optimizer instance.
# 2. Contain an EO Muon instance (composition).
#    The main problem is state ownership across two optimizer instances. One must
#    own the sharded momentum and parameter groups; updates and checkpoint
#    save/load must all use that same state, even when loading replaces it.
# 3. Call EO kernels from a standalone torch.optim.Optimizer.
#    Use Newton-Schulz and scaling functions directly, but maintain Muon's
#    configuration, defaults, and validation in FsdpMuon.
#
# We use inheritance for now. Composition (2) makes optimizer state ownership
# unclear, while direct kernel calls (3) duplicate EO optimizer code.
# Revisit this decision as the inheritance tax grows, such as when EO changes
# require more overrides or workarounds in FsdpMuon.
# See https://media.pragprog.com/titles/tpp20/inheritance-tax.pdf.
class FsdpMuon(Muon):
    """Muon for 2-D parameters with sharded optimizer buffers.

    Momentum remains sharded. Owner assignments balance orthogonalization across a
    fixed set of buffers and are computed during initialization. Each step gathers,
    orthogonalizes, and scatters each buffer. Transfers are asynchronous: the gathers of the next
    ``num_groups_in_flight`` buffers are posted before the current one is orthogonalized, and a
    buffer's updates are applied after ``num_groups_in_flight`` later buffers have been scattered,
    so each scatter overlaps their Newton-Schulz. Buffers whose tensors all sit on one rank need
    no communication; they are spread over the pipeline, the last one at the end, where their
    Newton-Schulz hides the transfers in flight.
    Full matrices exist only on their assigned owners.
    The FSDP groups may use different meshes; owners are planned jointly across all ranks, so
    every rank must construct FsdpMuon.
    Parameters and optimizer groups must have the same order on every DP rank, and
    ranks must agree on which parameters have gradients. Parameter membership and
    optimizer-group order must remain fixed after construction.

    This optimizer updates only the main weights. Wrap it with ``fully_shard_optimizer``, using
    ``precision_aware=True`` because it casts gradients itself, or with an MCore
    ``FullyShardedOptimizer`` to refresh the FSDP model weights after each step.
    Momentum is initialized lazily as DTensors in the standard optimizer state.

    Limitations:
        - Non-2D parameters are skipped. Use a separate optimizer for them and for
          matrices you exclude from Muon, such as embeddings and output projections.
        - Tensor parallelism is unsupported.

    Args:
        *args: Positional arguments passed unchanged to EO's ``Muon``.
        num_groups_in_flight: Number of buffers whose gathers are posted ahead of the one being
            orthogonalized, and whose scatters are posted before a buffer's updates are applied.
            Each buffer in flight adds its momentum inputs, full tensors, and received updates to
            the peak memory. Zero gives the non-overlapped order.
        **kwargs: Keyword arguments passed unchanged to EO's ``Muon``.
    """

    _fsdp_group_assignments: list[_FsdpGroupAssignment]

    def __init__(self, *args: Any, num_groups_in_flight: int = 8, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if num_groups_in_flight < 0:
            raise ValueError(
                f"num_groups_in_flight must be non-negative, got {num_groups_in_flight}."
            )
        self.num_groups_in_flight = num_groups_in_flight

        # Preserve discovery order: communication must be posted identically on every rank.
        fsdp_groups: dict[FsdpParameterGroup, None] = {}
        param_to_optimizer_group_index: dict[torch.nn.Parameter, int] = {}
        for group_index, group in enumerate(self.param_groups):
            for param in group["params"]:
                # Callers may include biases and other non-matrices for another optimizer.
                if param.ndim != 2:
                    continue
                fsdp_group: FsdpParameterGroup | None = get_containing_parameter_group(param)
                if fsdp_group is None:
                    raise ValueError("FsdpMuon requires parameters sharded by MFSDP v2.")
                fsdp_groups[fsdp_group] = None
                param_to_optimizer_group_index[param] = group_index
        self._fsdp_group_assignments = []
        for fsdp_group, tensor_to_owner_rank in zip(
            fsdp_groups, assign_owner_work(group.main_weight for group in fsdp_groups), strict=True
        ):
            self._fsdp_group_assignments.append(
                _FsdpGroupAssignment(
                    fsdp_group=fsdp_group,
                    tensor_to_optimizer_group_index={
                        i: param_to_optimizer_group_index[fp.sharded]
                        for i, fp in enumerate(fsdp_group.fsdp_parameters)
                        if fp.sharded in param_to_optimizer_group_index
                    },
                    tensor_to_owner_rank=tensor_to_owner_rank,
                    rank_ranges=get_sorted_rank_ranges(fsdp_group.main_weight),
                )
            )
        self._validate_parameter_groups()
        self._group_pipelines = [
            _GroupPipeline(assignment) for assignment in self._fsdp_group_assignments
        ]

    def _validate_parameter_groups(self) -> None:
        """Validate optimizer parameters and the FSDP buffers that contain them."""
        for assignment in self._fsdp_group_assignments:
            fsdp_group: FsdpParameterGroup = assignment.fsdp_group
            for buffer in (fsdp_group.main_weight, fsdp_group.main_grad):
                if not isinstance(buffer, DBuffer):
                    raise ValueError("FsdpMuon requires non-quantized main weights and gradients.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Update the sharded main weights."""
        if closure is not None:
            raise ValueError("closure is not supported")

        pipelines = [pipeline for pipeline in self._group_pipelines if pipeline.has_grad]
        _schedule(pipelines)

        # n = num_groups_in_flight; N: Newton-Schulz, G: gather, S: scatter, A: apply.
        #
        #   G(pipelines[:n]) | for each group i: G(i + n), N(i), S(i), A(i - n)
        #
        # Each reusable pipeline owns its scratch until the corresponding stage has consumed it.
        # Every rank posts transfers in the same order.
        for pipeline in pipelines[: self.num_groups_in_flight]:
            pipeline.gather(self)
        for i, pipeline in enumerate(pipelines):
            if i + self.num_groups_in_flight < len(pipelines):
                pipelines[i + self.num_groups_in_flight].gather(self)
            pipeline.orthogonalize(self)
            pipeline.scatter()
            if i >= self.num_groups_in_flight:
                pipelines[i - self.num_groups_in_flight].apply_updates(self)
        for pipeline in pipelines[max(0, len(pipelines) - self.num_groups_in_flight) :]:
            pipeline.apply_updates(self)

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Apply local weight decay, update momentum, and build inputs for orthogonalization."""
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        for i, group_index in tensor_to_optimizer_group_index.items():
            group: dict[str, Any] = self.param_groups[group_index]
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            state: dict[str, Any] = self.state[param]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(param)
            local_input: torch.Tensor = momentum_inputs.get_tensor_view(i)
            momentum: torch.Tensor = state["momentum_buffer"].to_local()
            grad: torch.Tensor = param.grad.to_local().to(momentum.dtype).reshape(momentum.shape)
            # Match EO's decay before momentum without calling its private helper.
            weight_decay: float = group["weight_decay"]
            if weight_decay != 0:
                weight: torch.Tensor = param.to_local()
                lr: float = group["lr"]
                match self.weight_decay_method:
                    case "decoupled":
                        weight.add_(weight, alpha=-(weight_decay * lr))
                    case "independent":
                        weight.add_(weight, alpha=-weight_decay)
                    case "l2":
                        grad.add_(weight, alpha=weight_decay)
                    case "palm":
                        weight.add_(weight, alpha=-(weight_decay * lr * lr))
                    case _:
                        raise ValueError(f"Invalid weight decay method: {self.weight_decay_method}")
            # EO's OrthogonalizedOptimizer.step implements momentum inline, without a helper.
            momentum.lerp_(grad, 1 - group["momentum"])
            if self.nesterov:
                torch.lerp(grad, momentum, group["momentum"], out=local_input.view_as(momentum))
            else:
                local_input.copy_(momentum.reshape(local_input.shape))
        return momentum_inputs

    def _orthogonalize_updates(
        self, assignment: _FsdpGroupAssignment, full_inputs: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Consume this rank's gathered inputs and produce its full updates."""
        fsdp_group = assignment.fsdp_group
        full_updates = {}
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for i in tuple(full_inputs):
                update = full_inputs.pop(i)
                param: torch.nn.Parameter = fsdp_group.fsdp_parameters[i].sharded
                group: dict[str, Any] = self.param_groups[
                    assignment.tensor_to_optimizer_group_index[i]
                ]
                kwargs: dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
                full_updates[i] = self.orthogonalize(param, update.float(), **kwargs).contiguous()
        return full_updates

    def _apply_local_updates(
        self,
        assignment: _FsdpGroupAssignment,
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer | None,
    ) -> None:
        """Apply owned result slices and received updates to local parameter shards."""
        fsdp_group = assignment.fsdp_group
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in assignment.tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            if param.grad is None:
                continue
            weight: torch.Tensor = param.to_local()
            if i in full_updates:
                # Scatter writes only remote-owned shards; owners keep their full result.
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            elif local_updates is not None:
                update = local_updates.get_tensor_view(i)
            else:
                # In a group without communication, only the owner holds the tensor.
                continue
            update = update.reshape(weight.shape).to(weight.dtype)
            # These hooks are inherited from EO's OrthogonalizedOptimizer, via Muon.
            self.pre_weight_update_fn_inplace(weight, update)
            weight.add_(update, alpha=-self.param_groups[group_index]["lr"])
            self.post_weight_update_fn_inplace(weight)
