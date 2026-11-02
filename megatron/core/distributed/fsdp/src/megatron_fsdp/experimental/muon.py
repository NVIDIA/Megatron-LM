# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

from .ce_transfer import CopyEngineExchange
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
    # The layout is fixed, so the ranges are cached here instead of being recomputed, once per rank,
    # on every gather and scatter of every step.
    rank_ranges: dict[int, Range]

    @cached_property
    def has_comm(self) -> bool:
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


@dataclass(frozen=True)
class _Transfer:
    """A posted gather or scatter of one FSDP group, with the buffers it uses.

    `sharded` is the DBuffer side of the transfer: the momentum inputs of a gather or the received
    updates of a scatter. `full_updates` holds the owners' full tensors on the other side. A group
    without communication has no works and no `sharded` buffer. The transfer is created on the
    stream it is posted on and records that stream's progress, because the works alone do not
    cover what it did before them, such as filling a gather's buffers.
    """

    works: list[dist.Work]
    assignment: _FsdpGroupAssignment
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner: dict[int, int]
    sharded: DBuffer | None
    full_updates: dict[int, torch.Tensor]
    ready: torch.cuda.Event = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ready", torch.cuda.current_stream().record_event())

    def wait(self) -> None:
        """Order the current stream after the transfer; the host does not block."""
        torch.cuda.current_stream().wait_event(self.ready)
        for work in self.works:
            work.wait()


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
    """Muon for 2-D parameters with sharded optimizer buffers on a one-dimensional DP mesh.

    Momentum remains sharded. Owner assignments balance orthogonalization across a
    fixed set of buffers and are computed during initialization. Each step gathers,
    orthogonalizes, and scatters each buffer. The momentum updates and gathers run on a side CUDA
    stream, the rest on the caller's stream. Transfers are asynchronous: the gathers of all buffers
    that need communication are posted before the first is orthogonalized and the updates are
    applied at the end of the step, so each scatter overlaps the Newton-Schulz of the later
    buffers. Every buffer in flight adds its full tensors and received updates to the peak memory.
    Buffers whose tensors all sit on one rank need no communication; the first runs before the
    gathers are posted and the others after the buffers that need it.
    Full matrices exist only on their assigned owners.
    The FSDP groups may use different meshes; owners are planned separately for each mesh.
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
        **kwargs: Keyword arguments passed unchanged to EO's ``Muon``.
    """

    _fsdp_group_assignments: list[_FsdpGroupAssignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._comm_stream = torch.cuda.Stream()

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
        # Experiment: move gathers and scatters with peer copies on the copy engines.
        self._exchanges: dict[FsdpParameterGroup, CopyEngineExchange] = {}
        if os.environ.get("FSDP_MUON_CE") == "1":
            for assignment in self._fsdp_group_assignments:
                if assignment.has_comm:
                    self._exchanges[assignment.fsdp_group] = CopyEngineExchange(
                        assignment.fsdp_group.main_weight, assignment.rank_ranges
                    )

    def _validate_parameter_groups(self) -> None:
        """Validate optimizer parameters and the FSDP buffers that contain them."""
        for assignment in self._fsdp_group_assignments:
            fsdp_group: FsdpParameterGroup = assignment.fsdp_group
            if fsdp_group.mesh.ndim != 1:
                raise NotImplementedError("FsdpMuon requires a one-dimensional DP mesh.")
            for buffer in (fsdp_group.main_weight, fsdp_group.main_grad):
                if not isinstance(buffer, DBuffer):
                    raise ValueError("FsdpMuon requires non-quantized main weights and gradients.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Update the sharded main weights."""
        if closure is not None:
            raise ValueError("closure is not supported")

        # Per FSDP group with gradients: its assignment, active tensors, and their owners.
        free: list[tuple[_FsdpGroupAssignment, dict[int, int], dict[int, int]]] = []
        comm: list[tuple[_FsdpGroupAssignment, dict[int, int], dict[int, int]]] = []
        for assignment in self._fsdp_group_assignments:
            fsdp_group: FsdpParameterGroup = assignment.fsdp_group
            active: dict[int, int] = {
                i: group_index
                for i, group_index in assignment.tensor_to_optimizer_group_index.items()
                if fsdp_group.fsdp_parameters[i].sharded.grad is not None
            }
            if not active:
                continue
            owners = {i: assignment.tensor_to_owner_rank[i] for i in active}
            (comm if assignment.has_comm else free).append((assignment, active, owners))

        # The gathers of the comm groups run on a side stream, everything else on the caller's, in
        # this order (N: Newton-Schulz, G: gather, S: scatter, A: apply the updates; a "free" group
        # needs no communication, a "comm" group does and is numbered i in the pipeline):
        #
        #   N(free[0]) | G(comm) | for each comm group i: wait for G(i), N(i), S(i) |
        #   N(free[1:]) | A(comm)
        #
        # The gathers start while N(free[0]) runs: on the caller's stream their momentum kernels
        # would queue behind it and the communication would start only after it. That group also
        # gives the GPU work while the host posts the gathers, which would otherwise leave it
        # idle. The scatters overlap the Newton-Schulz of the later groups, and the other free
        # groups hide the last ones. A free group has nothing to gather or scatter.
        self._comm_stream.wait_stream(torch.cuda.current_stream())  # The gradients are ready.
        for group in free[:1]:
            transfer = self._gather(*group)
            self._orthogonalize_updates(transfer)
            self._apply_scattered_updates(transfer)

        # The gathers of all groups with communication are in flight while the first is
        # orthogonalized, and so are the scatters while the later ones are, so communication
        # overlaps Newton-Schulz. Every rank posts them in the same order, whatever its other
        # groups are.
        gathers = [self._gather(*group) for group in comm]
        scatters: list[_Transfer] = []
        while gathers:
            gathered = gathers.pop(0)
            gathered.wait()
            self._orthogonalize_updates(gathered)
            scatters.append(self._scatter(gathered))
            del gathered  # Drop the gather, and with it the momentum inputs it holds.
        for group in free[1:]:
            transfer = self._gather(*group)
            self._orthogonalize_updates(transfer)
            self._apply_scattered_updates(transfer)
        for scattered in scatters:
            self._apply_scattered_updates(scattered)

    def _apply_scattered_updates(self, scattered: _Transfer) -> None:
        """Wait for a group's scatter, then apply its updates."""
        scattered.wait()
        self._apply_local_updates(
            scattered.assignment.fsdp_group,
            scattered.tensor_to_optimizer_group_index,
            scattered.full_updates,
            scattered.sharded,
        )

    def _gather(
        self,
        assignment: _FsdpGroupAssignment,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> _Transfer:
        """Compute momentum inputs and post their gather to the owners.

        A group with communication does both on the side stream. A group without has nothing to
        gather: the owner of a tensor is the only rank holding it, so its momentum input already is
        the full tensor, computed on the caller's stream.

        Args:
            assignment: The FSDP group to update and its cached assignments.
            tensor_to_optimizer_group_index: Active indices into ``fsdp_group.fsdp_parameters``
                mapped to indices into ``self.param_groups`` for their optimizer settings.
            tensor_to_owner: The same active tensor indices mapped to global owner ranks.

        Returns:
            The posted gather, holding the momentum inputs and the owners' full-tensor destinations.
        """
        fsdp_group: FsdpParameterGroup = assignment.fsdp_group
        this_rank: int = fsdp_group.mesh.get_rank()
        owned: list[int] = [i for i, owner in tensor_to_owner.items() if owner == this_rank]
        compute_stream = torch.cuda.current_stream()
        # The transfer is created on the stream that fills its buffers and records its progress.
        with torch.cuda.stream(self._comm_stream if assignment.has_comm else compute_stream):
            inputs = self._compute_momentum_inputs(fsdp_group, tensor_to_optimizer_group_index)
            works: list[dist.Work] = []
            sharded: DBuffer | None = None
            if assignment.has_comm:
                buffer: DBuffer = fsdp_group.main_weight
                exchange = self._exchanges.get(fsdp_group)
                sharded = DBuffer(
                    buffer.mesh,
                    buffer.placements,
                    buffer.layout,
                    buffer.dtype,
                    buffer.device,
                    local_buffer=None if exchange is None else exchange.source,
                )
                for i, local_input in inputs.items():
                    sharded.get_tensor_view(i).copy_(local_input)
                full_updates: dict[int, torch.Tensor] = {
                    i: torch.empty(
                        buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device
                    )
                    for i in owned
                }
                for update in full_updates.values():
                    update.record_stream(compute_stream)  # Newton-Schulz reads it on that stream.
                if exchange is None:
                    works = gather(
                        sharded,
                        full_updates,
                        tensor_to_owner=tensor_to_owner,
                        rank_ranges=assignment.rank_ranges,
                    )
                else:
                    exchange.gather(buffer.layout, full_updates, tensor_to_owner)
            else:
                full_updates = {i: inputs[i] for i in owned}
            return _Transfer(
                works=works,
                assignment=assignment,
                tensor_to_optimizer_group_index=tensor_to_optimizer_group_index,
                tensor_to_owner=tensor_to_owner,
                sharded=sharded,
                full_updates=full_updates,
            )

    def _scatter(self, gathered: _Transfer) -> _Transfer:
        """Post the scatter of a gathered group's orthogonalized updates.

        Returns:
            The same group's transfer, now holding the scatter works and the buffer that receives
            the updates of remote-owned tensors.
        """
        buffer: DBuffer = gathered.assignment.fsdp_group.main_weight
        exchange = self._exchanges.get(gathered.assignment.fsdp_group)
        if exchange is not None:
            compute_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._comm_stream):
                # Newton-Schulz, the last thing enqueued on the caller's stream, made the updates.
                self._comm_stream.wait_stream(compute_stream)
                local_updates = DBuffer(
                    buffer.mesh,
                    buffer.placements,
                    buffer.layout,
                    torch.float32,
                    buffer.device,
                    local_buffer=exchange.destination,
                )
                exchange.scatter(buffer.layout, gathered.full_updates, gathered.tensor_to_owner)
                return replace(gathered, works=[], sharded=local_updates)
        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        works = scatter(
            gathered.full_updates,
            local_updates,
            tensor_to_owner=gathered.tensor_to_owner,
            rank_ranges=gathered.assignment.rank_ranges,
        )
        return replace(gathered, works=works, sharded=local_updates)

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> dict[int, torch.Tensor]:
        """Apply local weight decay, update momentum, and return each tensor's local input.

        Without Nesterov the input is the momentum buffer itself, so callers must not modify it.
        """
        inputs: dict[int, torch.Tensor] = {}
        for i, group_index in tensor_to_optimizer_group_index.items():
            group: dict[str, Any] = self.param_groups[group_index]
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            state: dict[str, Any] = self.state[param]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(param)
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
            inputs[i] = torch.lerp(grad, momentum, group["momentum"]) if self.nesterov else momentum
        return inputs

    def _orthogonalize_updates(self, gathered: _Transfer) -> None:
        """Replace this rank's gathered inputs with full orthogonalized updates."""
        fsdp_group: FsdpParameterGroup = gathered.assignment.fsdp_group
        full_updates: dict[int, torch.Tensor] = gathered.full_updates
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for i, update in full_updates.items():
                param: torch.nn.Parameter = fsdp_group.fsdp_parameters[i].sharded
                group: dict[str, Any] = self.param_groups[
                    gathered.tensor_to_optimizer_group_index[i]
                ]
                kwargs: dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
                full_updates[i] = self.orthogonalize(param, update.float(), **kwargs).contiguous()

    def _apply_local_updates(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer | None,
    ) -> None:
        """Apply owned result slices and received updates to local parameter shards.

        `local_updates` is None for a group without communication: a tensor this rank does not own
        has no shard here and is skipped.
        """
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if i in full_updates:
                # Scatter writes only remote-owned shards; owners keep their full result.
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            elif local_updates is not None:
                update = local_updates.get_tensor_view(i)
            else:
                continue
            update = update.reshape(weight.shape).to(weight.dtype)
            # These hooks are inherited from EO's OrthogonalizedOptimizer, via Muon.
            self.pre_weight_update_fn_inplace(weight, update)
            weight.add_(update, alpha=-self.param_groups[group_index]["lr"])
            self.post_weight_update_fn_inplace(weight)
