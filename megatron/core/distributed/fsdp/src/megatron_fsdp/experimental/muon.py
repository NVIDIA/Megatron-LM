# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
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
from .gather_scatter import gather, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group


@dataclass(frozen=True)
class _FsdpGroupAssignment:
    """An FSDP group and its cached optimizer-group and owner-rank assignments."""

    fsdp_group: FsdpParameterGroup
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner_rank: dict[int, int]


@dataclass(frozen=True)
class _GroupEvents:
    """Completion events for the three ordered stages of one group's update."""

    gather: torch.cuda.Event
    orthogonalize: torch.cuda.Event
    scatter: torch.cuda.Event


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
    fixed set of buffers and are computed during initialization. Each group's momentum, gather,
    orthogonalization, scatter, and weight update run in sequence on one stream from a round-robin
    pool. Events order each gather, orthogonalization, and scatter after the same stage of the
    previous active group. The caller's stream waits for all group streams before continuing.
    Batched P2P still uses one internal NCCL stream, so this posting order can serialize
    communication with orthogonalization even though groups use separate caller streams.
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
        num_groups_in_flight: Number of group streams in the round-robin pool, capped at the
            number of FSDP groups. Zero runs groups serially on the caller's stream. This trial
            controls stream count rather than gather lookahead; pending NCCL work can delay
            temporary-buffer reuse even when a stream is reused.
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
                )
            )
        self._validate_parameter_groups()
        self._group_streams = [
            torch.cuda.Stream(device=self._fsdp_group_assignments[0].fsdp_group.main_weight.device)
            for _ in range(min(num_groups_in_flight, len(self._fsdp_group_assignments)))
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

        if not self._fsdp_group_assignments:
            return
        device = self._fsdp_group_assignments[0].fsdp_group.main_weight.device
        caller_stream = torch.cuda.current_stream(device)
        streams = self._group_streams or [caller_stream]
        for stream in streams:
            stream.wait_stream(caller_stream)

        previous: _GroupEvents | None = None
        for group_index, assignment in enumerate(self._fsdp_group_assignments):
            fsdp_group = assignment.fsdp_group
            group_indices = assignment.tensor_to_optimizer_group_index
            active = {
                i: optimizer_group_index
                for i, optimizer_group_index in group_indices.items()
                if fsdp_group.fsdp_parameters[i].sharded.grad is not None
            }
            if not active:
                continue
            owners = {i: assignment.tensor_to_owner_rank[i] for i in active}
            # Use the cached group index so skipping inactive groups does not move state
            # between streams on subsequent steps.
            with torch.cuda.stream(streams[group_index % len(streams)]):
                previous = self._step_group(fsdp_group, active, owners, previous)
        for stream in streams:
            caller_stream.wait_stream(stream)

    def _step_group(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
        previous: _GroupEvents | None,
    ) -> _GroupEvents:
        """Gather, orthogonalize, scatter, and apply one group's update on its current stream."""
        buffer: DBuffer = fsdp_group.main_weight
        stream = torch.cuda.current_stream(buffer.device)
        momentum_inputs: DBuffer = self._compute_momentum_inputs(
            fsdp_group, tensor_to_optimizer_group_index
        )
        this_rank: int = fsdp_group.mesh.get_rank()
        full_updates: dict[int, torch.Tensor] = {
            i: torch.empty(buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device)
            for i, owner in tensor_to_owner.items()
            if owner == this_rank
        }
        if previous is not None:
            stream.wait_event(previous.gather)
        gather(momentum_inputs, full_updates, tensor_to_owner=tensor_to_owner)
        gather_done = stream.record_event()

        if previous is not None:
            stream.wait_event(previous.orthogonalize)
        self._orthogonalize_updates(fsdp_group, tensor_to_optimizer_group_index, full_updates)
        orthogonalize_done = stream.record_event()

        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        if previous is not None:
            stream.wait_event(previous.scatter)
        scatter(full_updates, local_updates, tensor_to_owner=tensor_to_owner)
        scatter_done = stream.record_event()
        self._apply_local_updates(
            fsdp_group, tensor_to_optimizer_group_index, full_updates, local_updates
        )
        return _GroupEvents(gather_done, orthogonalize_done, scatter_done)

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Apply local weight decay, update momentum, and build inputs for orthogonalization."""
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        stream = torch.cuda.current_stream(buffer.device)
        for i, group_index in tensor_to_optimizer_group_index.items():
            group: dict[str, Any] = self.param_groups[group_index]
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            state: dict[str, Any] = self.state[param]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(param)
            local_input: torch.Tensor = momentum_inputs.get_tensor_view(i)
            momentum: torch.Tensor = state["momentum_buffer"].to_local()
            local_grad = param.grad.to_local()
            weight: torch.Tensor = param.to_local()
            # State restored from a checkpoint and caller-owned weights/gradients may have
            # been allocated on other streams. Keep their storage alive for this group's use.
            for tensor in (weight, local_grad, momentum):
                tensor.record_stream(stream)
            grad: torch.Tensor = local_grad.to(momentum.dtype).reshape(momentum.shape)
            # Match EO's decay before momentum without calling its private helper.
            weight_decay: float = group["weight_decay"]
            if weight_decay != 0:
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
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        full_updates: dict[int, torch.Tensor],
    ) -> None:
        """Replace this rank's gathered inputs with full orthogonalized updates."""
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for i, update in full_updates.items():
                param: torch.nn.Parameter = fsdp_group.fsdp_parameters[i].sharded
                group: dict[str, Any] = self.param_groups[tensor_to_optimizer_group_index[i]]
                kwargs: dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
                full_updates[i] = self.orthogonalize(param, update.float(), **kwargs).contiguous()

    def _apply_local_updates(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer,
    ) -> None:
        """Apply owned result slices and received updates to local parameter shards."""
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if i in full_updates:
                # Scatter writes only remote-owned shards; owners keep their full result.
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            else:
                update = local_updates.get_tensor_view(i)
            update = update.reshape(weight.shape).to(weight.dtype)
            # These hooks are inherited from EO's OrthogonalizedOptimizer, via Muon.
            self.pre_weight_update_fn_inplace(weight, update)
            weight.add_(update, alpha=-self.param_groups[group_index]["lr"])
            self.post_weight_update_fn_inplace(weight)
