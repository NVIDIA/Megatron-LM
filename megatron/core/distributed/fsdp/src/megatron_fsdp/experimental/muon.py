# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from .gather_scatter import gather, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group


@dataclass(frozen=True)
class _FsdpGroupAssignment:
    """An FSDP group and its cached optimizer-group and owner-rank assignments."""

    fsdp_group: FsdpParameterGroup
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner_rank: dict[int, int]
    has_comm: bool  # Whether any of the optimized tensors spans ranks.


@dataclass(frozen=True)
class _Transfer:
    """A posted gather or scatter of one FSDP group, with the buffers it uses.

    `sharded` is the DBuffer side of the transfer: the momentum inputs of a gather or the received
    updates of a scatter. `full_updates` holds the owners' full tensors on the other side.
    """

    works: list[dist.Work]
    fsdp_group: FsdpParameterGroup
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner: dict[int, int]
    sharded: DBuffer
    full_updates: dict[int, torch.Tensor]

    def wait(self) -> None:
        """Order the current stream after the transfer; the host does not block."""
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
    orthogonalizes, and scatters each buffer on the caller's CUDA stream. Transfers are
    asynchronous: before a buffer that needs communication is orthogonalized, the gathers of the
    next ``lookahead`` such buffers are posted, and its updates are applied after ``lookahead``
    later ones have been scattered, so each scatter overlaps their Newton-Schulz. Buffers whose
    tensors all sit on one rank need no communication; they run at both ends of the step.
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
        lookahead: Number of buffers that need communication whose gathers are posted ahead of
            the one being orthogonalized, and whose scatters are posted before a buffer's updates
            are applied. Each buffer in flight adds its momentum inputs, full tensors, and received
            updates to the peak memory. Zero gives the non-overlapped order.
        **kwargs: Keyword arguments passed unchanged to EO's ``Muon``.
    """

    _fsdp_group_assignments: list[_FsdpGroupAssignment]

    def __init__(self, *args: Any, lookahead: int = 8, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if lookahead < 0:
            raise ValueError(f"lookahead must be non-negative, got {lookahead}.")
        self.lookahead = lookahead

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
        for fsdp_group, plan in zip(
            fsdp_groups, assign_owner_work(group.main_weight for group in fsdp_groups), strict=True
        ):
            tensor_to_optimizer_group_index = {
                i: param_to_optimizer_group_index[fp.sharded]
                for i, fp in enumerate(fsdp_group.fsdp_parameters)
                if fp.sharded in param_to_optimizer_group_index
            }
            self._fsdp_group_assignments.append(
                _FsdpGroupAssignment(
                    fsdp_group=fsdp_group,
                    tensor_to_optimizer_group_index=tensor_to_optimizer_group_index,
                    tensor_to_owner_rank=plan.tensor_to_owner,
                    has_comm=not plan.split_tensors.isdisjoint(tensor_to_optimizer_group_index),
                )
            )
        self._validate_parameter_groups()

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

        # Per FSDP group with gradients: the group, its active tensors, and their owners.
        free: list[tuple[FsdpParameterGroup, dict[int, int], dict[int, int]]] = []
        comm: list[tuple[FsdpParameterGroup, dict[int, int], dict[int, int]]] = []
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
            (comm if assignment.has_comm else free).append((fsdp_group, active, owners))

        # A group without communication needs no gather, so its Newton-Schulz can hide the
        # communication of the others. Split those groups evenly between both ends: the first ones
        # hide the first gathers and the last ones hide the last scatters.
        front, back = free[: (len(free) + 1) // 2], free[(len(free) + 1) // 2 :]

        # The groups with communication form a software pipeline. While one is orthogonalized, the
        # gathers of the next `lookahead` groups and the scatters of the previous `lookahead`
        # groups are in flight, so communication overlaps Newton-Schulz and only those groups hold
        # buffers. Every rank posts them in the same order, whatever its other groups are.
        inflight_gathers: deque[_Transfer] = deque()
        inflight_scatters: deque[_Transfer] = deque()
        for group in front[:1]:
            self._orthogonalize_without_comm(*group)  # Runs while the gathers below are posted.
        inflight_gathers.extend(self._gather(*group) for group in comm[: self.lookahead])
        for group in front[1:]:
            self._orthogonalize_without_comm(*group)
        for i in range(len(comm)):
            if i + self.lookahead < len(comm):
                inflight_gathers.append(self._gather(*comm[i + self.lookahead]))
            gathered = inflight_gathers.popleft()
            gathered.wait()
            self._orthogonalize_updates(
                gathered.fsdp_group, gathered.tensor_to_optimizer_group_index, gathered.full_updates
            )
            inflight_scatters.append(self._scatter(gathered))
            del gathered  # Drop the gather, and with it the momentum inputs it holds.
            if len(inflight_scatters) > self.lookahead:
                self._apply_scattered_updates(inflight_scatters.popleft())
        for group in back:
            self._orthogonalize_without_comm(*group)
        for scattered in inflight_scatters:
            self._apply_scattered_updates(scattered)

    def _orthogonalize_without_comm(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> None:
        """Update a group none of whose tensors spans ranks, with no gather or scatter.

        The owner of a tensor is the only rank holding it, so its momentum input is already the
        full tensor: orthogonalize it directly.
        """
        this_rank: int = fsdp_group.mesh.get_rank()
        owned: dict[int, int] = {
            i: group_index
            for i, group_index in tensor_to_optimizer_group_index.items()
            if tensor_to_owner[i] == this_rank
        }
        inputs = self._compute_momentum_inputs(fsdp_group, tensor_to_optimizer_group_index)
        full_updates: dict[int, torch.Tensor] = {i: inputs[i] for i in owned}
        self._orthogonalize_updates(fsdp_group, owned, full_updates)
        self._apply_local_updates(fsdp_group, owned, full_updates, None)

    def _apply_scattered_updates(self, scattered: _Transfer) -> None:
        """Wait for a group's scatter, then apply its updates."""
        scattered.wait()
        self._apply_local_updates(
            scattered.fsdp_group,
            scattered.tensor_to_optimizer_group_index,
            scattered.full_updates,
            scattered.sharded,
        )

    def _gather(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> _Transfer:
        """Compute momentum inputs and post their gather to the owners.

        Args:
            fsdp_group: FSDP buffers and parameters to update.
            tensor_to_optimizer_group_index: Active indices into ``fsdp_group.fsdp_parameters``
                mapped to indices into ``self.param_groups`` for their optimizer settings.
            tensor_to_owner: The same active tensor indices mapped to global owner ranks.

        Returns:
            The posted gather, holding the momentum inputs and the owners' full-tensor destinations.
        """
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        for i, local_input in self._compute_momentum_inputs(
            fsdp_group, tensor_to_optimizer_group_index
        ).items():
            momentum_inputs.get_tensor_view(i).copy_(local_input)
        this_rank: int = fsdp_group.mesh.get_rank()
        full_updates: dict[int, torch.Tensor] = {
            i: torch.empty(buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device)
            for i, owner in tensor_to_owner.items()
            if owner == this_rank
        }
        works = gather(momentum_inputs, full_updates, tensor_to_owner=tensor_to_owner)
        return _Transfer(
            works=works,
            fsdp_group=fsdp_group,
            tensor_to_optimizer_group_index=tensor_to_optimizer_group_index,
            tensor_to_owner=tensor_to_owner,
            sharded=momentum_inputs,
            full_updates=full_updates,
        )

    def _scatter(self, gathered: _Transfer) -> _Transfer:
        """Post the scatter of a gathered group's orthogonalized updates.

        Returns:
            The same group's transfer, now holding the scatter works and the buffer that receives
            the updates of remote-owned tensors.
        """
        buffer: DBuffer = gathered.fsdp_group.main_weight
        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        works = scatter(
            gathered.full_updates, local_updates, tensor_to_owner=gathered.tensor_to_owner
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
        local_updates: DBuffer | None,
    ) -> None:
        """Apply owned result slices and received updates to local parameter shards.

        `local_updates` may be None when `full_updates` holds every tensor.
        """
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in tensor_to_optimizer_group_index.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if i in full_updates:
                # Scatter writes only remote-owned shards; owners keep their full result.
                offset: int = max(buffer.offset - buffer.layout.get_tensor_range(i).start, 0)
                update: torch.Tensor = full_updates[i].view(-1).narrow(0, offset, weight.numel())
            else:
                assert local_updates is not None
                update = local_updates.get_tensor_view(i)
            update = update.reshape(weight.shape).to(weight.dtype)
            # These hooks are inherited from EO's OrthogonalizedOptimizer, via Muon.
            self.pre_weight_update_fn_inplace(weight, update)
            weight.add_(update, alpha=-self.param_groups[group_index]["lr"])
            self.post_weight_update_fn_inplace(weight)
