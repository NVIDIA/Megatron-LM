# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

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
from .gather_scatter import find_split_tensors, gather, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group


@dataclass(frozen=True)
class _FsdpGroupAssignment:
    """An FSDP group and its cached optimizer-group and owner-rank assignments."""

    fsdp_group: FsdpParameterGroup
    tensor_to_optimizer_group_index: dict[int, int]
    tensor_to_owner_rank: dict[int, int]


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
    asynchronous: after a buffer is orthogonalized, the gathers of the next ``lookahead`` buffers
    that need communication are posted, and such a buffer's updates are applied after
    ``lookahead`` later ones have been scattered, so each scatter overlaps their Newton-Schulz.
    A buffer whose tensors all sit on one rank needs no communication and is handled when its
    turn comes.
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
        # Tensors that span ranks need communication; a group with none of them has none.
        self._split_tensors: dict[FsdpParameterGroup, frozenset[int]] = {
            a.fsdp_group: find_split_tensors(a.fsdp_group.main_weight)
            for a in self._fsdp_group_assignments
        }

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

        groups: list[tuple[FsdpParameterGroup, dict[int, int], dict[int, int]]] = []
        has_comm: list[bool] = []
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
            groups.append((fsdp_group, active, owners))
            has_comm.append(not self._split_tensors[fsdp_group].isdisjoint(active))

        # The Newton-Schulz of a group without communication needs no gather. Split those groups
        # evenly between both ends: the first ones hide the first gathers and the last ones hide
        # the last scatters. The groups with communication keep their order in between.
        free = [i for i in range(len(groups)) if not has_comm[i]]
        comm = [i for i in range(len(groups)) if has_comm[i]]
        half = (len(free) + 1) // 2
        order = free[:half] + comm + free[half:]
        groups = [groups[i] for i in order]
        has_comm = [has_comm[i] for i in order]

        # Software pipeline over groups. While a group is orthogonalized, the gathers of the next
        # `lookahead` groups with communication and the scatters of the previous `lookahead` ones
        # are in flight, so communication overlaps Newton-Schulz and only those groups hold
        # buffers. A group without communication computes its momentum when its turn comes and
        # applies its updates right after, so it holds nothing in between. The gathers are posted
        # after the Newton-Schulz of the current group is enqueued, so a leading group without
        # communication runs while they are posted. Every rank posts the transfers of the groups
        # with communication in the same order, whatever its groups without communication are.
        gathers: dict[int, _Transfer] = {}
        scatters: list[_Transfer] = []
        ahead = 0  # Groups before this position are gathered or need no gather ahead of time.
        for i, group in enumerate(groups):
            gathered = gathers.pop(i, None) or self._gather(*group)
            gathered.wait()
            self._orthogonalize_updates(
                gathered.fsdp_group, gathered.tensor_to_optimizer_group_index, gathered.full_updates
            )
            ahead = max(ahead, i + 1)
            while ahead < len(groups) and len(gathers) < self.lookahead:
                if has_comm[ahead]:
                    gathers[ahead] = self._gather(*groups[ahead])
                ahead += 1
            if has_comm[i]:
                scatters.append(self._scatter(gathered))
                if len(scatters) > self.lookahead:
                    self._apply_scattered_updates(scatters.pop(0))
            else:
                self._apply_scattered_updates(self._scatter(gathered))
            del gathered  # Drop the gather, and with it the momentum inputs it holds.
        for scattered in scatters:
            self._apply_scattered_updates(scattered)

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
        momentum_inputs: DBuffer = self._compute_momentum_inputs(
            fsdp_group, tensor_to_optimizer_group_index
        )
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
