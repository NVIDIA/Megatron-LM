# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from dataclasses import dataclass, replace
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

# Newton-Schulz temporaries grow with the batch, so a batch holds at most this much FP32 input.
_MAX_BATCH_BYTES = 1 << 30


def _batches_by_shape(tensors: dict[int, torch.Tensor]) -> list[list[int]]:
    """Group the tensors' indices by shape, in batches of at most ``_MAX_BATCH_BYTES`` each."""
    by_shape: dict[torch.Size, list[int]] = {}
    for i, tensor in tensors.items():
        by_shape.setdefault(tensor.shape, []).append(i)
    batches: list[list[int]] = []
    for shape, indices in by_shape.items():
        size = max(1, _MAX_BATCH_BYTES // (shape.numel() * 4))
        batches += [indices[start : start + size] for start in range(0, len(indices), size)]
    return batches


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
class _GroupStep:
    """One FSDP group's part of an optimizer step: its posted gather or scatter and the buffers.

    `sharded` is the DBuffer side of the transfer: the momentum inputs of a gather or the received
    updates of a scatter. `full_updates` holds the owners' full tensors on the other side. A group
    without communication has no works, and its full tensors are views of `sharded`.
    """

    works: list[dist.Work]
    assignment: _FsdpGroupAssignment
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
        # Same-shape tensors share one Newton-Schulz unless a subclass orthogonalizes differently
        # or SYRK is on, whose kernel takes 2-D inputs only.
        self._batch_same_shape = type(self).orthogonalize is Muon.orthogonalize and not kwargs.get(
            "use_syrk", False
        )

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

        # Per FSDP group with gradients: its assignment, active tensors, and their owners. A "free"
        # group needs no communication, a "comm" group does.
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

        # Spread the free groups evenly over the comm groups, the last one at the end: their
        # Newton-Schulz hides the transfers in flight, which the comm groups' own may be too short
        # for. A free group has nothing to scatter and is applied at once.
        positions = [(k + 1) / len(free) for k in range(len(free))]
        positions += [(i + 0.5) / len(comm) for i in range(len(comm))]
        order = [group for _, group in sorted(zip(positions, free + comm), key=lambda p: p[0])]

        # n = num_groups_in_flight; N: Newton-Schulz, G: gather, S: scatter, A: apply.
        #
        #   G(order[:n]) | for each group i: G(i + n), N(i), S(i), A(i - n)
        #
        # A group step is set to None once used, so only the groups in flight hold buffers.
        in_flight_gathers: list[_GroupStep | None] = [
            self._gather(*group) for group in order[: self.num_groups_in_flight]
        ]
        in_flight_scatters: list[_GroupStep | None] = []
        for i in range(len(order)):
            if i + self.num_groups_in_flight < len(order):
                in_flight_gathers.append(self._gather(*order[i + self.num_groups_in_flight]))
            in_flight_gathers[i].wait()
            self._orthogonalize_updates(in_flight_gathers[i])
            if in_flight_gathers[i].assignment.has_comm:
                in_flight_scatters.append(self._scatter(in_flight_gathers[i]))
            else:
                self._apply_updates(in_flight_gathers[i])
                in_flight_scatters.append(None)
            # Drop the gather, and with it the momentum inputs it holds.
            in_flight_gathers[i] = None
            if i >= self.num_groups_in_flight and in_flight_scatters[i - self.num_groups_in_flight]:
                self._apply_updates(in_flight_scatters[i - self.num_groups_in_flight])
                in_flight_scatters[i - self.num_groups_in_flight] = None
        for group_step in in_flight_scatters:
            if group_step is not None:
                self._apply_updates(group_step)

    def _gather(
        self,
        assignment: _FsdpGroupAssignment,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> _GroupStep:
        """Compute momentum inputs and post their gather to the owners.

        A group without communication has nothing to gather: the owner of a tensor is the only
        rank holding it, so its momentum input already is the full tensor.

        Args:
            assignment: The FSDP group to update and its cached assignments.
            tensor_to_optimizer_group_index: Active indices into ``fsdp_group.fsdp_parameters``
                mapped to indices into ``self.param_groups`` for their optimizer settings.
            tensor_to_owner: The same active tensor indices mapped to global owner ranks.

        Returns:
            The posted gather, holding the momentum inputs and the owners' full-tensor destinations.
        """
        fsdp_group: FsdpParameterGroup = assignment.fsdp_group
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = self._compute_momentum_inputs(
            fsdp_group, tensor_to_optimizer_group_index
        )
        this_rank: int = fsdp_group.mesh.get_rank()
        owned: list[int] = [i for i, owner in tensor_to_owner.items() if owner == this_rank]
        works: list[dist.Work] = []
        if assignment.has_comm:
            full_updates: dict[int, torch.Tensor] = {
                i: torch.empty(
                    buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device
                )
                for i in owned
            }
            works = gather(
                momentum_inputs,
                full_updates,
                tensor_to_owner=tensor_to_owner,
                rank_ranges=assignment.rank_ranges,
            )
        else:
            full_updates = {
                i: momentum_inputs.get_tensor_view(i).view(buffer.layout.tensor_shapes[i])
                for i in owned
            }
        return _GroupStep(
            works=works,
            assignment=assignment,
            tensor_to_optimizer_group_index=tensor_to_optimizer_group_index,
            tensor_to_owner=tensor_to_owner,
            sharded=momentum_inputs,
            full_updates=full_updates,
        )

    def _scatter(self, group_step: _GroupStep) -> _GroupStep:
        """Post the scatter of a gathered group's orthogonalized updates.

        Returns:
            The same group's step, now holding the scatter works and the buffer that receives
            the updates of remote-owned tensors.
        """
        buffer: DBuffer = group_step.assignment.fsdp_group.main_weight
        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        works = scatter(
            group_step.full_updates,
            local_updates,
            tensor_to_owner=group_step.tensor_to_owner,
            rank_ranges=group_step.assignment.rank_ranges,
        )
        return replace(group_step, works=works, sharded=local_updates)

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Apply local weight decay, update momentum, and build inputs for orthogonalization.

        Tensors sharing an optimizer group are updated together with foreach kernels, a few
        launches per group instead of a few per tensor.
        """
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        by_optimizer_group: dict[int, list[int]] = {}
        for i, group_index in tensor_to_optimizer_group_index.items():
            by_optimizer_group.setdefault(group_index, []).append(i)
        for group_index, indices in by_optimizer_group.items():
            group: dict[str, Any] = self.param_groups[group_index]
            weights: list[torch.Tensor] = []
            momenta: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            local_inputs: list[torch.Tensor] = []
            for i in indices:
                param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
                assert isinstance(param.grad, DTensor)
                state: dict[str, Any] = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum: torch.Tensor = state["momentum_buffer"].to_local()
                momenta.append(momentum)
                grads.append(param.grad.to_local().to(momentum.dtype).reshape(momentum.shape))
                weights.append(param.to_local())
                local_inputs.append(momentum_inputs.get_tensor_view(i).view_as(momentum))
            # Match EO's decay before momentum without calling its private helper.
            weight_decay: float = group["weight_decay"]
            if weight_decay != 0:
                lr: float = group["lr"]
                match self.weight_decay_method:
                    case "decoupled":
                        torch._foreach_add_(weights, weights, alpha=-(weight_decay * lr))
                    case "independent":
                        torch._foreach_add_(weights, weights, alpha=-weight_decay)
                    case "l2":
                        torch._foreach_add_(grads, weights, alpha=weight_decay)
                    case "palm":
                        torch._foreach_add_(weights, weights, alpha=-(weight_decay * lr * lr))
                    case _:
                        raise ValueError(f"Invalid weight decay method: {self.weight_decay_method}")
            # EO's OrthogonalizedOptimizer.step implements momentum inline, without a helper.
            torch._foreach_lerp_(momenta, grads, 1 - group["momentum"])
            if self.nesterov:
                # lerp(grad, momentum, momentum coefficient), written in place.
                torch._foreach_copy_(local_inputs, grads)
                torch._foreach_lerp_(local_inputs, momenta, group["momentum"])
            else:
                torch._foreach_copy_(local_inputs, momenta)
        return momentum_inputs

    def _orthogonalize_updates(self, group_step: _GroupStep) -> None:
        """Replace this rank's gathered inputs with full orthogonalized updates.

        Tensors of the same shape run one batched Newton-Schulz: a small matrix's Newton-Schulz is
        dozens of short kernels, too many for the host to launch one matrix at a time. A tensor
        alone with its shape keeps the per-tensor ``orthogonalize`` call.
        """
        fsdp_group: FsdpParameterGroup = group_step.assignment.fsdp_group
        tensor_to_optimizer_group_index: dict[int, int] = group_step.tensor_to_optimizer_group_index
        full_updates: dict[int, torch.Tensor] = group_step.full_updates
        batches: list[list[int]] = (
            _batches_by_shape(full_updates)
            if self._batch_same_shape
            else [[i] for i in full_updates]
        )
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for batch in batches:
                if len(batch) > 1:
                    inputs = torch.stack([full_updates[i].float() for i in batch])
                    updates = self.scaled_orthogonalize_fn(inputs).contiguous()
                    full_updates.update(zip(batch, updates.unbind()))
                    continue
                (i,) = batch
                param: torch.nn.Parameter = fsdp_group.fsdp_parameters[i].sharded
                group: dict[str, Any] = self.param_groups[tensor_to_optimizer_group_index[i]]
                kwargs: dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
                full_updates[i] = self.orthogonalize(
                    param, full_updates[i].float(), **kwargs
                ).contiguous()

    def _apply_updates(self, group_step: _GroupStep) -> None:
        """Wait for a group's scatter, then apply owned result slices and received updates."""
        group_step.wait()
        fsdp_group: FsdpParameterGroup = group_step.assignment.fsdp_group
        full_updates: dict[int, torch.Tensor] = group_step.full_updates
        local_updates: DBuffer = group_step.sharded
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in group_step.tensor_to_optimizer_group_index.items():
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
