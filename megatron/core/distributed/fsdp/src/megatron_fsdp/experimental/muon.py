# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

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
    fixed set of buffers and are computed during initialization. Each step
    gathers, orthogonalizes, scatters, and applies one buffer's updates at a time on
    the caller's CUDA stream. Full matrices exist only on their assigned owners.
    The mesh is inferred from the FSDP groups, which must all share the same mesh.
    Parameters and optimizer groups must have the same order on every DP rank, and
    ranks must agree on which parameters have gradients. Parameter membership and
    optimizer-group order must remain fixed after construction.

    Non-2D parameters are skipped. Use a separate optimizer for them and for matrices
    you exclude from Muon, such as embeddings and output projections. This optimizer
    refreshes its FSDP model weights itself, so it does not need
    ``fully_shard_optimizer``. Quantized buffers and tensor parallelism are unsupported.
    Weight decay is deferred: this custom step does not call EO's weight-decay
    implementation, and FSDP does not apply it elsewhere. It must therefore be zero.
    Momentum is initialized lazily as DTensors in the standard optimizer state.
    MCore's optimizer and distributed-checkpoint adapter is a separate integration.

    Args:
        params: FSDP-sharded parameters or optimizer parameter-group dictionaries.
        lr: Learning rate.
        momentum: Momentum coefficient.
        weight_decay: Must be zero, including in optimizer parameter-group dictionaries.
        **kwargs: Remaining keyword options accepted by EO's ``Muon``, such as ``nesterov``.
    """

    _fsdp_group_assignments: list[_FsdpGroupAssignment]

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

        # Preserve discovery order: communication must be posted identically on every rank.
        fsdp_groups: list[FsdpParameterGroup] = []
        param_to_optimizer_group_index: dict[torch.nn.Parameter, int] = {}
        for group_index, group in enumerate(self.param_groups):
            for param in group["params"]:
                # Callers may include biases and other non-matrices for another optimizer.
                if param.ndim != 2:
                    continue
                fsdp_group: FsdpParameterGroup | None = get_containing_parameter_group(param)
                if fsdp_group is None:
                    raise ValueError("FsdpMuon requires parameters sharded by MFSDP v2.")
                if fsdp_group not in fsdp_groups:
                    fsdp_groups.append(fsdp_group)
                param_to_optimizer_group_index[param] = group_index
        self._validate_param_groups(fsdp_groups)
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

    def _validate_param_groups(self, fsdp_groups: list[FsdpParameterGroup]) -> None:
        """Validate optimizer parameters and the FSDP buffers that contain them."""
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise NotImplementedError("FsdpMuon does not support nonzero weight decay yet.")
        for fsdp_group in fsdp_groups:
            if fsdp_group.mesh.ndim != 1:
                raise NotImplementedError("FsdpMuon requires a one-dimensional DP mesh.")
            for buffer in (fsdp_group.main_weight, fsdp_group.main_grad, fsdp_group.model_weight):
                if not isinstance(buffer, DBuffer):
                    raise ValueError("FsdpMuon requires non-quantized buffers.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Update sharded parameters and refresh their model weights."""
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise NotImplementedError("FsdpMuon does not support nonzero weight decay yet.")

        for assignment in self._fsdp_group_assignments:
            fsdp_group: FsdpParameterGroup = assignment.fsdp_group
            active: dict[int, int] = {
                i: group_index
                for i, group_index in assignment.tensor_to_optimizer_group_index.items()
                if fsdp_group.fsdp_parameters[i].sharded.grad is not None
            }
            if not active:
                continue
            self._step_group(
                fsdp_group, active, {i: assignment.tensor_to_owner_rank[i] for i in active}
            )
            fsdp_group.sync_model_weight_from_main_weight()

    def _step_group(
        self,
        fsdp_group: FsdpParameterGroup,
        tensor_to_optimizer_group_index: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> None:
        """Gather momentum, orthogonalize on owners, and scatter updates.

        Args:
            fsdp_group: FSDP buffers and parameters to update.
            tensor_to_optimizer_group_index: Active indices into ``fsdp_group.fsdp_parameters``
                mapped to indices into ``self.param_groups`` for their optimizer settings.
            tensor_to_owner: The same active tensor indices mapped to global owner ranks.
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
        gather(momentum_inputs, full_updates, tensor_to_owner=tensor_to_owner)
        self._orthogonalize_updates(fsdp_group, tensor_to_optimizer_group_index, full_updates)

        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        scatter(full_updates, local_updates, tensor_to_owner=tensor_to_owner)
        self._apply_local_updates(
            fsdp_group, tensor_to_optimizer_group_index, full_updates, local_updates
        )

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, tensor_to_optimizer_group_index: dict[int, int]
    ) -> DBuffer:
        """Update sharded momentum and build the inputs for orthogonalization."""
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
            # EO's OrthogonalizedOptimizer.step implements momentum inline, without a helper.
            momentum.lerp_(grad, 1 - group["momentum"])
            update: torch.Tensor = (
                grad.lerp(momentum, group["momentum"]) if self.nesterov else momentum
            )
            local_input.copy_(update.reshape(local_input.shape))
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
