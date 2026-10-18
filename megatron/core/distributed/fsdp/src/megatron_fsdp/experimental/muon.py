# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

try:
    import emerging_optimizers  # pylint: disable=unused-import
except ImportError as error:
    raise ImportError(
        "FsdpMuon requires Emerging-Optimizers. "
        "Install it with `pip install 'megatron-fsdp[emerging-optimizers]'`."
    ) from error

from emerging_optimizers import utils as eo_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from .dbuffer import DBuffer
from .gather_scatter import gather, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import RowAtomic


class FsdpMuon(Muon):
    """Muon for 2-D parameters on a one-dimensional, all-RowAtomic DP mesh.

    Momentum remains sharded. Owner assignments balance orthogonalization across a
    fixed set of buffers and are computed during initialization. Each step
    gathers, orthogonalizes, scatters, and applies one buffer's updates at a time on
    the caller's CUDA stream. Full matrices exist only on their assigned owners.
    Parameters and optimizer groups must have the same order on every DP rank, and
    ranks must agree on which parameters have gradients.

    Non-2D parameters are skipped. Use a separate optimizer for them and for matrices
    you exclude from Muon, such as embeddings and output projections. This optimizer
    refreshes its FSDP model weights itself, so it does not need
    ``fully_shard_optimizer``. Quantized buffers and tensor parallelism are unsupported.
    Weight decay defaults to zero; nonzero decay is unsupported.

    Args:
        params: FSDP-sharded parameters or optimizer parameter-group dictionaries.
        dp_mesh: The one-dimensional device mesh used to shard these parameters.
        **kwargs: Hyperparameters accepted by Emerging Optimizers' ``Muon``, including
            ``lr``, ``momentum``, and ``nesterov``. ``weight_decay`` must be zero,
            including in optimizer parameter-group dictionaries.
    """

    dp_mesh: DeviceMesh
    _fsdp_groups: list[FsdpParameterGroup]
    _owner_maps: list[dict[int, int]]

    def __init__(self, params: ParamsT, dp_mesh: DeviceMesh, **kwargs: Any) -> None:
        if dp_mesh.ndim != 1:
            raise NotImplementedError("FsdpMuon requires a one-dimensional DP mesh.")
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(params, **kwargs)
        self.dp_mesh = dp_mesh
        self._validate_param_groups()

        # Preserve discovery order: communication must be posted identically on every rank.
        self._fsdp_groups = []
        seen: set[FsdpParameterGroup] = set()
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    continue
                fsdp_group: FsdpParameterGroup | None = get_containing_parameter_group(param)
                assert fsdp_group is not None
                if fsdp_group not in seen:
                    seen.add(fsdp_group)
                    self._fsdp_groups.append(fsdp_group)
        self._owner_maps = assign_owner_work(group.main_weight for group in self._fsdp_groups)

    def _validate_param_groups(self) -> None:
        """Validate optimizer parameters and the FSDP buffers that contain them."""
        seen: set[FsdpParameterGroup] = set()
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise NotImplementedError("FsdpMuon does not support nonzero weight decay yet.")
            for param in group["params"]:
                if param.ndim != 2:
                    continue
                fsdp_group: FsdpParameterGroup | None = get_containing_parameter_group(param)
                if fsdp_group is None:
                    raise ValueError("FsdpMuon requires parameters sharded by MFSDP v2.")
                if fsdp_group in seen:
                    continue
                seen.add(fsdp_group)
                if fsdp_group.mesh is not self.dp_mesh:
                    raise ValueError("FsdpMuon parameters must use the optimizer's dp_mesh.")
                for buffer in (
                    fsdp_group.main_weight,
                    fsdp_group.main_grad,
                    fsdp_group.model_weight,
                ):
                    if not isinstance(buffer, DBuffer):
                        raise ValueError("FsdpMuon requires non-quantized buffers.")
                    if not all(isinstance(p, RowAtomic) for p in buffer.placements):
                        raise ValueError("FsdpMuon requires all-RowAtomic placements.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Update sharded parameters and refresh their model weights."""
        if closure is not None:
            raise ValueError("closure is not supported")

        param_to_group_index: dict[torch.nn.Parameter, int] = {}
        for group_index, group in enumerate(self.param_groups):
            if group["weight_decay"] != 0:
                raise NotImplementedError("FsdpMuon does not support nonzero weight decay yet.")
            for param in group["params"]:
                if param.ndim != 2 or param.grad is None:
                    continue
                state: dict[str, Any] = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                param_to_group_index[param] = group_index

        for fsdp_group, owners in zip(self._fsdp_groups, self._owner_maps, strict=True):
            group_indices: dict[int, int] = {
                i: param_to_group_index[fp.sharded]
                for i, fp in enumerate(fsdp_group.fsdp_parameters)
                if fp.sharded in param_to_group_index
            }
            if not group_indices:
                continue
            self._step_group(fsdp_group, group_indices, {i: owners[i] for i in group_indices})
            fsdp_group.sync_model_weight_from_main_weight()

    def _step_group(
        self,
        fsdp_group: FsdpParameterGroup,
        group_indices: dict[int, int],
        tensor_to_owner: dict[int, int],
    ) -> None:
        """Gather momentum, orthogonalize on owners, and scatter updates.

        Args:
            fsdp_group: FSDP buffers and parameters to update.
            group_indices: Active indices into ``fsdp_group.fsdp_parameters`` mapped
                to indices into ``self.param_groups`` for their optimizer settings.
            tensor_to_owner: The same active tensor indices mapped to global owner ranks.
        """
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = self._compute_momentum_inputs(fsdp_group, group_indices)
        this_rank: int = self.dp_mesh.get_rank()
        full_updates: dict[int, torch.Tensor] = {
            i: torch.empty(buffer.layout.tensor_shapes[i], dtype=buffer.dtype, device=buffer.device)
            for i, owner in tensor_to_owner.items()
            if owner == this_rank
        }
        gather(momentum_inputs, full_updates, tensor_to_owner=tensor_to_owner)
        self._orthogonalize_updates(fsdp_group, group_indices, full_updates)

        local_updates: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        scatter(full_updates, local_updates, tensor_to_owner=tensor_to_owner)
        self._apply_local_updates(fsdp_group, group_indices, full_updates, local_updates)

    def _compute_momentum_inputs(
        self, fsdp_group: FsdpParameterGroup, group_indices: dict[int, int]
    ) -> DBuffer:
        """Update sharded momentum and build the inputs for orthogonalization."""
        buffer: DBuffer = fsdp_group.main_weight
        momentum_inputs: DBuffer = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device
        )
        for i, group_index in group_indices.items():
            group: dict[str, Any] = self.param_groups[group_index]
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            local_input: torch.Tensor = momentum_inputs.get_tensor_view(i)
            momentum: torch.Tensor = self.state[param]["momentum_buffer"].to_local()
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
        group_indices: dict[int, int],
        full_updates: dict[int, torch.Tensor],
    ) -> None:
        """Replace this rank's gathered inputs with full orthogonalized updates."""
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for i, update in full_updates.items():
                param: torch.nn.Parameter = fsdp_group.fsdp_parameters[i].sharded
                group: dict[str, Any] = self.param_groups[group_indices[i]]
                kwargs: dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
                full_updates[i] = self.orthogonalize(param, update.float(), **kwargs).contiguous()

    def _apply_local_updates(
        self,
        fsdp_group: FsdpParameterGroup,
        group_indices: dict[int, int],
        full_updates: dict[int, torch.Tensor],
        local_updates: DBuffer,
    ) -> None:
        """Apply owned result slices and received updates to local parameter shards."""
        buffer: DBuffer = fsdp_group.main_weight
        for i, group_index in group_indices.items():
            param: DTensor = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight: torch.Tensor = param.to_local()
            if weight.numel() == 0:
                continue
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
