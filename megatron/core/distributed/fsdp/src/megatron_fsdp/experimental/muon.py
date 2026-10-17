# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with sharded momentum and owner-compute updates for MFSDP v2."""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT

try:
    from emerging_optimizers import utils as eo_utils
    from emerging_optimizers.orthogonalized_optimizers.muon import Muon
except ImportError as error:
    raise ImportError(
        "FsdpMuon requires Emerging-Optimizers. "
        "Install it with `pip install 'megatron-fsdp[emerging-optimizers]'`."
    ) from error

from .dbuffer import DBuffer
from .gather_scatter import gather, scatter
from .owner_planning import assign_owner_work
from .parameter_group import FsdpParameterGroup, get_containing_parameter_group
from .placement import RowAtomic
from .range import intersect_ranges


def _fsdp_group_of(param: torch.nn.Parameter) -> FsdpParameterGroup:
    group = get_containing_parameter_group(param)
    if group is None:
        raise ValueError("FsdpMuon requires parameters sharded by MFSDP v2.")
    return group


class FsdpMuon(Muon):
    """Muon for 2-D parameters on a one-dimensional, all-RowAtomic DP mesh.

    Momentum remains sharded. Each step balances orthogonalization across the buffers,
    then gathers, orthogonalizes, scatters, and applies one buffer's updates at a time
    on the caller's CUDA stream. Full matrices exist only on their assigned owners.
    Parameters and optimizer groups must have the same order on every DP rank, and
    ranks must agree on which parameters have gradients.

    Pass only the matrices to optimize with Muon; use a separate optimizer for other
    parameters, including biases, norms, embeddings, and output projections. This
    optimizer refreshes its FSDP model weights itself, so it does not need
    ``fully_shard_optimizer``. Quantized buffers and tensor parallelism are unsupported.

    Args:
        params: FSDP-sharded parameters or optimizer parameter-group dictionaries.
        dp_mesh: The one-dimensional device mesh used to shard these parameters.
        **kwargs: Hyperparameters accepted by Emerging Optimizers' ``Muon``, including
            ``lr``, ``momentum``, ``weight_decay``, and ``nesterov``.
    """

    def __init__(self, params: ParamsT, dp_mesh: DeviceMesh, **kwargs: Any) -> None:
        if dp_mesh.ndim != 1:
            raise NotImplementedError("FsdpMuon requires a one-dimensional DP mesh.")
        super().__init__(params, **kwargs)
        self.dp_mesh = dp_mesh
        seen = set()
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError("FsdpMuon only supports 2-D parameters.")
                fsdp_group = _fsdp_group_of(param)
                if fsdp_group in seen:
                    continue
                seen.add(fsdp_group)
                if fsdp_group.mesh is not dp_mesh:
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

        # Preserve discovery order: communication must be posted identically on every rank.
        active: dict[FsdpParameterGroup, dict[torch.nn.Parameter, dict[str, Any]]] = {}
        for group in self.param_groups:
            self._init_group(group)
            for param in group["params"]:
                if param.grad is not None:
                    active.setdefault(_fsdp_group_of(param), {})[param] = group

        # Count each buffer once, even when it contains multiple optimizer groups.
        # The planner estimates all matrices' work; only active parameters communicate.
        owner_maps = assign_owner_work(group.main_weight for group in active)
        for (fsdp_group, param_groups), owners in zip(active.items(), owner_maps, strict=True):
            groups_by_index = {
                i: param_groups[fp.sharded]
                for i, fp in enumerate(fsdp_group.fsdp_parameters)
                if fp.sharded in param_groups
            }
            self._step_group(fsdp_group, groups_by_index, {i: owners[i] for i in groups_by_index})
            fsdp_group.sync_model_weight_from_main_weight()

    def _step_group(
        self,
        fsdp_group: FsdpParameterGroup,
        groups_by_index: dict[int, dict[str, Any]],
        tensor_to_owner: dict[int, int],
    ) -> None:
        """Gather momentum inputs, compute full updates on owners, and apply local shards."""
        buffer = fsdp_group.main_weight
        source = DBuffer(buffer.mesh, buffer.placements, buffer.layout, buffer.dtype, buffer.device)
        for i, group in groups_by_index.items():
            param = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            assert isinstance(param.grad, DTensor)
            local = source.get_tensor_view(i)
            if local.numel() == 0:
                continue
            weight = param.to_local()
            momentum = self.state[param]["momentum_buffer"].to_local()
            grad = param.grad.to_local().to(momentum.dtype).reshape(momentum.shape)
            self._apply_weight_decay_inplace(weight, grad, group["lr"], group["weight_decay"])
            momentum.lerp_(grad, 1 - group["momentum"])
            inputs = grad.lerp(momentum, group["momentum"]) if self.nesterov else momentum
            local.copy_(inputs.reshape(local.shape))

        this_rank = self.dp_mesh.get_rank()
        full_inputs = {
            i: torch.empty(buffer.layout.tensor_shapes[i], dtype=source.dtype, device=source.device)
            for i, owner in tensor_to_owner.items()
            if owner == this_rank
        }
        gather(source, full_inputs, tensor_to_owner=tensor_to_owner)
        updates = {}
        with eo_utils.fp32_matmul_precision(self.fp32_matmul_prec):
            for i, inputs in full_inputs.items():
                param = fsdp_group.fsdp_parameters[i].sharded
                kwargs = {k: v for k, v in groups_by_index[i].items() if k != "params"}
                updates[i] = self.orthogonalize(param, inputs.float(), **kwargs).contiguous()

        scratch = DBuffer(
            buffer.mesh, buffer.placements, buffer.layout, torch.float32, buffer.device
        )
        scatter(updates, scratch, tensor_to_owner=tensor_to_owner)
        local_range = buffer.layout.get_local_range(buffer.mesh, buffer.placements)
        for i, owner in tensor_to_owner.items():
            tensor_range = buffer.layout.get_tensor_range(i)
            shard_range = intersect_ranges(tensor_range, local_range)
            if shard_range.numel == 0:
                continue
            if owner == this_rank:
                # Scatter writes only remote-owned shards; owners keep their full result.
                offset = shard_range.start - tensor_range.start
                update = updates[i].view(-1).narrow(0, offset, shard_range.numel)
            else:
                update = scratch.get_tensor_view(i)
            param = cast(DTensor, fsdp_group.fsdp_parameters[i].sharded)
            weight = param.to_local()
            update = update.reshape(weight.shape).to(weight.dtype)
            self.pre_weight_update_fn_inplace(weight, update)
            weight.add_(update, alpha=-groups_by_index[i]["lr"])
            self.post_weight_update_fn_inplace(weight)
