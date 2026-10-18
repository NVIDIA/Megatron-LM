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
from .range import intersect_ranges


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

    def __init__(self, params: ParamsT, dp_mesh: DeviceMesh, **kwargs: Any) -> None:
        if dp_mesh.ndim != 1:
            raise NotImplementedError("FsdpMuon requires a one-dimensional DP mesh.")
        kwargs.setdefault("weight_decay", 0.0)
        super().__init__(params, **kwargs)
        self.dp_mesh = dp_mesh
        self._validate_param_groups()

        # Preserve discovery order: communication must be posted identically on every rank.
        self._fsdp_groups: list[FsdpParameterGroup] = []
        seen = set()
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    continue
                fsdp_group = get_containing_parameter_group(param)
                assert fsdp_group is not None
                if fsdp_group not in seen:
                    seen.add(fsdp_group)
                    self._fsdp_groups.append(fsdp_group)
        self._owner_maps = assign_owner_work(group.main_weight for group in self._fsdp_groups)

    def _validate_param_groups(self) -> None:
        """Validate optimizer parameters and the FSDP buffers that contain them."""
        seen = set()
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise NotImplementedError("FsdpMuon does not support nonzero weight decay yet.")
            for param in group["params"]:
                if param.ndim != 2:
                    continue
                fsdp_group = get_containing_parameter_group(param)
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
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                param_to_group_index[param] = group_index

        for fsdp_group, owners in zip(self._fsdp_groups, self._owner_maps, strict=True):
            groups_by_index = {
                i: self.param_groups[param_to_group_index[fp.sharded]]
                for i, fp in enumerate(fsdp_group.fsdp_parameters)
                if fp.sharded in param_to_group_index
            }
            if not groups_by_index:
                continue
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
            momentum = self.state[param]["momentum_buffer"].to_local()
            grad = param.grad.to_local().to(momentum.dtype).reshape(momentum.shape)
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
