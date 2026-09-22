# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Distributed Checkpoint (DCP) save/load for the experimental Megatron-FSDP path.

These helpers wrap local parameters and optimizer state only for DCP and unwrap
loaded state before reinstalling it. Bare model/optimizer state_dict()
calls return local shards and must not be used as distributed checkpoints.

:func:`attach_uneven_dtensor_metadata` describes each
parameter's true position inside its packed parameter-group buffer. Without it the default planner
assumes canonical ``Shard(0)`` offsets and silently corrupts the checkpoint.
"""

import os
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor, Shard

from .module import FsdpModule
from .parameter_group import sync_model_weights_from_main_weights
from .uneven_dtensor import attach_uneven_dtensor_metadata

__all__ = ["save_checkpoint", "load_checkpoint"]


class _CheckpointState:
    """Translate between local runtime state and DCP state without replacing live tensors.

    Each state_dict() call creates new dictionaries and DTensor views sharing the local
    tensor storage. Only this adapter tracks which entries need unwrapping after a load.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self._local_fqns: set[str] = set()

    def state_dict(self) -> dict[str, Any]:
        """Build checkpoint dictionaries with global shapes and uneven-shard metadata."""
        model_state_dict = get_model_state_dict(self.model)
        optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer)
        # Optimizer.state_dict() can share these nested dictionaries with live state.
        optimizer_state_dict["state"] = {
            fqn: dict(state) for fqn, state in optimizer_state_dict.get("state", {}).items()
        }
        layouts = {}
        for module in self.model.modules():
            if isinstance(module, FsdpModule):
                for group in module.parameter_groups:
                    for index, parameter in enumerate(group.fsdp_parameters):
                        layouts[parameter.sharded] = (group.main_weight, index)
        self._local_fqns = set()
        for fqn, parameter in self.model.named_parameters(remove_duplicate=False):
            if parameter not in layouts:
                continue
            buffer, index = layouts[parameter]
            self._local_fqns.add(fqn)
            model_state_dict[fqn] = buffer.get_dtensor(index)
            for key, value in optimizer_state_dict.get("state", {}).get(fqn, {}).items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    continue
                if value.shape != parameter.shape:
                    raise NotImplementedError(
                        "Local-tensor checkpoints require parameter-shaped optimizer state."
                    )
                shape = buffer.layout.tensor_shapes[index]
                optimizer_state_dict["state"][fqn][key] = DTensor.from_local(
                    value,
                    buffer.mesh,
                    tuple(Shard(p.dim) if isinstance(p, Shard) else p for p in buffer.placements),
                    run_check=False,
                    shape=shape,
                    stride=torch.empty(shape, device="meta").stride(),
                )
        attach_uneven_dtensor_metadata(self.model, model_state_dict, optimizer_state_dict)
        return {"model": model_state_dict, "optimizer": optimizer_state_dict}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Install loaded local tensors, keeping checkpoint wrappers out of live state."""
        model_state_dict = dict(state_dict["model"])
        optimizer_state_dict = dict(state_dict["optimizer"])
        optimizer_state_dict["state"] = {
            fqn: dict(state) for fqn, state in optimizer_state_dict.get("state", {}).items()
        }
        for fqn in self._local_fqns:
            model_state_dict[fqn] = model_state_dict[fqn].to_local()
            for key, value in optimizer_state_dict["state"].get(fqn, {}).items():
                if isinstance(value, DTensor):
                    optimizer_state_dict["state"][fqn][key] = value.to_local()
        set_model_state_dict(self.model, model_state_dict)
        set_optimizer_state_dict(self.model, self.optimizer, optimizer_state_dict)


def _init_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Allocate optimizer state so a DCP load has DTensors to fill.

    :func:`get_optimizer_state_dict` initializes empty optimizer state via torch's
    ``_init_optim_state``, but that assigns a parameter-dtype gradient. A Megatron-FSDP sharded
    parameter advertises the FSDP gradient dtype through ``grad_dtype``, which differs from the
    (main-weight) parameter dtype under mixed precision, and rejects a mismatched gradient. So
    initialize the state here with a ``grad_dtype``-matched zero gradient; the subsequent load
    overwrites it. This is a no-op once the state exists (for example after a training step).

    TODO: this function becomes unnecessary once torch's ``_init_optim_state`` honors a parameter's
    ``grad_dtype`` when it allocates the placeholder gradient (``torch.zeros_like(param)`` in
    ``torch/distributed/checkpoint/state_dict.py``); an upstream issue is being filed.
    """
    if optimizer.state:
        return
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                grad_dtype = getattr(param, "grad_dtype", None) or param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str | os.PathLike
) -> None:
    """Save a ``fully_shard``-wrapped model and its optimizer as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters.
        checkpoint_dir: Destination directory for the DCP checkpoint.
    """
    checkpoint = _CheckpointState(model, optimizer)
    dcp.save(checkpoint.state_dict(), checkpoint_id=checkpoint_dir)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | os.PathLike,
    *,
    sync_model_weights: bool = True,
) -> None:
    """Load a DCP checkpoint into a ``fully_shard``-wrapped model and its optimizer.

    The model and optimizer must already be sharded with the same layout used at save time (the same
    module structure and mesh); DCP reshards the on-disk data to this rank's shards.
    Empty optimizer state is initialized before wrapping local tensors for DCP.
    After loading, the ``set_*`` helpers reinstall unwrapped local state.

    Args:
        model: A module tree sharded with :func:`fully_shard`, whose weights receive the load.
        optimizer: Optimizer whose state receives the load.
        checkpoint_dir: Source directory of the DCP checkpoint.
        sync_model_weights: Refresh compute weights from the loaded main weights afterwards.
    """
    _init_optimizer_state(optimizer)
    checkpoint = _CheckpointState(model, optimizer)
    state_dict = checkpoint.state_dict()
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)
    checkpoint.load_state_dict(state_dict)
    if sync_model_weights:
        sync_model_weights_from_main_weights(model.parameters())
