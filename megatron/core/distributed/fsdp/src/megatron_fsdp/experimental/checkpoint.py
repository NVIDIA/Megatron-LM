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
from torch.distributed.tensor import DTensor

from .dbuffer import DBuffer
from .module import FsdpModule
from .optimizer import init_optimizer_state
from .parameter_group import sync_model_weights_from_main_weights
from .uneven_dtensor import attach_uneven_dtensor_metadata

__all__ = ["save_checkpoint", "load_checkpoint"]


def parameter_layouts(model: torch.nn.Module) -> dict[torch.nn.Parameter, tuple[DBuffer, int]]:
    """Map each local runtime parameter to its main-weight buffer and logical index."""
    layouts = {}
    for module in model.modules():
        if not isinstance(module, FsdpModule):
            continue
        for group in module.parameter_groups:
            for index, parameter in enumerate(group.fsdp_parameters):
                layouts[parameter.sharded] = (group.main_weight, index)
    return layouts


def wrap_optimizer_state(state: dict[str, Any], buffer: DBuffer, index: int) -> dict[str, Any]:
    """Wrap elementwise state without changing the optimizer's live state dictionary."""
    return {
        key: (
            buffer.get_dtensor(index, local_tensor=value)
            if isinstance(value, torch.Tensor) and value.ndim > 0
            else value
        )
        for key, value in state.items()
    }


class CheckpointState:
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
        layouts = parameter_layouts(self.model)
        self._local_fqns = set()
        # Optimizer.state_dict() can share nested dictionaries with live state.
        optimizer_state_dict["state"] = dict(optimizer_state_dict.get("state", {}))
        for fqn, parameter in self.model.named_parameters(remove_duplicate=False):
            if parameter not in layouts:
                continue
            buffer, index = layouts[parameter]
            if fqn in model_state_dict:
                model_state_dict[fqn] = buffer.get_dtensor(index)
                self._local_fqns.add(fqn)
            if fqn in optimizer_state_dict["state"]:
                optimizer_state_dict["state"][fqn] = wrap_optimizer_state(
                    optimizer_state_dict["state"][fqn], buffer, index
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
        for state in optimizer_state_dict["state"].values():
            for key, value in state.items():
                if isinstance(value, DTensor):
                    state[key] = value.to_local()
        set_model_state_dict(self.model, model_state_dict)
        set_optimizer_state_dict(self.model, self.optimizer, optimizer_state_dict)


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str | os.PathLike
) -> None:
    """Save a ``fully_shard``-wrapped model and its optimizer as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters.
        checkpoint_dir: Destination directory for the DCP checkpoint.
    """
    checkpoint = CheckpointState(model, optimizer)
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
    init_optimizer_state(optimizer)
    checkpoint = CheckpointState(model, optimizer)
    state_dict = checkpoint.state_dict()
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)
    checkpoint.load_state_dict(state_dict)
    if sync_model_weights:
        sync_model_weights_from_main_weights(model.parameters())
