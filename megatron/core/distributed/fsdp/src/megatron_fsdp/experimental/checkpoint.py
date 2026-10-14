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

After :func:`fully_shard`, a module's parameters rest as ``DTensor`` views over the optimizer
(``main_weight``) buffers, and the optimizer's ``exp_avg``/``exp_avg_sq`` states are ``DTensor`` s
on the same device mesh. The standard DCP state-dict helpers
(:func:`torch.distributed.checkpoint.state_dict.get_model_state_dict` /
:func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`) expose those as FQN-keyed
DTensors and initialize the (empty) optimizer state on load, so we do not reimplement that here.

The one Megatron-FSDP-specific step is :func:`attach_uneven_dtensor_metadata`, which describes each
parameter's true position inside its packed parameter-group buffer. Without it the default planner
assumes canonical ``Shard(0)`` offsets and silently corrupts the checkpoint.
"""

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .optimizer import init_optimizer_state
from .parameter_group import sync_model_weights_from_main_weights
from .uneven_dtensor import attach_uneven_dtensor_metadata

__all__ = ["save_checkpoint", "load_checkpoint"]


def _rename_keys(state_dict: dict, mapping: Mapping[str, str]) -> dict:
    """Rename state entries without allowing two entries to overwrite one another."""
    renamed = {mapping.get(key, key): value for key, value in state_dict.items()}
    if len(renamed) != len(state_dict):
        raise ValueError("fqn_mapping must give each state entry a distinct checkpoint name.")
    return renamed


def _optimizer_key_mapping(state_dict: dict, mapping: Mapping[str, str]) -> dict[str, str]:
    """Rename parameter prefixes in torch's flattened optimizer state format."""
    result = {}
    for key in state_dict:
        section, _, remainder = key.partition(".")
        if section not in ("state", "param_groups"):
            continue
        # State entries can be nested. Search from the longest parameter prefix
        # instead of comparing every key against every model parameter.
        prefix, separator, _ = remainder.rpartition(".")
        while separator:
            if prefix in mapping:
                result[key] = f"{section}.{mapping[prefix]}{remainder[len(prefix):]}"
                break
            prefix, separator, _ = prefix.rpartition(".")
    return result


def _validate_global_names(model_state_dict: dict) -> None:
    """Reject ambiguous names shared by parameters on different meshes.

    DCP deduplicates entries by name and chunk offset. Distinct experts with the
    same local name on different expert-DP meshes would silently overwrite one
    another. Require the caller to name those experts globally instead.
    """
    names_by_mesh: dict[DeviceMesh, list[str]] = {}
    for key, value in model_state_dict.items():
        if isinstance(value, DTensor):
            names_by_mesh.setdefault(value.device_mesh, []).append(key)
    local_meshes = [
        (tuple(mesh.mesh.flatten().tolist()), names) for mesh, names in names_by_mesh.items()
    ]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_meshes)
    meshes_by_name: dict[str, tuple[int, ...]] = {}
    conflicts = set()
    for rank_meshes in gathered:
        for mesh, names in rank_meshes:
            for key in names:
                if meshes_by_name.setdefault(key, mesh) != mesh:
                    conflicts.add(key)
    if conflicts:
        raise ValueError(
            "Checkpoint names refer to different device meshes: "
            f"{sorted(conflicts)}. Supply fqn_mapping with globally unique parameter names."
        )


def _checkpoint_state_dict(model, optimizer, fqn_mapping):
    """Expose packed shards under checkpoint names with rank-local optimizer metadata."""
    model_state = get_model_state_dict(model)
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    attach_uneven_dtensor_metadata(model, model_state, optimizer_state)
    if fqn_mapping is not None:
        model_state = _rename_keys(model_state, fqn_mapping)
        # Flattening stores hyperparameters per parameter, so EP ranks with
        # different parameter lists can restore their own optimizer groups.
        optimizer_state = get_optimizer_state_dict(
            model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )
        optimizer_state = _rename_keys(
            optimizer_state, _optimizer_key_mapping(optimizer_state, fqn_mapping)
        )
    _validate_global_names(model_state)
    return {"model": model_state, "optimizer": optimizer_state}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | os.PathLike,
    *,
    fqn_mapping: Mapping[str, str] | None = None,
) -> None:
    """Save a ``fully_shard``-wrapped model and its optimizer as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters.
        checkpoint_dir: Destination directory for the DCP checkpoint.
        fqn_mapping: Local state-dict names to globally unique checkpoint names,
            required when distinct parameters on different meshes reuse local
            names (for example expert parallelism). Unmapped names are unchanged.
            Supply the corresponding mapping on load as well. This uses torch's
            flattened optimizer format to preserve per-parameter hyperparameters.
    """
    dcp.save(_checkpoint_state_dict(model, optimizer, fqn_mapping), checkpoint_id=checkpoint_dir)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | os.PathLike,
    *,
    sync_model_weights: bool = True,
    fqn_mapping: Mapping[str, str] | None = None,
) -> None:
    """Load a DCP checkpoint into a ``fully_shard``-wrapped model and its optimizer.

    The model and optimizer must already be sharded with the same layout used at save time (the same
    module structure and mesh); DCP reshards the on-disk data to this rank's shards.
    :func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` initializes the
    (empty) optimizer state so DCP has DTensors to load into in place, and the ``set_*`` helpers
    reinstall the loaded state.

    Args:
        model: A module tree sharded with :func:`fully_shard`, whose weights receive the load.
        optimizer: Optimizer whose state receives the load.
        checkpoint_dir: Source directory of the DCP checkpoint.
        sync_model_weights: Refresh compute weights from the loaded main weights afterwards.
        fqn_mapping: Local names to the global names used by :func:`save_checkpoint`.
    """
    init_optimizer_state(optimizer)
    state_dict = _checkpoint_state_dict(model, optimizer, fqn_mapping)
    dcp.load(state_dict, checkpoint_id=checkpoint_dir)
    model_state_dict, optimizer_state_dict = state_dict["model"], state_dict["optimizer"]
    if fqn_mapping is not None:
        reverse_mapping = {global_: local for local, global_ in fqn_mapping.items()}
        model_state_dict = _rename_keys(model_state_dict, reverse_mapping)
        optimizer_state_dict = _rename_keys(
            optimizer_state_dict, _optimizer_key_mapping(optimizer_state_dict, reverse_mapping)
        )
    set_model_state_dict(model, model_state_dict)
    set_optimizer_state_dict(
        model,
        optimizer,
        optimizer_state_dict,
        options=StateDictOptions(flatten_optimizer_state_dict=fqn_mapping is not None),
    )
    if sync_model_weights:
        sync_model_weights_from_main_weights(model.parameters())
