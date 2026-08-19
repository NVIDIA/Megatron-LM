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

The one Megatron-FSDP-specific step is :func:`preprocess_state_dict_for_uneven_dtensor`. A
``FsdpParameterGroup`` packs several parameters into one flat buffer with least-common-multiple row
padding, so a parameter's per-rank shard does not tile like torch's canonical ``Shard(0)`` (a rank
may own several rows of one parameter and none of the next). The helper attaches each DTensor's true
per-shard chunk offsets so DCP writes and reshards it correctly; without it the default planner
assumes canonical ``Shard(0)`` offsets and silently corrupts the checkpoint.
"""

import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from ..uneven_dtensor import preprocess_state_dict_for_uneven_dtensor
from .optimizer import init_optimizer_state
from .parameter_group import sync_model_weights_from_main_weights

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str | os.PathLike
) -> None:
    """Save a ``fully_shard``-wrapped model and its optimizer as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters.
        checkpoint_dir: Destination directory for the DCP checkpoint.
    """
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    preprocess_state_dict_for_uneven_dtensor(model_state_dict)
    preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
    dcp.save(
        {"model": model_state_dict, "optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir
    )


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
    :func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` initializes the
    (empty) optimizer state so DCP has DTensors to load into in place, and the ``set_*`` helpers
    reinstall the loaded state.

    Args:
        model: A module tree sharded with :func:`fully_shard`, whose weights receive the load.
        optimizer: Optimizer whose state receives the load.
        checkpoint_dir: Source directory of the DCP checkpoint.
        sync_model_weights: Refresh compute weights from the loaded main weights afterwards.
    """
    init_optimizer_state(optimizer)
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    preprocess_state_dict_for_uneven_dtensor(model_state_dict)
    preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
    dcp.load(
        {"model": model_state_dict, "optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir
    )
    set_model_state_dict(model, model_state_dict)
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)
    if sync_model_weights:
        sync_model_weights_from_main_weights(model.parameters())
