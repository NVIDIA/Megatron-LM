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

After :func:`fully_shard`, a module's parameters rest as ``DTensor`` views over the
optimizer (``main_weight``) buffers, and the optimizer's ``exp_avg``/``exp_avg_sq`` states
become ``DTensor`` s on the same device mesh. Those DTensors can be saved and loaded with
PyTorch DCP directly, with one caveat: a ``FsdpParameterGroup`` packs several parameters into
one flat buffer with least-common-multiple row padding, so a parameter's per-rank shard does
not tile like torch's canonical ``Shard(0)``. DCP's default planner would therefore compute
wrong global offsets. :func:`preprocess_state_dict_for_uneven_dtensor` fixes this by attaching
true per-shard chunk metadata to every DTensor before the save and load, exactly as the stable
Megatron-FSDP path does.
"""

import os
from typing import Optional, Union

import torch
import torch.distributed.checkpoint as dcp

from ..uneven_dtensor import preprocess_state_dict_for_uneven_dtensor
from .module import FsdpModule

__all__ = [
    "save_dcp_checkpoint",
    "load_dcp_checkpoint",
    "materialize_optimizer_state",
    "resync_compute_weights",
]


def _iter_fsdp_modules(model: torch.nn.Module):
    """Yield every ``FsdpModule`` in ``model`` (including ``model`` itself)."""
    for module in model.modules():
        if isinstance(module, FsdpModule):
            yield module


def _strip_extra_state(model_state_dict: dict) -> dict:
    """Drop TransformerEngine ``_extra_state`` entries.

    ``_extra_state`` holds opaque per-module bytes (for example FP8 scale history) that FSDP
    does not manage as DTensors. Following the stable Megatron-FSDP DCP path, these are not
    checkpointed here.
    """
    return {
        key: value for key, value in model_state_dict.items() if not key.endswith("_extra_state")
    }


def _build_state_dict(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> dict:
    state_dict = {"model": _strip_extra_state(model.state_dict())}
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    return state_dict


def _annotate_uneven_dtensors(state_dict: dict) -> dict:
    """Attach uneven-DTensor chunk metadata to the model and optimizer sub-dicts in place."""
    preprocess_state_dict_for_uneven_dtensor(state_dict["model"])
    if "optimizer" in state_dict:
        preprocess_state_dict_for_uneven_dtensor(state_dict["optimizer"])
    return state_dict


def materialize_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Allocate optimizer state slots so an in-place DCP load has destinations to fill.

    A freshly constructed optimizer has an empty ``optimizer.state``, so ``state_dict()["state"]``
    is empty and :func:`torch.distributed.checkpoint.load` would find nothing to load into. Run a
    single zero-gradient step to allocate the per-parameter state (for example Adam's ``exp_avg``,
    ``exp_avg_sq``, and ``step``); the subsequent load overwrites those slots in place, so this
    step's zero update and ``step == 1`` are discarded.

    Args:
        optimizer: Optimizer whose state should be materialized before loading.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                # A sharded parameter advertises the FSDP gradient dtype via ``grad_dtype``, which
                # can differ from the (main-weight) parameter dtype under mixed precision. Match it
                # so the temporary grad is consistent with the parameter's grad contract.
                grad_dtype = getattr(param, "grad_dtype", param.dtype)
                param.grad = torch.zeros_like(param, dtype=grad_dtype)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def resync_compute_weights(model: torch.nn.Module) -> None:
    """Refresh every FSDP group's compute weights from its (loaded) main weights.

    A load writes into the ``main_weight``-backed sharded DTensors. When mixed precision keeps a
    separate lower-precision compute buffer, that buffer is stale until the next forward pre-hook
    would resync it. Doing it here makes the post-load state deterministic (and correct if the
    caller inspects compute weights before running a forward pass). It is a no-op when the compute
    buffer aliases the main buffer.

    Args:
        model: Root module (or any module tree) containing ``FsdpModule`` instances.
    """
    for module in _iter_fsdp_modules(model):
        for parameter_group in module.parameter_groups:
            parameter_group.sync_model_weight_from_main_weight()


def save_dcp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_id: Union[str, os.PathLike],
) -> None:
    """Save a ``fully_shard``-wrapped model (and optionally its optimizer) as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters, or ``None`` to save only weights.
        checkpoint_id: Destination directory for the DCP checkpoint.
    """
    state_dict = _annotate_uneven_dtensors(_build_state_dict(model, optimizer))
    dcp.save(state_dict, checkpoint_id=str(checkpoint_id))


def load_dcp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_id: Union[str, os.PathLike],
    *,
    resync_model_weights: bool = True,
) -> None:
    """Load a DCP checkpoint into a ``fully_shard``-wrapped model (and optionally its optimizer).

    The model and optimizer must already be sharded with the same layout used at save time (the
    same module structure and mesh); DCP reshards the on-disk data to this rank's shards. Tensors
    are loaded in place into the resting sharded DTensors.

    Args:
        model: A module tree sharded with :func:`fully_shard`, whose weights receive the load.
        optimizer: Optimizer whose state receives the load, or ``None`` to load only weights.
        checkpoint_id: Source directory of the DCP checkpoint.
        resync_model_weights: Refresh compute weights from the loaded main weights afterwards.
    """
    if optimizer is not None:
        materialize_optimizer_state(optimizer)
    state_dict = _annotate_uneven_dtensors(_build_state_dict(model, optimizer))
    dcp.load(state_dict, checkpoint_id=str(checkpoint_id))
    # DCP wrote into the resting DTensors in place. Re-install the loaded model tensors (a no-op
    # for the shared storage, but it also restores any stripped-key structure) and the optimizer
    # state, mirroring the stable Megatron-FSDP load path.
    model.load_state_dict(state_dict["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
    if resync_model_weights:
        resync_compute_weights(model)
