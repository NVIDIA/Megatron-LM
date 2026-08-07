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

"""Distributed-checkpoint chunk metadata for Megatron-FSDP's packed parameter buffers.

An :class:`~.parameter_group.FsdpParameterGroup` packs several parameters into one flat
:class:`~.dbuffer.DBuffer` with least-common-multiple row padding, so a parameter's per-rank shard
does not tile the way torch's canonical ``Shard(0)`` does: a rank may own several rows of one
parameter and none of the next. :meth:`~.dbuffer.DBuffer.get_dtensor` presents each parameter as a
plain ``Shard(0)`` DTensor, which cannot express those offsets, so DCP's default planner would
mis-place every packed shard and silently corrupt the checkpoint.

DCP lets a local tensor override that placement through the ``__create_chunk_list__`` and
``__create_write_items__`` hooks, which :class:`~torch.distributed.tensor.DTensor` forwards to its
local tensor. This module derives each shard's true position from the
:class:`~.layout.GlobalLayout` that already backs the buffer -- it records every logical tensor's
global element offset, and the buffer records this rank's owned element range -- and attaches those
hooks. The derivation is pure rank-local arithmetic, so it needs no collective.
"""

from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.tensor import DTensor

from .dbuffer import DBuffer
from .layout import non_leading_numel
from .module import FsdpModule

__all__ = ["attach_uneven_dtensor_metadata", "chunk_metadata_by_fqn"]

# Key of the per-parameter optimizer state in a torch DCP optimizer state dict.
_OPTIMIZER_STATE_KEY = "state"


def _dbuffer_chunk_metadata(buffer: DBuffer, index: int) -> ChunkStorageMetadata:
    """Return the chunk of logical tensor ``index`` this rank holds in ``buffer``.

    Flat placements shard the flat buffer, and every logical tensor is row-aligned within it, so a
    rank always owns whole dim-0 rows and the chunk is a dim-0 range of the global tensor.

    Args:
        buffer: Buffer holding the logical tensor.
        index: Index of the logical tensor within ``buffer``.

    Returns:
        This rank's chunk offset and size, in global tensor coordinates.
    """
    shape = buffer.layout.tensor_shapes[index]
    row_size = non_leading_numel(shape)
    owned_range = buffer._get_owned_range(index)
    if owned_range is None:
        # Flat placements hand ranks contiguous buffer ranges in rank order, so the rows this rank
        # does not own but that precede its range belong to lower ranks. Reporting that count keeps
        # an empty chunk's offset equal to what a gather of all shard sizes would produce; DCP
        # ignores it either way, because a zero-size chunk neither writes nor reads anything.
        preceding_numel = buffer.offset - buffer.layout.tensor_to_offset[index]
        preceding_numel = min(max(preceding_numel, 0), shape.numel())
        return _chunk(row_offset=preceding_numel // row_size, rows=0, shape=shape)

    return _chunk(
        row_offset=owned_range.tensor_relative_offset // row_size,
        rows=owned_range.numel // row_size,
        shape=shape,
    )


def chunk_metadata_by_fqn(model: nn.Module) -> dict[str, ChunkStorageMetadata]:
    """Return this rank's chunk of every Megatron-FSDP parameter in ``model``, keyed by FQN.

    The keys are the parameter FQNs that torch's DCP state-dict helpers use for both the model
    state dict and the per-parameter entries of the optimizer state dict.

    Args:
        model: A module tree that has been sharded with :func:`~.fully_shard.fully_shard`.

    Returns:
        Chunk metadata for each sharded parameter, and nothing for parameters FSDP does not own.
        Tied parameters appear once per FQN, all sharing the one chunk of the buffer entry that
        backs them.
    """
    # Parameters key this map by identity because tensor equality is elementwise. The map is
    # consumed below while ``model`` still holds every parameter, so the ids stay valid.
    metadata_by_parameter = {
        id(fsdp_parameter.sharded): _dbuffer_chunk_metadata(parameter_group.main_weight, index)
        for module in model.modules()
        if isinstance(module, FsdpModule)
        for parameter_group in module.parameter_groups
        for index, fsdp_parameter in enumerate(parameter_group.fsdp_parameters)
    }
    # Tied parameters share one nn.Parameter under several FQNs, and the state dict carries an
    # entry for each of them, so iterate without deduplicating.
    return {
        fqn: metadata_by_parameter[id(parameter)]
        for fqn, parameter in model.named_parameters(remove_duplicate=False)
        if id(parameter) in metadata_by_parameter
    }


def attach_uneven_dtensor_metadata(
    model: nn.Module, model_state_dict: dict[str, Any], optimizer_state_dict: dict[str, Any]
) -> None:
    """Attach Megatron-FSDP chunk metadata to the sharded DTensors DCP will save or load.

    Optimizer state is allocated as ``zeros_like`` of a sharded parameter, so it shares that
    parameter's chunk.

    Args:
        model: The sharded module tree the state dicts were taken from.
        model_state_dict: Model state dict from
            :func:`~torch.distributed.checkpoint.state_dict.get_model_state_dict`.
        optimizer_state_dict: Optimizer state dict from
            :func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`.
    """
    metadata_by_fqn = chunk_metadata_by_fqn(model)
    _attach_to_parameter_state(model_state_dict, metadata_by_fqn)
    _attach_to_parameter_state(optimizer_state_dict[_OPTIMIZER_STATE_KEY], metadata_by_fqn)


def _chunk(row_offset: int, rows: int, shape: torch.Size) -> ChunkStorageMetadata:
    """Build chunk metadata for a dim-0 row range of a tensor of shape ``shape``."""
    # Flat placements only shard dim 0, so every other dimension is fully owned from offset 0.
    return ChunkStorageMetadata(
        offsets=(row_offset, *(0,) * (len(shape) - 1)), sizes=(rows, *shape[1:])
    )


def _attach_to_parameter_state(
    state: dict[str, Any], metadata_by_fqn: dict[str, ChunkStorageMetadata]
) -> None:
    """Attach chunk metadata to the DTensors of a parameter-FQN-keyed state dict.

    Each value is either a parameter's tensor, as in a model state dict, or a mapping of that
    parameter's state tensors, as in an optimizer state dict.
    """
    for fqn, value in state.items():
        tensors = value.values() if isinstance(value, dict) else (value,)
        for tensor in tensors:
            if not isinstance(tensor, DTensor):
                continue
            if fqn not in metadata_by_fqn:
                raise KeyError(
                    f"[Megatron-FSDP] No chunk metadata for DTensor {fqn!r}. Distributed "
                    "checkpointing supports DTensors sharded by Megatron-FSDP only."
                )
            _attach_dcp_hooks(tensor, metadata_by_fqn[fqn])


def _attach_dcp_hooks(dtensor: DTensor, chunk: ChunkStorageMetadata) -> None:
    """Override a DTensor's DCP placement with ``chunk``."""
    local_shape = tuple(dtensor.to_local().shape)
    if tuple(chunk.sizes) != local_shape:
        raise RuntimeError(
            f"[Megatron-FSDP] Chunk size {tuple(chunk.sizes)} does not match the local shard "
            f"shape {local_shape} of a DTensor with global shape {tuple(dtensor.shape)}."
        )

    def create_chunk_list() -> list[ChunkStorageMetadata]:
        return [chunk]

    def create_write_items(fqn: str, tensor: DTensor) -> list[WriteItem]:
        local_tensor = tensor.to_local()
        if local_tensor.numel() == 0:
            # This rank holds none of the tensor, so it writes nothing.
            return []
        return [
            WriteItem(
                type=WriteItemType.SHARD,
                index=MetadataIndex(fqn, chunk.offsets),
                tensor_data=TensorWriteData(
                    chunk=chunk,
                    properties=TensorProperties.create_from_tensor(local_tensor),
                    size=tensor.size(),
                ),
            )
        ]

    dtensor._local_tensor.__create_chunk_list__ = create_chunk_list
    dtensor._local_tensor.__create_write_items__ = create_write_items
