# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Iterable, List, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard


def gather_and_compute_chunk_metadata(dtensor: DTensor) -> ChunkStorageMetadata:
    """
    Gather chunk metadata for a DTensor across all ranks and compute the
    offsets and sizes of each chunk. This is necessary for handling uneven
    sharding in distributed tensors.
    """
    local_tensor = dtensor.to_local()
    local_shape = local_tensor.shape
    device_mesh = dtensor.device_mesh

    offsets = [0] * len(local_shape)
    cumulative_shape = list(local_shape).copy()

    def _update_offsets_and_cumulative_shape(
        mesh_dim: int, offsets: List[int], cumulative_shape: List[int]
    ):
        shard_group = device_mesh.get_group(mesh_dim)
        shard_dim = p.dim

        # Synchronize local shard dimensions across ranks
        world_size = dist.get_world_size(shard_group)
        global_shapes = [None] * world_size
        dist.all_gather_object(global_shapes, cumulative_shape, group=shard_group)

        # Calculate global offset for current rank's shard
        rank = dist.get_rank(shard_group)
        offset = sum(s[shard_dim] for s in global_shapes[:rank])
        # TODO: add documentation for the offset calculation
        # Add on the offset of the current mesh dimension
        offsets[shard_dim] += offset
        cumulative_shape[shard_dim] = sum(s[shard_dim] for s in global_shapes)

    # Get the shard placements order.
    shard_order = getattr(device_mesh, "_shard_order", None)
    if shard_order is None:
        shard_order = []
        reversed_shard_order = []
        mesh_dims = list(range(len(dtensor.placements)))
        strided_shard_count = 0
        for mesh_dim, p in enumerate(dtensor.placements):
            if isinstance(p, _StridedShard):
                reversed_shard_order.append(mesh_dim)
                mesh_dims.remove(mesh_dim)
                strided_shard_count += 1
        if strided_shard_count > 1:
            raise ValueError(
                f"DTensor has multiple strided shards ({strided_shard_count}), "
                "which is not supported."
            )
        reversed_shard_order += mesh_dims
        shard_order = list(reversed(reversed_shard_order))

    for mesh_dim in reversed(shard_order):
        p = dtensor.placements[mesh_dim]
        if isinstance(p, (Shard, _StridedShard)):
            _update_offsets_and_cumulative_shape(mesh_dim, offsets, cumulative_shape)
        elif isinstance(p, Replicate):
            # If we have a replicate placement, we do not need to update offsets
            # or cumulative shape, as it does not affect the chunk metadata.
            continue
        else:
            raise ValueError(f"Unsupported placement type {type(p)} in DTensor: {dtensor}")

    return ChunkStorageMetadata(offsets=tuple(offsets), sizes=tuple(local_shape))


def update_uneven_dtensor_chunk_metadata(dtensor: DTensor) -> dict:
    """
    Update the DTensor's chunk metadata to handle uneven sharding.
    This function modifies the DTensor in-place to include chunk metadata
    and write items closures for saving and loading.
    """

    def _chunk_list_closure(chunk_meta):
        return lambda: chunk_meta

    def _write_items_closure(uneven_chunk_meta):
        def _write_items(fqn: str, tensor: DTensor) -> List[WriteItem]:
            if tensor.to_local().numel() == 0:
                # If the tensor is empty, return an empty list
                return []

            return [
                WriteItem(
                    type=WriteItemType.SHARD,
                    index=MetadataIndex(fqn, uneven_chunk_meta.offsets),
                    tensor_data=TensorWriteData(
                        chunk=uneven_chunk_meta,
                        properties=TensorProperties.create_from_tensor(tensor.to_local()),
                        size=tensor.size(),
                    ),
                )
            ]

        return _write_items

    # Get uneven chunk metadata for the DTensor
    # TODO: Optimize gather_and_compute_chunk_metadata synchronization:
    # 1. Add pre-check validation to verify tensor shape consistency
    #    across devices before entering barrier (prevents potential hangs)
    # 2. Implement batched barrier using grouped collectives
    #    to amortize synchronization overhead
    uneven_chunk_meta = gather_and_compute_chunk_metadata(dtensor)

    # Set the chunk list and write items closure for the DTensor
    dtensor._local_tensor.__create_chunk_list__ = _chunk_list_closure([uneven_chunk_meta])
    dtensor._local_tensor.__create_write_items__ = _write_items_closure(uneven_chunk_meta)


def validate_uneven_dtensor(dtensor: DTensor) -> None:
    """
    Validates the chunk metadata of an uneven DTensor to ensure correctness and boundary coverage.

    Notes:
    - `gather_and_compute_chunk_metadata` will ensure that all chunks do not overlap.

    This function performs the following checks:
      - All chunk offsets and sizes are within the tensor shape bounds.
      - All boundaries of each dimension are actually covered by shard placements.

    Args:
        dtensor (DTensor): The distributed tensor to validate.

    Raises:
        AssertionError: If any chunk falls out of bounds or not all boundaries are touched.
    """

    # gather_and_compute_chunk_metadata will ensure that all chunks do not overlap.
    chunk_meta = gather_and_compute_chunk_metadata(dtensor)

    # Validate that each chunk's metadata is within bounds.
    assert all(
        [
            0 <= offset and offset + size <= dtensor.shape[dim]
            for (dim, (offset, size)) in enumerate(zip(chunk_meta.offsets, chunk_meta.sizes))
        ]
    ), (
        "[Megatron-FSDP] DTensor chunk metadata is invalid. "
        f"Offsets: {chunk_meta.offsets}, "
        f"Sizes: {chunk_meta.sizes}, "
        f"Global shape: {dtensor.shape}, "
        f"Local shape: {dtensor.to_local().shape}, "
        f"Device mesh: {dtensor.device_mesh}."
    )

    # Check that all boundaries (start and end) are touched.
    boundary_checks = torch.tensor(
        [
            [offset == 0, offset + size == dtensor.shape[dim]]
            for (dim, (offset, size)) in enumerate(zip(chunk_meta.offsets, chunk_meta.sizes))
        ],
        dtype=torch.int,
    ).cuda()

    for i, p in enumerate(dtensor.placements):
        if isinstance(p, Shard) or isinstance(p, _StridedShard):
            torch.distributed.all_reduce(
                boundary_checks,
                op=torch.distributed.ReduceOp.MAX,
                group=dtensor.device_mesh.get_group(i),
            )
    assert torch.all(boundary_checks), (
        "[Megatron-FSDP] DTensor chunk metadata boundary check failed. "
        f"Offsets: {chunk_meta.offsets}, "
        f"Sizes: {chunk_meta.sizes}, "
        f"Global shape: {dtensor.shape}, "
        f"Local shape: {dtensor.to_local().shape}, "
        f"Device mesh: {dtensor.device_mesh}."
    )


def filter_unflattened_state_dict(state_dict, key_chain=[], visit_condition=lambda x: False):
    """
    Recursively traverses an unflattened state_dict and collects keys
    of items that meet the visit_condition. The keys are returned as lists
    of strings representing the path to each item in the state_dict.
    """
    visit_items = []
    for key, value in state_dict.items():
        if isinstance(value, dict):
            # Recurse into nested dictionaries
            visit_items += filter_unflattened_state_dict(
                value, key_chain=key_chain + [key], visit_condition=visit_condition
            )
        elif visit_condition(value):
            # If the value meets the visit condition, process it
            visit_items.append(key_chain + [key])
    return visit_items


def get_unflattened_state_dict(state_dict, key_chain=[]):
    """Get a value from an unflattened state_dict at the specified key chain."""
    current = state_dict
    for key in key_chain:
        if isinstance(current, dict) and key in current:
            # Navigate through the nested dictionary
            current = current[key]
        else:
            raise KeyError(f"Key {key_chain} not found in state_dict")

    return current


def preprocess_state_dict_for_uneven_dtensor(state_dict: dict) -> dict:
    """
    Preprocess the state_dict to prepare it for saving or loading unevenly sharded DTensors.
    This function modifies the DTensors in the state_dict to include chunk metadata
    and write items closures.
    """
    visit_dtensor = filter_unflattened_state_dict(
        state_dict, visit_condition=lambda x: isinstance(x, DTensor)
    )
    for key_chain in visit_dtensor:
        # Get the DTensor at the key chain
        dtensor = get_unflattened_state_dict(state_dict, key_chain)
        update_uneven_dtensor_chunk_metadata(dtensor)
    return state_dict


def uneven_dtensor_to_full_tensor(dtensor: DTensor) -> torch.Tensor:
    """
    Gather a DTensor with potentially uneven sharding across ranks into a full tensor.

    This function handles DTensors with uneven shards (where different ranks may have
    different-sized chunks) by gathering chunk metadata and local tensors across all
    ranks, then reconstructing the complete tensor.

    Args:
        dtensor (DTensor): The distributed tensor to gather. Must have chunk metadata
            available (either pre-existing or will be computed).

    Returns:
        torch.Tensor: The fully reconstructed tensor with shape matching the original
            DTensor's global shape.

    Raises:
        TypeError: If input is not a DTensor.
        ValueError: If chunk metadata is malformed (expected exactly one chunk per rank).
        AssertionError: If an unexpected placement type is encountered after processing
            Shard placements.

    Note:
        - This function performs collective operations (all_gather_object, all_gather)
          across the device mesh, requiring synchronization across ranks.
        - Works with Shard and _StridedShard placements, and expects Replicate placements
          for non-sharded dimensions.
        - The function modifies the DTensor in-place by adding chunk metadata if missing.

    Example:
        >>> mesh = DeviceMesh("cuda", [0, 1, 2, 3])
        >>> # Create a DTensor with uneven sharding
        >>> dtensor = DTensor(..., placements=[Shard(0)])
        >>> full_tensor = gather_uneven_dtensor_to_full_tensor(dtensor)
        >>> assert full_tensor.shape == dtensor.shape
    """
    # Validate input type
    if not isinstance(dtensor, DTensor):
        raise TypeError(f"Input must be a DTensor, got {type(dtensor).__name__}.")

    # Ensure chunk metadata is available for uneven shards
    if not hasattr(dtensor._local_tensor, "__create_chunk_list__"):
        update_uneven_dtensor_chunk_metadata(dtensor)

    # Retrieve and validate chunk metadata
    chunk_metadata_list = dtensor.__create_chunk_list__()
    if len(chunk_metadata_list) != 1:
        raise ValueError(
            f"Expected exactly one chunk metadata per rank, got {len(chunk_metadata_list)}."
        )
    local_chunk_metadata = chunk_metadata_list[0]

    # Prepare local chunk information for gathering
    local_chunks_info = [
        {
            "shape": dtensor.to_local().shape,
            "offset": getattr(local_chunk_metadata, "offsets", [0] * len(dtensor.shape)),
        }
    ]
    local_buffer = dtensor.to_local().contiguous().view(-1)

    # Iterate through device mesh dimensions and gather across sharded dimensions
    for mesh_dim, placement in enumerate(dtensor.placements):
        if isinstance(placement, (Shard, _StridedShard)):
            # Get the process group for this mesh dimension
            shard_group = dtensor.device_mesh.get_group(mesh_dim)

            # Gather chunk metadata from all ranks in this dimension
            group_chunks_info = [None] * shard_group.size()
            dist.all_gather_object(group_chunks_info, local_chunks_info, group=shard_group)

            # Prepare buffers for gathering tensors from all ranks
            group_tensors = [
                torch.empty(
                    sum(chunk["shape"].numel() for chunk in chunks_info),
                    dtype=dtensor.dtype,
                    device=dtensor.device,
                )
                for chunks_info in group_chunks_info
            ]

            # Gather actual tensor data from all ranks
            dist.all_gather(group_tensors, local_buffer, group=shard_group)

            # Flatten the gathered metadata and concatenate tensors
            local_chunks_info = [item for sublist in group_chunks_info for item in sublist]
            local_buffer = torch.cat(group_tensors)
        elif not isinstance(placement, Replicate):
            raise ValueError(
                f"Unexpected placement {placement} at mesh dimension {mesh_dim}. "
                f"Expected Shard, _StridedShard, or Replicate."
            )

    # Split the gathered buffer back into individual chunks
    all_local_chunks = []
    buffer_offset = 0
    for chunk_info in local_chunks_info:
        chunk_shape = chunk_info["shape"]
        chunk_numel = chunk_shape.numel()
        chunk_tensor = local_buffer[buffer_offset : buffer_offset + chunk_numel].view(chunk_shape)
        all_local_chunks.append(chunk_tensor)
        buffer_offset += chunk_numel

    debug_slices = []
    for chunk_info, local_chunk in zip(local_chunks_info, all_local_chunks):
        offset = chunk_info["offset"]
        slices = tuple(slice(o, o + s) for o, s in zip(offset, local_chunk.shape))
        debug_slices.append(slices)

    # Reconstruct the full tensor by placing chunks at their correct offsets
    full_tensor = torch.zeros(dtensor.shape, dtype=dtensor.dtype, device=dtensor.device)
    for chunk_info, local_chunk in zip(local_chunks_info, all_local_chunks):
        offset = chunk_info["offset"]
        slices = tuple(slice(o, o + s) for o, s in zip(offset, local_chunk.shape))
        full_tensor[slices] = local_chunk

    return full_tensor


def redistribute_uneven_dtensor_to_replicated(dtensor: DTensor) -> DTensor:
    """
    Redistribute an unevenly sharded DTensor to a fully replicated DTensor.

    This function first gathers the unevenly sharded DTensor into a full tensor
    and then redistributes it as a replicated DTensor across all ranks.

    Args:
        dtensor (DTensor): The unevenly sharded DTensor to redistribute.
    Returns:
        DTensor: A replicated DTensor with the same data as the input DTensor.
    """
    full_tensor = uneven_dtensor_to_full_tensor(dtensor)
    replicated_dtensor = DTensor.from_local(
        full_tensor,
        placements=[Replicate()] * len(dtensor.placements),
        device_mesh=dtensor.device_mesh,
    )
    return replicated_dtensor


def _intersection(s1, s2):
    # Only works for step=1
    start = max(s1.start, s2.start)
    stop = min(s1.stop, s2.stop)
    if start >= stop:
        return slice(0, 0)  # Empty slice if no intersection
    return slice(start, stop)


def _offset_slice(s, offset):
    return slice(s.start + offset, s.stop + offset)


def split_dtensor(
    dtensor: DTensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0,
    update_uneven_dtensor_chunk_meta: bool = False,
) -> Iterable[DTensor]:
    """
    Splits a DTensor into smaller DTensors along a specified dimension.

    This function manages uneven sharding by accurately assigning chunk metadata
    for each split. Unlike the native PyTorch DTensor split functionality,
    it does not redistribute `Replicate` placements, which helps avoid Out-Of-Memory (OOM) issues.

    Args:
        dtensor (DTensor): The DTensor to split.
        split_size_or_sections (int or list of int): If int, defines the size of each chunk.
            If a list, specifies the sizes of each chunk in order.
        dim (int, optional): The axis along which to split. Default is 0.
        update_uneven_dtensor_chunk_meta (bool, optional): Whether to update chunk
            metadata for each resulting DTensor. Default is False.

    Yields:
        DTensor: Sub-DTensor resulting from the split, maintaining correct metadata.

    Example:
        >>> for chunk in split_dtensor(dt, 3, dim=1):
        ...     print(chunk)
    """
    tensor_size = dtensor.shape[dim]

    # Calculate boundary indices for each split
    if isinstance(split_size_or_sections, int):
        split_points = list(range(0, tensor_size, split_size_or_sections))
        split_points.append(tensor_size)
    else:
        split_points = [0]
        for size in split_size_or_sections:
            split_points.append(split_points[-1] + size)

    chunk_meta = gather_and_compute_chunk_metadata(dtensor)
    chunk_slice = slice(chunk_meta.offsets[dim], chunk_meta.offsets[dim] + chunk_meta.sizes[dim])
    local_offset = chunk_meta.offsets[dim]
    local_tensor = dtensor.to_local()

    # Create chunks using manual slicing
    for i in range(len(split_points) - 1):
        split_slice = slice(split_points[i], split_points[i + 1])
        s = _intersection(split_slice, chunk_slice)
        if s.start < s.stop:
            s = _offset_slice(s, -local_offset)

        if s.start < 0 or s.stop < s.start and torch.distributed.get_rank() == 0:
            raise ValueError(
                f"Invalid split slice {s} for DTensor with shape {dtensor.shape} "
                f"and local offset {local_offset} on dimension {dim}."
            )

        # Slice the local tensor
        sliced_tensor = local_tensor.narrow(dim, s.start, s.stop - s.start)
        out_shape = list(dtensor.shape)
        out_shape[dim] = split_slice.stop - split_slice.start

        new_dtensor = DTensor.from_local(
            sliced_tensor,
            shape=tuple(out_shape),
            stride=sliced_tensor.stride(),
            placements=dtensor.placements,
            device_mesh=dtensor.device_mesh,
        )

        if update_uneven_dtensor_chunk_meta:
            update_uneven_dtensor_chunk_metadata(new_dtensor)

        yield new_dtensor
