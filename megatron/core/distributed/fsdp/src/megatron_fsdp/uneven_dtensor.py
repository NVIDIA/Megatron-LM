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

import hashlib
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

from megatron.core.dist_checkpointing.core import CheckpointingException


def _get_backend_name(group: dist.ProcessGroup) -> str:
    """Return a normalized process group backend name."""
    return str(dist.get_backend(group)).lower().removeprefix("backend.")


def _get_collective_device(
    group: dist.ProcessGroup, device_type: str | None = None
) -> torch.device:
    """Return a device supported by the process group's backend."""
    # DeviceMesh process groups can report an ``undefined`` aggregate backend when
    # PyTorch dispatches to a device-specific backend at collective time. Prefer the
    # mesh's declared device type instead of trying to infer it from that name.
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if device_type == "cpu":
        return torch.device("cpu")
    backend = _get_backend_name(group)
    if backend == "nccl" or (backend == "undefined" and torch.cuda.is_available()):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _all_gather_int64(
    values: Iterable[int], group: dist.ProcessGroup, *, device_type: str | None = None
) -> List[List[int]]:
    """Gather a fixed-size list of integers without serializing Python objects."""
    values = list(values)
    if _get_backend_name(group) == "fake":
        return [values.copy() for _ in range(dist.get_world_size(group))]

    local_values = torch.tensor(
        values,
        dtype=torch.int64,
        device=_get_collective_device(group, device_type),
    )
    gathered_values = [
        torch.empty_like(local_values) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered_values, local_values, group=group)
    return [value.cpu().tolist() for value in gathered_values]


def _placement_contract(placement: object) -> str:
    """Return a stable representation of a DTensor placement."""
    if isinstance(placement, _StridedShard):
        return f"strided_shard:{placement.dim}:{int(placement.split_factor)}"
    if isinstance(placement, Shard):
        return f"shard:{placement.dim}"
    if isinstance(placement, Replicate):
        return "replicate"
    placement_type = type(placement)
    return f"unsupported:{placement_type.__module__}.{placement_type.__qualname__}:{placement!r}"


def _dtensor_contract(dtensor: DTensor, context: str | None = None) -> str:
    """Build the rank-invariant part of an uneven DTensor's metadata contract."""
    local_shape = tuple(int(size) for size in dtensor.to_local().shape)
    global_shape = tuple(int(size) for size in dtensor.shape)
    mesh_shape = tuple(int(size) for size in dtensor.device_mesh.shape)
    mesh_ranks = tuple(int(rank) for rank in dtensor.device_mesh.mesh.flatten().tolist())
    placements = tuple(
        _placement_contract(placement) for placement in dtensor.placements
    )
    return (
        f"context={context};local_ndim={len(local_shape)};global_shape={global_shape};"
        f"mesh_shape={mesh_shape};mesh_ranks={mesh_ranks};placements={placements}"
    )


def _contract_digest(contract: str) -> List[int]:
    """Encode a contract as four signed int64 values for fixed-size collectives."""
    digest = hashlib.sha256(contract.encode("utf-8")).digest()
    return [
        int.from_bytes(digest[offset : offset + 8], byteorder="big", signed=True)
        for offset in range(0, len(digest), 8)
    ]


def _mesh_has_failure(local_failure: bool, device_mesh) -> bool:
    """Propagate a local validation failure to every rank in a Cartesian device mesh."""
    failure = local_failure
    for mesh_dim in range(device_mesh.ndim):
        group = device_mesh.get_group(mesh_dim)
        if _get_backend_name(group) == "fake":
            continue
        failure_tensor = torch.tensor(
            [int(failure)],
            dtype=torch.int64,
            device=_get_collective_device(group, device_mesh.device_type),
        )
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX, group=group)
        failure = bool(failure_tensor.item())
    return failure


def _validate_dtensor_contract(dtensor: DTensor, context: str | None = None) -> None:
    """Fail symmetrically when ranks disagree on rank-invariant DTensor metadata."""
    contract = _dtensor_contract(dtensor, context)
    digest = _contract_digest(contract)
    mismatched_groups = []

    # Every rank traverses every mesh dimension before deciding whether to fail. This keeps
    # placement mismatches from making one rank enter a shape collective while a peer skips it.
    for mesh_dim in range(dtensor.device_mesh.ndim):
        group = dtensor.device_mesh.get_group(mesh_dim)
        group_digests = _all_gather_int64(
            digest, group, device_type=dtensor.device_mesh.device_type
        )
        if any(group_digest != group_digests[0] for group_digest in group_digests[1:]):
            mismatched_groups.append((mesh_dim, group_digests))

    if _mesh_has_failure(bool(mismatched_groups), dtensor.device_mesh):
        details = (
            f"Mismatched mesh groups: {mismatched_groups}."
            if mismatched_groups
            else "A mismatch was detected by another rank in the device mesh."
        )
        raise CheckpointingException(
            "[Megatron-FSDP] Uneven DTensor metadata contract mismatch before chunk "
            f"collective. Local contract: {contract}. {details}"
        )


def _compute_chunk_metadata(
    dtensor: DTensor,
) -> tuple[ChunkStorageMetadata, tuple[int, ...], tuple[int, ...], list]:
    """Collect local shapes and return metadata plus shape-validation details."""
    local_tensor = dtensor.to_local()
    local_shape = local_tensor.shape
    device_mesh = dtensor.device_mesh
    offsets = [0] * len(local_shape)
    cumulative_shape = list(local_shape).copy()
    gathered_shapes = []

    def _update_offsets_and_cumulative_shape(
        mesh_dim: int, offsets: List[int], cumulative_shape: List[int]
    ):
        shard_group = device_mesh.get_group(mesh_dim)
        shard_dim = p.dim

        # The contract preflight guarantees that every rank uses the same number of
        # dimensions, so this fixed-shape tensor collective cannot disagree on sizes.
        global_shapes = _all_gather_int64(
            cumulative_shape,
            shard_group,
            device_type=device_mesh.device_type,
        )
        gathered_shapes.append((mesh_dim, global_shapes))

        # Calculate global offset for current rank's shard
        rank = dist.get_rank(shard_group)
        offset = sum(s[shard_dim] for s in global_shapes[:rank])
        # TODO: add documentation for the offset calculation
        # Add on the offset of the current mesh dimension
        offsets[shard_dim] += offset
        # Calculate the global shape using the sum of the sharding dim sizes.
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

    expected_shape = tuple(int(size) for size in dtensor.shape)
    actual_shape = tuple(cumulative_shape)
    return (
        ChunkStorageMetadata(offsets=tuple(offsets), sizes=tuple(local_shape)),
        expected_shape,
        actual_shape,
        gathered_shapes,
    )


def _raise_for_invalid_shard_composition(
    dtensor: DTensor,
    expected_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
    gathered_shapes: list,
) -> None:
    """Raise symmetrically when local shards do not compose the logical shape."""
    shape_mismatch = actual_shape != expected_shape
    if _mesh_has_failure(shape_mismatch, dtensor.device_mesh):
        details = (
            f"Computed shape: {actual_shape}; gathered shapes: {gathered_shapes}."
            if shape_mismatch
            else "An invalid shard composition was detected by another rank in the device mesh."
        )
        raise CheckpointingException(
            "[Megatron-FSDP] Uneven DTensor local shards do not compose the declared "
            f"global shape {expected_shape}. Local shape: {tuple(dtensor.to_local().shape)}. "
            f"{details}"
        )


def gather_and_compute_chunk_metadata(
    dtensor: DTensor, *, context: str | None = None
) -> ChunkStorageMetadata:
    """
    Gather chunk metadata for a DTensor across all ranks and compute the
    offsets and sizes of each chunk. This is necessary for handling uneven
    sharding in distributed tensors.
    """
    _validate_dtensor_contract(dtensor, context)
    chunk_metadata, expected_shape, actual_shape, gathered_shapes = (
        _compute_chunk_metadata(dtensor)
    )
    _raise_for_invalid_shard_composition(
        dtensor, expected_shape, actual_shape, gathered_shapes
    )
    return chunk_metadata


def _install_uneven_dtensor_chunk_metadata(
    dtensor: DTensor, uneven_chunk_meta: ChunkStorageMetadata
) -> None:
    """Install PyTorch DCP chunk and write-item hooks on a DTensor."""
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

    dtensor._local_tensor.__create_chunk_list__ = _chunk_list_closure([uneven_chunk_meta])
    dtensor._local_tensor.__create_write_items__ = _write_items_closure(uneven_chunk_meta)


def update_uneven_dtensor_chunk_metadata(
    dtensor: DTensor, *, context: str | None = None
) -> dict:
    """
    Update the DTensor's chunk metadata to handle uneven sharding.
    This function modifies the DTensor in-place to include chunk metadata
    and write items closures for saving and loading.
    """
    uneven_chunk_meta = gather_and_compute_chunk_metadata(dtensor, context=context)
    _install_uneven_dtensor_chunk_metadata(dtensor, uneven_chunk_meta)


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
    # Skip under fake process group — all_reduce is a no-op so only rank 0's
    # boundaries are visible, which makes the end-boundary check always fail.
    if torch.distributed.is_initialized() and torch.distributed.get_backend() == 'fake':
        return

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


def _validate_state_dict_contract(dtensors: list[tuple[list, str, DTensor]]):
    """Validate all state-dict keys and DTensor contracts with one collective."""
    if not dtensors:
        return None

    contract_entries = tuple(
        f"key_chain={key_path};{_dtensor_contract(dtensor)}"
        for _, key_path, dtensor in dtensors
    )
    manifest_digest = hashlib.sha256(repr(contract_entries).encode("utf-8")).hexdigest()
    contract_digest = [len(contract_entries), *_contract_digest(repr(contract_entries))]
    anchor_mesh = dtensors[0][2].device_mesh
    mismatched_groups = []
    for mesh_dim in range(anchor_mesh.ndim):
        group = anchor_mesh.get_group(mesh_dim)
        group_contracts = _all_gather_int64(
            contract_digest, group, device_type=anchor_mesh.device_type
        )
        if any(contract != group_contracts[0] for contract in group_contracts[1:]):
            mismatched_groups.append((mesh_dim, group_contracts))

    if _mesh_has_failure(bool(mismatched_groups), anchor_mesh):
        details = (
            f"Mismatched mesh groups: {mismatched_groups}."
            if mismatched_groups
            else "A mismatch was detected by another rank in the device mesh."
        )
        raise CheckpointingException(
            "[Megatron-FSDP] Uneven DTensor metadata contract mismatch before chunk "
            f"collective. Local manifest: entries={len(contract_entries)};"
            f"sha256={manifest_digest}. {details}"
        )
    return anchor_mesh


def _get_state_dict_dtensors(state_dict: dict) -> list[tuple[list, str, DTensor]]:
    """Return DTensors in deterministic state-dict key order."""
    key_chains = sorted(
        filter_unflattened_state_dict(
            state_dict, visit_condition=lambda value: isinstance(value, DTensor)
        )
    )
    return [
        (
            key_chain,
            repr(tuple(key_chain)),
            get_unflattened_state_dict(state_dict, key_chain),
        )
        for key_chain in key_chains
    ]


def validate_state_dict_for_uneven_dtensor(state_dict: dict) -> dict:
    """Validate a state dict before transformations that may enter DTensor collectives."""
    dtensors = _get_state_dict_dtensors(state_dict)
    if dtensors:
        _validate_state_dict_contract(dtensors)
    return state_dict


def preprocess_state_dict_for_uneven_dtensor(state_dict: dict) -> dict:
    """
    Preprocess the state_dict to prepare it for saving or loading unevenly sharded DTensors.
    This function modifies the DTensors in the state_dict to include chunk metadata
    and write items closures.
    """
    dtensors = _get_state_dict_dtensors(state_dict)

    if not dtensors:
        return state_dict

    # Validate the complete key/metadata manifest once. Once this passes, every rank
    # is guaranteed to enter the same fixed-size shape collectives in the same order.
    anchor_mesh = _validate_state_dict_contract(dtensors)

    computed_metadata = []
    local_shape_failures = []
    for _, key_path, dtensor in dtensors:
        chunk_metadata, expected_shape, actual_shape, gathered_shapes = (
            _compute_chunk_metadata(dtensor)
        )
        computed_metadata.append((dtensor, chunk_metadata))
        if actual_shape != expected_shape:
            local_shape_failures.append(
                f"key_chain={key_path};expected_shape={expected_shape};"
                f"actual_shape={actual_shape};gathered_shapes={gathered_shapes}"
            )

    # Shape collectives are now complete, so one mesh reduction can propagate every
    # tensor's composition failure instead of reducing once per tensor.
    if _mesh_has_failure(bool(local_shape_failures), anchor_mesh):
        details = (
            f"Local failures: {local_shape_failures}."
            if local_shape_failures
            else "An invalid shard composition was detected by another rank."
        )
        raise CheckpointingException(
            "[Megatron-FSDP] Uneven DTensor local shards do not compose their declared "
            f"global shapes. {details}"
        )

    for dtensor, chunk_metadata in computed_metadata:
        _install_uneven_dtensor_chunk_metadata(dtensor, chunk_metadata)
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


def gather_uneven_dtensor_to_full_tensor(dtensor: DTensor) -> DTensor:
    """
    Deprecated: use `redistribute_uneven_dtensor_to_replicated` instead.
    """
    return redistribute_uneven_dtensor_to_replicated(dtensor)


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
