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

from typing import Iterable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType
from torch.distributed.checkpoint.state_dict import get_state_dict as _get_state_dict
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard, _StridedShard


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

    return ChunkStorageMetadata(offsets=tuple(offsets), sizes=tuple(local_shape))


def update_uneven_dtensor_chunk_metadata(dtensor: DTensor, source: str = "init") -> dict:
    """
    Update the DTensor's chunk metadata to handle uneven sharding.
    This function modifies the DTensor in-place to include chunk metadata
    and write items closures for saving and loading.
    """
    # Get uneven chunk metadata for the DTensor
    # TODO: Optimize gather_and_compute_chunk_metadata synchronization:
    # 1. Add pre-check validation to verify tensor shape consistency
    #    across devices before entering barrier (prevents potential hangs)
    # 2. Implement batched barrier using grouped collectives
    #    to amortize synchronization overhead
    uneven_chunk_meta = gather_and_compute_chunk_metadata(dtensor)

    # Set the chunk list and write items closure for the DTensor
    _set_chunk_metadata(dtensor, uneven_chunk_meta.offsets, uneven_chunk_meta.sizes, source=source)


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


def preprocess_state_dict_for_uneven_dtensor(state_dict: dict) -> dict:
    """
    Preprocess the state_dict to prepare it for saving or loading unevenly sharded DTensors.
    This function modifies the DTensors in the state_dict to include chunk metadata
    and write items closures.
    """
    visit_dtensor = filter_unflattened_state_dict(
        state_dict, visit_condition=lambda x: isinstance(x, DTensor)
    )
    # Sort the keys, since some state dictionaries are mocked
    # and extended to include empty global keys.
    for key_chain in sorted(visit_dtensor):
        # Get the DTensor at the key chain
        dtensor = get_unflattened_state_dict(state_dict, key_chain)
        update_uneven_dtensor_chunk_metadata(dtensor, source="preprocess")
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
    Split a DTensor into smaller DTensors along a specified dimension.

    This function handles uneven sharding correctly by computing chunk metadata
    for every output DTensor without extra collective operations.  All subsequent
    per-split metadata is derived locally from that result using pure integer
    arithmetic.

    Unlike the native PyTorch ``DTensor.split``, this function does **not**
    redistribute ``Replicate`` placements, which avoids OOM issues when the
    full tensor is large.

    Chunk metadata assignment strategy
    ------------------------------------
    For each split window ``[split_start, split_end)`` along *dim*:

    * The rank's local chunk covers the global interval
      ``[local_start, local_end)`` where::

          local_start = chunk_meta.offsets[dim]
          local_end   = local_start + chunk_meta.sizes[dim]

    * The overlap with the split window is::

          overlap_start = max(local_start, split_start)
          overlap_end   = min(local_end,   split_end)

    * If ``overlap_start < overlap_end`` the rank owns part of this split:

      - ``new_offsets[dim] = overlap_start - split_start``
        (offset is relative to the split's own global origin)
      - ``new_sizes[dim]   = overlap_end - overlap_start``

    * Otherwise the rank owns nothing in this split:

      - ``new_offsets[dim] = 0``, ``new_sizes[dim] = 0``

    All other dimensions are copied unchanged from the parent's chunk metadata.

    Args:
        dtensor (DTensor): The DTensor to split.  Must be compatible with
            ``gather_and_compute_chunk_metadata`` (placements are ``Shard``,
            ``_StridedShard``, or ``Replicate``).
        split_size_or_sections (int | list[int]): If an ``int``, each chunk
            has this size (the last chunk may be smaller).  If a ``list``,
            each element is the exact size of the corresponding chunk; the
            sizes must sum to ``dtensor.shape[dim]``.
        dim (int, optional): Dimension along which to split.  Default: ``0``.
        update_uneven_dtensor_chunk_meta (bool, optional): If ``True``, call
            ``update_uneven_dtensor_chunk_metadata`` for every output DTensor
            instead of using the locally-derived metadata.  This triggers one
            collective per split chunk and is only needed when you require a
            full round-trip validation or the parent's chunk metadata cannot
            be trusted.  Default: ``False``.

    Yields:
        DTensor: Split sub-DTensors in order.  Each yielded DTensor:

        * Shares the same ``placements`` and ``device_mesh`` as *dtensor*.
        * Has shape ``dtensor.shape`` with ``shape[dim]`` replaced by the
          split-window size.
        * Has ``__create_chunk_list__`` / ``__create_write_items__`` set on
          its local tensor (via ``_set_chunk_metadata``), so checkpoint
          writers can consume it directly without further collectives.

    Raises:
        ValueError: If the locally-computed split slice is internally
            inconsistent (negative start after offset correction).

    Note:
        ``gather_and_compute_chunk_metadata`` is called exactly **once**,
        regardless of the number of splits.  When
        ``update_uneven_dtensor_chunk_meta=False`` (the default) no further
        collective operations are performed.

    Example::

        # Split a [1024, 512] DTensor sharded on dim-0 across 4 ranks
        # into four [256, 512] chunks — one collective, zero extra collectives
        # for metadata.
        for chunk in split_dtensor(dt, split_size_or_sections=256, dim=0):
            process(chunk)

        # Variable-size split, chunk metadata derived locally
        for chunk in split_dtensor(dt, split_size_or_sections=[300, 300, 424], dim=0):
            process(chunk)
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

    # One collective call — result reused for all splits below.
    assert hasattr(dtensor._local_tensor, "__create_chunk_list__"), (
        "DTensor local tensor is missing chunk metadata."
    )
    chunk_meta = dtensor._local_tensor.__create_chunk_list__()[0]
    chunk_slice = slice(chunk_meta.offsets[dim], chunk_meta.offsets[dim] + chunk_meta.sizes[dim])
    local_offset = chunk_meta.offsets[dim]
    local_tensor = dtensor.to_local()

    for i in range(len(split_points) - 1):
        split_slice = slice(split_points[i], split_points[i + 1])

        # Compute the intersection of this rank's local chunk with the split
        # window.  _intersection returns slice(0, 0) when there is no overlap,
        # which produces an empty tensor via narrow() — that is the correct
        # behaviour for ranks that own nothing in this split window.
        s = _intersection(split_slice, chunk_slice)
        if s.start < s.stop:
            # Translate to local-tensor coordinates.
            s = _offset_slice(s, -local_offset)

        if s.start < 0:
            raise ValueError(
                f"Invalid split slice {s} for DTensor with shape {dtensor.shape} "
                f"and local offset {local_offset} on dimension {dim}. "
                "This is a bug — please report it."
            )

        # Slice the local tensor along `dim`.
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
            # Triggers one collective per split — use only when a validated
            # round-trip is required.
            update_uneven_dtensor_chunk_metadata(new_dtensor)
        else:
            # Derive chunk metadata locally from the already-computed
            # chunk_meta — zero additional collectives.
            #
            # Compute the intersection in *global* coordinates so that the
            # resulting offset is expressed relative to the split's own origin.
            global_local_start = chunk_meta.offsets[dim]
            global_local_end = global_local_start + chunk_meta.sizes[dim]
            global_split_start = split_points[i]
            global_split_end = split_points[i + 1]

            overlap_start = max(global_local_start, global_split_start)
            overlap_end = min(global_local_end, global_split_end)

            new_offsets = list(chunk_meta.offsets)
            new_sizes = list(chunk_meta.sizes)

            if overlap_start < overlap_end:
                # This rank owns [overlap_start, overlap_end) of the global
                # tensor; the offset within the split chunk is the distance
                # from the split's start.
                new_offsets[dim] = overlap_start - global_split_start
                new_sizes[dim] = overlap_end - overlap_start
            else:
                # This rank owns nothing in this split window.
                new_offsets[dim] = 0
                new_sizes[dim] = 0

            _set_chunk_metadata(new_dtensor, tuple(new_offsets), tuple(new_sizes), source="split")

        yield new_dtensor


def make_uneven_dtensor(
    local_tensor: torch.Tensor,
    shape: torch.Size,
    dp_mesh: DeviceMesh,
    placements: List[Placement],
    *,
    post_process_uneven: bool = False,
    copy_chunk_meta_from: Optional[DTensor] = None,
    chunk_metadata: Optional[tuple] = None,
):
    """Create a DTensor from a possibly uneven local shard with known global shape.

    Args:
        local_tensor: Local shard tensor.
        shape: Global shape of the full DTensor.
        dp_mesh: 1D device mesh.
        placements: DTensor placements (e.g., [Shard(0)]).
        post_process_uneven: If True, call ``update_uneven_dtensor_chunk_metadata``.
        copy_chunk_meta_from: If set, copy ``__create_chunk_list__`` /
            ``__create_write_items__`` from this DTensor.
        chunk_metadata: ``(offsets, sizes)`` tuple where *offsets* and *sizes*
            are tuples of ints (one per dimension).  Sets chunk metadata
            closures without collectives.
    """
    assert dp_mesh.ndim == 1, "Only 1D mesh is supported for now"
    if local_tensor.numel() == 0:
        local_shape = (0,) + tuple(shape[1:]) if len(shape) > 1 else (0,)
        local_tensor = local_tensor.reshape(local_shape)
    else:
        local_tensor = local_tensor.view(-1, *shape[1:])
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=dp_mesh,
        placements=placements,
        run_check=False,
        shape=shape,
        stride=torch.empty(shape, device="meta").stride(),
    )
    if post_process_uneven:
        update_uneven_dtensor_chunk_metadata(dtensor)
    elif copy_chunk_meta_from is not None:
        # This branch is used for the case where we are creating a new DTensor that has the same
        # sharding as an existing uneven DTensor, so we can copy the chunk metadata from the
        # existing uneven DTensor instead of recomputing it.
        copy_chunk_metadata(copy_chunk_meta_from, dtensor)
    elif chunk_metadata is not None:
        _set_chunk_metadata(dtensor, *chunk_metadata, source="make_uneven")
    return dtensor


def get_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    *,
    submodules: Optional[set[nn.Module]] = None,
    options: Optional["StateDictOptions"] = None,
) -> tuple[dict[str, "ValueType"], "OptimizerStateType"]:
    """Produce model and optimizer state dicts with uneven DTensor preprocessing.

    PyTorch's ``get_state_dict`` clones every DTensor, so the returned
    tensors lack ``__create_chunk_list__`` / ``__create_write_items__``.
    Instead of recomputing metadata via ``all_gather_object``, we copy it
    from the model's ``dist_params`` (which carry metadata from FSDP init)
    to the corresponding state-dict entries — a zero-collective operation.
    """
    for param in model.parameters():
        assert isinstance(param, DTensor), "Expected all parameters to be DTensors"

    model_state_dict, optimizer_state_dict = _get_state_dict(
        model=model, optimizers=optimizers, submodules=submodules, options=options
    )

    # Build FQN → model DTensor mapping (dist_params carry chunk metadata).
    param_by_fqn: dict[str, DTensor] = {}
    for fqn, p in model.named_parameters():
        if isinstance(p, DTensor) and hasattr(p._local_tensor, "__create_chunk_list__"):
            param_by_fqn[fqn] = p

    # Copy chunk metadata into model state-dict entries.
    for fqn, dt in model_state_dict.items():
        if isinstance(dt, DTensor) and not hasattr(dt._local_tensor, "__create_chunk_list__"):
            src = param_by_fqn.get(fqn)
            if src is not None:
                copy_chunk_metadata(src, dt)

    # Copy chunk metadata into optimizer state-dict entries.
    optim_state = optimizer_state_dict.get("state", {})
    for fqn, state_tensors in optim_state.items():
        src = param_by_fqn.get(fqn)
        if src is None:
            continue
        for key, dt in state_tensors.items():
            if isinstance(dt, DTensor) and not hasattr(dt._local_tensor, "__create_chunk_list__"):
                copy_chunk_metadata(src, dt)

    return model_state_dict, optimizer_state_dict


# ------------------------------------------------------------------
# Chunk metadata helpers (zero-collective)
# ------------------------------------------------------------------


def copy_chunk_metadata(src: DTensor, dst: DTensor) -> None:
    """Copy ``__create_chunk_list__`` / ``__create_write_items__`` from *src* to *dst*."""
    dst._local_tensor.__create_chunk_list__ = src._local_tensor.__create_chunk_list__
    dst._local_tensor.__create_write_items__ = src._local_tensor.__create_write_items__
    src_source = getattr(src._local_tensor, "_chunk_meta_source", None)
    if src_source is not None:
        dst._local_tensor._chunk_meta_source = f"propagate:{src_source}"
    else:
        dst._local_tensor._chunk_meta_source = "propagate:unknown"


def get_chunk_meta_source(dtensor: DTensor) -> str:
    """Return the source tag for *dtensor*'s chunk metadata, or ``"none"``."""
    return getattr(dtensor._local_tensor, "_chunk_meta_source", "none")


def compute_split_offsets_and_sizes(dist_param, split_dim, comp_idx, total_split, comp_data):
    """Compute chunk offsets/sizes for a split component, derived from *dist_param*'s metadata.

    Pure local computation — no collectives.
    """
    chunk_list = dist_param._local_tensor.__create_chunk_list__()
    orig = chunk_list[0]
    global_shape = list(dist_param.size())

    comp_size = global_shape[split_dim] // total_split
    comp_start = comp_idx * comp_size
    comp_end = comp_start + comp_size

    offsets = list(orig.offsets)
    sizes = list(comp_data.shape)

    o = offsets[split_dim]
    s = orig.sizes[split_dim]

    if o < comp_end and o + s > comp_start:
        offsets[split_dim] = max(o, comp_start) - comp_start
        sizes[split_dim] = min(o + s, comp_end) - max(o, comp_start)
    else:
        offsets[split_dim] = 0
        sizes[split_dim] = 0

    return tuple(offsets), tuple(sizes)


def get_fsdp_slice_from_uneven_dtensor(dist_param: DTensor) -> slice:
    """Compute the FSDP slice (flattened range) from a v2 DTensor.

    Uses the uneven chunk metadata (``__create_chunk_list__``) attached by
    ``update_uneven_dtensor_chunk_metadata`` to correctly handle uneven
    sharding where ranks may own different-sized slices.

    The DTensor must have ``__create_chunk_list__`` set on its local tensor
    (via ``preprocess_state_dict_for_uneven_dtensor`` or
    ``update_uneven_dtensor_chunk_metadata``) before calling this function.
    """
    local_numel = dist_param._local_tensor.numel()
    if local_numel == 0:
        return slice(0, 0)

    assert hasattr(dist_param._local_tensor, "__create_chunk_list__"), (
        "get_fsdp_slice_from_uneven_dtensor requires the DTensor to have "
        "__create_chunk_list__ metadata. Call update_uneven_dtensor_chunk_metadata "
        "or preprocess_state_dict_for_uneven_dtensor first."
    )

    chunk_list = dist_param._local_tensor.__create_chunk_list__()
    assert len(chunk_list) == 1, f"Expected exactly one chunk per rank, got {len(chunk_list)}"
    chunk_meta = chunk_list[0]
    offsets = chunk_meta.offsets
    sizes = chunk_meta.sizes

    global_shape = dist_param.size()
    strides = torch.empty(global_shape, device="meta").stride()

    start = sum(o * s for o, s in zip(offsets, strides))
    return slice(start, start + local_numel)


def _set_chunk_metadata(
    dtensor: DTensor, offsets: tuple, sizes: tuple, source: str = "unknown"
) -> None:
    """Set ``__create_chunk_list__`` / ``__create_write_items__`` closures on *dtensor*.

    No collective ops — *offsets* and *sizes* are computed locally.

    Args:
        source: A tag describing where the metadata was set, e.g. ``"init"``,
            ``"preprocess"``, ``"split"``.  Stored on the local tensor as
            ``_chunk_meta_source`` for diagnostic tracing.
    """
    chunk_meta = ChunkStorageMetadata(offsets=tuple(offsets), sizes=tuple(sizes))

    def _write_items(fqn: str, tensor: DTensor) -> list:
        if tensor.to_local().numel() == 0:
            return []
        return [
            WriteItem(
                type=WriteItemType.SHARD,
                index=MetadataIndex(fqn, chunk_meta.offsets),
                tensor_data=TensorWriteData(
                    chunk=chunk_meta,
                    properties=TensorProperties.create_from_tensor(tensor.to_local()),
                    size=tensor.size(),
                ),
            )
        ]

    dtensor._local_tensor.__create_chunk_list__ = lambda: [chunk_meta]
    dtensor._local_tensor.__create_write_items__ = _write_items
    dtensor._local_tensor._chunk_meta_source = source
