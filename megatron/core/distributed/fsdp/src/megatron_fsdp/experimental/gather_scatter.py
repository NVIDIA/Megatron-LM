# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute P2P gather/scatter for MFSDP v2.

- `gather` moves each participating parameter's local chunks to its owner and reconstructs the full
  tensor there;
- `scatter` sends the owner's full result back as per-rank flat chunks.

The sharded side is a `DBuffer`, which supplies the data, mesh, layout, and placements.
`tensor_to_owner` selects participating tensors and maps their indices to owner global ranks.
Both operations support one-dimensional meshes with sharded placements.
Each tensor's owner holds a shard, so any peer pair shares at most one tensor across their
disjoint buffer ranges. Tensor traversal order therefore need not match across ranks.

Operations are posted on the current CUDA stream and return their P2P works without waiting.
Waiting on a work orders the stream it is called on after the transfer, so the caller decides
when to wait. Multiple parameter groups can be pipelined by posting several before waiting, or by
executing the functions in different streams. Callers explicitly wait for inputs to be ready on
each stream and for results to be ready before consuming them on another stream.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard

from .dbuffer import DBuffer
from .range import Range, intersect_ranges


def _get_sorted_rank_ranges(buffer: DBuffer) -> dict[int, Range]:
    """Return sharded buffer ranges keyed by global rank, in buffer offset order."""
    mesh = buffer.mesh
    if mesh.ndim != 1 or not isinstance(buffer.placements[0], Shard):
        raise NotImplementedError("Owner gather/scatter requires a 1-D sharded DBuffer.")
    rank_ranges = buffer.layout.get_rank_ranges(mesh, buffer.placements)
    return dict(sorted(rank_ranges.items(), key=lambda item: item[1].start))


def find_split_tensors(buffer: DBuffer) -> frozenset[int]:
    """Return the indices of tensors whose elements span more than one rank.

    Gather and scatter post P2P only for these tensors. The result depends only on the layout,
    so every rank of the mesh computes the same set.
    """
    rank_ranges = _get_sorted_rank_ranges(buffer).values()
    layout = buffer.layout
    return frozenset(
        i
        for i in range(len(layout.tensor_shapes))
        if sum(intersect_ranges(layout.get_tensor_range(i), r).numel > 0 for r in rank_ranges) > 1
    )


def gather(
    source: DBuffer, destination: dict[int, torch.Tensor], *, tensor_to_owner: dict[int, int]
) -> list[dist.Work]:
    """Gather full tensors from local chunks to their owners via P2P.

    The source buffers must share dtype and device type across ranks.

    For each participating parameter (a key of `tensor_to_owner`):

    - If this rank owns it, each shard is placed at its tensor-relative offset in
      `destination[i]`. Fully local parameters are copied straight into `destination[i]`.
    - Otherwise, this rank's local chunk is sent to the owner.

    Args:
        source: Sharded buffer containing this rank's local parameter chunks. A separate
            DBuffer with the same layout can hold computed inputs such as momentum updates.
        destination: Preallocated contiguous full tensor per owned parameter, with the
            source's dtype and device and the matching number of elements in any shape.
            Receives write directly into these tensors. Entries for parameters this rank
            does not own are never written to.
        tensor_to_owner: Participating tensor index to owner global rank. Each owner must
            hold a nonempty shard of its tensor; see `assign_owner_work`.

    Returns:
        The posted P2P works, empty if this rank has none. Wait on each before using the
        destination, and keep the buffers alive until then.
    """
    mesh = source.mesh
    this_rank = mesh.get_rank()
    layout = source.layout
    rank_ranges = _get_sorted_rank_ranges(source)
    group = mesh.get_group()

    # Build the P2P ops: send non-owned chunks to their owners, receive owned ones.
    ops: list[dist.P2POp] = []
    for i, owner_rank in tensor_to_owner.items():
        tensor_range = layout.get_tensor_range(i)
        if owner_rank != this_rank:
            if intersect_ranges(tensor_range, rank_ranges[this_rank]).numel > 0:
                chunk = source.get_tensor_view(i).flatten()
                ops.append(dist.P2POp(dist.isend, chunk, peer=owner_rank, group=group))
            continue

        target = destination[i]
        if (
            not target.is_contiguous()
            or target.dtype != source.dtype
            or target.device != source.device
        ):
            raise ValueError(
                f"destination[{i}] must be contiguous and match the source's dtype and device."
            )
        full = target.view(tensor_range.numel)

        for src_rank, rank_range in rank_ranges.items():
            shard_range = intersect_ranges(tensor_range, rank_range)
            if shard_range.numel == 0:
                continue
            offset = shard_range.start - tensor_range.start
            chunk = full.narrow(0, offset, shard_range.numel)
            if src_rank == this_rank:
                chunk.copy_(source.get_tensor_view(i).flatten())
            else:
                ops.append(dist.P2POp(dist.irecv, chunk, peer=src_rank, group=group))

    return dist.batch_isend_irecv(ops) if ops else []


def scatter(
    source: dict[int, torch.Tensor], destination: DBuffer, *, tensor_to_owner: dict[int, int]
) -> list[dist.Work]:
    """Scatter full result tensors from the owners to the other ranks via P2P.

    All source tensors and destination buffers must share dtype and device type across ranks.

    For each participating parameter (a key of `tensor_to_owner`):

    - If this rank owns it, `source[i]` is viewed flat and each rank's
      shard is sliced out and sent to it directly. The owner keeps its own result chunk.
    - Otherwise, if this rank holds elements of the parameter, the chunk is received directly
      into `destination.get_tensor_view(i)`. The owner's local destination shard is unchanged.

    Args:
        source: Contiguous full result tensor per owned parameter, with the destination's dtype
            and device and the matching number of elements in any shape. Entries for parameters
            this rank does not own are never read.
        destination: Sharded result buffer. P2P receives write only remote-owned participating
            tensor shards; other tensors and padding are unchanged. This may
            be a separate scratch DBuffer for updates the caller will apply later.
        tensor_to_owner: Participating tensor index to owner global rank. Each owner must
            hold a nonempty shard of its tensor; see `assign_owner_work`.

    Returns:
        The posted P2P works, empty if this rank has none. Wait on each before using the
        destination, and keep the buffers alive until then.
    """
    mesh = destination.mesh
    this_rank = mesh.get_rank()
    layout = destination.layout
    rank_ranges = _get_sorted_rank_ranges(destination)
    group = mesh.get_group()

    # Build the P2P ops: send owned result chunks, receive the ones held here.
    ops: list[dist.P2POp] = []
    for i, owner in tensor_to_owner.items():
        tensor_range = layout.get_tensor_range(i)
        if owner != this_rank:
            if intersect_ranges(tensor_range, rank_ranges[this_rank]).numel > 0:
                chunk = destination.get_tensor_view(i).flatten()
                ops.append(dist.P2POp(dist.irecv, chunk, peer=owner, group=group))
            continue

        tensor = source[i]
        if (
            not tensor.is_contiguous()
            or tensor.dtype != destination.dtype
            or tensor.device != destination.device
        ):
            raise ValueError(
                f"source[{i}] must be contiguous and match the destination's dtype and device."
            )
        flat = tensor.view(tensor_range.numel)
        for dest in rank_ranges:
            if dest == this_rank:
                continue
            shard_range = intersect_ranges(tensor_range, rank_ranges[dest])
            numel = shard_range.numel
            offset = shard_range.start - tensor_range.start
            if numel == 0:
                continue
            ops.append(
                dist.P2POp(dist.isend, flat[offset : offset + numel], peer=dest, group=group)
            )

    return dist.batch_isend_irecv(ops) if ops else []
