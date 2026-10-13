# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute P2P gather/scatter for MFSDP v2.

- `gather` moves each participating parameter's local chunks to its owner and reconstructs the full
  tensor there;
- `scatter` sends the owner's full result back as per-rank flat chunks.

The sharded side is a `DBuffer`, which supplies the data, mesh, layout, and placements.
`tensor_to_owner` selects participating tensors and maps their indices to owner global ranks.
Both operations support one-dimensional meshes with sharded placements.

Operations run on the current CUDA stream. Multiple parameter groups can be pipelined by
executing the functions in different streams. Callers explicitly wait for inputs to be ready on
each stream and for results to be ready before consuming them on another stream.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard

from .dbuffer import DBuffer
from .range import Range, intersect_ranges


def _get_rank_ranges(buffer: DBuffer) -> dict[int, Range]:
    """Return sharded buffer ranges keyed by global rank, in buffer offset order."""
    mesh = buffer.mesh
    if mesh.ndim != 1 or not isinstance(buffer.placements[0], Shard):
        raise NotImplementedError("Owner gather/scatter requires a 1-D sharded DBuffer.")
    ranges = {
        rank: buffer.layout.get_rank_range(mesh, buffer.placements, rank)
        for rank in mesh.mesh.flatten().tolist()
    }
    return dict(sorted(ranges.items(), key=lambda item: item[1].start))


def gather(
    source: DBuffer, destination: dict[int, torch.Tensor], *, tensor_to_owner: dict[int, int]
) -> None:
    """Gather full tensors from local chunks to their owners via P2P.

    The source buffers must share dtype and device type across ranks.

    For each participating parameter (a key of `tensor_to_owner`):

    - If this rank owns it, `destination[i]` is filled with the full flat tensor, reconstructed by
      concatenating the per-rank chunks in buffer order. Fully local
      parameters (i.e., non-boundary parameters) are copied straight into `destination[i]`.
    - Otherwise, this rank's local chunk is sent to the owner.

    Args:
        source: Sharded buffer containing this rank's local parameter chunks. A separate
            DBuffer with the same layout can hold computed inputs such as momentum updates.
        destination: Preallocated full tensor per owned parameter, in any shape with the
            matching number of elements; filled with the flat content viewed into the
            destination's shape. Entries for parameters this rank does not own are never
            written to.
        tensor_to_owner: Participating tensor index to owner global rank. Each owner must
            hold a nonempty shard of its tensor; see `assign_owner_work`.
    """
    mesh = source.mesh
    this_rank = mesh.get_rank()
    layout = source.layout
    rank_ranges = _get_rank_ranges(source)
    group = mesh.get_group()
    owned_tensor_indices = [i for i, owner in sorted(tensor_to_owner.items()) if owner == this_rank]

    # Allocate the flat recv buffers (one per (tensor index, source rank) pair).
    recv_chunks: dict[tuple[int, int], torch.Tensor] = {}
    for i in owned_tensor_indices:
        for src_rank in rank_ranges:
            if src_rank == this_rank:
                continue
            numel = intersect_ranges(layout.get_tensor_range(i), rank_ranges[src_rank]).numel
            if numel > 0:
                recv_chunks[(i, src_rank)] = torch.empty(
                    numel, dtype=source.dtype, device=source.device
                )

    # Build the P2P ops: send non-owned chunks to their owners, receive owned ones.
    ops: list[dist.P2POp] = []
    for i, owner_rank in sorted(tensor_to_owner.items()):
        # Don't send already local or empty chunks.
        if (
            owner_rank == this_rank
            or intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel == 0
        ):
            continue
        chunk = source.get_tensor_view(i).flatten()
        ops.append(dist.P2POp(dist.isend, chunk, peer=owner_rank, group=group))
    for (_, src_rank), buf in recv_chunks.items():
        ops.append(dist.P2POp(dist.irecv, buf, peer=src_rank, group=group))

    works = dist.batch_isend_irecv(ops) if ops else []
    for work in works:
        work.wait()

    # Reconstruct the full flat tensors for owned parameters into `destination`.
    for i in owned_tensor_indices:
        chunks: list[torch.Tensor] = []
        for src_rank in rank_ranges:
            if src_rank == this_rank:
                chunk = source.get_tensor_view(i)
                if chunk.numel() > 0:
                    chunks.append(chunk.flatten())
            else:
                chunk = recv_chunks.get((i, src_rank))
                if chunk is not None:
                    chunks.append(chunk)
        # Avoid a copy if we have only one chunk element.
        full = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        destination[i].copy_(full.view(destination[i].shape))


def scatter(
    source: dict[int, torch.Tensor], destination: DBuffer, *, tensor_to_owner: dict[int, int]
) -> None:
    """Scatter full result tensors from the owners to the other ranks via P2P.

    All source tensors and destination buffers must share dtype and device type across ranks.

    For each participating parameter (a key of `tensor_to_owner`):

    - If this rank owns it, `source[i]` is flattened (a view on contiguous tensors) and each rank's
      shard is sliced out and sent to it directly. The owner keeps its own result chunk.
    - Otherwise, if this rank holds elements of the parameter, the received flat chunk is copied
      into `destination.get_tensor_view(i)`. The owner's local destination shard is unchanged.

    Args:
        source: Full result tensor per owned parameter, in any shape with the matching number of
            elements; flattened for slicing. Entries for parameters this rank does not own
            are never read.
        destination: Sharded result buffer. Only remote-owned participating tensor shards are
            written after the P2P completes; other tensors and padding are unchanged. This may
            be a separate scratch DBuffer for updates the caller will apply later.
        tensor_to_owner: Participating tensor index to owner global rank. Each owner must
            hold a nonempty shard of its tensor; see `assign_owner_work`.
    """
    mesh = destination.mesh
    this_rank = mesh.get_rank()
    layout = destination.layout
    rank_ranges = _get_rank_ranges(destination)
    group = mesh.get_group()
    held_tensor_indices = [
        i
        for i, owner in sorted(tensor_to_owner.items())
        if owner != this_rank
        and intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel > 0
    ]

    # Allocate the recv buffers.
    recv_chunks: dict[int, torch.Tensor] = {}
    for i in held_tensor_indices:
        numel = intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel
        recv_chunks[i] = torch.empty(numel, dtype=destination.dtype, device=destination.device)

    # Build the P2P ops: send owned result chunks, receive the ones held here.
    ops: list[dist.P2POp] = []
    for i, owner in sorted(tensor_to_owner.items()):
        # Don't send local chunks.
        if owner != this_rank:
            continue
        flat = source[i].flatten().contiguous()
        for dest in rank_ranges:
            if dest == this_rank:
                continue
            shard_range = intersect_ranges(layout.get_tensor_range(i), rank_ranges[dest])
            numel = shard_range.numel
            offset = shard_range.start - layout.tensor_to_offset[i]
            if numel == 0:
                continue
            ops.append(
                dist.P2POp(dist.isend, flat[offset : offset + numel], peer=dest, group=group)
            )
    for i in held_tensor_indices:
        ops.append(dist.P2POp(dist.irecv, recv_chunks[i], peer=tensor_to_owner[i], group=group))

    works = dist.batch_isend_irecv(ops) if ops else []
    for work in works:
        work.wait()

    for i in held_tensor_indices:
        local_view = destination.get_tensor_view(i)
        local_view.copy_(recv_chunks[i].view(local_view.shape))
