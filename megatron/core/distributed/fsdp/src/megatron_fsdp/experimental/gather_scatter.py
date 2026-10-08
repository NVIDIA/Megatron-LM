# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute P2P gather/scatter for MFSDP v2.

- `gather` moves each participating parameter's local chunks to its owner and reconstructs the full
  tensor there;
- `scatter` sends the owner's full result back as per-rank flat chunks.

Both take the caller-built `GroupOwnerLayout` (who owns which parameter, and which rank holds which
flat range), operating on one set of parameters at a time.

Operations run on the current CUDA stream. Multiple parameter groups can be pipelined by
executing the functions in different streams. Callers explicitly wait for inputs to be ready on
each stream and for results to be ready before consuming them on another stream.
"""

import torch
import torch.distributed as dist

from .owner_planning import GroupOwnerLayout
from .placement import RowAtomic
from .range import intersect_ranges


def gather(
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    *,
    owner_layout: GroupOwnerLayout,
) -> None:
    """Gather full tensors from local chunks to their owners via P2P.

    All tensors in `source` must share dtype and device (as in a `DBuffer`), also across ranks.

    For each participating parameter (a key of `owner_layout.tensor_to_owner`):

    - If this rank owns it, `destination[i]` is filled with the full flat tensor, reconstructed by
      concatenating the per-rank chunks in buffer order. Fully local
      parameters (i.e., non-boundary parameters) are copied straight into `destination[i]`.
    - Otherwise, this rank's local chunk is sent to the owner.

    Args:
        source: Local shard per parameter this rank holds elements of, in any shape. Entries for
            zero-element parameters are never read.
        destination: Preallocated full tensor per owned parameter, in any shape with the
            matching number of elements; filled with the flat content viewed into the
            destination's shape. Entries for parameters this rank does not own are never
            written to.
        owner_layout: The group's owner layout.
    """
    mesh = owner_layout.mesh
    this_rank = mesh.get_rank()
    layout = owner_layout.layout
    rank_ranges = {
        rank: layout.get_rank_range(mesh, [RowAtomic()] * mesh.ndim, rank)
        for rank in mesh.mesh.flatten().tolist()
    }
    rank_ranges = dict(sorted(rank_ranges.items(), key=lambda item: item[1].start))
    group = mesh.get_group()
    owned_tensor_indices = [
        i for i, owner in sorted(owner_layout.tensor_to_owner.items()) if owner == this_rank
    ]

    # Allocate the flat recv buffers (one per (tensor index, source rank) pair).
    recv_chunks: dict[tuple[int, int], torch.Tensor] = {}
    if owned_tensor_indices:
        try:
            # All tensors share dtype and device; take any available one for the recv buffers.
            reference = next(iter(source.values()))
        except StopIteration:
            # Owners hold elements of their parameters, so the source must be non-empty.
            raise RuntimeError("`gather` needs at least one source chunk to infer dtype")

        for i in owned_tensor_indices:
            for src_rank in rank_ranges:
                if src_rank == this_rank:
                    continue
                numel = intersect_ranges(layout.get_tensor_range(i), rank_ranges[src_rank]).numel
                if numel > 0:
                    recv_chunks[(i, src_rank)] = torch.empty(
                        numel, dtype=reference.dtype, device=reference.device
                    )

    # Build the P2P ops: send non-owned chunks to their owners, receive owned ones.
    ops: list[dist.P2POp] = []
    for i, owner_rank in sorted(owner_layout.tensor_to_owner.items()):
        # Don't send already local or empty chunks.
        if (
            owner_rank == this_rank
            or intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel == 0
        ):
            continue
        chunk = source[i].flatten().contiguous()
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
                chunk = source[i]
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
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    *,
    owner_layout: GroupOwnerLayout,
) -> None:
    """Scatter full result tensors from the owners to the other ranks via P2P.

    All tensors in `source` and `destination` must share dtype and device (as in a `DBuffer`), also
    across ranks.

    For each participating parameter (a key of `owner_layout.tensor_to_owner`):

    - If this rank owns it, `source[i]` is flattened (a view on contiguous tensors) and each rank's
      shard is sliced out and sent to it directly. The owner keeps its own result chunk.
    - Otherwise, if this rank holds elements of the parameter, the received flat chunk is copied
      into `destination[i]` viewed into the destination's shape. To write into a `DBuffer`'s local
      views, pass `{i: dbuffer.get_tensor_view(i) for ...}`.

    Args:
        source: Full result tensor per owned parameter, in any shape with the matching number of
            elements; flattened as a view for slicing. Entries for parameters this rank does not own
            are never read.
        destination: Result-shard tensor per parameter this rank holds elements of but does not own,
            in any shape with the matching number of elements; written to after the P2P completes.
            Entries for parameters this rank does not hold elements of are never written.
        owner_layout: The group's owner layout.
    """
    mesh = owner_layout.mesh
    this_rank = mesh.get_rank()
    layout = owner_layout.layout
    rank_ranges = {
        rank: layout.get_rank_range(mesh, [RowAtomic()] * mesh.ndim, rank)
        for rank in mesh.mesh.flatten().tolist()
    }
    rank_ranges = dict(sorted(rank_ranges.items(), key=lambda item: item[1].start))
    group = mesh.get_group()
    held_tensor_indices = [
        i
        for i, owner in sorted(owner_layout.tensor_to_owner.items())
        if owner != this_rank
        and intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel > 0
    ]

    # Allocate the recv buffers.
    recv_chunks: dict[int, torch.Tensor] = {}
    if held_tensor_indices:
        try:
            # All tensors share dtype and device; take any available one for the recv buffers.
            reference = next(iter(source.values() if source else destination.values()))
        except StopIteration:
            # Every rank must send or receive at least one parameter.
            raise RuntimeError("`scatter` needs at least one tensor to infer dtype")

        for i in held_tensor_indices:
            numel = intersect_ranges(layout.get_tensor_range(i), rank_ranges[this_rank]).numel
            recv_chunks[i] = torch.empty(numel, dtype=reference.dtype, device=reference.device)

    # Build the P2P ops: send owned result chunks, receive the ones held here.
    ops: list[dist.P2POp] = []
    for i, owner in sorted(owner_layout.tensor_to_owner.items()):
        # Don't send local chunks.
        if owner != this_rank:
            continue
        flat = source[i].flatten()
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
        ops.append(
            dist.P2POp(
                dist.irecv, recv_chunks[i], peer=owner_layout.tensor_to_owner[i], group=group
            )
        )

    works = dist.batch_isend_irecv(ops) if ops else []
    for work in works:
        work.wait()

    for i in held_tensor_indices:
        destination[i].copy_(recv_chunks[i].view(destination[i].shape))
