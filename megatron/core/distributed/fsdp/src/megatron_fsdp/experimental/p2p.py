# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Owner-compute P2P gather/scatter for MFSDP v2.

- `gather` moves each participating parameter's local chunks to its owner and reconstructs the full
  tensor there;
- `scatter` sends the owner's full result back as per-rank flat chunks.

Both take the caller-built `GroupOwnerLayout` (who owns which parameter, and which rank holds which
flat range) and a CUDA `stream`, operating on one set of parameters at a time. This module uses
stream-scoped execution, which allows for pipelining: each function issues `batch_isend_irecv` and
assembles the results on the given `stream`, waiting there. The caller's stream is never blocked, so
multiple param groups can be pipelined by passing different streams. The caller needs to wait on the
streams before reading the results.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from .owner_planning import GroupOwnerLayout


@contextmanager
def _waiting_stream_scope(stream: torch.cuda.Stream | None) -> Iterator[None]:
    """Create a context running PyTorch ops on `stream`, which waits on the caller's stream at
    entry.

    Callers must wait on `stream` before reading the results. The caller's stream itself is never
    blocked.

    Args:
        stream: Which stream to use for PyTorch ops inside the context. If `None`, everything runs
            on the caller's current stream with no switching.
    """
    if stream is None:
        yield
        return
    default_stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        yield


def gather(
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    *,
    owner_layout: GroupOwnerLayout,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Gather full tensors from local chunks to their owners via P2P.

    Callers must wait on `stream` before reading `destination`. The caller's stream is never
    blocked.

    All tensors in `source` must share dtype and device (as in a `DBuffer`), also across ranks.

    For each participating parameter (a key of `owner_layout.layouts`):

    - If this rank owns it, `destination[i]` is filled with the full flat tensor, reconstructed by
      concatenating the per-rank chunks in rank order (i.e., global element order). Fully local
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
        stream: CUDA stream for the P2P ops and result assembly. `None` means the current stream.
    """
    mesh = owner_layout.mesh
    world_size = mesh.size()
    this_rank = mesh.get_local_rank()
    group = mesh.get_group()
    owned_tensor_indices = [i for i in owner_layout.layouts if owner_layout.owners[i] == this_rank]

    with _waiting_stream_scope(stream):
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
                layout = owner_layout.layouts[i]
                for src_rank in range(world_size):
                    if src_rank == this_rank:
                        continue
                    numel = layout.rank_numel(src_rank)
                    if numel > 0:
                        recv_chunks[(i, src_rank)] = torch.empty(
                            numel, dtype=reference.dtype, device=reference.device
                        )

        # Build the P2P ops: send non-owned chunks to their owners, receive owned ones.
        ops: list[dist.P2POp] = []
        for i, layout in owner_layout.layouts.items():
            owner_rank = owner_layout.owners[i]
            # Don't send already local or empty chunks.
            if owner_rank == this_rank or layout.rank_numel(this_rank) == 0:
                continue
            chunk = source[i]
            ops.append(dist.P2POp(dist.isend, chunk, peer=owner_rank, group=group))
        for (_, src_rank), buf in recv_chunks.items():
            ops.append(dist.P2POp(dist.irecv, buf, peer=src_rank, group=group))

        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()

        # Reconstruct the full flat tensors for owned parameters into `destination`.
        for i in owned_tensor_indices:
            chunks: list[torch.Tensor] = []
            for src_rank in range(world_size):
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
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Scatter full result tensors from the owners to the other ranks via P2P.

    Callers must wait on `stream` before reading the `destination`. The caller's stream is never
    blocked.

    All tensors in `source` and `destination` must share dtype and device (as in a `DBuffer`), also
    across ranks.

    For each participating parameter (a key of `owner_layout.layouts`):

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
        stream: CUDA stream for the P2P ops and result assembly. `None` means the current stream.
    """
    mesh = owner_layout.mesh
    world_size = mesh.size()
    this_rank = mesh.get_local_rank()
    group = mesh.get_group()
    held_tensor_indices = [
        i
        for i in owner_layout.layouts
        if owner_layout.owners[i] != this_rank and owner_layout.layouts[i].rank_numel(this_rank) > 0
    ]

    with _waiting_stream_scope(stream):
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
                numel = owner_layout.layouts[i].rank_numel(this_rank)
                recv_chunks[i] = torch.empty(numel, dtype=reference.dtype, device=reference.device)

        # Build the P2P ops: send owned result chunks, receive the ones held here.
        ops: list[dist.P2POp] = []
        for i in owner_layout.layouts:
            # Don't send local chunks.
            if owner_layout.owners[i] != this_rank:
                continue
            flat = source[i].flatten()
            layout = owner_layout.layouts[i]
            for dest in range(world_size):
                if dest == this_rank:
                    continue
                numel = layout.rank_numel(dest)
                if numel == 0:
                    continue
                offset = layout.rank_offset(dest)
                ops.append(
                    dist.P2POp(dist.isend, flat[offset : offset + numel], peer=dest, group=group)
                )
        for i in held_tensor_indices:
            ops.append(
                dist.P2POp(dist.irecv, recv_chunks[i], peer=owner_layout.owners[i], group=group)
            )

        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()

        for i in held_tensor_indices:
            destination[i].copy_(recv_chunks[i].view(destination[i].shape))
