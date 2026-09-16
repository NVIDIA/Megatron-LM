# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner-compute P2P gather/scatter for M-FSDPv2 `DBuffer`s.

`gather` moves each participating parameter's local chunks to its owner and reconstructs the
full tensor there; `scatter` sends the owner's full result back as per-rank flat chunks. Both
take the caller-built `GroupOwnerLayout` (who owns which parameter, and which rank holds which
flat range) and a CUDA `stream`, operating on one `DBuffer` at a time so multiple param groups
can be pipelined on different streams.

Design principles:

1. No packing: `DBuffer` local views are used directly as P2P send buffers, and result chunks
   are flat slices of the owner's full result, so no intermediate flat buffers or offset
   bookkeeping are allocated. (`OwnerGatherPlan`/`OwnerScatterPlan` in `owner_planning`
   implement the packed alternative, which trades fewer P2P ops for a copy per buffer.)

2. Stream-scoped execution: each function issues `batch_isend_irecv` and assembles the
   results on the given `stream`, waiting there. The caller's stream is never blocked, so
   multiple param groups can be pipelined by passing different streams. No futures are
   returned — the caller waits on the streams before reading the results.

3. The caller owns the plan: the `GroupOwnerLayout` is built once per group with the
   caller's cost and eligibility policy (`GroupOwnerLayout.from_group`) and reused across
   steps. These functions only execute communication.

All dicts are keyed by tensor index and all communicated tensors are flat element ranges,
matching `owner_planning`. Peers are mesh-local ranks in the mesh's process group; a 1-D DP
mesh is assumed.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

from .dbuffer import DBuffer
from .owner_planning import GroupOwnerLayout


@contextmanager
def _waiting_stream_scope(stream: torch.cuda.Stream | None) -> Iterator[None]:
    """Run the P2P ops and result assembly on `stream`, which waits on the caller's stream
    at entry; the caller's stream itself is never blocked.

    The caller's stream is captured before switching: inside a `torch.cuda.stream` block,
    `current_stream()` returns the switched-to stream, so ordering against the caller's
    stream must be set up first. Callers must wait on `stream` before reading the results.
    """
    default_stream = torch.cuda.current_stream() if stream is not None else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        if stream is not None:
            assert default_stream is not None
            stream.wait_stream(default_stream)
        yield


def gather(
    source: DBuffer,
    destination: dict[int, torch.Tensor],
    *,
    plan: GroupOwnerLayout,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Gather full tensors from a `DBuffer` to their owners via P2P.

    For each participating parameter (a key of `plan.layouts`):

    - If this rank owns it, `destination[i]` is filled with the full flat tensor,
      reconstructed by concatenating the per-rank chunks in rank order (which is global
      element order).
    - Otherwise this rank's local chunk is sent to the owner.

    No packing: `source.get_local_tensor(i)` views are used directly as send buffers; recv
    buffers are flat tensors allocated inside this function.

    Args:
        source: `DBuffer` whose local chunks are gathered to the owners. Its layout must
            match `plan`'s (the same group's buffers).
        destination: Preallocated full tensor per owned parameter, in any shape with the
            matching number of elements; filled with the flat content viewed into the
            destination's shape. Entries for parameters this rank does not own are never
            written.
        plan: The group's owner layout, built with `GroupOwnerLayout.from_group`.
        stream: CUDA stream for the P2P ops and result assembly. Issues and waits on this
            stream; the caller's stream is never blocked. `None` means the current stream.
            Wait on `stream` before reading `destination`.
    """
    mesh = source.mesh
    world_size = mesh.size()
    this_rank = mesh.get_local_rank()
    group = mesh.get_group()
    device = source.local_buffer.device
    owned = [i for i in plan.layouts if plan.owners[i] == this_rank]

    # Allocate flat recv buffers for owned parameters (one per (parameter, source) pair).
    recv_chunks: dict[tuple[int, int], torch.Tensor] = {}
    for i in owned:
        layout = plan.layouts[i]
        for src in range(world_size):
            if src == this_rank:
                continue
            numel = layout.rank_numel(src)
            if numel > 0:
                recv_chunks[(i, src)] = torch.empty(numel, dtype=source.dtype, device=device)

    # Build the P2P ops: send non-owned chunks to their owners, receive owned ones.
    ops: list[dist.P2POp] = []
    for i in plan.layouts:
        if plan.owners[i] == this_rank:
            continue
        chunk = source.get_local_tensor(i)
        if chunk.numel() == 0:
            continue
        ops.append(dist.P2POp(dist.isend, chunk, peer=plan.owners[i], group=group))
    for (i, src), buf in recv_chunks.items():
        ops.append(dist.P2POp(dist.irecv, buf, peer=src, group=group))

    with _waiting_stream_scope(stream):
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()

        # Reconstruct the full flat tensors for owned parameters into `destination`.
        for i in owned:
            layout = plan.layouts[i]
            chunks: list[torch.Tensor] = []
            for src in range(world_size):
                if src == this_rank:
                    own = source.get_local_tensor(i)
                    if own.numel() > 0:
                        chunks.append(own.reshape(-1))
                else:
                    chunk = recv_chunks.get((i, src))
                    if chunk is not None:
                        chunks.append(chunk)
            full = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
            destination[i].copy_(full.view(destination[i].shape))


def scatter(
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    *,
    plan: GroupOwnerLayout,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Scatter full result tensors from the owners to the other ranks via P2P.

    For each participating parameter (a key of `plan.layouts`):

    - If this rank owns it, `source[i]` is flattened (a view on contiguous tensors) and each
      rank's flat range is sliced out and sent to it directly. The owner keeps its own
      result chunk and applies it itself.
    - Otherwise, if this rank holds elements of the parameter, the received flat chunk is
      copied into `destination[i]` viewed into the destination's shape — the
      update-application site is fully caller-owned. To write into a `DBuffer`'s local
      views, pass `{i: dbuffer.get_local_tensor(i) for ...}`.

    No packing: result chunks are flat slices of the owner's full result, sent directly;
    recv buffers are flat tensors allocated inside this function. All tensors must share
    dtype and device (as in a `DBuffer`).

    Args:
        source: Full result tensor per owned parameter, in any shape with the matching
            number of elements; flattened as a view for slicing. Pass the
            orthogonalization kernel's output as-is. Entries for parameters this rank does
            not own are never read.
        destination: Result-shard tensor per parameter this rank holds elements of but
            does not own, in any shape with the matching number of elements; written after
            the P2P completes. Entries for other parameters are never written.
        plan: The group's owner layout, built with `GroupOwnerLayout.from_group`.
        stream: CUDA stream for the P2P ops and result assembly. Issues and waits on this
            stream; the caller's stream is never blocked. Wait on `stream` before reading
            `destination`.
    """
    mesh = plan.mesh
    world_size = mesh.size()
    this_rank = mesh.get_local_rank()
    group = mesh.get_group()
    held = [
        i
        for i in plan.layouts
        if plan.owners[i] != this_rank and plan.layouts[i].rank_numel(this_rank) > 0
    ]

    # All tensors share dtype and device; take any available one for the recv buffers.
    reference = next(iter(source.values() if source else destination.values()), None)
    recv_chunks: dict[int, torch.Tensor] = {}
    if held:
        assert reference is not None, "scatter needs at least one tensor to infer dtype"
        for i in held:
            numel = plan.layouts[i].rank_numel(this_rank)
            recv_chunks[i] = torch.empty(numel, dtype=reference.dtype, device=reference.device)

    # Build the P2P ops: send owned result chunks, receive the ones held here.
    ops: list[dist.P2POp] = []
    for i in plan.layouts:
        if plan.owners[i] != this_rank:
            continue
        flat = source[i].reshape(-1)
        layout = plan.layouts[i]
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
    for i in held:
        ops.append(dist.P2POp(dist.irecv, recv_chunks[i], peer=plan.owners[i], group=group))

    with _waiting_stream_scope(stream):
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()

        for i in held:
            destination[i].copy_(recv_chunks[i].view(destination[i].shape))
