# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner gather and scatter through peer copies on the copy engines (no SM copy kernels).

Each FSDP group keeps two staging buffers in PyTorch symmetric memory: the momentum inputs a
gather reads and the updates a scatter writes. Every rank of the mesh can address its peers'
copies, so a plain ``copy_`` between a local tensor and a peer's view is a peer memcpy that runs on
a copy engine instead of an SM. A barrier on the symmetric memory handle orders the exchange: a
gather pulls after every rank has filled its source, a scatter pushes and every rank applies after
all pushes have landed. Both are enqueued on the current CUDA stream, so they need no works.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .dbuffer import DBuffer
from .range import Range, intersect_ranges


class CopyEngineExchange:
    """Persistent symmetric staging buffers and peer views of one FSDP group."""

    def __init__(self, buffer: DBuffer, rank_ranges: dict[int, Range]) -> None:
        """Allocate the buffers. Every rank of the mesh must construct this in the same order."""
        mesh = buffer.mesh
        # Rendezvous over the default group: symmetric memory needs an NCCL communicator on the
        # group, which sub-groups only get lazily. Handles are then indexed by global rank.
        group = dist.group.WORLD
        self.this_rank: int = mesh.get_rank()
        self.rank_ranges = rank_ranges
        # Symmetric allocations use one size on every rank; shards of other ranks may be larger.
        numel = max(rank_range.numel for rank_range in rank_ranges.values())
        local_numel = rank_ranges[self.this_rank].numel
        source = symm_mem.empty(numel, dtype=buffer.dtype, device=buffer.device)
        destination = symm_mem.empty(numel, dtype=torch.float32, device=buffer.device)
        self.source_handle = symm_mem.rendezvous(source, group)
        self.destination_handle = symm_mem.rendezvous(destination, group)
        self.source: torch.Tensor = source[:local_numel]
        self.destination: torch.Tensor = destination[:local_numel]
        self._remote_source = {
            rank: self.source_handle.get_buffer(rank, (numel,), buffer.dtype)
            for rank in rank_ranges
        }
        self._remote_destination = {
            rank: self.destination_handle.get_buffer(rank, (numel,), torch.float32)
            for rank in rank_ranges
        }

    def gather(
        self,
        layout,
        destination: dict[int, torch.Tensor],
        tensor_to_owner: dict[int, int],
    ) -> None:
        """Pull the shards of the tensors this rank owns from their holders' sources.

        ``self.source`` must hold this rank's momentum inputs, written on the current stream.
        Every rank of the mesh must call this, owner or not, because of the barrier.
        """
        self.source_handle.barrier()
        for i, owner in tensor_to_owner.items():
            if owner != self.this_rank:
                continue
            tensor_range = layout.get_tensor_range(i)
            full = destination[i].view(tensor_range.numel)
            for rank, rank_range in self.rank_ranges.items():
                shard = intersect_ranges(tensor_range, rank_range)
                if shard.numel == 0:
                    continue
                full.narrow(0, shard.start - tensor_range.start, shard.numel).copy_(
                    self._remote_source[rank].narrow(0, shard.start - rank_range.start, shard.numel)
                )

    def scatter(
        self, layout, source: dict[int, torch.Tensor], tensor_to_owner: dict[int, int]
    ) -> None:
        """Push the shards of the tensors this rank owns into their holders' destinations.

        The owner keeps its own shard in ``source``. Every rank of the mesh must call this, owner
        or not, because of the barrier; afterwards ``self.destination`` holds the received shards.
        """
        for i, owner in tensor_to_owner.items():
            if owner != self.this_rank:
                continue
            tensor_range = layout.get_tensor_range(i)
            flat = source[i].view(tensor_range.numel)
            for rank, rank_range in self.rank_ranges.items():
                if rank == self.this_rank:
                    continue
                shard = intersect_ranges(tensor_range, rank_range)
                if shard.numel == 0:
                    continue
                self._remote_destination[rank].narrow(
                    0, shard.start - rank_range.start, shard.numel
                ).copy_(flat.narrow(0, shard.start - tensor_range.start, shard.numel))
        self.destination_handle.barrier()
