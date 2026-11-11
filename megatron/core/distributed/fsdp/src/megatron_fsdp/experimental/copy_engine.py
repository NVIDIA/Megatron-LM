# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Owner gather and scatter on the copy engines, through PyTorch symmetric memory.

The NCCL P2P of `gather_scatter` runs as kernels, which take SMs from the Newton-Schulz it
overlaps. Here the owner moves the data itself with peer memcpys, which run on copy engines: a
gather pulls each holder's chunk out of the holder's sharded buffer, and a scatter pushes each
result chunk into it. The sharded buffers therefore live in slots of symmetric memory, where every
rank can address a peer's chunk from the layout alone.

Ranks signal each other without SMs either. A holder writes READY into the owner's flags once its
momentum inputs are in its slot; the owner writes DONE into the holder's flags once its results
have landed there. A flag holds the slot's use count, so it never needs resetting, and is written
with a 4-byte peer memcpy: stream memory operations cannot address another node's memory. Each
rank waits on its own flags with cuStreamWaitValue32, which blocks only the copy stream.

Every peer must be reachable by peer memcpy: one node, or one multi-node NVLink domain.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .dbuffer import DBuffer
from .range import Range, intersect_ranges

_READY, _DONE = 0, 1


class CopyEngineWork:
    """Completion of work posted on the copy stream; like `dist.Work`, `wait` does not block."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self._event = torch.cuda.Event()
        self._event.record(stream)

    def wait(self) -> None:
        """Order the current stream after the work."""
        torch.cuda.current_stream().wait_event(self._event)


class CopyEngineP2P:
    """Symmetric slots for in-flight sharded buffers, and owner gather and scatter between them.

    `empty` hands out slots round robin. A slot must stay in use from its group's gather until
    its received updates are applied, so `num_slots` must cover every group in flight at once.
    A group's scatter reuses the slot of its gather: by then the owners have read their inputs.
    Every rank must construct this, and call `empty`, `gather`, and `scatter` in the same order.
    """

    def __init__(self, num_slots: int, slot_numel: int, device: torch.device) -> None:
        """Allocate the slots and flags. A collective over the default group."""
        # Deferred: only this experimental transport needs the CUDA driver bindings.
        from cuda.bindings import driver  # pylint: disable=import-outside-toplevel

        self._driver = driver
        # Symmetric allocations need one size on every rank.
        size = torch.tensor([slot_numel], device=device)
        dist.all_reduce(size, op=dist.ReduceOp.MAX)
        self.num_slots = num_slots
        self.slot_numel = int(size.item())
        self.this_rank = dist.get_rank()
        world_size = dist.get_world_size()
        slots = symm_mem.empty(num_slots * self.slot_numel, dtype=torch.float32, device=device)
        flags = symm_mem.empty((num_slots, 2, world_size), dtype=torch.int32, device=device)
        flags.zero_()
        torch.cuda.synchronize(device)
        # Rendezvous over the default group: a sub-group has no NCCL communicator to bootstrap from
        # until it runs a collective. Peers are then addressed by global rank.
        slot_handle = symm_mem.rendezvous(slots, dist.group.WORLD)
        flag_handle = symm_mem.rendezvous(flags, dist.group.WORLD)
        self._slots = slots.view(num_slots, self.slot_numel)
        self._flags = flags
        self._peer_slots = [
            slot_handle.get_buffer(rank, (num_slots, self.slot_numel), torch.float32)
            for rank in range(world_size)
        ]
        self._peer_flags = [
            flag_handle.get_buffer(rank, (num_slots, 2, world_size), torch.int32)
            for rank in range(world_size)
        ]
        self._uses = [0] * num_slots
        # Each slot's use count on the device, the source of its signal memcpys.
        self._signal_values = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self._next_slot = 0
        self.stream = torch.cuda.Stream(device)

    def empty(self, like: DBuffer) -> DBuffer:
        """Return a float32 DBuffer like `like` whose local buffer is the next slot."""
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.num_slots
        self._uses[slot] += 1
        # Ordered after the slot's previous signals, which also run on the copy stream.
        self._stream_memory_op(
            self._driver.cuStreamWriteValue32,
            self._signal_values[slot].data_ptr(),
            self._uses[slot],
            0,
        )
        numel = like.layout.get_local_range(like.mesh, like.placements).numel
        return DBuffer(
            like.mesh,
            like.placements,
            like.layout,
            torch.float32,
            like.device,
            local_buffer=self._slots[slot, :numel],
        )

    def gather(
        self,
        source: DBuffer,
        destination: dict[int, torch.Tensor],
        *,
        tensor_to_owner: dict[int, int],
        rank_ranges: dict[int, Range],
    ) -> list[CopyEngineWork]:
        """Pull the chunks of owned tensors from their holders' slots; see `gather_scatter.gather`.

        `source` must come from `empty`. Holders signal READY to the owners they hold chunks for.
        """
        slot = self._slot_of(source)
        chunks, owners = self._chunks(source, slot, destination, tensor_to_owner, rank_ranges)
        # The inputs and the destinations' memory come from the current stream.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # Signal before waiting: owners wait for each other's chunks.
            for owner in owners:
                self._signal(owner, slot, _READY)
            for holder, chunk, holder_chunk in chunks:
                if holder != self.this_rank:
                    self._wait(holder, slot, _READY)
                chunk.copy_(holder_chunk)
        return [CopyEngineWork(self.stream)]

    def scatter(
        self,
        source: dict[int, torch.Tensor],
        destination: DBuffer,
        *,
        tensor_to_owner: dict[int, int],
        rank_ranges: dict[int, Range],
    ) -> list[CopyEngineWork]:
        """Push owned results into their holders' slots; see `gather_scatter.scatter`.

        `destination` must be the gather's `source`. Owners signal DONE to the holders they push to.
        """
        slot = self._slot_of(destination)
        chunks, owners = self._chunks(destination, slot, source, tensor_to_owner, rank_ranges)
        # The owner keeps its own result chunk.
        chunks = [chunk for chunk in chunks if chunk[0] != self.this_rank]
        # The results come from the current stream.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _, chunk, holder_chunk in chunks:
                holder_chunk.copy_(chunk)
            for holder in {chunk[0] for chunk in chunks}:
                self._signal(holder, slot, _DONE)
            for owner in owners:
                self._wait(owner, slot, _DONE)
        return [CopyEngineWork(self.stream)]

    def _chunks(
        self,
        sharded: DBuffer,
        slot: int,
        full_tensors: dict[int, torch.Tensor],
        tensor_to_owner: dict[int, int],
        rank_ranges: dict[int, Range],
    ) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], set[int]]:
        """Match the chunks of this rank's full tensors with their holders' slots.

        Returns:
            `(holder, chunk of a full tensor, the holder's chunk in its slot)` for every chunk of
            the tensors this rank owns, and the owners of the other tensors this rank holds.
        """
        chunks: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        owners: set[int] = set()
        for i, owner in tensor_to_owner.items():
            tensor_range = sharded.layout.get_tensor_range(i)
            if owner != self.this_rank:
                if intersect_ranges(tensor_range, rank_ranges[self.this_rank]).numel > 0:
                    owners.add(owner)
                continue
            full = full_tensors[i].view(tensor_range.numel)
            for holder, rank_range in rank_ranges.items():
                shard = intersect_ranges(tensor_range, rank_range)
                if shard.numel > 0:
                    chunk = full.narrow(0, shard.start - tensor_range.start, shard.numel)
                    holder_chunk = self._peer_slots[holder][slot].narrow(
                        0, shard.start - rank_range.start, shard.numel
                    )
                    chunks.append((holder, chunk, holder_chunk))
        return chunks, owners

    def _slot_of(self, buffer: DBuffer) -> int:
        """Index of the slot holding `buffer`'s local buffer."""
        offset = buffer.local_buffer.data_ptr() - self._slots.data_ptr()
        slot, remainder = divmod(offset, self._slots.stride(0) * self._slots.element_size())
        if remainder or not 0 <= slot < self.num_slots:
            raise ValueError("The buffer was not allocated by CopyEngineP2P.empty.")
        return slot

    def _signal(self, peer: int, slot: int, kind: int) -> None:
        """Copy this slot's use count into `peer`'s flag for this rank, after prior copy work."""
        flag = self._peer_flags[peer][slot, kind, self.this_rank : self.this_rank + 1]
        flag.copy_(self._signal_values[slot : slot + 1])

    def _wait(self, peer: int, slot: int, kind: int) -> None:
        """Hold the copy stream until `peer` has signaled this slot's current use."""
        self._stream_memory_op(
            self._driver.cuStreamWaitValue32,
            self._flags[slot, kind, peer].data_ptr(),
            self._uses[slot],
            self._driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
        )

    def _stream_memory_op(self, op, address: int, value: int, flags) -> None:
        """Enqueue a 32-bit stream memory operation on local memory on the copy stream."""
        (result,) = op(self.stream.cuda_stream, address, value, flags)
        if result != self._driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA stream memory operation failed: {result}.")
