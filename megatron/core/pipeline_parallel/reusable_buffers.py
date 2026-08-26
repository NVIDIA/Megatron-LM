# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Caller-owned CUDA output buffers with schedule-aware lifetime tracking."""

from dataclasses import dataclass

import torch

StorageKey = tuple[int, int]


def _storage_key(tensor: torch.Tensor) -> StorageKey:
    """Return a device-qualified key shared by every view of a CUDA storage."""
    return (tensor.device.index, tensor.untyped_storage().data_ptr())


@dataclass
class _BufferSlot:
    tensor: torch.Tensor
    event: torch.cuda.Event | None = None
    in_use: bool = False


_buffer_owners: dict[StorageKey, tuple["ReusableOutputBufferPool", int]] = {}


class ReusableOutputBufferPool:
    """A fixed-size CUDA tensor ring whose slots are released by schedule consumers.

    The pool is intended for external kernels that can write into caller-provided output
    tensors. A slot is reused only after the stream consuming its previous contents records
    a completion event. Unlike ``Tensor.record_stream``, the storage stays persistent and
    does not enter the caching allocator's pending-free queue.
    """

    def __init__(self, name: str):
        self.name = name
        self.num_slots = 0
        self._slots: list[_BufferSlot] = []
        self._signature = None
        self._cursor = 0

    def configure(self, num_slots: int) -> None:
        """Configure the ring before its first allocation.

        Args:
            num_slots: Number of persistent tensors in the ring. Zero disables the pool.
        """
        if num_slots < 0:
            raise ValueError(f"{self.name} buffer count must be non-negative, got {num_slots}")
        if self._slots and num_slots != self.num_slots:
            raise RuntimeError(
                f"Cannot resize initialized {self.name} pool from {self.num_slots} to {num_slots}"
            )
        self.num_slots = num_slots

    @property
    def enabled(self) -> bool:
        """Return whether the pool should provide caller-owned output tensors."""
        return self.num_slots > 0

    def acquire(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor | None:
        """Acquire an alias of the next persistent slot on the current CUDA stream.

        Args:
            shape: Static output shape shared by every pool use.
            dtype: Output data type.
            device: CUDA device on which to allocate the pool.

        Returns:
            A fresh tensor alias of a persistent slot, or ``None`` when disabled.
        """
        if not self.enabled:
            return None

        signature = (tuple(shape), dtype, torch.device(device))
        if self._signature is None:
            self._signature = signature
            for index in range(self.num_slots):
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self._slots.append(_BufferSlot(tensor=tensor))
                _buffer_owners[_storage_key(tensor)] = (self, index)
        elif signature != self._signature:
            raise RuntimeError(
                f"{self.name} requires a static output signature; expected {self._signature}, "
                f"got {signature}"
            )

        slot_index = self._cursor % self.num_slots
        self._cursor += 1
        slot = self._slots[slot_index]
        if slot.in_use:
            raise RuntimeError(
                f"{self.name} slot {slot_index} was reused before its consumer released it; "
                "increase the configured buffer count"
            )
        if slot.event is not None:
            torch.cuda.current_stream(device).wait_event(slot.event)
        slot.in_use = True
        # A fresh TensorImpl prevents autograd metadata from one dispatch from being reused by
        # another dispatch while retaining the same persistent storage.
        return slot.tensor.detach()

    def _release(self, slot_index: int, stream: torch.cuda.Stream) -> None:
        slot = self._slots[slot_index]
        if not slot.in_use:
            # Some final consumers own their lifetime and release immediately after enqueueing
            # work, while ScheduleNode subsequently visits the same input in its generic
            # free-input path. Treat that second visit as an ownership query: the persistent
            # storage is already safe and must not be resized or record_stream'ed.
            return
        if slot.event is None:
            # External events become explicit CUDA graph event nodes. This is required when
            # capture begins after eager warmup: an internal event wait would otherwise create
            # a forbidden dependency on uncaptured work from the warmup stream.
            slot.event = torch.cuda.Event(enable_timing=False, external=True)
        slot.event.record(stream)
        slot.in_use = False

    def reset(self) -> None:
        """Drop persistent tensors and their storage-owner registrations."""
        for slot in self._slots:
            _buffer_owners.pop(_storage_key(slot.tensor), None)
        self.num_slots = 0
        self._slots.clear()
        self._signature = None
        self._cursor = 0


def release_reusable_output_buffer(tensor: torch.Tensor, stream: torch.cuda.Stream) -> bool:
    """Release a registered persistent output after its last scheduled consumer.

    Args:
        tensor: Tensor whose storage may belong to a reusable output pool.
        stream: CUDA stream containing the tensor's final consumer.

    Returns:
        ``True`` when the tensor belongs to a reusable pool. The caller must then avoid
        resizing its storage or calling ``record_stream``, because the pool owns the storage
        and records the precise consumer completion event itself. Releasing an already-released
        registered storage is an idempotent ownership query, so a final consumer may release it
        before ``ScheduleNode`` reaches its generic free-input path.
    """
    owner = _buffer_owners.get(_storage_key(tensor))
    if owner is None:
        return False
    pool, slot_index = owner
    pool._release(slot_index, stream)
    return True
