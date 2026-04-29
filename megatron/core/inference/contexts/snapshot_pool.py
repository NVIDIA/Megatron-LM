# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fixed-address GPU snapshot buffers for async dynamic inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from .gpu_view import ContextGPUView

SNAPSHOT_FREE = "free"
SNAPSHOT_POPULATING = "populating"
SNAPSHOT_READY = "ready"
SNAPSHOT_FORWARD_IN_FLIGHT = "forward_in_flight"
SNAPSHOT_RETIRING = "retiring"


@dataclass
class GpuSnapshotSlot:
    """One fixed-address CPU/GPU bookkeeping snapshot slot."""

    slot_id: int
    gpu_view: ContextGPUView
    cpu_mirror: torch.Tensor
    cpu_view: ContextGPUView
    metadata_ready_event: torch.cuda.Event
    state: str = SNAPSHOT_FREE
    owning_step_id: Optional[int] = None
    last_gpu_read_event: Optional[Any] = None
    journal_retired: bool = True
    graph_capture_count: int = 0
    graph_capture_bytes: int = 0

    @property
    def is_free(self) -> bool:
        """Return whether this slot can be acquired immediately."""
        return self.state == SNAPSHOT_FREE


class GpuSnapshotBufferPool:
    """Owns fixed-address GPU snapshot slots and pinned CPU mirrors."""

    def __init__(
        self,
        *,
        slot_count: int,
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        device,
        max_mamba_chunks: int = 0,
    ):
        if slot_count < 1:
            raise ValueError(f"slot_count must be >= 1, got {slot_count}")
        self.slot_count = int(slot_count)
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self.max_kv_blocks = int(max_kv_blocks)
        self.device = device
        self.max_mamba_chunks = int(max_mamba_chunks)
        self.slots = tuple(self._make_slot(slot_id) for slot_id in range(self.slot_count))

    def _make_slot(self, slot_id: int) -> GpuSnapshotSlot:
        gpu_view = ContextGPUView(
            max_requests=self.max_requests,
            max_tokens=self.max_tokens,
            max_kv_blocks=self.max_kv_blocks,
            device=self.device,
            max_mamba_chunks=self.max_mamba_chunks,
        )
        gpu_view.current_snapshot_slot_id = slot_id

        cpu_mirror = torch.empty(
            gpu_view.total_bytes, dtype=torch.uint8, device='cpu', pin_memory=True,
        )
        cpu_mirror.zero_()
        cpu_view = ContextGPUView(
            max_requests=self.max_requests,
            max_tokens=self.max_tokens,
            max_kv_blocks=self.max_kv_blocks,
            device=torch.device('cpu'),
            max_mamba_chunks=self.max_mamba_chunks,
            backing_buffer=cpu_mirror,
            zero_initialize=False,
        )
        cpu_view.current_snapshot_slot_id = slot_id

        return GpuSnapshotSlot(
            slot_id=slot_id,
            gpu_view=gpu_view,
            cpu_mirror=cpu_mirror,
            cpu_view=cpu_view,
            metadata_ready_event=torch.cuda.Event(),
        )

    def acquire(self, step_id: int) -> GpuSnapshotSlot:
        """Acquire a free snapshot slot for a scheduled step."""
        self.poll_retired()
        for slot in self.slots:
            if slot.is_free:
                slot.state = SNAPSHOT_POPULATING
                slot.owning_step_id = int(step_id)
                slot.journal_retired = False
                slot.last_gpu_read_event = None
                slot.gpu_view.current_dynamic_step_id = int(step_id)
                slot.cpu_view.current_dynamic_step_id = int(step_id)
                return slot
        raise RuntimeError("No reusable GPU snapshot slots are available")

    def mark_ready(
        self, slot_or_id, metadata_ready_event: Optional[torch.cuda.Event] = None
    ) -> GpuSnapshotSlot:
        """Mark a populated snapshot ready for the forward pass."""
        slot = self.get_slot(slot_or_id)
        if slot.state != SNAPSHOT_POPULATING:
            raise RuntimeError(f"Snapshot slot {slot.slot_id} is not being populated")
        if metadata_ready_event is not None:
            slot.metadata_ready_event = metadata_ready_event
        slot.state = SNAPSHOT_READY
        return slot

    def mark_forward_in_flight(
        self, slot_or_id, gpu_read_event: Optional[Any] = None
    ) -> GpuSnapshotSlot:
        """Mark a ready snapshot as being read by a GPU forward pass."""
        slot = self.get_slot(slot_or_id)
        if slot.state not in (SNAPSHOT_POPULATING, SNAPSHOT_READY):
            raise RuntimeError(f"Snapshot slot {slot.slot_id} cannot enter forward_in_flight")
        if gpu_read_event is None:
            gpu_read_event = torch.cuda.Event()
            gpu_read_event.record(torch.cuda.current_stream())
        slot.last_gpu_read_event = gpu_read_event
        slot.state = SNAPSHOT_FORWARD_IN_FLIGHT
        return slot

    def release(self, slot_or_id) -> GpuSnapshotSlot:
        """Mark the slot's journal retired and free it once GPU reads are done."""
        slot = self.get_slot(slot_or_id)
        slot.journal_retired = True
        if self._can_reuse(slot):
            self._mark_free(slot)
        else:
            slot.state = SNAPSHOT_RETIRING
        return slot

    def poll_retired(self) -> None:
        """Free retiring slots whose GPU-read event has completed."""
        for slot in self.slots:
            if slot.state == SNAPSHOT_RETIRING and self._can_reuse(slot):
                self._mark_free(slot)

    def get_slot(self, slot_or_id) -> GpuSnapshotSlot:
        """Return a snapshot slot by handle or slot id."""
        if isinstance(slot_or_id, GpuSnapshotSlot):
            return slot_or_id
        slot_id = int(slot_or_id)
        if slot_id < 0 or slot_id >= self.slot_count:
            raise IndexError(f"snapshot slot id {slot_id} out of range")
        return self.slots[slot_id]

    def register_graph_capture(self, slot_or_id, byte_count: int = 0) -> None:
        """Record CUDA graph capture accounting associated with a slot."""
        slot = self.get_slot(slot_or_id)
        slot.graph_capture_count += 1
        slot.graph_capture_bytes += int(byte_count)

    def memory_accounting(self) -> Dict[str, int]:
        """Return memory accounting for snapshot metadata and graph captures."""
        graph_capture_count = sum(slot.graph_capture_count for slot in self.slots)
        graph_capture_bytes = sum(slot.graph_capture_bytes for slot in self.slots)
        return {
            "snapshot_slot_count": self.slot_count,
            "metadata_buffer_bytes": sum(slot.gpu_view.total_bytes for slot in self.slots),
            "pinned_mirror_bytes": sum(slot.cpu_mirror.numel() for slot in self.slots),
            "graph_capture_count": graph_capture_count,
            "graph_capture_bytes": graph_capture_bytes,
        }

    @property
    def active_slot_ids(self) -> Sequence[int]:
        """Return slot IDs not currently free."""
        return tuple(slot.slot_id for slot in self.slots if not slot.is_free)

    def _can_reuse(self, slot: GpuSnapshotSlot) -> bool:
        if not slot.journal_retired:
            return False
        event = slot.last_gpu_read_event
        return event is None or bool(event.query())

    def _mark_free(self, slot: GpuSnapshotSlot) -> None:
        slot.state = SNAPSHOT_FREE
        slot.owning_step_id = None
        slot.last_gpu_read_event = None
        slot.journal_retired = True
