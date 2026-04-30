# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Snapshot buffer pool for the async-overlap inference pipeline (v3 plan
§2.5).

Each pool slot is a fixed-address GPU bookkeeping buffer wrapped by a
``ContextGPUView``. Slots are acquired by step ID and released back to the
free list once any in-flight GPU read of the slot has completed (gated on a
``cuda.Event``). The architectural primitive that resolves
immutability vs. CUDA-graph fixed addresses; ``buffer_count = max_concurrent_steps
+ 1`` so prepare-next-step always has a free slot to populate.

This module is wired in commit 9. With ``max_concurrent_steps=1`` the pool
holds a single in-flight slot and the same slot is acquired every step;
commit 18 raises the queue depth.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .gpu_view import ContextGPUView


class GpuSnapshotBufferPool:
    """Pool of ``ContextGPUView`` buffers, one per pool slot.

    The pool is sized at ``max_concurrent_steps + 1`` to ensure
    ``prepare_next_step_optimistic`` always has a free slot to populate. With
    ``max_concurrent_steps=1`` (today) the pool has 2 slots; the second is
    spare so that the engine can prepare step N+1 while step N's slot is still
    referenced by retirement.
    """

    def __init__(
        self,
        max_concurrent_steps: int,
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        device: torch.device,
        max_mamba_chunks: int = 0,
    ) -> None:
        from .gpu_view import ContextGPUView

        self.max_concurrent_steps = max_concurrent_steps
        self.buffer_count = max_concurrent_steps + 1
        self._slots: List["ContextGPUView"] = [
            ContextGPUView(
                max_requests=max_requests,
                max_tokens=max_tokens,
                max_kv_blocks=max_kv_blocks,
                device=device,
                max_mamba_chunks=max_mamba_chunks,
            )
            for _ in range(self.buffer_count)
        ]
        # Owning step-id for each slot (None when free).
        self._owning_step_ids: List[Optional[int]] = [None] * self.buffer_count
        # Last GPU read event recorded against each slot. release(slot, event)
        # defers the slot's return until the event has signaled.
        self._pending_release_events: List[Optional[torch.cuda.Event]] = [
            None
        ] * self.buffer_count
        # Free list kept in ascending slot order so acquire() yields the
        # lowest-numbered free slot. Sorted insert via ``bisect.insort`` on
        # release; pool sizes are small (≤3 in practice) so O(N) is fine.
        self._free: List[int] = list(range(self.buffer_count))
        self._per_slot_bytes = self._slots[0]._buf.numel()

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    @property
    def per_slot_bytes(self) -> int:
        """Bytes allocated for one slot's bookkeeping buffer."""
        return self._per_slot_bytes

    @property
    def total_bytes(self) -> int:
        """Total GPU memory held by the pool (slot_count × per-slot bytes)."""
        return self.buffer_count * self._per_slot_bytes

    @property
    def free_slot_count(self) -> int:
        # Reclaim any pending releases whose events have signaled.
        self._reclaim_signaled()
        return len(self._free)

    def slot(self, slot_idx: int) -> "ContextGPUView":
        """Direct slot accessor (used by metadata bindings in commit 10)."""
        return self._slots[slot_idx]

    def slots(self) -> List["ContextGPUView"]:
        """All slots in pool order."""
        return list(self._slots)

    def acquire(self, step_id: int) -> Tuple[int, "ContextGPUView"]:
        """Acquire the lowest-numbered free slot and bind it to ``step_id``.

        Returns the ``(slot_idx, view)`` pair. Raises if no slot is free.
        Today (max_concurrent_steps=1) the engine never asks for more slots
        than the pool holds; commit 18 adds the blocking-acquire path.
        """
        self._reclaim_signaled()
        if not self._free:
            raise RuntimeError(
                f"GpuSnapshotBufferPool exhausted (buffer_count={self.buffer_count}). "
                "Commit 18 introduces blocking acquire under queue-depth pressure."
            )
        slot_idx = self._free.pop(0)
        self._owning_step_ids[slot_idx] = step_id
        return slot_idx, self._slots[slot_idx]

    def release(
        self,
        slot_idx: int,
        after_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Return ``slot_idx`` to the free list.

        If ``after_event`` is provided the slot stays "draining" until the
        event has signaled (i.e. all GPU reads of the slot have completed).
        Otherwise the slot is freed immediately.
        """
        if self._owning_step_ids[slot_idx] is None:
            raise RuntimeError(
                f"GpuSnapshotBufferPool: slot {slot_idx} is not currently acquired."
            )
        self._owning_step_ids[slot_idx] = None
        if after_event is None or after_event.query():
            bisect.insort(self._free, slot_idx)
        else:
            self._pending_release_events[slot_idx] = after_event

    def owning_step_id(self, slot_idx: int) -> Optional[int]:
        return self._owning_step_ids[slot_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reclaim_signaled(self) -> None:
        """Move any draining slots whose events have signaled back to free."""
        for slot_idx, event in enumerate(self._pending_release_events):
            if event is not None and event.query():
                self._pending_release_events[slot_idx] = None
                bisect.insort(self._free, slot_idx)
