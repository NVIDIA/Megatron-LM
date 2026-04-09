# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-chunk TMS management for contiguous tensors.

Uses torch_memory_saver's per-chunk CUDA VMM support to selectively free and
restore physical pages within a single contiguous tensor. The tensor's virtual
address space is preserved — CUDA graphs and kernel indexing work unchanged.

Usage:
    tracker = ChunkTracker(total_slots=1024, num_chunks=8, tms_tag="mamba_conv")
    # When a slot becomes active:
    tracker.activate_slot(slot_idx)
    # When a slot becomes inactive:
    tracker.deactivate_slot(slot_idx)  # auto-pauses chunk if fully free
"""

import logging
import math
from typing import List, Optional

try:
    from torch_memory_saver import torch_memory_saver

    _HAVE_TMS = True
except ImportError:
    _HAVE_TMS = False


class ChunkTracker:
    """Tracks per-chunk utilization and manages TMS pause/resume.

    Maps a flat index space (e.g., request slots or KV blocks) to chunks.
    When all slots in a chunk become inactive, the chunk is TMS-paused
    (physical pages freed). When a slot in a paused chunk is activated,
    the chunk is TMS-resumed first.

    Args:
        total_slots: Total number of slots in the index space.
        num_chunks: Number of TMS chunks.
        tms_tag: The TMS region tag used when allocating the tensor.
        auto_pause: If True, automatically pause chunks when they become
            fully free. If False, caller must call pause_free_chunks().
    """

    def __init__(
        self,
        total_slots: int,
        num_chunks: int,
        tms_tag: str,
        auto_pause: bool = True,
    ):
        self.total_slots = total_slots
        self.num_chunks = min(num_chunks, total_slots)
        self.tms_tag = tms_tag
        self.auto_pause = auto_pause

        self.chunk_size = math.ceil(total_slots / self.num_chunks)
        # Recompute num_chunks in case rounding made some empty.
        self.num_chunks = math.ceil(total_slots / self.chunk_size)

        self.chunk_active_counts: List[int] = [0] * self.num_chunks
        self.chunk_paused: List[bool] = [False] * self.num_chunks

    def _chunk_of(self, slot_idx: int) -> int:
        return slot_idx // self.chunk_size

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def activate_slot(self, slot_idx: int) -> None:
        """Mark a slot as active. Resumes its chunk if paused."""
        chunk_idx = self._chunk_of(slot_idx)
        if self.chunk_paused[chunk_idx]:
            self._resume_chunk(chunk_idx)
        self.chunk_active_counts[chunk_idx] += 1

    def deactivate_slot(self, slot_idx: int) -> None:
        """Mark a slot as inactive. Pauses the chunk if fully free."""
        chunk_idx = self._chunk_of(slot_idx)
        self.chunk_active_counts[chunk_idx] = max(
            0, self.chunk_active_counts[chunk_idx] - 1
        )
        if (
            self.auto_pause
            and self.chunk_active_counts[chunk_idx] == 0
            and not self.chunk_paused[chunk_idx]
        ):
            self._pause_chunk(chunk_idx)

    # ------------------------------------------------------------------
    # TMS management
    # ------------------------------------------------------------------

    def _pause_chunk(self, chunk_idx: int) -> None:
        if not _HAVE_TMS:
            return
        torch_memory_saver.pause_chunks(
            tag=self.tms_tag, chunk_indices=[chunk_idx]
        )
        self.chunk_paused[chunk_idx] = True
        logging.debug(
            "ChunkTracker[%s]: paused chunk %d", self.tms_tag, chunk_idx
        )

    def _resume_chunk(self, chunk_idx: int) -> None:
        if not _HAVE_TMS:
            self.chunk_paused[chunk_idx] = False
            return
        torch_memory_saver.resume_chunks(
            tag=self.tms_tag, chunk_indices=[chunk_idx]
        )
        self.chunk_paused[chunk_idx] = False
        logging.debug(
            "ChunkTracker[%s]: resumed chunk %d", self.tms_tag, chunk_idx
        )

    def pause_free_chunks(self) -> None:
        """Pause all chunks with no active slots."""
        for i in range(self.num_chunks):
            if self.chunk_active_counts[i] == 0 and not self.chunk_paused[i]:
                self._pause_chunk(i)

    def resume_all(self) -> None:
        """Resume all paused chunks."""
        if not _HAVE_TMS:
            for i in range(self.num_chunks):
                self.chunk_paused[i] = False
            return
        paused = [i for i in range(self.num_chunks) if self.chunk_paused[i]]
        if paused:
            torch_memory_saver.resume_chunks(
                tag=self.tms_tag, chunk_indices=paused
            )
            for i in paused:
                self.chunk_paused[i] = False

    def reset(self) -> None:
        """Reset all tracking state. Does not pause/resume — caller manages."""
        self.chunk_active_counts = [0] * self.num_chunks

    @property
    def num_paused(self) -> int:
        return sum(1 for p in self.chunk_paused if p)

    @property
    def num_active_chunks(self) -> int:
        return self.num_chunks - self.num_paused
