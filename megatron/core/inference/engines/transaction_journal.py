# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transaction journal for the async-overlap inference pipeline.

The journal is the single audit trail for every state change attempted during
``prepare_next_step_optimistic``. Each entry is keyed by ``step_id`` and is
either committed or rolled back during ``commit_step_transaction``. No allocator
releases a resource without a journal-driven commit or rollback.

See v3 plan §2.4 in
``lawrence/reports/20260429-context-cpu-async-schedule-claude-v3.md``.

This module is introduced in commit 2 with no live consumers; it is wired up
in commit 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from megatron.core.inference.engines.async_pipeline_types import Reservation, ReservationState

if TYPE_CHECKING:
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


@dataclass
class JournalEntry:
    """One step's transactional record.

    Holds the before/after slot maps, the placeholder deltas, the list of
    resource reservations, the input scatter map, and the snapshot pool slot
    that owns the H2D bookkeeping copy for this step.
    """

    step_id: int
    request_slot_map_before: Dict[int, int] = field(default_factory=dict)
    request_slot_map_after: Dict[int, int] = field(default_factory=dict)
    placeholder_deltas: Dict[int, int] = field(default_factory=dict)
    reservations: List[Reservation] = field(default_factory=list)
    paused_resumed_evicted: List[Any] = field(default_factory=list)
    decode_input_slot_map: Dict[int, int] = field(default_factory=dict)
    graph_batch_dimensions: Optional["InferenceBatchDimensions"] = None
    snapshot_buffer_id: int = -1


class TransactionJournal:
    """Append-only journal of step-scoped state changes.

    Steps are committed or rolled back in step-id order via
    ``commit_step_transaction`` / ``rollback_step_transaction``. The journal
    raises if asked to commit/rollback an unknown step or to double-commit.
    """

    def __init__(self) -> None:
        self._entries: Dict[int, JournalEntry] = {}
        self._next_journal_id: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def begin_step_transaction(self, step_id: int) -> JournalEntry:
        """Open a journal entry for ``step_id``. Idempotent within a step.

        Subsequent calls in the same step return the same entry; the engine
        opens once at ``prepare_next_step_optimistic`` and the allocator
        helpers append into it via ``record_resource_reservation``.
        """
        if step_id in self._entries:
            return self._entries[step_id]
        entry = JournalEntry(step_id=step_id)
        self._entries[step_id] = entry
        return entry

    def get_entry(self, step_id: int) -> JournalEntry:
        """Fetch the (already-opened) entry for ``step_id``."""
        return self._entries[step_id]

    def has_entry(self, step_id: int) -> bool:
        """Return True iff a journal entry exists for ``step_id``."""
        return step_id in self._entries

    # ------------------------------------------------------------------
    # Reservation recording
    # ------------------------------------------------------------------
    def issue_journal_id(self) -> int:
        """Allocate a fresh journal ID for a new ``Reservation``."""
        jid = self._next_journal_id
        self._next_journal_id += 1
        return jid

    def record_resource_reservation(
        self, step_id: int, reservation: Reservation
    ) -> None:
        """Append a reservation to the entry for ``step_id``."""
        self._entries[step_id].reservations.append(reservation)

    # ------------------------------------------------------------------
    # Commit / rollback
    # ------------------------------------------------------------------
    def commit_step_transaction(self, step_id: int) -> JournalEntry:
        """Mark all reservations for ``step_id`` as committed; pop the entry.

        Returns the popped entry so the caller can drive resource release for
        any reservations whose ``must_outlast_snapshot_step_id`` is now retired.
        Raises ``KeyError`` if no such entry exists.
        """
        entry = self._entries.pop(step_id)
        for r in entry.reservations:
            if r.state is ReservationState.RESERVED:
                r.state = ReservationState.COMMITTED
        return entry

    def rollback_step_transaction(self, step_id: int) -> JournalEntry:
        """Mark all reservations for ``step_id`` as rolled-back; pop the entry."""
        entry = self._entries.pop(step_id)
        for r in entry.reservations:
            if r.state is ReservationState.RESERVED:
                r.state = ReservationState.ROLLED_BACK
        return entry

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def open_step_count(self) -> int:
        """Number of journal entries currently open (uncommitted, unrolled-back)."""
        return len(self._entries)

    def open_step_ids(self) -> List[int]:
        """Step IDs of currently open journal entries, sorted ascending."""
        return sorted(self._entries.keys())
