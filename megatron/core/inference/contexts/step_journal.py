# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transaction journal scaffolding for dynamic inference steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def _step_id_value(step_id) -> int:
    """Return the integer value for a dynamic step ID."""
    if hasattr(step_id, "value"):
        return int(step_id.value)
    if step_id < 0:
        raise ValueError(f"step_id must be >= 0, got {step_id}")
    return int(step_id)


def _int_tuple(values: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Return an immutable tuple of ints."""
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _str_tuple(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Return an immutable tuple of strings."""
    if values is None:
        return ()
    return tuple(str(value) for value in values)


def _freeze_mapping(value: Optional[Mapping[Any, Any]]) -> Mapping[Any, Any]:
    """Return an immutable mapping copy."""
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, kw_only=True)
class StepJournalEntry:
    """Immutable journal record for one dynamic step."""

    step_id: int
    snapshot_slot_id: int = 0
    request_slots_before: Sequence[int] = field(default_factory=tuple)
    request_slots_after: Sequence[int] = field(default_factory=tuple)
    active_request_ids: Sequence[str] = field(default_factory=tuple)
    placeholder_token_counts: Mapping[str, int] = field(default_factory=dict)
    reserved_kv_blocks: Sequence[int] = field(default_factory=tuple)
    reserved_mamba_slots: Sequence[int] = field(default_factory=tuple)
    prefix_cache_refcount_deltas: Mapping[Any, int] = field(default_factory=dict)
    pause_resume_evictions: Sequence[str] = field(default_factory=tuple)
    decode_input_destination_indices: Sequence[int] = field(default_factory=tuple)
    cuda_graph_batch_dimensions: Optional[Any] = None
    resources_waiting_on_snapshot: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError(f"step_id must be >= 0, got {self.step_id}")
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "request_slots_before", _int_tuple(self.request_slots_before))
        object.__setattr__(self, "request_slots_after", _int_tuple(self.request_slots_after))
        object.__setattr__(self, "active_request_ids", _str_tuple(self.active_request_ids))
        object.__setattr__(
            self, "placeholder_token_counts", _freeze_mapping(self.placeholder_token_counts)
        )
        object.__setattr__(self, "reserved_kv_blocks", _int_tuple(self.reserved_kv_blocks))
        object.__setattr__(self, "reserved_mamba_slots", _int_tuple(self.reserved_mamba_slots))
        object.__setattr__(
            self,
            "prefix_cache_refcount_deltas",
            _freeze_mapping(self.prefix_cache_refcount_deltas),
        )
        object.__setattr__(
            self, "pause_resume_evictions", _str_tuple(self.pause_resume_evictions)
        )
        object.__setattr__(
            self,
            "decode_input_destination_indices",
            _int_tuple(self.decode_input_destination_indices),
        )
        object.__setattr__(
            self, "resources_waiting_on_snapshot", tuple(self.resources_waiting_on_snapshot)
        )


@dataclass
class _MutableStepJournalEntry:
    """Mutable journal entry used while a step is open."""

    step_id: int
    snapshot_slot_id: int = 0
    request_slots_before: Tuple[int, ...] = field(default_factory=tuple)
    request_slots_after: Tuple[int, ...] = field(default_factory=tuple)
    active_request_ids: Tuple[str, ...] = field(default_factory=tuple)
    placeholder_token_counts: Dict[str, int] = field(default_factory=dict)
    reserved_kv_blocks: list[int] = field(default_factory=list)
    reserved_mamba_slots: list[int] = field(default_factory=list)
    prefix_cache_refcount_deltas: Dict[Any, int] = field(default_factory=dict)
    pause_resume_evictions: list[str] = field(default_factory=list)
    decode_input_destination_indices: Tuple[int, ...] = field(default_factory=tuple)
    cuda_graph_batch_dimensions: Optional[Any] = None
    resources_waiting_on_snapshot: list[Any] = field(default_factory=list)

    def freeze(self) -> StepJournalEntry:
        """Return an immutable public view of this journal entry."""
        return StepJournalEntry(
            step_id=self.step_id,
            snapshot_slot_id=self.snapshot_slot_id,
            request_slots_before=self.request_slots_before,
            request_slots_after=self.request_slots_after,
            active_request_ids=self.active_request_ids,
            placeholder_token_counts=self.placeholder_token_counts,
            reserved_kv_blocks=self.reserved_kv_blocks,
            reserved_mamba_slots=self.reserved_mamba_slots,
            prefix_cache_refcount_deltas=self.prefix_cache_refcount_deltas,
            pause_resume_evictions=self.pause_resume_evictions,
            decode_input_destination_indices=self.decode_input_destination_indices,
            cuda_graph_batch_dimensions=self.cuda_graph_batch_dimensions,
            resources_waiting_on_snapshot=self.resources_waiting_on_snapshot,
        )


class StepJournal:
    """Ordered transaction journal for scheduled dynamic steps."""

    def __init__(self):
        self._open_entries: Dict[int, _MutableStepJournalEntry] = {}
        self._committed_entries: Dict[int, StepJournalEntry] = {}
        self._rolled_back_entries: Dict[int, StepJournalEntry] = {}

    @property
    def open_step_ids(self) -> Tuple[int, ...]:
        """Step IDs with open journal entries."""
        return tuple(sorted(self._open_entries))

    @property
    def open_entry_count(self) -> int:
        """Number of open journal entries."""
        return len(self._open_entries)

    def has_open_entries(self) -> bool:
        """Return whether any journal entry is still open."""
        return bool(self._open_entries)

    def begin_step_journal(
        self,
        step_id,
        *,
        snapshot_slot_id: int = 0,
        request_slots_before: Optional[Sequence[int]] = None,
        active_request_ids: Optional[Sequence[Any]] = None,
    ) -> StepJournalEntry:
        """Open a journal entry for one dynamic step."""
        step_value = _step_id_value(step_id)
        if step_value in self._open_entries:
            raise RuntimeError(f"Step journal entry {step_value} is already open")
        if step_value in self._committed_entries or step_value in self._rolled_back_entries:
            raise RuntimeError(f"Step journal entry {step_value} is already terminal")

        entry = _MutableStepJournalEntry(
            step_id=step_value,
            snapshot_slot_id=int(snapshot_slot_id),
            request_slots_before=_int_tuple(request_slots_before),
            active_request_ids=_str_tuple(active_request_ids),
        )
        self._open_entries[step_value] = entry
        return entry.freeze()

    def record_placeholder_delta(
        self, step_id, request_id: Any, token_count_delta: int
    ) -> StepJournalEntry:
        """Record placeholder-token accounting for a request."""
        entry = self._require_open_entry(step_id)
        request_key = str(request_id)
        new_count = entry.placeholder_token_counts.get(request_key, 0) + int(token_count_delta)
        if new_count == 0:
            entry.placeholder_token_counts.pop(request_key, None)
        else:
            entry.placeholder_token_counts[request_key] = new_count
        return entry.freeze()

    def record_resource_reservation(self, step_id, reservation) -> StepJournalEntry:
        """Record resources reserved for the step."""
        entry = self._require_open_entry(step_id)
        entry.resources_waiting_on_snapshot.append(reservation)
        entry.reserved_kv_blocks.extend(
            int(block_id) for block_id in getattr(reservation, "kv_block_ids", ())
        )
        entry.reserved_mamba_slots.extend(
            int(slot_id) for slot_id in getattr(reservation, "mamba_slot_ids", ())
        )
        for key, delta in getattr(reservation, "prefix_cache_refcount_deltas", {}).items():
            entry.prefix_cache_refcount_deltas[key] = (
                entry.prefix_cache_refcount_deltas.get(key, 0) + int(delta)
            )
        return entry.freeze()

    def record_snapshot_slot(
        self,
        step_id,
        snapshot_slot_id: int,
        *,
        cuda_graph_batch_dimensions: Optional[Any] = None,
    ) -> StepJournalEntry:
        """Record the snapshot slot bound to this step."""
        entry = self._require_open_entry(step_id)
        entry.snapshot_slot_id = int(snapshot_slot_id)
        if cuda_graph_batch_dimensions is not None:
            entry.cuda_graph_batch_dimensions = cuda_graph_batch_dimensions
        return entry.freeze()

    def record_request_slot_transition(
        self,
        step_id,
        *,
        request_slots_before: Optional[Sequence[int]] = None,
        request_slots_after: Optional[Sequence[int]] = None,
        active_request_ids: Optional[Sequence[Any]] = None,
        pause_resume_evictions: Optional[Sequence[Any]] = None,
        decode_input_destination_indices: Optional[Sequence[int]] = None,
    ) -> StepJournalEntry:
        """Record request-slot movement for this step."""
        entry = self._require_open_entry(step_id)
        if request_slots_before is not None:
            entry.request_slots_before = _int_tuple(request_slots_before)
        if request_slots_after is not None:
            entry.request_slots_after = _int_tuple(request_slots_after)
        if active_request_ids is not None:
            entry.active_request_ids = _str_tuple(active_request_ids)
        if pause_resume_evictions is not None:
            entry.pause_resume_evictions = list(_str_tuple(pause_resume_evictions))
        if decode_input_destination_indices is not None:
            entry.decode_input_destination_indices = _int_tuple(
                decode_input_destination_indices
            )
        return entry.freeze()

    def commit_step_journal(
        self,
        step_id,
        *,
        request_slots_after: Optional[Sequence[int]] = None,
        active_request_ids: Optional[Sequence[Any]] = None,
    ) -> StepJournalEntry:
        """Commit an open journal entry."""
        step_value = _step_id_value(step_id)
        entry = self._open_entries.pop(step_value)
        if request_slots_after is not None:
            entry.request_slots_after = _int_tuple(request_slots_after)
        if active_request_ids is not None:
            entry.active_request_ids = _str_tuple(active_request_ids)
        frozen_entry = entry.freeze()
        self._committed_entries[step_value] = frozen_entry
        return frozen_entry

    def rollback_step_journal(
        self, step_id, *, reason: Optional[str] = None
    ) -> StepJournalEntry:
        """Roll back an open journal entry."""
        del reason
        step_value = _step_id_value(step_id)
        entry = self._open_entries.pop(step_value)
        frozen_entry = entry.freeze()
        self._rolled_back_entries[step_value] = frozen_entry
        return frozen_entry

    def rollback_all_open(self, *, reason: Optional[str] = None) -> Tuple[StepJournalEntry, ...]:
        """Roll back every open entry in step order."""
        return tuple(
            self.rollback_step_journal(step_id, reason=reason) for step_id in self.open_step_ids
        )

    def get_open_entry(self, step_id) -> Optional[StepJournalEntry]:
        """Return an immutable view of an open entry."""
        entry = self._open_entries.get(_step_id_value(step_id))
        return None if entry is None else entry.freeze()

    def get_committed_entry(self, step_id) -> Optional[StepJournalEntry]:
        """Return a committed entry."""
        return self._committed_entries.get(_step_id_value(step_id))

    def get_rolled_back_entry(self, step_id) -> Optional[StepJournalEntry]:
        """Return a rolled-back entry."""
        return self._rolled_back_entries.get(_step_id_value(step_id))

    def debug_dump_open_entries(self) -> Mapping[int, StepJournalEntry]:
        """Return immutable open-entry views keyed by step ID."""
        return {step_id: entry.freeze() for step_id, entry in self._open_entries.items()}

    def _require_open_entry(self, step_id) -> _MutableStepJournalEntry:
        """Return an open mutable entry or raise a targeted error."""
        step_value = _step_id_value(step_id)
        try:
            return self._open_entries[step_value]
        except KeyError as exc:
            raise RuntimeError(f"Step journal entry {step_value} is not open") from exc
