# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Typed contracts for dynamic async-overlap inference steps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Tuple


def _as_tuple(value: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
    """Return an immutable tuple copy of a possibly missing sequence."""
    if value is None:
        return ()
    return tuple(value)


def _freeze_mapping(value: Optional[Mapping[Any, Any]]) -> Mapping[Any, Any]:
    """Return an immutable mapping copy of a possibly missing mapping."""
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, order=True)
class DynamicStepId:
    """Monotonic dynamic inference step identity."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"DynamicStepId must be >= 0, got {self.value}")


@dataclass(frozen=True, kw_only=True)
class DynamicStepRequestPlan:
    """Scheduler-visible request plan for one dynamic step."""

    step_id: DynamicStepId
    active_request_ids: Sequence[str] = field(default_factory=tuple)
    decode_request_ids: Sequence[str] = field(default_factory=tuple)
    prefill_request_ids: Sequence[str] = field(default_factory=tuple)
    speculative_request_ids: Sequence[str] = field(default_factory=tuple)
    placeholder_token_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_request_ids", _as_tuple(self.active_request_ids))
        object.__setattr__(self, "decode_request_ids", _as_tuple(self.decode_request_ids))
        object.__setattr__(self, "prefill_request_ids", _as_tuple(self.prefill_request_ids))
        object.__setattr__(
            self, "speculative_request_ids", _as_tuple(self.speculative_request_ids)
        )
        object.__setattr__(
            self, "placeholder_token_counts", _freeze_mapping(self.placeholder_token_counts)
        )


@dataclass(frozen=True, kw_only=True)
class SnapshotSlotHandle:
    """Ownership handle for a fixed-address GPU snapshot slot."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    metadata_ready_event: Optional[Any] = None
    input_ready_event: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")


@dataclass(frozen=True, kw_only=True)
class StepInputPlan:
    """GPU input preparation plan for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    request_ids: Sequence[str] = field(default_factory=tuple)
    decode_request_ids: Sequence[str] = field(default_factory=tuple)
    prefill_request_ids: Sequence[str] = field(default_factory=tuple)
    decode_input_destination_indices: Sequence[int] = field(default_factory=tuple)
    input_ready_event: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "request_ids", _as_tuple(self.request_ids))
        object.__setattr__(self, "decode_request_ids", _as_tuple(self.decode_request_ids))
        object.__setattr__(self, "prefill_request_ids", _as_tuple(self.prefill_request_ids))
        object.__setattr__(
            self,
            "decode_input_destination_indices",
            _as_tuple(self.decode_input_destination_indices),
        )


@dataclass(frozen=True, kw_only=True)
class DynamicStepContextSnapshot:
    """Immutable view of the context snapshot owned by one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    request_plan: DynamicStepRequestPlan
    metadata_ready_event: Optional[Any] = None
    input_ready_event: Optional[Any] = None
    gpu_view: Optional[Any] = None
    cpu_staging_buffer: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")


@dataclass(frozen=True, kw_only=True)
class DynamicStepGpuLaunch:
    """Forward/sampling launch contract for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    metadata_ready_event: Optional[Any] = None
    input_ready_event: Optional[Any] = None
    compute_done_event: Optional[Any] = None
    cuda_graph_batch_dimensions: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")


@dataclass(frozen=True, kw_only=True)
class AsyncStepOutput:
    """CPU-visible output copy contract for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    compute_done_event: Optional[Any] = None
    output_ready_event: Optional[Any] = None
    sampled_tokens: Optional[Any] = None
    sampled_tokens_cpu: Optional[Any] = None
    accepted_token_counts_cpu: Optional[Any] = None
    logprob_payload_cpu: Optional[Any] = None
    routing_payload_cpu: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")


@dataclass(frozen=True, kw_only=True)
class ResourceReservation:
    """Allocator resources reserved for a scheduled-but-not-retired step."""

    step_id: DynamicStepId
    request_id: str
    snapshot_slot_id: int
    kv_block_ids: Sequence[int] = field(default_factory=tuple)
    mamba_slot_ids: Sequence[int] = field(default_factory=tuple)
    prefix_cache_refcount_deltas: Mapping[Any, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "kv_block_ids", _as_tuple(self.kv_block_ids))
        object.__setattr__(self, "mamba_slot_ids", _as_tuple(self.mamba_slot_ids))
        object.__setattr__(
            self,
            "prefix_cache_refcount_deltas",
            _freeze_mapping(self.prefix_cache_refcount_deltas),
        )


@dataclass(frozen=True, kw_only=True)
class StepJournalEntry:
    """Ordered commit/rollback record for one scheduled dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
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
    resources_waiting_on_snapshot: Sequence[ResourceReservation] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "request_slots_before", _as_tuple(self.request_slots_before))
        object.__setattr__(self, "request_slots_after", _as_tuple(self.request_slots_after))
        object.__setattr__(self, "active_request_ids", _as_tuple(self.active_request_ids))
        object.__setattr__(
            self, "placeholder_token_counts", _freeze_mapping(self.placeholder_token_counts)
        )
        object.__setattr__(self, "reserved_kv_blocks", _as_tuple(self.reserved_kv_blocks))
        object.__setattr__(self, "reserved_mamba_slots", _as_tuple(self.reserved_mamba_slots))
        object.__setattr__(
            self,
            "prefix_cache_refcount_deltas",
            _freeze_mapping(self.prefix_cache_refcount_deltas),
        )
        object.__setattr__(
            self, "pause_resume_evictions", _as_tuple(self.pause_resume_evictions)
        )
        object.__setattr__(
            self,
            "decode_input_destination_indices",
            _as_tuple(self.decode_input_destination_indices),
        )
        object.__setattr__(
            self, "resources_waiting_on_snapshot", _as_tuple(self.resources_waiting_on_snapshot)
        )


@dataclass(frozen=True, kw_only=True)
class StepRetirementResult:
    """Ordered retirement result for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    output_ready_event: Optional[Any] = None
    committed_request_ids: Sequence[str] = field(default_factory=tuple)
    completed_request_ids: Sequence[str] = field(default_factory=tuple)
    rolled_back_request_ids: Sequence[str] = field(default_factory=tuple)
    reservation_commits: int = 0
    reservation_rollbacks: int = 0

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "committed_request_ids", _as_tuple(self.committed_request_ids))
        object.__setattr__(self, "completed_request_ids", _as_tuple(self.completed_request_ids))
        object.__setattr__(
            self, "rolled_back_request_ids", _as_tuple(self.rolled_back_request_ids)
        )
