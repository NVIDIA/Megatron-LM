# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Typed contracts for dynamic async-overlap inference steps."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Deque, Optional, Tuple


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


def _as_int_pair_tuple(value: Optional[Sequence[Sequence[int]]]) -> Tuple[Tuple[int, int], ...]:
    """Return an immutable tuple of integer pairs."""
    if value is None:
        return ()
    pairs = []
    for pair in value:
        if len(pair) != 2:
            raise ValueError(f"expected a pair, got {pair}")
        start, end = pair
        if int(start) < 0 or int(end) < int(start):
            raise ValueError(f"invalid token range ({start}, {end})")
        pairs.append((int(start), int(end)))
    return tuple(pairs)


@dataclass(frozen=True, order=True)
class DynamicStepId:
    """Monotonic dynamic inference step identity."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"DynamicStepId must be >= 0, got {self.value}")


class DynamicAsyncPipeline:
    """Queue-depth-limited dynamic-step launch and retirement pipeline."""

    def __init__(self, *, engine: Any, retirement_service: Any, queue_depth: int):
        if queue_depth <= 0:
            raise ValueError(f"queue_depth must be > 0, got {queue_depth}")
        self.engine = engine
        self.retirement_service = retirement_service
        self.queue_depth = int(queue_depth)
        self.snapshot_pool = engine.context.snapshot_pool
        self.in_flight_launches: Deque[Tuple[Any, Any, float]] = deque()

    @property
    def pending_launch_count(self) -> int:
        """Number of launched steps awaiting ordered retirement."""
        return len(self.in_flight_launches)

    async def step(self):
        """Run one queue-depth-limited pipeline step."""
        if self.pending_launch_count >= self.queue_depth:
            await self.drain_next()
        self.in_flight_launches.append(await self.engine.async_forward())
        return await self.drain_next()

    async def drain_next(self):
        """Retire the oldest launched step, if one is pending."""
        if not self.in_flight_launches:
            return None
        step_result, context_state, step_time = self.in_flight_launches.popleft()
        return await self.engine.async_bookkeep(step_result, context_state, step_time)

    async def drain_all(self):
        """Retire every pending launched step."""
        result = None
        while self.in_flight_launches:
            result = await self.drain_next()
        return result

    def drain_for_shutdown(self) -> None:
        """Validate no launched step is left unretired before shutdown."""
        if self.in_flight_launches:
            raise RuntimeError("Cannot synchronously shutdown with unretired launched steps")
        self.retirement_service.drain_for_shutdown()

    def drain_for_suspend(self) -> None:
        """Validate no launched step is left unretired before suspend."""
        if self.in_flight_launches:
            raise RuntimeError("Cannot synchronously suspend with unretired launched steps")
        self.retirement_service.drain_for_suspend()

    def drain_for_request_reuse(self, request_id: int) -> None:
        """Validate no launched step can still reference a reused request ID."""
        if self.in_flight_launches:
            raise RuntimeError(
                f"Cannot reuse request {request_id} with unretired launched steps"
            )
        self.retirement_service.drain_for_request_reuse(request_id)


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
    source_step_id: Optional[DynamicStepId] = None
    request_ids: Sequence[str] = field(default_factory=tuple)
    decode_request_slots: Sequence[int] = field(default_factory=tuple)
    decode_request_ids: Sequence[str] = field(default_factory=tuple)
    prefill_request_ids: Sequence[str] = field(default_factory=tuple)
    decode_input_destination_indices: Sequence[int] = field(default_factory=tuple)
    prefill_prompt_token_ranges: Sequence[Sequence[int]] = field(default_factory=tuple)
    speculative_width: int = 0
    debug_expected_input_ids: Optional[Any] = None
    input_ready_event: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        if self.speculative_width < 0:
            raise ValueError(f"speculative_width must be >= 0, got {self.speculative_width}")
        object.__setattr__(self, "request_ids", _as_tuple(self.request_ids))
        object.__setattr__(self, "decode_request_slots", _as_tuple(self.decode_request_slots))
        object.__setattr__(self, "decode_request_ids", _as_tuple(self.decode_request_ids))
        object.__setattr__(self, "prefill_request_ids", _as_tuple(self.prefill_request_ids))
        object.__setattr__(
            self,
            "decode_input_destination_indices",
            _as_tuple(self.decode_input_destination_indices),
        )
        object.__setattr__(
            self,
            "prefill_prompt_token_ranges",
            _as_int_pair_tuple(self.prefill_prompt_token_ranges),
        )
        if len(self.decode_request_slots) != len(self.decode_input_destination_indices):
            raise ValueError(
                "decode_request_slots and decode_input_destination_indices must have "
                "the same length"
            )


@dataclass(frozen=True, kw_only=True)
class DynamicStepContextSnapshot:
    """Immutable view of the context snapshot owned by one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    request_plan: DynamicStepRequestPlan
    active_request_count: int = 0
    cuda_graph_request_count: Optional[int] = None
    metadata_ready_event: Optional[Any] = None
    input_ready_event: Optional[Any] = None
    input_ids: Optional[Any] = None
    position_ids: Optional[Any] = None
    gpu_view: Optional[Any] = None
    cpu_staging_buffer: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        if self.active_request_count < 0:
            raise ValueError(
                f"active_request_count must be >= 0, got {self.active_request_count}"
            )


def assert_snapshot_gpu_view_bound(
    snapshot: DynamicStepContextSnapshot, live_gpu_view: Optional[Any], *, debug_enabled: bool
) -> None:
    """Guard that dynamic GPU work is bound to the prepared snapshot view."""
    if not debug_enabled:
        return
    if snapshot.gpu_view is None:
        raise RuntimeError("Dynamic GPU launch requires a snapshot-bound gpu_view")
    if snapshot.gpu_view is not live_gpu_view:
        raise RuntimeError(
            "Dynamic GPU launch would read mutable context.gpu_view instead of the prepared "
            "snapshot gpu_view"
        )


@dataclass(frozen=True, kw_only=True)
class DynamicStepGpuLaunch:
    """Forward/sampling launch contract for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    metadata_ready_event: Optional[Any] = None
    input_ready_event: Optional[Any] = None
    compute_done_event: Optional[Any] = None
    input_ids: Optional[Any] = None
    position_ids: Optional[Any] = None
    gpu_view: Optional[Any] = None
    logits: Optional[Any] = None
    routing_indices_per_request: Optional[Any] = None
    cuda_graph_batch_dimensions: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")


@dataclass(frozen=True, kw_only=True)
class AsyncStepOutput:
    """CPU-visible output copy contract for one dynamic step."""

    step_id: DynamicStepId
    snapshot_slot_id: int
    active_request_count: int = 0
    compute_done_event: Optional[Any] = None
    output_ready_event: Optional[Any] = None
    active_requests_mask: Optional[Any] = None
    sampled_tokens: Optional[Any] = None
    sampled_tokens_cpu: Optional[Any] = None
    sampled_mtp_tokens_cpu: Optional[Any] = None
    accepted_token_counts_cpu: Optional[Any] = None
    logprob_payload_cpu: Optional[Any] = None
    routing_payload_cpu: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        if self.active_request_count < 0:
            raise ValueError(
                f"active_request_count must be >= 0, got {self.active_request_count}"
            )


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
    context_update_result: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.snapshot_slot_id < 0:
            raise ValueError(f"snapshot_slot_id must be >= 0, got {self.snapshot_slot_id}")
        object.__setattr__(self, "committed_request_ids", _as_tuple(self.committed_request_ids))
        object.__setattr__(self, "completed_request_ids", _as_tuple(self.completed_request_ids))
        object.__setattr__(
            self, "rolled_back_request_ids", _as_tuple(self.rolled_back_request_ids)
        )
        if self.context_update_result is not None:
            object.__setattr__(
                self, "context_update_result", _freeze_mapping(self.context_update_result)
            )
