# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed contract structures for the async-overlap inference pipeline.

These dataclasses are the only currency that crosses async boundaries in the
async-overlap engine. No anonymous dictionaries. See v3 plan §2.3 in
``lawrence/reports/20260429-context-cpu-async-schedule-claude-v3.md``.

This module is introduced in commit 2 and has no production consumers yet;
subsequent commits import these types as the pipeline scaffolding lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


class ResourceKind(Enum):
    """Kinds of resources that participate in the transaction journal."""

    KV_BLOCK = "kv_block"
    MAMBA_SLOT = "mamba_slot"
    PREFIX_CACHE_REF = "prefix_cache_ref"
    REQUEST_SLOT = "request_slot"


class ReservationState(Enum):
    """Lifecycle states of a journal-tracked resource reservation."""

    RESERVED = "reserved"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Reservation:
    """A journal-tracked resource reservation.

    Reservations are taken during ``prepare_next_step_optimistic`` and either
    committed or rolled back during ``commit_step_transaction``. The release
    of the underlying resource is deferred until the snapshot referencing it
    has retired (``must_outlast_snapshot_step_id``).
    """

    journal_id: int
    resource_kind: ResourceKind
    resource_handle: Any  # opaque (block_id, slot_index, prefix_hash, etc.)
    state: ReservationState = ReservationState.RESERVED
    must_outlast_snapshot_step_id: int = -1


@dataclass
class StepInputPlan:
    """Sample-independent description of inputs for one step."""

    decode_request_slots: Tuple[int, ...]
    decode_token_destination_indices: Tuple[int, ...]
    prefill_cpu_token_ranges: Tuple[Tuple[int, int], ...]
    speculative_width: int
    previous_sample_source_step: Optional[int]


@dataclass
class DynamicStepPlan:
    """Output of ``prepare_next_step_optimistic`` — the optimistic plan for a step."""

    step_id: int
    request_slots: Tuple[int, ...]
    placeholder_deltas: Tuple[int, ...]
    input_plan: StepInputPlan
    resource_reservation_ids: List[int]
    intended_batch_dimensions: "InferenceBatchDimensions"


@dataclass
class DynamicStepSnapshot:
    """A versioned, GPU-resident snapshot of context state for one step.

    See v3 plan §2.5 (snapshot pool). The snapshot's ``buffer_pool_slot``
    handle binds it to fixed GPU addresses (required for CUDA graphs).

    The events ``metadata_ready_event`` and ``input_ready_event`` are the
    GPU-side dependencies the next forward kernel waits on. ``cpu_owner_step_count``
    is a snapshot-version counter incremented every time the slot is reused.
    """

    step_id: int
    buffer_pool_slot: int
    active_request_slot_view: Any  # slice or index tensor
    attention_metadata: Any  # MHA metadata bound to snapshot's MHA fields
    mamba_metadata: Optional[Any]  # bound to snapshot's Mamba fields (None if not hybrid)
    graph_match: Optional[Any]  # CUDA graph chosen for this snapshot's (slot, batch_dims)
    metadata_ready_event: Optional["torch.cuda.Event"] = None
    input_ready_event: Optional["torch.cuda.Event"] = None
    cpu_owner_step_count: int = 0


@dataclass
class AsyncStepOutput:
    """Output bundle for one step's CPU-visible D2H copies.

    All payload tensors live in pinned destinations. ``d2h_done_event`` fires
    when all copies complete; CPU consumers must synchronize on it before
    reading ``cpu_view``.
    """

    step_id: int
    source_gpu_tensors: Dict[str, Any] = field(default_factory=dict)
    pinned_destinations: Dict[str, Any] = field(default_factory=dict)
    d2h_done_event: Optional["torch.cuda.Event"] = None
    payload_metadata: Dict[str, bool] = field(default_factory=dict)
    # `cpu_view` is implemented as a property on AsyncStepOutput in commit 4;
    # not stored here.

    def has_payload(self, name: str) -> bool:
        """True if this output bundle includes the named payload."""
        return self.payload_metadata.get(name, False)


@dataclass
class DynamicStepLaunch:
    """One in-flight step — the pipeline's ``self._inflight`` deque element."""

    step_id: int
    snapshot: DynamicStepSnapshot
    forward_done_event: Optional["torch.cuda.Event"] = None
    sample_done_event: Optional["torch.cuda.Event"] = None
    output: Optional[AsyncStepOutput] = None
    journal_id: int = -1


@dataclass
class StepRetirementResult:
    """Result of committing one step's journal entry."""

    step_id: int
    finished_request_records: List[Any] = field(default_factory=list)
    paused_request_ids: List[int] = field(default_factory=list)
    evicted_request_ids: List[int] = field(default_factory=list)
    reservation_commit_count: int = 0
    reservation_rollback_count: int = 0
    discarded_lookahead_token_count: int = 0
