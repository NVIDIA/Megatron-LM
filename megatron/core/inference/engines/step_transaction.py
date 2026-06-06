# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transactional primitives for async-scheduling (launch-before-commit) decode.

These data structures make one decode step a *transaction* so the dynamic engine
can launch the next forward before committing the current one
(``enable_async_scheduling``), while keeping resource lifetimes provably safe:

- :class:`StepTxn` -- one decode step's transaction. Holds the snapshot of the
  request set the forward consumes, the resources it *leases* (KV blocks, reserved
  boundary blocks, mamba slots), and its two CUDA fences (``h2d_done_event``,
  ``forward_done_event``).
- :class:`RetireQueue` -- a two-step deferred-free queue. A resource freed during
  step K is returned to its allocator only after the forward that may still read
  it has completed (its ``forward_done_event``) *and* two steps have elapsed.
- :class:`StepTxnDiagnostics` -- cheap counters (prepared / launched / adopted /
  sync-steps / barrier-skips / retired / guard-failures / per-reason skips).
- :class:`StepTransactionManager` -- owns the retire queue + diagnostics and the
  begin / commit / publish / retire / barrier lifecycle.

Design invariants (enforced here and asserted by tests):

* There are deliberately **no** global ``_async_reserved_*`` / ``_async_deferred_*``
  arrays. Every resource a forward can touch is a lease field on its
  :class:`StepTxn`; freeing goes through the :class:`RetireQueue`.
* The structures are inert until ``enable_async_scheduling`` is set: the manager is
  only constructed when the flag is on, so the disabled decode path is unchanged.

The commit that introduces this module only wraps the existing *serial* step as an
always-adopted transaction (no speculative child launch); :meth:`prestage` and
:meth:`adopt` (the launch-before-commit machinery) arrive in later commits.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LaunchEligibility(str, Enum):
    """Whether a step may launch a speculative child forward before committing."""

    ELIGIBLE = "eligible"  # a speculative child forward may be launched before commit
    SYNC = "sync"  # layout-changing / unsupported: must run as a synchronous step


class LaunchGateReason(str, Enum):
    """Why a step is (in)eligible for a speculative launch.

    These are the *static* gates and the explicit barrier/sync reasons from the design
    (sections 1.2 and 1.9). There are deliberately no per-step "disabled" vetoes beyond
    these. Note that finish / stop-word / cancel are absent: a launched forward absorbs a
    finish (it discards the finished row; compaction is a pure row permutation), so those
    events do NOT force a sync step and are never a reason here.
    """

    PURE_DECODE = "pure_decode"  # eligible: pure decode for the committed members
    NO_ACTIVE_REQUESTS = "no_active_requests"
    NOT_DECODE_ONLY = "not_decode_only"  # prefill / mixed step
    GRAPH_INELIGIBLE = "graph_ineligible"  # not running under a (block-scope) CUDA graph
    GRAPH_RECAPTURE = "graph_recapture"  # a graph recapture barrier this step
    CHUNKED_PREFILL = "chunked_prefill"
    PENDING_ADMISSION = "pending_admission"  # a waiting request can be admitted
    RESUME = "resume"
    EVICT = "evict"
    FORCED_PAUSE_OVERFLOW = "forced_pause_overflow"
    MTP_DEPENDENT_LAYOUT = "mtp_dependent_layout"  # MTP verify/rewind changes layout
    KV_RESERVATION_UNAVAILABLE = "kv_reservation_unavailable"  # boundary block can't be reserved


@dataclass
class LaunchSignals:
    """The (static) facts needed to decide if a speculative child launch is eligible.

    Absorbable events (finish / stop-word / cancel) are intentionally NOT inputs: a
    launched forward discards a finished row without re-running, so they never block a
    launch. Only changes a launched forward could not absorb appear here.
    """

    decode_only: bool  # this step is decode-only (no prefill/mixed)
    active_request_count: int  # number of active (non-paused) requests
    using_cuda_graph: bool  # the forward runs under a (block-scope) CUDA graph
    pending_admission: bool = False  # a waiting request could be admitted
    resume_pending: bool = False  # a paused request is being resumed
    evict_pending: bool = False  # a request is being evicted
    forced_pause_overflow: bool = False  # active requests force-paused on KV overflow
    chunked_prefill_active: bool = False  # chunked prefill in progress
    graph_recapture: bool = False  # a CUDA-graph recapture barrier this step
    mtp_layout_change: bool = False  # MTP verify/rewind changes layout this step
    kv_reservation_fits: bool = True  # the <=1 boundary block per crosser fits pre-launch


@dataclass
class LaunchDecision:
    """The result of :func:`classify_launch_eligibility`."""

    eligibility: LaunchEligibility
    reason: LaunchGateReason

    @property
    def eligible(self) -> bool:
        """True iff a speculative child forward may be launched this step."""
        return self.eligibility is LaunchEligibility.ELIGIBLE


def classify_launch_eligibility(signals: LaunchSignals) -> LaunchDecision:
    """Statically classify whether a step may launch a speculative child forward.

    A speculative launch is attempted ONLY when every static gate holds (feature on +
    CUDA graph + decode-only + active>0) and no layout-changing / barrier condition is
    present (admission, resume, evict, forced-pause, chunked prefill, graph recapture,
    MTP-dependent layout, unfittable KV reservation). Otherwise the step runs
    synchronously. This mirrors the design's launch gate (1.2) and static eligibility
    gate (1.9): no per-step dynamic vetoes beyond these.

    The order of checks is fixed so the reported reason is deterministic when several
    conditions hold at once (the most fundamental gate wins).
    """
    if signals.active_request_count <= 0:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.NO_ACTIVE_REQUESTS)
    if not signals.decode_only:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.NOT_DECODE_ONLY)
    if not signals.using_cuda_graph:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.GRAPH_INELIGIBLE)
    if signals.graph_recapture:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.GRAPH_RECAPTURE)
    if signals.chunked_prefill_active:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.CHUNKED_PREFILL)
    if signals.pending_admission:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.PENDING_ADMISSION)
    if signals.resume_pending:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.RESUME)
    if signals.evict_pending:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.EVICT)
    if signals.forced_pause_overflow:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.FORCED_PAUSE_OVERFLOW)
    if signals.mtp_layout_change:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.MTP_DEPENDENT_LAYOUT)
    if not signals.kv_reservation_fits:
        return LaunchDecision(LaunchEligibility.SYNC, LaunchGateReason.KV_RESERVATION_UNAVAILABLE)
    return LaunchDecision(LaunchEligibility.ELIGIBLE, LaunchGateReason.PURE_DECODE)


class TxnPhase(str, Enum):
    """Lifecycle phase of a :class:`StepTxn`."""

    PRESTAGED = "prestaged"  # metadata + H2D built into the free slot, not yet launched
    LAUNCHED = "launched"  # forward enqueued speculatively, in flight
    ADOPTED = "adopted"  # launched forward's outputs consumed this step
    COMMITTED = "committed"  # update_requests applied for this step
    RETIRED = "retired"  # resources released
    SERIAL = "serial"  # always-adopted serial wrap (no speculative launch)


@dataclass
class StepTxn:
    """A single decode step's transaction.

    The launched forward is consumed by ``request_ids`` (the snapshotted
    pre-compaction active request ids), never by row index -- compaction only
    permutes rows, so physical KV-block ids and mamba-slot ids stay bonded to each
    surviving request (see the design's consume-by-request-id rule).
    """

    step_id: int
    phase: TxnPhase = TxnPhase.SERIAL

    # Snapshot of the active request set the forward consumes (pre-compaction).
    request_ids: Optional[Any] = None  # torch.Tensor or None
    active_request_count: int = 0

    # CUDA-graph bucket / padded batch dimensions this forward was built for.
    bucket: Any = None

    # Whether the forward was launched speculatively (False for the serial wrap).
    speculative: bool = False

    # --- Resource leases (no global arrays; the txn owns what its forward touches). ---
    # Physical KV block ids leased for this step.
    kv_block_leases: List[int] = field(default_factory=list)
    # Boundary blocks reserved for crossers, keyed by (request_id, block_column).
    reserved_boundary_blocks: Dict[Tuple[int, int], int] = field(default_factory=dict)
    # Mamba state slot ids leased for this step.
    mamba_slot_leases: List[int] = field(default_factory=list)

    # --- The two (and only two) per-transaction CUDA fences. ---
    # Signalled when the coalesced bookkeeping H2D into this txn's slot completes.
    h2d_done_event: Any = None  # torch.cuda.Event or None
    # Signalled when this txn's forward completes; gates retire-queue frees + barriers.
    forward_done_event: Any = None  # torch.cuda.Event or None

    # Output handles (logits / sample buffers) produced by the forward.
    output_handles: Dict[str, Any] = field(default_factory=dict)

    def lease_kv_block(self, block_id: int) -> None:
        """Record that this transaction owns ``block_id`` for the duration of its forward."""
        self.kv_block_leases.append(int(block_id))

    def reserve_boundary_block(self, request_id: int, block_column: int, block_id: int) -> None:
        """Reserve the (request_id, block_column) boundary block this forward may cross into."""
        self.reserved_boundary_blocks[(int(request_id), int(block_column))] = int(block_id)

    def lease_mamba_slot(self, slot_id: int) -> None:
        """Record that this transaction owns mamba slot ``slot_id`` for its forward."""
        self.mamba_slot_leases.append(int(slot_id))

    def leased_kv_block_ids(self) -> List[int]:
        """All KV block ids this transaction owns (direct leases + reserved boundary blocks)."""
        return list(self.kv_block_leases) + list(self.reserved_boundary_blocks.values())

    def has_leases(self) -> bool:
        """Whether this transaction owns any resource."""
        return bool(self.kv_block_leases or self.reserved_boundary_blocks or self.mamba_slot_leases)


@dataclass
class _RetireEntry:
    """One deferred free: release ``payload`` once the fence clears and the delay elapses."""

    enqueue_step: int
    event: Any  # torch.cuda.Event-like (exposes .query()) or None
    release: Callable[[], None]
    tag: str = ""


class RetireQueue:
    """Two-step deferred-free queue gated by a CUDA fence.

    A resource enqueued at step K is released only once **both** conditions hold:

    1. at least :attr:`RETIRE_DELAY_STEPS` engine steps have elapsed
       (``current_step - enqueue_step >= RETIRE_DELAY_STEPS``), and
    2. its ``event`` has completed (``event.query()`` is True), or it has no event.

    This guarantees a freed KV block / mamba slot is never handed back to its
    allocator while a still-in-flight forward could read it.
    """

    RETIRE_DELAY_STEPS = 2

    def __init__(self) -> None:
        self._entries: "deque[_RetireEntry]" = deque()

    def enqueue(
        self,
        *,
        enqueue_step: int,
        release: Callable[[], None],
        event: Any = None,
        tag: str = "",
    ) -> None:
        """Defer ``release`` until the fence clears and the two-step delay elapses."""
        self._entries.append(
            _RetireEntry(enqueue_step=enqueue_step, event=event, release=release, tag=tag)
        )

    @classmethod
    def _ready(cls, entry: _RetireEntry, current_step: int) -> bool:
        if current_step - entry.enqueue_step < cls.RETIRE_DELAY_STEPS:
            return False
        if entry.event is not None and not entry.event.query():
            return False
        return True

    def drain(self, current_step: int) -> int:
        """Release every ready entry; keep the rest in order. Returns count released.

        Scans all entries (not just the front) so a ready entry is never blocked
        behind a not-yet-ready one, regardless of fence-completion ordering.
        """
        if not self._entries:
            return 0
        remaining: "deque[_RetireEntry]" = deque()
        released = 0
        for entry in self._entries:
            if self._ready(entry, current_step):
                entry.release()
                released += 1
            else:
                remaining.append(entry)
        self._entries = remaining
        return released

    def drain_all_blocking(self) -> int:
        """Synchronize every pending fence and release everything (shutdown / barrier)."""
        released = 0
        for entry in self._entries:
            if entry.event is not None:
                entry.event.synchronize()
            entry.release()
            released += 1
        self._entries.clear()
        return released

    def pending(self) -> int:
        """Number of not-yet-released entries."""
        return len(self._entries)


@dataclass
class StepTxnDiagnostics:
    """Cheap, always-on counters for the async-scheduling pipeline.

    All increments are O(1) integer/Counter updates; nothing here is allocated or
    touched when ``enable_async_scheduling`` is off (the manager is not constructed).
    """

    prepared: int = 0  # child metadata prestaged
    launched: int = 0  # forwards launched speculatively (before commit)
    adopted: int = 0  # launched children whose outputs were consumed
    serial_steps: int = 0  # always-adopted serial wraps (no speculative launch)
    sync_steps: int = 0  # layout-changing steps that ran synchronously
    barrier_skips: int = 0  # steps where a barrier prevented a child launch
    retired: int = 0  # resources released from the retire queue
    guard_failures: int = 0  # adopt-guard assertion failures (expected to stay 0)
    skip_reasons: Counter = field(default_factory=Counter)  # reason -> count

    def as_dict(self) -> Dict[str, Any]:
        """A plain-dict snapshot suitable for logging."""
        return {
            "prepared": self.prepared,
            "launched": self.launched,
            "adopted": self.adopted,
            "serial_steps": self.serial_steps,
            "sync_steps": self.sync_steps,
            "barrier_skips": self.barrier_skips,
            "retired": self.retired,
            "guard_failures": self.guard_failures,
            "skip_reasons": dict(self.skip_reasons),
        }


class StepTransactionManager:
    """Owns the retire queue + diagnostics and the per-step transaction lifecycle.

    Constructed by the engine only when ``enable_async_scheduling`` is on. In the
    commit that introduces it, the manager wraps the existing serial step as an
    always-adopted transaction via :meth:`begin_serial_txn`; the speculative
    prestage / launch / adopt path is added in later commits.
    """

    def __init__(self, context: Any) -> None:
        self.context = context
        self.retire_queue = RetireQueue()
        self.diagnostics = StepTxnDiagnostics()
        # The currently-running (adopted) transaction and the in-flight child (later).
        self.current_txn: Optional[StepTxn] = None
        self.child_txn: Optional[StepTxn] = None

    # --- Retire ---------------------------------------------------------------

    def retire(self, current_step: int) -> int:
        """Drain the retire queue for ``current_step``; returns the count released."""
        released = self.retire_queue.drain(current_step)
        self.diagnostics.retired += released
        return released

    def enqueue_retire(
        self,
        *,
        enqueue_step: int,
        release: Callable[[], None],
        event: Any = None,
        tag: str = "",
    ) -> None:
        """Defer ``release`` behind the two-step + fence gate (see :class:`RetireQueue`)."""
        self.retire_queue.enqueue(
            enqueue_step=enqueue_step, release=release, event=event, tag=tag
        )

    # --- Lifecycle ------------------------------------------------------------

    def begin_serial_txn(
        self,
        *,
        step_id: int,
        active_request_count: int,
        request_ids: Any = None,
        bucket: Any = None,
    ) -> StepTxn:
        """Begin an always-adopted serial transaction (no speculative launch).

        Used to route the legacy serial step through the transaction lifecycle
        without changing what is computed.
        """
        txn = StepTxn(
            step_id=step_id,
            phase=TxnPhase.ADOPTED,
            request_ids=request_ids,
            active_request_count=active_request_count,
            bucket=bucket,
            speculative=False,
        )
        self.diagnostics.prepared += 1
        self.diagnostics.adopted += 1
        self.diagnostics.serial_steps += 1
        self.current_txn = txn
        return txn

    def commit(self, txn: StepTxn) -> None:
        """Mark ``txn`` committed (``update_requests`` has been applied for its step)."""
        txn.phase = TxnPhase.COMMITTED

    def publish(self, txn: StepTxn) -> None:
        """Publish ``txn``'s artifacts by request id. Filled in by the publication commit."""
        # Publication (logprobs / top-n / routing / coordinator replies) is keyed by
        # request id and happens after commit; wired up in a later commit.

    def note_sync_step(self, reason: str) -> None:
        """Record that this step ran synchronously (a layout-changing step)."""
        self.diagnostics.sync_steps += 1
        self.diagnostics.skip_reasons[reason] += 1

    def barrier(self, reason: str) -> None:
        """Record a barrier that prevented a speculative child launch for one step."""
        self.diagnostics.barrier_skips += 1
        self.diagnostics.skip_reasons[reason] += 1

    def note_skip(self, reason: str) -> None:
        """Record that a speculative launch was skipped for ``reason`` (diagnostics only)."""
        self.diagnostics.skip_reasons[reason] += 1

    def note_guard_failure(self) -> None:
        """Record an adopt-guard failure (expected to stay 0; recovery is a barrier)."""
        self.diagnostics.guard_failures += 1
