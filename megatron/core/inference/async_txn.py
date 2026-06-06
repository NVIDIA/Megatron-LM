# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Async decode transaction primitives.

This module intentionally contains only small, explicit ownership objects.  The
dynamic engine remains the source of truth for request scheduling and persistent
CPU state; these helpers describe which GPU-side resources are tied to a forward
and when they may be reused.
"""

from __future__ import annotations

import zlib
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Iterable, Optional, Sequence

import torch


class AsyncTxnSkipReason(str, Enum):
    """Concrete reasons a decode child was not eligible for async launch."""

    ASYNC_DISABLED = "async_disabled"
    NOT_DECODE_ONLY = "not_decode_only"
    MTP_ACTIVE = "mtp_active"
    CHUNKED_PREFILL = "chunked_prefill"
    PENDING_ADMISSION = "pending_admission"
    PAUSED_REQUESTS = "paused_requests"
    KV_RESERVATION_UNAVAILABLE = "kv_reservation_unavailable"
    GRAPH_RECAPTURE_BARRIER = "graph_recapture_barrier"
    LOG_INTERVAL_BARRIER = "log_interval_barrier"
    RESUME_BARRIER = "resume_barrier"
    EVICT_BARRIER = "evict_barrier"
    FORCE_PAUSE_BARRIER = "force_pause_barrier"
    CHILD_SLOT_BUSY = "child_slot_busy"
    HYBRID_PRESTAGE_DEFERRED = "hybrid_prestage_deferred"
    BOUNDARY_PRESTAGE_DEFERRED = "boundary_prestage_deferred"
    TERMINAL_CHECK_REQUIRED = "terminal_check_required"
    LOGPROBS_DEFERRED = "logprobs_deferred"
    CUDA_GRAPH_DEFERRED = "cuda_graph_deferred"
    MOE_EP_DEFERRED = "moe_ep_deferred"
    SKIP_BOOKKEEPING = "skip_bookkeeping"
    ACTIVE_COUNT_CHANGED = "active_count_changed"
    SERIAL_WRAPPED = "serial_wrapped"


@dataclass
class AsyncTxnDiagnostics:
    """Low-overhead counters for async transaction decisions."""

    enabled: bool = False
    prepared: int = 0
    launched: int = 0
    adopted: int = 0
    sync_steps: int = 0
    barrier_skips: int = 0
    retired: int = 0
    guard_failures: int = 0
    chain_attempts: int = 0
    chain_launches: int = 0
    presampled_commits: int = 0
    prepare_under_forward: int = 0
    h2d_ready_before_sampling: int = 0
    sample_to_launch_latency_us: float = 0.0
    commit_duration_us: float = 0.0
    child_prestage_duration_us: float = 0.0
    ep_handoff_duration_us: float = 0.0
    child_graph_shape_duration_us: float = 0.0
    child_forward_gpu_us: float = 0.0
    controller_wall_us: float = 0.0
    sampling_block_us: float = 0.0
    engine_forward_wall_us: float = 0.0
    engine_bookkeep_wall_us: float = 0.0
    engine_postprocess_us: float = 0.0
    retire_queue_depth: int = 0
    eligibility_skip_reasons: Counter = field(default_factory=Counter)

    def record_prepared(self, *, under_forward: bool = False) -> None:
        if not self.enabled:
            return
        self.prepared += 1
        if under_forward:
            self.prepare_under_forward += 1

    def record_launched(
        self,
        *,
        h2d_ready_before_sampling: bool = False,
        sample_to_launch_latency_us: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        self.launched += 1
        if h2d_ready_before_sampling:
            self.h2d_ready_before_sampling += 1
        if sample_to_launch_latency_us is not None:
            self.sample_to_launch_latency_us = max(0.0, float(sample_to_launch_latency_us))

    def record_adopted(self) -> None:
        if self.enabled:
            self.adopted += 1

    def record_commit_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.commit_duration_us = max(0.0, float(duration_us))

    def record_child_prestage_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.child_prestage_duration_us = max(0.0, float(duration_us))

    def record_ep_handoff_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.ep_handoff_duration_us = max(0.0, float(duration_us))

    def record_child_graph_shape_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.child_graph_shape_duration_us = max(0.0, float(duration_us))

    def record_child_forward_gpu_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.child_forward_gpu_us = max(0.0, float(duration_us))

    def record_controller_wall_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.controller_wall_us = max(0.0, float(duration_us))

    def record_sampling_block_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.sampling_block_us = max(0.0, float(duration_us))

    def record_engine_forward_wall_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.engine_forward_wall_us = max(0.0, float(duration_us))

    def record_engine_bookkeep_wall_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.engine_bookkeep_wall_us = max(0.0, float(duration_us))

    def record_engine_postprocess_duration(self, duration_us: float) -> None:
        if self.enabled:
            self.engine_postprocess_us = max(0.0, float(duration_us))

    def record_sync_step(self, reason: AsyncTxnSkipReason | str) -> None:
        if not self.enabled:
            return
        self.sync_steps += 1
        self.record_skip(reason)

    def record_barrier_skip(self, reason: AsyncTxnSkipReason | str) -> None:
        if not self.enabled:
            return
        self.barrier_skips += 1
        self.record_skip(reason)

    def record_guard_failure(self) -> None:
        if self.enabled:
            self.guard_failures += 1

    def record_chain_attempt(self, *, launched: bool = False) -> None:
        if not self.enabled:
            return
        self.chain_attempts += 1
        if launched:
            self.chain_launches += 1

    def record_presampled_commit(self) -> None:
        if self.enabled:
            self.presampled_commits += 1

    def record_retired(self, count: int = 1) -> None:
        if self.enabled:
            self.retired += count

    def record_skip(self, reason: AsyncTxnSkipReason | str) -> None:
        if not self.enabled:
            return
        key = reason.value if isinstance(reason, AsyncTxnSkipReason) else str(reason)
        self.eligibility_skip_reasons[key] += 1

    def top_skip_reason(self) -> Optional[str]:
        if not self.eligibility_skip_reasons:
            return None
        return self.eligibility_skip_reasons.most_common(1)[0][0]

    def snapshot(self) -> dict:
        """Return a plain dict suitable for logging or assertions."""

        return {
            "enabled": self.enabled,
            "prepared": self.prepared,
            "launched": self.launched,
            "adopted": self.adopted,
            "sync_steps": self.sync_steps,
            "barrier_skips": self.barrier_skips,
            "retired": self.retired,
            "guard_failures": self.guard_failures,
            "chain_attempts": self.chain_attempts,
            "chain_launches": self.chain_launches,
            "presampled_commits": self.presampled_commits,
            "prepare_under_forward": self.prepare_under_forward,
            "h2d_ready_before_sampling": self.h2d_ready_before_sampling,
            "sample_to_launch_latency_us": self.sample_to_launch_latency_us,
            "commit_duration_us": self.commit_duration_us,
            "child_prestage_duration_us": self.child_prestage_duration_us,
            "ep_handoff_duration_us": self.ep_handoff_duration_us,
            "child_graph_shape_duration_us": self.child_graph_shape_duration_us,
            "child_forward_gpu_us": self.child_forward_gpu_us,
            "controller_wall_us": self.controller_wall_us,
            "sampling_block_us": self.sampling_block_us,
            "engine_forward_wall_us": self.engine_forward_wall_us,
            "engine_bookkeep_wall_us": self.engine_bookkeep_wall_us,
            "engine_postprocess_us": self.engine_postprocess_us,
            "retire_queue_depth": self.retire_queue_depth,
            "eligibility_skip_reasons": dict(self.eligibility_skip_reasons),
            "top_skip_reason": self.top_skip_reason(),
        }


def _event_done(event) -> bool:
    """Return whether a CUDA-like event is complete.

    Tests use tiny fake events with a ``query()`` method.  ``None`` means there
    is no outstanding GPU dependency.
    """

    if event is None:
        return True
    if hasattr(event, "query"):
        return bool(event.query())
    return bool(event)


@dataclass
class KVBlockLease:
    """A KV block reserved for a request before an async child launches."""

    request_id: int
    block_column: int
    block_id: int

    def key(self) -> tuple[int, int]:
        return (int(self.request_id), int(self.block_column))


@dataclass
class StepTxn:
    """Ownership record for one decode forward step."""

    step_id: int
    request_ids: Sequence[int]
    slot_id: int = 0
    decode_only: bool = True
    cuda_graph_key: Optional[Any] = None
    h2d_done_event: object = None
    forward_done_event: object = None
    forward_timing_start_event: object = None
    forward_timing_done_event: object = None
    sample_done_event: object = None
    cpu_bookkeeping_buf: Optional[torch.Tensor] = None
    kv_block_ids: tuple[int, ...] = ()
    kv_block_leases: tuple[KVBlockLease, ...] = ()
    mamba_slot_ids: tuple[int, ...] = ()
    terminal_request_ids: tuple[int, ...] = ()
    committed_request_ids: tuple[int, ...] = ()
    launched: bool = False
    adopted: bool = False
    retired: bool = False

    def __post_init__(self) -> None:
        self.request_ids = tuple(int(r) for r in self.request_ids)
        self.kv_block_leases = tuple(
            lease
            if isinstance(lease, KVBlockLease)
            else KVBlockLease(
                request_id=int(lease[0]), block_column=int(lease[1]), block_id=int(lease[2])
            )
            for lease in self.kv_block_leases
        )
        self.terminal_request_ids = tuple(int(r) for r in self.terminal_request_ids)
        self.committed_request_ids = tuple(int(r) for r in self.committed_request_ids)
        self.mamba_slot_ids = tuple(int(slot_id) for slot_id in self.mamba_slot_ids)

    @property
    def request_id_set(self) -> set[int]:
        return set(self.request_ids)

    @property
    def terminal_request_id_set(self) -> set[int]:
        return set(self.terminal_request_ids)

    @property
    def kv_lease_map(self) -> dict[tuple[int, int], int]:
        return {lease.key(): int(lease.block_id) for lease in self.kv_block_leases}

    def get_kv_lease(self, request_id: int, block_column: int) -> Optional[int]:
        """Return a reserved block for ``(request_id, block_column)``, if any."""

        return self.kv_lease_map.get((int(request_id), int(block_column)))

    def mark_committed(
        self,
        committed_request_ids: Iterable[int],
        *,
        terminal_request_ids: Iterable[int] = (),
    ) -> None:
        """Record CPU commit results keyed by request id."""

        self.committed_request_ids = tuple(int(r) for r in committed_request_ids)
        self.terminal_request_ids = tuple(int(r) for r in terminal_request_ids)

    def unused_kv_leases(self) -> tuple[KVBlockLease, ...]:
        """Return leased blocks that did not become part of committed survivors."""

        if not self.kv_block_leases:
            return ()
        committed = set(self.committed_request_ids)
        return tuple(lease for lease in self.kv_block_leases if lease.request_id not in committed)

    def committed_row_indices(self, committed_request_ids: Iterable[int]) -> tuple[int, ...]:
        """Map committed request ids back to launched row indices."""

        row_by_request_id = {request_id: row for row, request_id in enumerate(self.request_ids)}
        return tuple(row_by_request_id[int(request_id)] for request_id in committed_request_ids)

    def h2d_ready(self) -> bool:
        return self.cpu_bookkeeping_buf is None and _event_done(self.h2d_done_event)

    def forward_done(self) -> bool:
        return _event_done(self.forward_done_event)

    def guard_adoption(
        self,
        current_request_ids: Iterable[int],
        *,
        terminal_request_ids: Iterable[int] = (),
        decode_only: bool = True,
        cuda_graph_key: Optional[Any] = None,
    ) -> bool:
        """Check the no-reject-after-launch adoption invariant.

        Survivors must be a subset of launched rows.  Any launched row that is
        missing from the current committed set must be terminal; no non-terminal
        survivor may disappear after child launch.
        """

        if not self.launched:
            return False
        if self.decode_only and not decode_only:
            return False
        if self.cuda_graph_key is not None and cuda_graph_key is not None:
            if self.cuda_graph_key != cuda_graph_key:
                return False

        current = set(int(r) for r in current_request_ids)
        terminal = set(int(r) for r in terminal_request_ids)
        launched = self.request_id_set
        if not current.issubset(launched):
            return False
        missing = launched - current
        return missing.issubset(terminal)


@dataclass
class _RetireItem:
    event: object
    callback: Callable[[], None]
    label: str = ""


class TxnRetireQueue:
    """FIFO retire queue for resources owned by in-flight forwards."""

    def __init__(self, diagnostics: Optional[AsyncTxnDiagnostics] = None) -> None:
        self._items: Deque[_RetireItem] = deque()
        self._diagnostics = diagnostics

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, event, callback: Callable[[], None], *, label: str = "") -> None:
        self._items.append(_RetireItem(event=event, callback=callback, label=label))
        self._update_depth()

    def drain_ready(self) -> int:
        """Run callbacks for completed items at the queue head."""

        retired = 0
        while self._items and _event_done(self._items[0].event):
            item = self._items.popleft()
            item.callback()
            retired += 1
        if retired and self._diagnostics is not None:
            self._diagnostics.record_retired(retired)
        self._update_depth()
        return retired

    def drain_all_ready(self) -> int:
        """Drain all ready items, including ready entries behind pending ones."""

        kept: Deque[_RetireItem] = deque()
        retired = 0
        while self._items:
            item = self._items.popleft()
            if _event_done(item.event):
                item.callback()
                retired += 1
            else:
                kept.append(item)
        self._items = kept
        if retired and self._diagnostics is not None:
            self._diagnostics.record_retired(retired)
        self._update_depth()
        return retired

    def _update_depth(self) -> None:
        if self._diagnostics is not None and self._diagnostics.enabled:
            self._diagnostics.retire_queue_depth = len(self._items)


@dataclass
class AsyncDecodeSlot:
    """One GPU metadata slot in the decode ring."""

    slot_id: int
    gpu_view: object
    cpu_bookkeeping_buf: object = None
    h2d_done_event: object = None
    forward_done_event: object = None

    def can_reuse(self) -> bool:
        return _event_done(self.h2d_done_event) and _event_done(self.forward_done_event)

    def copy_bookkeeping_from_cpu(
        self,
        cpu_bookkeeping_buf: torch.Tensor,
        *,
        non_blocking: bool = True,
        stream: object = None,
    ) -> object:
        """Copy CPU bookkeeping into this slot and record the H2D dependency."""

        if stream is not None and getattr(self.gpu_view._buf, "is_cuda", False):
            with torch.cuda.stream(stream):
                self.gpu_view._buf.copy_(cpu_bookkeeping_buf, non_blocking=non_blocking)
            self.h2d_done_event = self._record_event_if_cuda(stream=stream)
        else:
            self.gpu_view._buf.copy_(cpu_bookkeeping_buf, non_blocking=non_blocking)
            self.h2d_done_event = self._record_event_if_cuda()
        return self.h2d_done_event

    def record_forward_done(self) -> object:
        """Record the forward completion dependency for this slot."""

        self.forward_done_event = self._record_event_if_cuda()
        return self.forward_done_event

    def _record_event_if_cuda(self, stream: object = None) -> object:
        buf = getattr(self.gpu_view, "_buf", None)
        if buf is None or not getattr(buf, "is_cuda", False):
            return None
        event = torch.cuda.Event()
        event.record(stream or torch.cuda.current_stream(buf.device))
        return event


class AsyncDecodeSlotRing:
    """Depth-2 decode metadata slot owner."""

    def __init__(self, slots: Sequence[AsyncDecodeSlot]):
        if len(slots) != 2:
            raise ValueError("async decode transaction ring requires exactly two slots")
        self._slots = tuple(slots)
        self.current_index = 0

    @property
    def slots(self) -> tuple[AsyncDecodeSlot, AsyncDecodeSlot]:
        return self._slots

    @property
    def current(self) -> AsyncDecodeSlot:
        return self._slots[self.current_index]

    @property
    def child(self) -> AsyncDecodeSlot:
        return self._slots[1 - self.current_index]

    def promote_child(self) -> AsyncDecodeSlot:
        if not self.child.can_reuse():
            raise RuntimeError("cannot promote child slot before its prior work has retired")
        self.current_index = 1 - self.current_index
        return self.current

    def adopt_child(self) -> AsyncDecodeSlot:
        """Mark the prepared child slot as the current in-flight forward slot."""

        self.current_index = 1 - self.current_index
        return self.current


@dataclass(frozen=True)
class AsyncLaunchEligibility:
    """Result of the child-launch gate."""

    eligible: bool
    reason: Optional[AsyncTxnSkipReason] = None
    boundary_request_ids: tuple[int, ...] = ()
    required_boundary_blocks: int = 0


@dataclass(frozen=True)
class EPDecodeBroadcastPlan:
    """Rank-uniform source and shape for EP decode post-forward collectives."""

    active_request_count: int
    src_group_rank: int
    has_real_work: bool


def _group_size(group) -> int:
    if group is None:
        return 1
    if hasattr(group, "size"):
        return int(group.size())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size(group))
    return 1


def _group_rank(group) -> int:
    if group is None:
        return 0
    if hasattr(group, "rank"):
        return int(group.rank())
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank(group))
    return 0


def _global_src_rank(group, src_group_rank: int) -> int:
    if (
        group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and hasattr(torch.distributed, "get_process_group_ranks")
    ):
        return int(torch.distributed.get_process_group_ranks(group)[src_group_rank])
    return int(src_group_rank)


def _broadcast_tensor(tensor: torch.Tensor, group, src_group_rank: int, broadcast_fn=None) -> None:
    if _group_size(group) <= 1:
        return
    src_rank = _global_src_rank(group, src_group_rank)
    if broadcast_fn is None:
        broadcast_fn = torch.distributed.broadcast
    broadcast_fn(tensor, src=src_rank, group=group)


def resolve_ep_decode_broadcast_plan(
    active_request_count: int,
    group,
    *,
    has_real_work: bool,
    sync_all_reduce_max_fn=None,
) -> EPDecodeBroadcastPlan:
    """Resolve the one real EP source rank that owns decode sampling results.

    Coordinator EP mode has one rank with persistent request state and peer ranks
    running dummy model work to satisfy MoE collectives. Post-forward collectives
    must still be entered by every EP rank with an identical source rank and
    tensor shape. This CPU-side plan keeps that contract explicit before the
    NCCL broadcast is issued.
    """

    local_count = int(active_request_count) if has_real_work else 0
    group_size = _group_size(group)
    if group_size <= 1:
        return EPDecodeBroadcastPlan(
            active_request_count=local_count,
            src_group_rank=0,
            has_real_work=local_count > 0,
        )

    if sync_all_reduce_max_fn is None:
        return EPDecodeBroadcastPlan(
            active_request_count=int(active_request_count),
            src_group_rank=0,
            has_real_work=local_count > 0,
        )

    group_rank = _group_rank(group)
    local_src_max = group_rank if local_count > 0 else -1
    local_neg_src_min = -group_rank if local_count > 0 else -(group_size + 1)
    global_count, max_src, neg_min_src = sync_all_reduce_max_fn(
        local_count, local_src_max, local_neg_src_min
    )
    global_count = int(global_count)
    max_src = int(max_src)
    min_src = -int(neg_min_src)

    if global_count <= 0:
        return EPDecodeBroadcastPlan(
            active_request_count=0,
            src_group_rank=0,
            has_real_work=False,
        )
    if max_src != min_src:
        raise RuntimeError(
            "EP decode broadcast requires exactly one real source rank per EP group; "
            f"got min/max real source ranks {min_src}/{max_src}"
        )

    return EPDecodeBroadcastPlan(
        active_request_count=global_count,
        src_group_rank=max_src,
        has_real_work=True,
    )


def broadcast_ep_sampled_tokens(
    sampled_tokens: torch.Tensor,
    active_request_count: int,
    group,
    *,
    src_group_rank: int = 0,
    broadcast_fn=None,
) -> torch.Tensor:
    """Broadcast canonical sampled tokens across an EP group.

    Only row-ordered token values are exchanged. Request ids, row maps, KV block
    ids, and Mamba slot ids remain local.
    """

    if sampled_tokens is None or int(active_request_count) <= 0:
        return sampled_tokens
    token_slice = sampled_tokens[: int(active_request_count)]
    _broadcast_tensor(token_slice, group, src_group_rank, broadcast_fn=broadcast_fn)
    return sampled_tokens


def broadcast_ep_stop_word_finished_ids(
    active_request_ids: Sequence[int] | torch.Tensor,
    finished_request_ids: Iterable[int],
    group,
    *,
    src_group_rank: int = 0,
    device: Optional[torch.device | int | str] = None,
    broadcast_fn=None,
) -> set[int]:
    """Broadcast stop-word finishes as an active-row mask.

    The mask lets every EP rank derive its local finished-id set without
    exchanging request ids.
    """

    if isinstance(active_request_ids, torch.Tensor):
        request_id_list = [int(request_id) for request_id in active_request_ids.tolist()]
    else:
        request_id_list = [int(request_id) for request_id in active_request_ids]
    active_count = len(request_id_list)
    if active_count == 0:
        return set()

    if device is None:
        device = (
            active_request_ids.device
            if isinstance(active_request_ids, torch.Tensor)
            else torch.device("cpu")
        )
    finished_set = {int(request_id) for request_id in finished_request_ids}
    mask = torch.tensor(
        [1 if request_id in finished_set else 0 for request_id in request_id_list],
        dtype=torch.int32,
        device=device,
    )
    _broadcast_tensor(mask, group, src_group_rank, broadcast_fn=broadcast_fn)
    mask_cpu = mask.cpu()
    return {
        request_id
        for request_id, finished in zip(request_id_list, mask_cpu.tolist())
        if int(finished) != 0
    }


def broadcast_ep_accepted_counts(
    accepted_counts: torch.Tensor,
    active_request_count: int,
    group,
    *,
    src_group_rank: int = 0,
    broadcast_fn=None,
) -> torch.Tensor:
    """Broadcast canonical MTP accepted-count rows across an EP group."""

    if accepted_counts is None or int(active_request_count) <= 0:
        return accepted_counts
    counts_slice = accepted_counts[: int(active_request_count)]
    _broadcast_tensor(counts_slice, group, src_group_rank, broadcast_fn=broadcast_fn)
    return accepted_counts


def _phase_code(phase: str) -> int:
    return zlib.adler32(phase.encode("utf-8")) & 0x7FFFFFFF


def assert_ep_phase_tag(
    phase: str,
    step_id: int,
    active_request_count: int,
    group,
    *,
    device: Optional[torch.device | int | str] = None,
    all_gather_fn=None,
) -> torch.Tensor:
    """Verify that EP ranks enter the same phase and active shape.

    This is a debug/proof hook. It exchanges only ``(phase, step_id,
    active_count)`` tags and raises a clear error instead of letting skew hang in
    a later collective.
    """

    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    local = torch.tensor(
        [_phase_code(phase), int(step_id), int(active_request_count)],
        dtype=torch.int64,
        device=device,
    )
    size = _group_size(group)
    if size <= 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(size)]
    if all_gather_fn is None:
        all_gather_fn = torch.distributed.all_gather
    all_gather_fn(gathered, local, group=group)
    expected = gathered[0]
    mismatched = [idx for idx, tag in enumerate(gathered) if not torch.equal(tag, expected)]
    if mismatched:
        tags = [tag.cpu().tolist() for tag in gathered]
        raise RuntimeError(
            "EP async transaction phase mismatch: "
            f"phase={phase!r}, local_tag={local.cpu().tolist()}, gathered_tags={tags}"
        )
    return local


def boundary_crossing_request_ids(context) -> tuple[int, ...]:
    """Return active request ids that require one more KV block next decode step."""

    active_count = context.total_request_count - context.paused_request_count
    if active_count <= 0:
        return ()
    start = context.paused_request_count
    end = context.total_request_count
    offsets = context.request_last_kv_block_offset[start:end]
    threshold = context.block_size_tokens - 1 - context.num_speculative_tokens
    mask = offsets >= threshold
    if not bool(mask.any()):
        return ()
    req_ids = context.request_ids[start:end][mask]
    return tuple(int(r) for r in req_ids.tolist())


def classify_decode_child_launch(
    context,
    *,
    async_enabled: bool,
    pending_admission: bool = False,
    graph_recapture_barrier: bool = False,
    log_interval_barrier: bool = False,
    resume_barrier: bool = False,
    evict_barrier: bool = False,
    force_pause_barrier: bool = False,
    mtp_active: Optional[bool] = None,
) -> AsyncLaunchEligibility:
    """Classify whether the next plain-decode child may be launched."""

    if not async_enabled:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.ASYNC_DISABLED)
    if mtp_active is None:
        mtp_active = getattr(context, "num_speculative_tokens", 0) > 0
    if mtp_active:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.MTP_ACTIVE)
    if not context.is_decode_only():
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.NOT_DECODE_ONLY)
    if getattr(context, "chunked_prefill_request_id", -1) != -1:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.CHUNKED_PREFILL)
    if pending_admission:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.PENDING_ADMISSION)
    if getattr(context, "paused_request_count", 0) > 0:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.PAUSED_REQUESTS)
    if graph_recapture_barrier:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.GRAPH_RECAPTURE_BARRIER)
    if log_interval_barrier:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.LOG_INTERVAL_BARRIER)
    if resume_barrier:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.RESUME_BARRIER)
    if evict_barrier:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.EVICT_BARRIER)
    if force_pause_barrier:
        return AsyncLaunchEligibility(False, AsyncTxnSkipReason.FORCE_PAUSE_BARRIER)

    boundary_ids = boundary_crossing_request_ids(context)
    required = len(boundary_ids)
    if required and not context.kv_block_allocator.is_memory_available(required):
        return AsyncLaunchEligibility(
            False,
            AsyncTxnSkipReason.KV_RESERVATION_UNAVAILABLE,
            boundary_request_ids=boundary_ids,
            required_boundary_blocks=required,
        )

    return AsyncLaunchEligibility(
        True, boundary_request_ids=boundary_ids, required_boundary_blocks=required
    )


class RequestRNGStore:
    """Request-id keyed RNG store for stochastic decode sampling."""

    def __init__(self, base_seed: int, *, device: Optional[torch.device | int | str] = None):
        self.base_seed = int(base_seed)
        self.device = device
        self._generators: dict[int, torch.Generator] = {}

    def get(self, request_id: int) -> torch.Generator:
        request_id = int(request_id)
        generator = self._generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self._seed_for(request_id))
            self._generators[request_id] = generator
        return generator

    def remove(self, request_id: int) -> None:
        self._generators.pop(int(request_id), None)

    def remove_many(self, request_ids: Iterable[int]) -> None:
        for request_id in request_ids:
            self.remove(int(request_id))

    def _seed_for(self, request_id: int) -> int:
        # SplitMix64-style integer mixing keeps nearby request ids from sharing
        # nearby RNG streams while staying deterministic across processes.
        x = (self.base_seed + 0x9E3779B97F4A7C15 + int(request_id)) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x = x ^ (x >> 31)
        return int(x % (2**63 - 1))
