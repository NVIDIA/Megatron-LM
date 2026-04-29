# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Committed and optimistic request ledgers for dynamic inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class RequestLedgerRequestState:
    """Per-request view inside a dynamic request ledger."""

    request_id: int
    slot: int
    is_paused: bool
    query_length: int
    in_prefill: bool


@dataclass(frozen=True)
class RequestLedgerState:
    """Snapshot of scheduler-visible dynamic request state."""

    total_request_count: int = 0
    paused_request_count: int = 0
    active_token_count: int = 0
    chunked_prefill_request_id: int = -1
    request_ids: Tuple[int, ...] = field(default_factory=tuple)
    request_query_lengths: Tuple[int, ...] = field(default_factory=tuple)
    request_in_prefill_status: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_context(cls, context) -> "RequestLedgerState":
        """Capture the current request state from a DynamicInferenceContext."""
        total_request_count = int(context.total_request_count)
        active_slice = slice(0, total_request_count)
        return cls(
            total_request_count=total_request_count,
            paused_request_count=int(context.paused_request_count),
            active_token_count=int(context.active_token_count),
            chunked_prefill_request_id=int(context.chunked_prefill_request_id),
            request_ids=tuple(int(v) for v in context.request_ids[active_slice].tolist()),
            request_query_lengths=tuple(
                int(v) for v in context.request_query_lengths[active_slice].tolist()
            ),
            request_in_prefill_status=tuple(
                int(v)
                for v in context.request_in_prefill_status_tensor[active_slice].tolist()
            ),
        )

    def get_request_state(self, request_id: int) -> Optional[RequestLedgerRequestState]:
        """Return per-request state if the request is present in this ledger."""
        for slot, current_request_id in enumerate(self.request_ids):
            if current_request_id == request_id:
                return RequestLedgerRequestState(
                    request_id=request_id,
                    slot=slot,
                    is_paused=slot < self.paused_request_count,
                    query_length=self.request_query_lengths[slot],
                    in_prefill=bool(self.request_in_prefill_status[slot]),
                )
        return None


class DynamicRequestLedgers:
    """Pair of committed and optimistic request ledgers."""

    def __init__(self):
        empty_state = RequestLedgerState()
        self.committed = empty_state
        self.optimistic = empty_state

    def sync_from_context_for_queue_depth_one(self, context) -> None:
        """Keep both ledgers identical to the live context in serial mode."""
        snapshot = RequestLedgerState.from_context(context)
        self.committed = snapshot
        self.optimistic = snapshot

    def assert_committed_matches_optimistic(self) -> None:
        """Assert the queue-depth-one invariant."""
        if self.committed != self.optimistic:
            raise AssertionError(
                "Committed and optimistic request ledgers diverged: "
                f"committed={self.committed}, optimistic={self.optimistic}"
            )

    def get_committed_request_state(
        self, request_id: Optional[int] = None
    ) -> RequestLedgerState | Optional[RequestLedgerRequestState]:
        """Return the committed ledger or one committed request state."""
        if request_id is None:
            return self.committed
        return self.committed.get_request_state(request_id)

    def get_optimistic_request_state(
        self, request_id: Optional[int] = None
    ) -> RequestLedgerState | Optional[RequestLedgerRequestState]:
        """Return the optimistic ledger or one optimistic request state."""
        if request_id is None:
            return self.optimistic
        return self.optimistic.get_request_state(request_id)
