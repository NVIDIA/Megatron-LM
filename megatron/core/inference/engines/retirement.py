# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordered retirement of in-flight inference steps.

See v3 plan §2.8 in
``lawrence/reports/20260429-context-cpu-async-schedule-claude-v3.md``. The
service is the only path from ``AsyncStepOutput`` to user-visible state. It
preserves per-request ordering, owns drain semantics on shutdown / suspend /
cancel, and gates request-id reuse.

At commit 5 the queue depth is 1 (serial-equivalent): the engine retires
each step synchronously inside the same coroutine that launched it.
Subsequent commits raise the queue depth and wire the deque-based ordering
contract established here.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Deque, List, Optional, Set, Tuple

if TYPE_CHECKING:
    # AsyncStepOutput lives under ``engines.async_pipeline_types``; the dataclass
    # is consumed by reference here. Quoted to avoid eager import at module
    # load.
    from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput


FinalizeCallback = Callable[..., Awaitable[Any]]


@dataclass
class _InflightEntry:
    """One in-flight step awaiting retirement."""

    step_id: int
    output: Optional["AsyncStepOutput"]
    payload: Tuple[Any, ...]
    request_ids: Tuple[int, ...] = ()


class StepRetirementService:
    """Owns ordered retirement of in-flight steps.

    The service holds a deque of ``_InflightEntry`` in step-id order and
    delegates the actual finalization (``post_process_requests``,
    detokenization, coordinator emission, metrics, console logging) to the
    engine through ``finalize_callback``.

    The contract is intentionally minimal at commit 5:

    * ``enqueue`` registers an in-flight entry without blocking.
    * ``retire`` blocks on the entry's ``d2h_done_event`` (if any), runs the
      finalize callback, returns its result.
    * ``retire_all_ready`` drains the prefix of the queue whose events have
      already fired.
    * ``drain`` synchronously retires every remaining entry — used at
      shutdown, suspend, cancellation, and request-id reuse.

    Request-id reuse is gated by ``await_request_id_release`` which waits
    until no in-flight entry references the id.
    """

    def __init__(self, finalize_callback: FinalizeCallback) -> None:
        self._finalize = finalize_callback
        self._inflight: Deque[_InflightEntry] = collections.deque()
        self._closed = False

    # ------------------------------------------------------------------
    # Queue manipulation
    # ------------------------------------------------------------------

    def enqueue(
        self,
        step_id: int,
        output: Optional["AsyncStepOutput"],
        payload: Tuple[Any, ...],
        request_ids: Tuple[int, ...] = (),
    ) -> None:
        """Append an in-flight entry. Cheap; the actual finalization runs
        later in ``retire`` / ``retire_all_ready`` / ``drain``."""
        if self._closed:
            raise RuntimeError("StepRetirementService is closed; cannot enqueue.")
        self._inflight.append(
            _InflightEntry(
                step_id=step_id, output=output, payload=payload, request_ids=request_ids
            )
        )

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def __len__(self) -> int:
        return len(self._inflight)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Mark the service closed. Subsequent ``enqueue`` calls raise."""
        self._closed = True

    # ------------------------------------------------------------------
    # Retirement
    # ------------------------------------------------------------------

    async def retire(
        self,
        output: Optional["AsyncStepOutput"],
        payload: Tuple[Any, ...],
    ) -> Any:
        """Block on ``output.d2h_done_event`` then run the finalize callback.

        ``output`` may be ``None`` for engine paths that have no CPU-visible
        D2H bundle yet (today's serial path resolves the bundle inside the
        controller, so the engine sees ``None`` and the finalize callback
        receives the already-CPU step result).
        """
        if output is not None and output.d2h_done_event is not None:
            output.d2h_done_event.synchronize()
        return await self._finalize(*payload)

    async def retire_all_ready(self) -> List[Any]:
        """Non-blocking poll: retire each prefix entry whose event has fired."""
        results: List[Any] = []
        while self._inflight:
            entry = self._inflight[0]
            if not self._is_entry_ready(entry):
                break
            self._inflight.popleft()
            results.append(await self.retire(entry.output, entry.payload))
        return results

    async def drain(self) -> List[Any]:
        """Synchronously retire every remaining entry in step-id order."""
        results: List[Any] = []
        while self._inflight:
            entry = self._inflight.popleft()
            results.append(await self.retire(entry.output, entry.payload))
        return results

    async def await_request_id_release(self, request_id: int) -> List[Any]:
        """Drain prefix entries until no in-flight entry references
        ``request_id``. Caller must call this before reusing the id.
        """
        results: List[Any] = []
        while self._has_inflight_request(request_id):
            entry = self._inflight.popleft()
            results.append(await self.retire(entry.output, entry.payload))
        return results

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def _is_entry_ready(self, entry: _InflightEntry) -> bool:
        if entry.output is None or entry.output.d2h_done_event is None:
            return True
        # ``query()`` returns True when every command in the recording
        # stream up to the event has completed.
        return bool(entry.output.d2h_done_event.query())

    def _has_inflight_request(self, request_id: int) -> bool:
        return any(request_id in entry.request_ids for entry in self._inflight)

    def referenced_request_ids(self) -> Set[int]:
        """All request ids currently referenced by in-flight entries."""
        ids: Set[int] = set()
        for entry in self._inflight:
            ids.update(entry.request_ids)
        return ids
