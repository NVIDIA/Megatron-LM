# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DP coordinator step-boundary admission gate (v3 plan §commit 29).

Makes cross-rank optimistic-ledger divergence impossible by *prevention*:
request admissions are received by all ranks at the same step boundary
via the coordinator's broadcast. A rank that hasn't received the
broadcast for ``step_id`` blocks at the boundary until it does. There is
no rank-local admission path that bypasses the coordinator.

This module is consumed by the engine's distributed run loop. It owns
nothing about the coordinator wire format; the coordinator client
publishes admission sets per step ID via ``publish``, the engine waits
on them via ``wait_for_admission``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class _AdmissionSet:
    request_ids: Tuple[int, ...]
    event: asyncio.Event = field(default_factory=asyncio.Event)


class StepBoundaryAdmissionGate:
    """Owns per-step admission sets and the await-until-published protocol.

    The engine's run loop calls ``wait_for_admission(step_id)`` at the
    top of each step; the call returns the admission set once the
    coordinator broadcast for that step has arrived. The coordinator
    client (or a test fixture) calls ``publish(step_id, request_ids)``
    when the broadcast is received locally.

    With the coordinator off the gate is trivially open — engines that
    don't use the coordinator can register an empty admission set per
    step or skip the wait entirely.
    """

    def __init__(self) -> None:
        self._admission_sets: Dict[int, _AdmissionSet] = {}

    def publish(self, step_id: int, request_ids: Iterable[int]) -> None:
        """Record the coordinator-broadcast admission set for ``step_id``
        and wake any waiters."""
        slot = self._admission_sets.setdefault(
            step_id, _AdmissionSet(request_ids=tuple(request_ids))
        )
        # If the slot already existed, refresh it (idempotent: same key
        # publishes the same broadcast).
        slot.request_ids = tuple(request_ids)
        slot.event.set()

    async def wait_for_admission(
        self, step_id: int, timeout: Optional[float] = None
    ) -> Tuple[int, ...]:
        """Block until the coordinator publishes the admission set for
        ``step_id``. Returns the admitted request IDs. Raises
        ``asyncio.TimeoutError`` if ``timeout`` elapses first.
        """
        slot = self._admission_sets.setdefault(step_id, _AdmissionSet(request_ids=()))
        if not slot.event.is_set():
            if timeout is None:
                await slot.event.wait()
            else:
                await asyncio.wait_for(slot.event.wait(), timeout=timeout)
        return slot.request_ids

    def is_admitted(self, step_id: int) -> bool:
        """True iff the admission set for ``step_id`` has been published."""
        slot = self._admission_sets.get(step_id)
        return slot is not None and slot.event.is_set()

    def known_step_ids(self) -> List[int]:
        return sorted(self._admission_sets.keys())

    def reset(self) -> None:
        """Drop all admission state — used by suspend/resume."""
        self._admission_sets.clear()
