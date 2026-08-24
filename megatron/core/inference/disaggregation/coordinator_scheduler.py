# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Coordinator scheduling for native prefill/decode disaggregation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

PREFILL = "prefill"
DECODE = "decode"


@dataclass(frozen=True)
class QueuedPrefillRequest:
    """Prefill request waiting for SSM state capacity."""

    request_id: int
    slot_cost: int


@dataclass(frozen=True)
class QueuedDecodeHandoff:
    """Decode handoff waiting for SSM state capacity."""

    request_id: int
    payload: bytes
    slot_cost: int


@dataclass
class _RoleState:
    role: str
    engines: List[Any] = field(default_factory=list)
    round_robin_offset: int = 0
    assignments: Dict[int, Any] = field(default_factory=dict)  # request_id -> selected engine
    usage: Dict[Any, int] = field(default_factory=dict)
    counts: Dict[Any, int] = field(default_factory=dict)
    reservations: Dict[int, Tuple[Any, int]] = field(
        default_factory=dict
    )  # request_id -> (engine, cost)
    queues: Dict[Any, Deque] = field(default_factory=dict)


class DisaggCoordinatorScheduler:
    """Select engines and reserve capacity for prefill and decode work."""

    def __init__(self) -> None:
        self._capacity: Dict[Any, int] = {}
        self._request_capacity: Dict[Any, int] = {}
        self._prefill_slot_cost: Dict[Any, int] = {}
        self._prefill = _RoleState(PREFILL)
        self._decode = _RoleState(DECODE)

    def _state(self, role: str) -> _RoleState:
        if role == PREFILL:
            return self._prefill
        if role == DECODE:
            return self._decode
        raise ValueError(f"unknown disaggregated role {role!r}")

    @staticmethod
    def _metadata_entries(instance_meta) -> list[dict]:
        if not isinstance(instance_meta, list):
            raise ValueError(
                "Engine transfer metadata must be a per-model-parallel-rank list, got "
                f"{type(instance_meta).__name__}"
            )
        if not all(isinstance(entry, dict) for entry in instance_meta):
            raise ValueError("Engine transfer metadata must contain one mapping per rank")
        return instance_meta

    @staticmethod
    def _request_capacity_from_entries(entries: list[dict]) -> int:
        """Return the conservative request capacity across model-parallel ranks."""

        capacities = [
            int(entry["request_capacity"])
            for entry in entries
            if entry.get("request_capacity") is not None
        ]
        if not capacities or len(capacities) != len(entries):
            raise ValueError(
                "Request capacity is missing from part of an engine's model-parallel metadata"
            )
        capacity = min(capacities)
        if capacity < 1:
            raise ValueError(f"Request capacity must be positive, got {capacity}")
        return capacity

    @staticmethod
    def _capacity_from_entries(entries: list[dict]) -> Optional[int]:
        """Return the conservative live-state capacity across model-parallel ranks."""

        capacities = [
            int(entry["ssm_slot_capacity"])
            for entry in entries
            if entry.get("ssm_slot_capacity") is not None
        ]
        if capacities and len(capacities) != len(entries):
            raise ValueError(
                "SSM slot capacity is missing from part of an engine's model-parallel metadata"
            )
        capacity = min(capacities) if capacities else None
        if capacity is not None and capacity < 1:
            raise ValueError(f"SSM slot capacity must be positive, got {capacity}")
        return capacity

    @staticmethod
    def _prefill_slot_cost_from_entries(entries: list[dict]) -> int:
        """Return the per-request handoff bound advertised by a prefill engine."""

        hybrid_entries = [entry for entry in entries if entry.get("ssm_slot_capacity") is not None]
        if not hybrid_entries:
            return 0
        costs = [
            int(entry["ssm_handoff_max_slots"])
            for entry in hybrid_entries
            if entry.get("ssm_handoff_max_slots") is not None
        ]
        if len(costs) != len(hybrid_entries):
            raise ValueError(
                "SSM handoff slot demand is missing from part of a prefill "
                "engine's model-parallel metadata"
            )
        cost = max(costs)
        if cost < 1:
            raise ValueError(f"SSM handoff slot demand must be positive, got {cost}")
        return cost

    def register_engine(self, identity, role: str, instance_meta) -> Optional[int]:
        """Register an engine's capacity and return the parsed limit."""

        state = self._state(role)
        if identity in state.counts:
            raise ValueError(f"engine {identity!r} is already registered")
        entries = self._metadata_entries(instance_meta)
        capacity = self._capacity_from_entries(entries)
        request_capacity = self._request_capacity_from_entries(entries)
        prefill_slot_cost = (
            self._prefill_slot_cost_from_entries(entries) if role == PREFILL else None
        )

        self._request_capacity[identity] = request_capacity
        if capacity is not None:
            self._capacity[identity] = capacity
        if prefill_slot_cost is not None:
            self._prefill_slot_cost[identity] = prefill_slot_cost
        state.engines.append(identity)
        state.usage[identity] = 0
        state.counts[identity] = 0
        return capacity

    def remove_engine(self, identity) -> None:
        """Forget an engine after its queued and in-flight requests are dropped."""

        self._capacity.pop(identity, None)
        self._request_capacity.pop(identity, None)
        self._prefill_slot_cost.pop(identity, None)
        for state in (self._prefill, self._decode):
            if identity in state.engines:
                state.engines.remove(identity)
            state.usage.pop(identity, None)
            state.counts.pop(identity, None)
            state.queues.pop(identity, None)
            for request_id, (reserved_identity, _) in list(state.reservations.items()):
                if reserved_identity == identity:
                    state.reservations.pop(request_id)

    def select_engine(
        self, role: str, request_id: int, score: Callable[[Any], tuple] | None = None
    ) -> Any:
        """Select an engine using load-aware routing and round-robin tie breaking."""

        state = self._state(role)
        if not state.engines:
            raise RuntimeError(f"no {role} engines registered")
        offset = state.round_robin_offset
        state.round_robin_offset += 1
        if score is None:
            identity = state.engines[offset % len(state.engines)]
        else:
            # Rotate the pool so min() breaks equal-score ties round-robin.
            candidates = (
                state.engines[(offset + index) % len(state.engines)]
                for index in range(len(state.engines))
            )
            identity = min(candidates, key=score)
        state.assignments[request_id] = identity
        return identity

    def assigned_engine(self, role: str, request_id: int) -> Any | None:
        """Return the engine assigned to a request, if any."""

        return self._state(role).assignments.get(request_id)

    def reserved_engine(self, role: str, request_id: int) -> Any | None:
        """Return the engine holding capacity for an active request, if any."""

        reservation = self._state(role).reservations.get(request_id)
        return reservation[0] if reservation is not None else None

    def forget_assignment(self, role: str, request_id: int) -> None:
        """Drop one role's assignment for a request."""

        self._state(role).assignments.pop(request_id, None)

    def forget_request(self, request_id: int) -> None:
        """Drop all engine assignments for a request."""

        self._prefill.assignments.pop(request_id, None)
        self._decode.assignments.pop(request_id, None)

    def requests_involving(self, identity) -> List[int]:
        """Return requests assigned to an engine on either hop."""

        request_ids = {
            request_id
            for state in (self._prefill, self._decode)
            for request_id, assigned_identity in state.assignments.items()
            if assigned_identity == identity
        }
        return list(request_ids)

    def capacity(self, identity) -> Optional[int]:
        """Return an engine's advertised live recurrent-state capacity, if any."""

        return self._capacity.get(identity)

    def decode_usage(self, identity) -> int:
        """Return currently reserved decode slots."""

        return self._decode.usage.get(identity, 0)

    def prefill_usage(self, identity) -> int:
        """Return currently reserved prefill slots."""

        return self._prefill.usage.get(identity, 0)

    @staticmethod
    def _load(state: _RoleState, identity) -> tuple[int, int, int]:
        return (
            len(state.queues.get(identity, ())),
            state.counts.get(identity, 0),
            state.usage.get(identity, 0),
        )

    def prefill_load(self, identity) -> tuple[int, int, int]:
        """Return queued, active, and weighted live-state prefill load."""

        return self._load(self._prefill, identity)

    def decode_load(self, identity) -> tuple[int, int, int]:
        """Return queued, active, and weighted live-state decode load."""

        return self._load(self._decode, identity)

    def available_fraction(self, identity, role: str) -> float:
        """Return the binding free-capacity fraction for routing."""

        state = self._state(role)

        request_capacity = self._request_capacity.get(identity)
        if request_capacity is None:
            return 0.0
        request_fraction = (
            max(
                0.0,
                request_capacity
                - state.counts.get(identity, 0)
                - len(state.queues.get(identity, ())),
            )
            / request_capacity
        )
        state_capacity = self._capacity.get(identity)
        if state_capacity is None:
            return request_fraction
        state_fraction = max(0.0, state_capacity - state.usage.get(identity, 0)) / state_capacity
        return min(request_fraction, state_fraction)

    def prefill_slot_cost(self, identity) -> int:
        """Return the engine-advertised live-state handoff demand per request."""

        return self._prefill_slot_cost.get(identity, 0)

    def _try_reserve(self, state: _RoleState, identity, request_id: int, slot_cost: int) -> bool:
        if slot_cost < 0:
            raise ValueError(f"SSM slot cost cannot be negative, got {slot_cost}")
        if request_id in state.reservations:
            raise RuntimeError(
                f"{state.role.capitalize()} request {request_id} already holds an SSM reservation"
            )
        count = state.counts.get(identity, 0)
        if count >= self._request_capacity.get(identity, 0):
            return False
        capacity = self._capacity.get(identity)
        usage = state.usage.get(identity, 0)
        if capacity is not None and usage + slot_cost > capacity:
            return False
        state.usage[identity] = usage + slot_cost
        state.counts[identity] = count + 1
        state.reservations[request_id] = (identity, slot_cost)
        return True

    def try_reserve_prefill(self, identity, request_id: int, slot_cost: int) -> bool:
        """Reserve one prefill request by count and weighted slot demand."""

        return self._try_reserve(self._prefill, identity, request_id, slot_cost)

    def enqueue_prefill(self, identity, request_id: int, slot_cost: int) -> None:
        """Append a prefill request to an engine's FIFO capacity queue."""

        self._prefill.queues.setdefault(identity, deque()).append(
            QueuedPrefillRequest(request_id=request_id, slot_cost=slot_cost)
        )

    def has_queued_prefill(self, identity) -> bool:
        """Return whether an engine already has older queued prefills."""

        return bool(self._prefill.queues.get(identity))

    def pop_next_prefill(self, identity) -> Optional[QueuedPrefillRequest]:
        """Reserve and return the next queued prefill, if it fits."""

        return self._pop_next(self._prefill, identity)

    @staticmethod
    def _release(state: _RoleState, request_id: int):
        reservation = state.reservations.get(request_id)
        if reservation is None:
            return None
        identity, slot_cost = reservation
        usage = state.usage.get(identity, 0)
        count = state.counts.get(identity, 0)
        if usage < slot_cost:
            raise RuntimeError(
                f"{state.role.capitalize()} SSM slot accounting underflow on {identity!r}: "
                f"used={usage}, release={slot_cost}"
            )
        if count < 1:
            raise RuntimeError(
                f"{state.role.capitalize()} request accounting underflow on {identity!r}"
            )
        state.reservations.pop(request_id)
        state.usage[identity] = usage - slot_cost
        state.counts[identity] = count - 1
        return identity

    def release_prefill(self, request_id: int):
        """Release a request's prefill reservation and return its identity."""

        return self._release(self._prefill, request_id)

    @staticmethod
    def slot_cost_from_handoff(handoff) -> int:
        """Return the live recurrent-state demand carried by a handoff."""

        kv_meta = handoff.get("kv_meta") if isinstance(handoff, dict) else None
        ssm_meta = kv_meta.get("ssm") if isinstance(kv_meta, dict) else None
        return int(bool(ssm_meta))

    def can_ever_fit(self, identity, slot_cost: int) -> bool:
        """Return whether one request can fit the engine's advertised capacity."""

        capacity = self._capacity.get(identity)
        return capacity is None or slot_cost <= capacity

    def try_reserve(self, identity, request_id: int, slot_cost: int) -> bool:
        """Reserve decode capacity, or report that the request must wait."""

        return self._try_reserve(self._decode, identity, request_id, slot_cost)

    def has_queued(self, identity) -> bool:
        """Return whether an engine already has older queued handoffs."""

        return bool(self._decode.queues.get(identity))

    def enqueue(self, identity, request_id: int, payload: bytes, slot_cost: int) -> None:
        """Append a decode handoff to an engine's FIFO capacity queue."""

        self._decode.queues.setdefault(identity, deque()).append(
            QueuedDecodeHandoff(request_id=request_id, payload=payload, slot_cost=slot_cost)
        )

    def pop_next_admissible(self, identity) -> Optional[QueuedDecodeHandoff]:
        """Reserve and return the next FIFO handoff, if it fits.

        Admission is intentionally one-at-a-time so the coordinator can send
        each handoff before reserving another. If that send discovers a dead
        engine, no later queued request is left with an orphan reservation.
        """

        return self._pop_next(self._decode, identity)

    def _pop_next(self, state: _RoleState, identity):
        queue = state.queues.get(identity)
        if not queue:
            return None
        request = queue[0]
        if not self._try_reserve(state, identity, request.request_id, request.slot_cost):
            return None
        queue.popleft()
        if not queue:
            state.queues.pop(identity, None)
        return request

    def release_decode(self, request_id: int):
        """Release a request's reservation and return its decode identity."""

        return self._release(self._decode, request_id)

    def remove_queued(self, request_id: int) -> bool:
        """Remove a queued request and report whether one was found."""

        for state in (self._prefill, self._decode):
            for identity, queue in list(state.queues.items()):
                remaining = deque(item for item in queue if item.request_id != request_id)
                if len(remaining) == len(queue):
                    continue
                if remaining:
                    state.queues[identity] = remaining
                else:
                    state.queues.pop(identity, None)
                return True
        return False

    def pop_queued_for_engine(self, identity) -> List[int]:
        """Remove and return all requests queued for an engine."""

        prefills = self._prefill.queues.pop(identity, ())
        handoffs = self._decode.queues.pop(identity, ())
        return [request.request_id for request in prefills] + [
            handoff.request_id for handoff in handoffs
        ]

    def reservations_for_engine(self, identity) -> List[int]:
        """Return request IDs currently holding capacity on an engine."""

        prefills = [
            request_id
            for request_id, (reserved_identity, _) in self._prefill.reservations.items()
            if reserved_identity == identity
        ]
        decodes = [
            request_id
            for request_id, (reserved_identity, _) in self._decode.reservations.items()
            if reserved_identity == identity
        ]
        return prefills + decodes
