# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recurrent-state flow control for coordinator-native disaggregation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


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


class DisaggStateFlowControl:
    """Reserve recurrent-state slots for prefill and decode handoffs."""

    def __init__(self) -> None:
        self._capacity: Dict[Any, int] = {}
        self._request_capacity: Dict[Any, int] = {}
        self._prefill_slot_cost: Dict[Any, int] = {}
        self._prefill_usage: Dict[Any, int] = {}
        self._prefill_counts: Dict[Any, int] = {}
        self._prefill_reservations: Dict[int, Tuple[Any, int]] = {}
        self._prefill_queues: Dict[Any, Deque[QueuedPrefillRequest]] = {}
        self._decode_usage: Dict[Any, int] = {}
        self._decode_counts: Dict[Any, int] = {}
        self._decode_reservations: Dict[int, Tuple[Any, int]] = {}
        self._decode_queues: Dict[Any, Deque[QueuedDecodeHandoff]] = {}

    @staticmethod
    def _request_capacity_from_instance_meta(instance_meta) -> int:
        """Return the conservative request capacity across model-parallel ranks."""

        entries = (
            [entry for entry in instance_meta if isinstance(entry, dict)]
            if isinstance(instance_meta, list)
            else []
        )
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
    def _capacity_from_instance_meta(instance_meta) -> Optional[int]:
        """Return the conservative live-state capacity across model-parallel ranks."""

        if not isinstance(instance_meta, list):
            return None
        entries = [entry for entry in instance_meta if isinstance(entry, dict)]
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
    def _prefill_slot_cost_from_instance_meta(instance_meta) -> int:
        """Return the per-request handoff bound advertised by a prefill engine."""

        if not isinstance(instance_meta, list):
            return 0
        entries = [entry for entry in instance_meta if isinstance(entry, dict)]
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

        capacity = self._capacity_from_instance_meta(instance_meta)
        request_capacity = self._request_capacity_from_instance_meta(instance_meta)
        # A reconnect may reuse an identity with new limits.
        self._capacity.pop(identity, None)
        self._request_capacity[identity] = request_capacity
        self._prefill_slot_cost.pop(identity, None)
        if capacity is not None:
            self._capacity[identity] = capacity
        if role == "prefill":
            self._prefill_slot_cost[identity] = self._prefill_slot_cost_from_instance_meta(
                instance_meta
            )
            self._prefill_usage.setdefault(identity, 0)
            self._prefill_counts.setdefault(identity, 0)
        if role == "decode":
            self._decode_usage.setdefault(identity, 0)
            self._decode_counts.setdefault(identity, 0)
        return capacity

    def remove_engine(self, identity) -> None:
        """Forget an engine after its queued and in-flight requests are dropped."""

        self._capacity.pop(identity, None)
        self._request_capacity.pop(identity, None)
        self._prefill_slot_cost.pop(identity, None)
        self._prefill_usage.pop(identity, None)
        self._prefill_counts.pop(identity, None)
        self._prefill_queues.pop(identity, None)
        self._decode_usage.pop(identity, None)
        self._decode_counts.pop(identity, None)
        self._decode_queues.pop(identity, None)
        for reservations in (self._prefill_reservations, self._decode_reservations):
            for request_id, (reserved_identity, _) in list(reservations.items()):
                if reserved_identity == identity:
                    reservations.pop(request_id)

    def capacity(self, identity) -> Optional[int]:
        """Return an engine's advertised live recurrent-state capacity, if any."""

        return self._capacity.get(identity)

    def decode_usage(self, identity) -> int:
        """Return currently reserved decode slots."""

        return self._decode_usage.get(identity, 0)

    def prefill_usage(self, identity) -> int:
        """Return currently reserved prefill slots."""

        return self._prefill_usage.get(identity, 0)

    def prefill_load(self, identity) -> tuple[int, int, int]:
        """Return queued, active, and weighted live-state prefill load."""

        return (
            len(self._prefill_queues.get(identity, ())),
            self._prefill_counts.get(identity, 0),
            self.prefill_usage(identity),
        )

    def decode_load(self, identity) -> tuple[int, int, int]:
        """Return queued, active, and weighted live-state decode load."""

        return (
            len(self._decode_queues.get(identity, ())),
            self._decode_counts.get(identity, 0),
            self.decode_usage(identity),
        )

    def prefill_slot_cost(self, identity) -> int:
        """Return the engine-advertised live-state handoff demand per request."""

        return self._prefill_slot_cost.get(identity, 0)

    def try_reserve_prefill(self, identity, request_id: int, slot_cost: int) -> bool:
        """Reserve one prefill request by count and weighted slot demand."""

        if slot_cost < 0:
            raise ValueError(f"SSM slot cost cannot be negative, got {slot_cost}")
        if request_id in self._prefill_reservations:
            raise RuntimeError(f"Prefill request {request_id} already holds an SSM reservation")
        count = self._prefill_counts.get(identity, 0)
        if count >= self._request_capacity[identity]:
            return False
        capacity = self._capacity.get(identity)
        usage = self._prefill_usage.get(identity, 0)
        if capacity is not None and usage + slot_cost > capacity:
            return False
        self._prefill_usage[identity] = usage + slot_cost
        self._prefill_counts[identity] = count + 1
        self._prefill_reservations[request_id] = (identity, slot_cost)
        return True

    def enqueue_prefill(self, identity, request_id: int, slot_cost: int) -> None:
        """Append a prefill request to an engine's FIFO capacity queue."""

        self._prefill_queues.setdefault(identity, deque()).append(
            QueuedPrefillRequest(request_id=request_id, slot_cost=slot_cost)
        )

    def has_queued_prefill(self, identity) -> bool:
        """Return whether an engine already has older queued prefills."""

        return bool(self._prefill_queues.get(identity))

    def pop_next_prefill(self, identity) -> Optional[QueuedPrefillRequest]:
        """Reserve and return the next queued prefill, if it fits."""

        queue = self._prefill_queues.get(identity)
        if not queue:
            return None
        request = queue[0]
        if not self.try_reserve_prefill(identity, request.request_id, request.slot_cost):
            return None
        queue.popleft()
        if not queue:
            self._prefill_queues.pop(identity, None)
        return request

    def release_prefill(self, request_id: int):
        """Release a request's prefill reservation and return its identity."""

        reservation = self._prefill_reservations.get(request_id)
        if reservation is None:
            return None
        identity, slot_cost = reservation
        usage = self._prefill_usage.get(identity, 0)
        count = self._prefill_counts.get(identity, 0)
        if usage < slot_cost:
            raise RuntimeError(
                f"Prefill SSM slot accounting underflow on {identity!r}: "
                f"used={usage}, release={slot_cost}"
            )
        if count < 1:
            raise RuntimeError(f"Prefill request accounting underflow on {identity!r}")
        self._prefill_reservations.pop(request_id)
        self._prefill_usage[identity] = usage - slot_cost
        self._prefill_counts[identity] = count - 1
        return identity

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

        if slot_cost < 0:
            raise ValueError(f"SSM slot cost cannot be negative, got {slot_cost}")
        capacity = self._capacity.get(identity)
        if request_id in self._decode_reservations:
            raise RuntimeError(f"Decode request {request_id} already holds an SSM reservation")
        count = self._decode_counts.get(identity, 0)
        if count >= self._request_capacity[identity]:
            return False
        usage = self._decode_usage.get(identity, 0)
        if capacity is not None and usage + slot_cost > capacity:
            return False
        self._decode_usage[identity] = usage + slot_cost
        self._decode_counts[identity] = count + 1
        self._decode_reservations[request_id] = (identity, slot_cost)
        return True

    def has_queued(self, identity) -> bool:
        """Return whether an engine already has older queued handoffs."""

        return bool(self._decode_queues.get(identity))

    def enqueue(self, identity, request_id: int, payload: bytes, slot_cost: int) -> None:
        """Append a decode handoff to an engine's FIFO capacity queue."""

        self._decode_queues.setdefault(identity, deque()).append(
            QueuedDecodeHandoff(request_id=request_id, payload=payload, slot_cost=slot_cost)
        )

    def pop_next_admissible(self, identity) -> Optional[QueuedDecodeHandoff]:
        """Reserve and return the next FIFO handoff, if it fits.

        Admission is intentionally one-at-a-time so the coordinator can send
        each handoff before reserving another. If that send discovers a dead
        engine, no later queued request is left with an orphan reservation.
        """
        queue = self._decode_queues.get(identity)
        if not queue:
            return None
        handoff = queue[0]
        if not self.try_reserve(identity, handoff.request_id, handoff.slot_cost):
            return None
        queue.popleft()
        if not queue:
            self._decode_queues.pop(identity, None)
        return handoff

    def release_decode(self, request_id: int):
        """Release a request's reservation and return its decode identity."""

        reservation = self._decode_reservations.get(request_id)
        if reservation is None:
            return None
        identity, slot_cost = reservation
        usage = self._decode_usage.get(identity, 0)
        count = self._decode_counts.get(identity, 0)
        if usage < slot_cost:
            raise RuntimeError(
                f"Decode SSM slot accounting underflow on {identity!r}: "
                f"used={usage}, release={slot_cost}"
            )
        if count < 1:
            raise RuntimeError(f"Decode request accounting underflow on {identity!r}")
        self._decode_reservations.pop(request_id)
        self._decode_usage[identity] = usage - slot_cost
        self._decode_counts[identity] = count - 1
        return identity

    def remove_queued(self, request_id: int) -> bool:
        """Remove a queued request and report whether one was found."""

        for queues in (self._prefill_queues, self._decode_queues):
            for identity, queue in list(queues.items()):
                remaining = deque(item for item in queue if item.request_id != request_id)
                if len(remaining) == len(queue):
                    continue
                if remaining:
                    queues[identity] = remaining
                else:
                    queues.pop(identity, None)
                return True
        return False

    def pop_queued_for_engine(self, identity) -> List[int]:
        """Remove and return all requests queued for an engine."""

        prefills = self._prefill_queues.pop(identity, ())
        handoffs = self._decode_queues.pop(identity, ())
        return [request.request_id for request in prefills] + [
            handoff.request_id for handoff in handoffs
        ]

    def reservations_for_engine(self, identity) -> List[int]:
        """Return request IDs currently holding capacity on an engine."""

        prefills = [
            request_id
            for request_id, (reserved_identity, _) in self._prefill_reservations.items()
            if reserved_identity == identity
        ]
        decodes = [
            request_id
            for request_id, (reserved_identity, _) in self._decode_reservations.items()
            if reserved_identity == identity
        ]
        return prefills + decodes
