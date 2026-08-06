# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recurrent-state flow control for coordinator-native disaggregation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class QueuedPrefillRequest:
    """Prefill request waiting for durable Mamba capacity."""

    request_id: int
    prompt: Any
    sampling_params: Any
    slot_cost: int


@dataclass(frozen=True)
class QueuedDecodeHandoff:
    """Decode handoff waiting for enough durable Mamba cache slots."""

    request_id: int
    payload: bytes
    slot_cost: int


class DisaggStateFlowControl:
    """Weighted prefill/decode admission based on durable recurrent-state slots.

    Engines advertise their durable slot capacity with their disaggregation
    transfer metadata. Prefill reservations conservatively use prompt block
    count; decode reservations use the exact number of Mamba block positions
    in each handoff. Reservations remain held while the corresponding engine
    owns the request's cache state.
    """

    def __init__(self) -> None:
        self._capacity: Dict[Any, int] = {}
        self._prefill_slot_cost: Dict[Any, int] = {}
        self._prefill_usage: Dict[Any, int] = {}
        self._prefill_reservations: Dict[int, Tuple[Any, int]] = {}
        self._prefill_queues: Dict[Any, Deque[QueuedPrefillRequest]] = {}
        self._decode_usage: Dict[Any, int] = {}
        self._decode_reservations: Dict[int, Tuple[Any, int]] = {}
        self._decode_queues: Dict[Any, Deque[QueuedDecodeHandoff]] = {}

    @staticmethod
    def _capacity_from_instance_meta(instance_meta) -> Optional[int]:
        """Return the conservative durable capacity across model-parallel ranks."""

        if not isinstance(instance_meta, list):
            return None
        entries = [entry for entry in instance_meta if isinstance(entry, dict)]
        capacities = [
            int(entry.get("ssm_slot_capacity", entry.get("mamba_slot_capacity")))
            for entry in entries
            if entry.get("ssm_slot_capacity", entry.get("mamba_slot_capacity")) is not None
        ]
        if capacities and len(capacities) != len(entries):
            raise ValueError(
                "SSM slot capacity is missing from part of an engine's " "model-parallel metadata"
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
        hybrid_entries = [
            entry
            for entry in entries
            if entry.get("ssm_slot_capacity", entry.get("mamba_slot_capacity")) is not None
        ]
        if not hybrid_entries:
            return 0
        costs = [
            int(entry.get("ssm_handoff_max_slots", entry.get("mamba_handoff_max_slots")))
            for entry in hybrid_entries
            if entry.get("ssm_handoff_max_slots", entry.get("mamba_handoff_max_slots")) is not None
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
        # A reconnect may reuse an identity with different metadata. Replace
        # the old limits instead of retaining values that are no longer
        # advertised by the new engine instance.
        self._capacity.pop(identity, None)
        self._prefill_slot_cost.pop(identity, None)
        if capacity is not None:
            self._capacity[identity] = capacity
        if role == "prefill":
            self._prefill_slot_cost[identity] = self._prefill_slot_cost_from_instance_meta(
                instance_meta
            )
            self._prefill_usage.setdefault(identity, 0)
        if role == "decode":
            self._decode_usage.setdefault(identity, 0)
        return capacity

    def remove_engine(self, identity) -> None:
        """Forget an engine after its queued and in-flight requests are dropped."""

        self._capacity.pop(identity, None)
        self._prefill_slot_cost.pop(identity, None)
        self._prefill_usage.pop(identity, None)
        self._prefill_queues.pop(identity, None)
        self._decode_usage.pop(identity, None)
        self._decode_queues.pop(identity, None)
        for reservations in (self._prefill_reservations, self._decode_reservations):
            for request_id, (reserved_identity, _) in list(reservations.items()):
                if reserved_identity == identity:
                    reservations.pop(request_id)

    def capacity(self, identity) -> Optional[int]:
        """Return an engine's advertised durable capacity, if any."""

        return self._capacity.get(identity)

    def decode_usage(self, identity) -> int:
        """Return currently reserved decode slots."""

        return self._decode_usage.get(identity, 0)

    def prefill_usage(self, identity) -> int:
        """Return currently reserved prefill slots."""

        return self._prefill_usage.get(identity, 0)

    def prefill_slot_cost(self, identity) -> int:
        """Return the engine-advertised durable handoff demand per request."""

        return self._prefill_slot_cost.get(identity, 0)

    def _prefill_count(self, identity) -> int:
        return sum(
            reserved_identity == identity
            for reserved_identity, _ in self._prefill_reservations.values()
        )

    def try_reserve_prefill(
        self, identity, request_id: int, slot_cost: int, max_requests: int
    ) -> bool:
        """Reserve one prefill request by count and weighted slot demand."""

        if slot_cost < 0:
            raise ValueError(f"SSM slot cost cannot be negative, got {slot_cost}")
        if request_id in self._prefill_reservations:
            raise RuntimeError(f"Prefill request {request_id} already holds an SSM reservation")
        if self._prefill_count(identity) >= max_requests:
            return False
        capacity = self._capacity.get(identity)
        usage = self._prefill_usage.get(identity, 0)
        if capacity is not None and usage + slot_cost > capacity:
            return False
        self._prefill_usage[identity] = usage + slot_cost
        self._prefill_reservations[request_id] = (identity, slot_cost)
        return True

    def enqueue_prefill(
        self, identity, request_id: int, prompt, sampling_params, slot_cost: int
    ) -> None:
        """Append a prefill request to an engine's FIFO capacity queue."""

        self._prefill_queues.setdefault(identity, deque()).append(
            QueuedPrefillRequest(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                slot_cost=slot_cost,
            )
        )

    def has_queued_prefill(self, identity) -> bool:
        """Return whether an engine already has older queued prefills."""

        return bool(self._prefill_queues.get(identity))

    def pop_next_prefill(self, identity, max_requests: int) -> Optional[QueuedPrefillRequest]:
        """Reserve and return the next queued prefill, if it fits."""

        queue = self._prefill_queues.get(identity)
        if not queue:
            return None
        request = queue[0]
        if not self.try_reserve_prefill(
            identity, request.request_id, request.slot_cost, max_requests
        ):
            return None
        queue.popleft()
        if not queue:
            self._prefill_queues.pop(identity, None)
        return request

    def release_prefill(self, request_id: int):
        """Release a request's prefill reservation and return its identity."""

        reservation = self._prefill_reservations.pop(request_id, None)
        if reservation is None:
            return None
        identity, slot_cost = reservation
        usage = self._prefill_usage.get(identity, 0)
        if usage < slot_cost:
            raise RuntimeError(
                f"Prefill SSM slot accounting underflow on {identity!r}: "
                f"used={usage}, release={slot_cost}"
            )
        self._prefill_usage[identity] = usage - slot_cost
        return identity

    @staticmethod
    def slot_cost_from_handoff(handoff) -> int:
        """Return the exact durable-slot demand advertised by a handoff."""

        kv_meta = handoff.get("kv_meta") if isinstance(handoff, dict) else None
        ssm_meta = kv_meta.get("ssm") if isinstance(kv_meta, dict) else None
        positions = ssm_meta.get("positions", []) if isinstance(ssm_meta, dict) else []
        return len({int(position) for position in positions})

    def can_ever_fit(self, identity, slot_cost: int) -> bool:
        """Return whether one request can fit the engine's advertised capacity."""

        capacity = self._capacity.get(identity)
        return capacity is None or slot_cost <= capacity

    def try_reserve(self, identity, request_id: int, slot_cost: int) -> bool:
        """Reserve decode capacity, or report that the request must wait."""

        if slot_cost < 0:
            raise ValueError(f"SSM slot cost cannot be negative, got {slot_cost}")
        capacity = self._capacity.get(identity)
        if capacity is None or slot_cost == 0:
            return True
        if request_id in self._decode_reservations:
            raise RuntimeError(f"Decode request {request_id} already holds an SSM reservation")
        usage = self._decode_usage.get(identity, 0)
        if usage + slot_cost > capacity:
            return False
        self._decode_usage[identity] = usage + slot_cost
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

        reservation = self._decode_reservations.pop(request_id, None)
        if reservation is None:
            return None
        identity, slot_cost = reservation
        usage = self._decode_usage.get(identity, 0)
        if usage < slot_cost:
            raise RuntimeError(
                f"Decode SSM slot accounting underflow on {identity!r}: "
                f"used={usage}, release={slot_cost}"
            )
        self._decode_usage[identity] = usage - slot_cost
        return identity

    def remove_queued(self, request_id: int) -> None:
        """Remove a request from a prefill or decode capacity queue."""

        for queues in (self._prefill_queues, self._decode_queues):
            for identity, queue in list(queues.items()):
                remaining = deque(item for item in queue if item.request_id != request_id)
                if len(remaining) == len(queue):
                    continue
                if remaining:
                    queues[identity] = remaining
                else:
                    queues.pop(identity, None)
                return

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
