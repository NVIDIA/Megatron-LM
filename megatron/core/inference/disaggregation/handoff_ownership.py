# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-owned handoffs: bounded offers and explicitly acknowledged readers."""

import time
from dataclasses import dataclass


@dataclass
class _Handoff:
    engine: bytes
    expires_at: float | None
    owner: str | None = None


class HandoffOwnership:
    """Keep ownership beside the source allocation, independently of decode workers.

    A supervisor may retire an owner only after its adapter and all decode ranks
    have exited and cannot restart. Losing discovery or a heartbeat is not proof.
    """

    def __init__(self, offer_timeout: float = 300.0) -> None:
        if offer_timeout <= 0:
            raise ValueError("Handoff offer timeout must be positive")
        self.offer_timeout = offer_timeout
        self._handoffs: dict[int, _Handoff] = {}
        self._owners: dict[str, set[int]] = {}
        self._terminated: set[str] = set()

    def offer(self, request_id: int, engine: bytes, *, require_claim: bool = False) -> None:
        """Register source state; only claim-aware consumers permit offer expiry."""
        self._handoffs.setdefault(
            request_id,
            _Handoff(engine, time.monotonic() + self.offer_timeout if require_claim else None),
        )

    def claim(self, request_id: int, owner: str) -> bool:
        """Claim once before reading; duplicate consumers must not share a release."""
        handoff = self._handoffs.get(request_id)
        if not owner or owner in self._terminated or handoff is None:
            return False
        if handoff.owner is not None or (
            handoff.expires_at is not None and time.monotonic() >= handoff.expires_at
        ):
            return False
        handoff.owner = owner
        self._owners.setdefault(owner, set()).add(request_id)
        return True

    def expired_unclaimed(self) -> list[tuple[int, bytes]]:
        """List expired offers for release, retaining failed deliveries for retry.

        Claims are fenced by the same deadline, even before the coordinator's
        next sweep. Claimed reads NEVER expire: time is not a completion barrier.
        """
        now = time.monotonic()
        return [
            (request_id, handoff.engine)
            for request_id, handoff in self._handoffs.items()
            if handoff.owner is None
            and handoff.expires_at is not None
            and now >= handoff.expires_at
        ]

    def source_engine(self, request_id: int) -> bytes | None:
        """Return the source without forgetting a release that may need retrying."""
        handoff = self._handoffs.get(request_id)
        return handoff.engine if handoff is not None else None

    def release(self, request_id: int) -> bytes | None:
        """Forget a completed handoff and return its source engine."""
        handoff = self._handoffs.pop(request_id, None)
        if handoff is None:
            return None
        if handoff.owner is not None:
            requests = self._owners[handoff.owner]
            requests.discard(request_id)
            if not requests:
                del self._owners[handoff.owner]
        return handoff.engine

    def confirm_terminated(self, owner: str) -> list[tuple[int, bytes]]:
        """Fence delayed claims and list state to release after successful delivery."""
        if not owner:
            raise ValueError("A nonempty handoff owner is required")
        self._terminated.add(owner)
        return [
            (request_id, self._handoffs[request_id].engine)
            for request_id in self._owners.get(owner, ())
        ]
