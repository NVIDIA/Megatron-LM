# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-owned handoff lifetimes; no persistent state or timeout-based reclamation."""

from dataclasses import dataclass


@dataclass
class _Handoff:
    engine: bytes
    owner: str | None = None


class HandoffOwnership:
    """Keep ownership beside the source allocation, independently of decode workers.

    A supervisor may retire an owner only after its adapter and all decode ranks
    have exited and cannot restart. Losing discovery or a heartbeat is not proof.
    """

    def __init__(self) -> None:
        self._handoffs: dict[int, _Handoff] = {}
        self._owners: dict[str, set[int]] = {}
        self._terminated: set[str] = set()

    def offer(self, request_id: int, engine: bytes) -> None:
        """Register retained source state before publishing its handoff."""
        self._handoffs.setdefault(request_id, _Handoff(engine))

    def claim(self, request_id: int, owner: str) -> bool:
        """Claim once before reading; duplicate consumers must not share a release."""
        handoff = self._handoffs.get(request_id)
        if not owner or owner in self._terminated or handoff is None:
            return False
        if handoff.owner is not None:
            return False
        handoff.owner = owner
        self._owners.setdefault(owner, set()).add(request_id)
        return True

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
