# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure 2-hop routing state for coordinator-native prefill->decode disagg."""

from __future__ import annotations

from typing import Callable

PREFILL = "prefill"
DECODE = "decode"


class DisaggRouting:
    """Load-aware prefill-to-decode router with round-robin tie breaking."""

    def __init__(self) -> None:
        self.prefill_engines: list = []
        self.decode_engines: list = []
        self._prefill_rr = 0
        self._decode_rr = 0
        self._req_prefill: dict[int, object] = {}  # request_id -> prefill identity
        self._req_decode: dict[int, object] = {}  # request_id -> decode identity

    def register(self, identity, role: str) -> None:
        """Record an engine and its disagg role (idempotent)."""
        if role == PREFILL:
            pool = self.prefill_engines
        elif role == DECODE:
            pool = self.decode_engines
        else:
            raise ValueError(f"disagg engine role must be 'prefill'/'decode'; got {role!r}")
        if identity not in pool:
            pool.append(identity)

    def remove(self, identity) -> None:
        """Drop a disconnected engine from both pools."""
        for pool in (self.prefill_engines, self.decode_engines):
            if identity in pool:
                pool.remove(identity)

    def route_submit(self, request_id: int, score: Callable | None = None):
        """Hop 1: pick the prefill engine for a newly submitted request."""
        # TODO: Keep related requests on the same prefill engine when a routing key is available.
        if not self.prefill_engines:
            raise RuntimeError("no prefill engines registered")
        ident = self._pick(self.prefill_engines, self._prefill_rr, score)
        self._prefill_rr += 1
        self._req_prefill[request_id] = ident
        return ident

    def route_prefill_done(self, request_id: int, score: Callable | None = None):
        """Pick and remember the decode engine for a completed prefill."""
        if not self.decode_engines:
            raise RuntimeError("no decode engines registered")
        dec = self._pick(self.decode_engines, self._decode_rr, score)
        self._decode_rr += 1
        self._req_decode[request_id] = dec
        return dec

    def forget(self, request_id: int) -> None:
        """Drop per-request state once the reply has been routed to the client."""
        self._req_prefill.pop(request_id, None)
        self._req_decode.pop(request_id, None)

    def requests_involving(self, identity) -> list[int]:
        """Request ids routed through `identity` on either hop (snapshot)."""
        rids = {rid for rid, ident in self._req_prefill.items() if ident == identity}
        rids.update(rid for rid, ident in self._req_decode.items() if ident == identity)
        return list(rids)

    def decode_for_request(self, request_id: int):
        """Return the decode engine assigned to a request, if any."""

        return self._req_decode.get(request_id)

    @staticmethod
    def _pick(pool: list, offset: int, score: Callable | None):
        if score is None:
            return pool[offset % len(pool)]
        # Rotate the pool so min() breaks equal-score ties round-robin.
        return min((pool[(offset + i) % len(pool)] for i in range(len(pool))), key=score)
