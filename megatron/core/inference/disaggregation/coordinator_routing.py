# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure 2-hop routing state for coordinator-native prefill->decode disagg."""

from __future__ import annotations

import abc
from typing import Callable, Dict, List, Tuple

PREFILL = "prefill"
DECODE = "decode"


class DisaggRouter(abc.ABC):
    """Routing policy for coordinator-native disaggregation."""

    @abc.abstractmethod
    def register(self, identity, role: str) -> None:
        """Record an engine and its role ("prefill"/"decode")."""

    @abc.abstractmethod
    def remove(self, identity) -> None:
        """Drop a disconnected engine."""

    @abc.abstractmethod
    def route_submit(self, request_id: int):
        """Hop 1: pick (and remember) the prefill engine for a new request."""

    @abc.abstractmethod
    def route_prefill_done(self, request_id: int) -> Tuple[object, object]:
        """Hop 2: pick the decode engine; return (prefill_id, decode_id)."""

    @abc.abstractmethod
    def forget(self, request_id: int) -> None:
        """Drop per-request state once the reply has been routed home."""

    def requests_involving(self, identity) -> List[int]:
        """Return requests routed through an engine."""
        return []


class DisaggRouting(DisaggRouter):
    """Round-robin prefill-to-decode router."""

    def __init__(self) -> None:
        self.prefill_engines: List = []
        self.decode_engines: List = []
        self._prefill_rr = 0
        self._decode_rr = 0
        self._req_prefill: Dict[int, object] = {}  # request_id -> prefill identity
        self._req_decode: Dict[int, object] = {}  # request_id -> decode identity

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

    def route_submit(self, request_id: int):
        """Hop 1: pick the prefill engine for a newly submitted request."""
        # TODO: load- and prefix-locality-aware selection instead of round-robin.
        if not self.prefill_engines:
            raise RuntimeError("no prefill engines registered")
        ident = self.prefill_engines[self._prefill_rr % len(self.prefill_engines)]
        self._prefill_rr += 1
        self._req_prefill[request_id] = ident
        return ident

    def route_prefill_done(self, request_id: int) -> Tuple[object, object]:
        """Pick decode after prefill and return both engine identities."""
        if not self.decode_engines:
            raise RuntimeError("no decode engines registered")
        dec = self._pick_decode(request_id)
        self._req_decode[request_id] = dec
        prefill = self._req_prefill.get(request_id)
        return prefill, dec

    def forget(self, request_id: int) -> None:
        """Drop per-request state once the reply has been routed to the client."""
        self._req_prefill.pop(request_id, None)
        self._req_decode.pop(request_id, None)

    def requests_involving(self, identity) -> List[int]:
        """Request ids routed through `identity` on either hop (snapshot)."""
        rids = {rid for rid, ident in self._req_prefill.items() if ident == identity}
        rids.update(rid for rid, ident in self._req_decode.items() if ident == identity)
        return list(rids)

    def _pick_decode(self, request_id: int):
        # TODO: load- and handoff-cost-aware selection instead of round-robin.
        dec = self.decode_engines[self._decode_rr % len(self.decode_engines)]
        self._decode_rr += 1
        return dec


# Named policies survive the coordinator spawn boundary.
_DISAGG_ROUTERS: Dict[str, Callable[[], DisaggRouter]] = {}


def register_disagg_router(name: str, factory: Callable[[], DisaggRouter]) -> None:
    """Register a DisaggRouter factory under `name` (call at import time)."""
    _DISAGG_ROUTERS[name] = factory


def make_disagg_router(name: str = "round_robin") -> DisaggRouter:
    """Instantiate the router registered under `name`."""
    try:
        return _DISAGG_ROUTERS[name]()
    except KeyError:
        raise KeyError(f"unknown disagg router {name!r}; registered: {sorted(_DISAGG_ROUTERS)}")


register_disagg_router("round_robin", DisaggRouting)
