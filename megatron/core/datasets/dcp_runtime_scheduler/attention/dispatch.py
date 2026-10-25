# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

import heapq
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

_BACKEND = ContextVar("attention_dispatch_backend", default="python")


@contextmanager
def dispatch_backend(name):
    if name not in ("python", "numba"):
        raise ValueError("dispatch backend must be python or numba")
    token = _BACKEND.set(name)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def _chain(units, slots, rho):
    count = len(units)
    if count == 0:
        return 0.0, 0, 0, 0, 0
    maximum = int(max(units))
    if count <= slots:
        return rho + float(maximum), 1, maximum, count, maximum
    if rho + float(maximum) > 0 and min(units) == maximum:
        waves = (count + slots - 1) // slots
        end = 0.0
        for _ in range(waves):
            end += rho + float(maximum)
        return end, waves, waves * maximum, count, maximum
    heap = [(0.0, i) for i in range(slots)]
    chains = [0] * slots
    tiles = [0] * slots
    end, critical = 0.0, 0
    for value in units:
        finish, index = heap[0]
        finish += rho + float(value)
        chains[index] += 1
        tiles[index] += int(value)
        heapq.heapreplace(heap, (finish, index))
        if finish > end:
            end, critical = finish, index
    return end, chains[critical], tiles[critical], count, maximum


@lru_cache(maxsize=1)
def _compiled():
    try:
        from numba import njit
    except ImportError as exc:
        raise RuntimeError(
            "numba dispatch requires optional numba; use kernel_backend='python' otherwise"
        ) from exc
    return njit(cache=True, fastmath=False, parallel=False)(_chain)


def compiled_chain(units, slots, rho):
    """Integer tile units only; return the Python oracle's exact five quantities."""
    import numpy as np

    if slots <= 0:
        raise ValueError("slots must be positive")
    return _compiled()(np.asarray(units, dtype=np.int64), int(slots), float(rho))


def accelerated_chain(units, slots, rho):
    if _BACKEND.get() == "python":
        return None
    return compiled_chain(units, slots, rho)


def dispatch_chain(units, slots, rho):
    """Shared integer-tile dispatch for Python and optional Numba execution."""
    result = accelerated_chain(units, slots, rho)
    return _chain(units, int(slots), float(rho)) if result is None else result
