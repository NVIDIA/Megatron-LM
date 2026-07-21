# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in wall-time and payload accounting for rollout weight synchronization."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.distributed as dist

_ENV_NAME = "MLITE_WEIGHT_SYNC_PROBE"
_REPORT_PREFIX = "MLITE_WEIGHT_SYNC_PROBE "


def weight_sync_probe_enabled() -> bool:
    return os.getenv(_ENV_NAME, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _StageStats:
    calls: int = 0
    bytes: int = 0
    wall_s: float = 0.0


@dataclass
class _Session:
    backend: str
    started_at: float
    depth: int = 1
    stages: dict[str, _StageStats] = field(default_factory=dict)


@dataclass
class ProbeSample:
    nbytes: int = 0


class WeightSyncProbe:
    def __init__(self) -> None:
        self._session: ContextVar[_Session | None] = ContextVar(
            "mlite_weight_sync_probe_session", default=None
        )

    @property
    def active(self) -> bool:
        return weight_sync_probe_enabled() and self._session.get() is not None

    @contextmanager
    def session(self, backend: str) -> Iterator[None]:
        if not weight_sync_probe_enabled():
            yield
            return

        current = self._session.get()
        if current is not None:
            current.depth += 1
            try:
                yield
            finally:
                current.depth -= 1
            return

        session = _Session(backend=backend, started_at=time.perf_counter())
        token = self._session.set(session)
        try:
            yield
        finally:
            total_wall_s = time.perf_counter() - session.started_at
            self._session.reset(token)
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            report = {
                "backend": session.backend,
                "rank": rank,
                "stages": {
                    name: {
                        "calls": stats.calls,
                        "bytes": stats.bytes,
                        "wall_s": stats.wall_s,
                    }
                    for name, stats in sorted(session.stages.items())
                },
                "total_wall_s": total_wall_s,
            }
            print(_REPORT_PREFIX + json.dumps(report, sort_keys=True), flush=True)

    @contextmanager
    def measure(
        self, stage: str, *, nbytes: int = 0, device: torch.device | str | None = None
    ) -> Iterator[ProbeSample]:
        session = self._session.get()
        if not weight_sync_probe_enabled() or session is None:
            yield ProbeSample(nbytes=nbytes)
            return

        if device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        sample = ProbeSample(nbytes=nbytes)
        try:
            yield sample
        finally:
            if device is not None and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            stats = session.stages.setdefault(stage, _StageStats())
            stats.calls += 1
            stats.bytes += int(sample.nbytes)
            stats.wall_s += time.perf_counter() - started_at


_PROBE = WeightSyncProbe()


def get_weight_sync_probe() -> WeightSyncProbe:
    return _PROBE


@contextmanager
def weight_sync_probe_session(backend: str) -> Iterator[None]:
    with _PROBE.session(backend):
        yield


__all__ = [
    "ProbeSample",
    "WeightSyncProbe",
    "get_weight_sync_probe",
    "weight_sync_probe_enabled",
    "weight_sync_probe_session",
]
