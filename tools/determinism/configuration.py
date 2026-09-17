# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime configuration shared by replay and numerical-check observations."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator

_CONFIGURATION: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "determinism_replay_configuration", default=None
)


@contextlib.contextmanager
def replay_configuration(configuration: dict) -> Iterator[None]:
    """Attach observed runtime configuration to replay and numerical checks."""
    token = _CONFIGURATION.set({**(_CONFIGURATION.get() or {}), **configuration})
    try:
        yield
    finally:
        _CONFIGURATION.reset(token)


def configured_signature(signature: dict) -> dict:
    """Snapshot the signature with the configuration active when a check starts."""
    return {**signature, **(_CONFIGURATION.get() or {})}
