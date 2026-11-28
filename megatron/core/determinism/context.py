# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Scoped execution identity for rank-local boundary observers.

Callers establish context at execution boundaries and explicitly pass its fields
into the recorder. This module neither records tensors nor changes selection or
capture semantics. Missing-context diagnostics belong in tensor metadata (or
in fields for ordinary events).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DeterminismContext:
    """Immutable identity that can be saved by a forward call for backward.

    Use ``dataclasses.replace`` to derive a nested scope or to set a saved
    forward snapshot's phase to ``"backward"``. A scope installs the complete
    snapshot, so absent fields cannot accidentally inherit another microbatch.
    """

    iteration: int | None = None
    microbatch: int | None = None
    phase: str = "unspecified"


_CONTEXT: ContextVar[DeterminismContext] = ContextVar(
    "determinism_context", default=DeterminismContext()
)


def get_determinism_context() -> DeterminismContext:
    """Return the current immutable snapshot, or an unspecified identity."""
    return _CONTEXT.get()


@contextmanager
def determinism_context(context: DeterminismContext) -> Iterator[DeterminismContext]:
    """Install a snapshot and restore the previous context even on exceptions.

    Context is local to the current thread or asynchronous task. New threads
    must explicitly install a saved snapshot; async tasks inherit the snapshot
    at creation. Autograd callbacks must explicitly bind their forward snapshot
    rather than rely on the context active when backward executes.
    """
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)
