# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Replay only the data movement of an activation-recompute re-run of a MoE layer's dispatch.

Under activation recompute (``--recompute-granularity selective`` with ``moe``, or ``full``), the
backward re-runs the MoE layer's forward.  The re-run's expert dispatch is DeepEP's normal-mode
``fused_dispatch``: ``get_dispatch_layout`` and ``notify_dispatch`` kernels, the dispatch kernel,
and a HOST WAIT for the receive counts (``num_recv_tokens_per_expert_list``) -- all to rebuild
routing bookkeeping the forward already produced from the same router output (the same saved
input, a deterministic top-k).  This module lets the re-run reuse the forward's bookkeeping:

* :func:`checkpoint_scope` -- a context ``CheckpointFunction`` (``tensor_parallel/random.py``)
  enters around the forward run of a checkpointed function (``recompute=False``) and around its
  re-run in the backward (``recompute=True``), keyed by a per-call token stored on the ctx, so a
  layer can pair its forward with the exact re-run whatever the schedule (1F1B keeps several
  microbatches in flight).
* :func:`current` -- the innermost scope ``(key, recompute)`` or ``None`` outside a checkpoint.
* :func:`stash` / :func:`take` -- the per-key stash: the forward's ``_DeepepManager.dispatch``
  stores the DeepEP handle, the dispatched routing (indices and probs in DeepEP's received
  layout) and the host counts; the re-run takes them and issues DeepEP's CACHED dispatch through
  the handle (``fused_a2a.cached_fused_dispatch``: the dispatch kernel alone, no layout, no
  notify, no host wait) -- the same path the combine's backward already takes.

The recomputed values are bitwise the full dispatch's: the same handle delivers the same tokens in
the same received order.  Gated by ``TransformerConfig.moe_cached_recompute_dispatch``.
"""

import contextlib
import threading
from typing import Any, Optional

_state = threading.local()


def _scopes() -> list:
    scopes = getattr(_state, "scopes", None)
    if scopes is None:
        scopes = _state.scopes = []
    return scopes


def _stash() -> dict:
    stash = getattr(_state, "stash", None)
    if stash is None:
        stash = _state.stash = {}
    return stash


@contextlib.contextmanager
def checkpoint_scope(key: Any, recompute: bool):
    """Mark the run (``recompute=False``) or the re-run (``recompute=True``) of one checkpointed
    function call, identified by ``key`` (an object the checkpoint ctx keeps)."""
    scopes = _scopes()
    scopes.append((key, recompute))
    try:
        yield
    finally:
        scopes.pop()


def current() -> Optional[tuple]:
    """The innermost ``(key, recompute)`` scope, or None outside any checkpointed call."""
    scopes = _scopes()
    return scopes[-1] if scopes else None


def in_recompute() -> bool:
    """True while a checkpointed function re-runs to build the graph for its backward."""
    scope = current()
    return scope is not None and scope[1]


def stash(key: Any, layer: Any, value: Any) -> None:
    """Keep ``value`` for the re-run of the checkpointed call ``key`` on ``layer``."""
    _stash()[(id(key), id(layer))] = (key, value)  # the key object kept alive with the value


def take(key: Any, layer: Any) -> Any:
    """The value stashed for (``key``, ``layer``), removed; None when nothing was stashed."""
    entry = _stash().pop((id(key), id(layer)), None)
    return None if entry is None else entry[1]


def stashed_count() -> int:
    """How many forward dispatches await their re-run (a leak check for tests)."""
    return len(_stash())


class DispatchCache:
    """What a forward DeepEP dispatch leaves for its re-run (per layer and checkpointed call)."""

    __slots__ = ("handle", "tokens_per_expert", "dispatched_indices", "dispatched_probs")

    def __init__(self, handle, tokens_per_expert, dispatched_indices, dispatched_probs):
        self.handle = handle
        self.tokens_per_expert = tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs
