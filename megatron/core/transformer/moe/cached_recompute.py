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
* :func:`stash` / :func:`take` -- the per-key stash (shared by every thread: the forward fills it
  on the model's thread, the re-run takes from it on the autograd engine's worker thread): the
  forward's ``_DeepepManager.dispatch`` stores the DeepEP handle, the dispatched routing (indices
  and probs in DeepEP's received layout) and the host counts; the re-run takes them and issues
  DeepEP's CACHED dispatch through the handle (``fused_a2a.cached_fused_dispatch``: the dispatch kernel alone, no layout, no
  notify, no host wait) -- the same path the combine's backward already takes.

The recomputed values are bitwise the full dispatch's: the same handle delivers the same tokens in
the same received order.  Gated by ``TransformerConfig.moe_cached_recompute_dispatch``.
"""

import contextlib
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The scope stack is per thread: a checkpointed forward runs on the thread that calls the model,
# its re-run on the autograd engine's worker thread, and each pushes its own scope around its
# own run.  The STASH is shared by every thread (with a lock): the forward fills it on the main
# thread and the re-run takes from it on the worker thread.
_state = threading.local()
_STASH: dict = {}
_LOCK = threading.Lock()
_LIVE_BYTES = 0
_PEAK_ENTRIES = 0
_REPORTED = False


def _scopes() -> list:
    scopes = getattr(_state, "scopes", None)
    if scopes is None:
        scopes = _state.scopes = []
    return scopes


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


def _bytes(value: Any) -> int:
    total = 0
    for t in getattr(value, "tensors", lambda: ())():
        total += t.numel() * t.element_size()
    return total


def stash(key: Any, layer: Any, value: Any) -> None:
    """Keep ``value`` for the re-run of the checkpointed call ``key`` on ``layer``."""
    global _LIVE_BYTES, _PEAK_ENTRIES, _REPORTED
    size = _bytes(value)
    with _LOCK:
        _STASH[(id(key), id(layer))] = (key, value, size)  # the key object kept alive with it
        _LIVE_BYTES += size
        n = len(_STASH)
        _PEAK_ENTRIES = max(_PEAK_ENTRIES, n)
        report = not _REPORTED
        _REPORTED = True
    if report:
        logger.info(
            "moe_cached_recompute_dispatch: first stash %d bytes (DeepEP handle + dispatched"
            " indices / probs); entries are taken by the re-run",
            size,
        )
    if n == 256 or n == 1024:  # a stash that keeps growing is a forward whose re-run never takes
        logger.warning(
            "moe_cached_recompute_dispatch: %d dispatches stashed and not yet re-run"
            " (%.1f MB): forwards outnumber re-runs",
            n,
            _LIVE_BYTES / 2**20,
        )


def take(key: Any, layer: Any) -> Any:
    """The value stashed for (``key``, ``layer``), removed; None when nothing was stashed."""
    global _LIVE_BYTES
    with _LOCK:
        entry = _STASH.pop((id(key), id(layer)), None)
        if entry is not None:
            _LIVE_BYTES -= entry[2]
    return None if entry is None else entry[1]


def stashed_count() -> int:
    """How many forward dispatches await their re-run (a leak check for tests)."""
    with _LOCK:
        return len(_STASH)


def live_bytes() -> int:
    """The bytes of every stashed value not yet taken."""
    with _LOCK:
        return _LIVE_BYTES


class DispatchCache:
    """What a forward DeepEP dispatch leaves for its re-run (per layer and checkpointed call)."""

    __slots__ = ("handle", "tokens_per_expert", "dispatched_indices", "dispatched_probs")

    def __init__(self, handle, tokens_per_expert, dispatched_indices, dispatched_probs):
        self.handle = handle
        self.tokens_per_expert = tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

    def tensors(self):
        """The device tensors the cache keeps alive (for the byte accounting)."""
        import torch

        out = [t for t in (self.dispatched_indices, self.dispatched_probs) if torch.is_tensor(t)]
        if isinstance(self.handle, tuple):
            out.extend(t for t in self.handle if torch.is_tensor(t))
        return out
