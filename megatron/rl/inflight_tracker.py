# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-global counter of in-flight RL rollouts (rank 0 only).

A rollout is "in-flight" from when its group's generation starts until the group
is consumed into a training batch. This therefore spans every stage a rollout can
be in:
  (a) actively decoding/prefilling in the inference engine,
  (b) admitted to the engine but waiting in its queue,
  (c) finished generating but buffered, waiting for the rest of its group/batch
      before it can be packaged into a training batch.

The agent generator increments the counter when a group starts generating and the
rollout consumer decrements it on consumption (filtered groups are decremented
where they are dropped). The inference step tracer samples the counter each engine
step so the total in-flight count can be plotted against the engine batch size.

Only the rank-0 process runs the rollout generator/consumer, so this counter is
meaningful on rank 0 only.
"""

import threading

_lock = threading.Lock()
_inflight = 0
_submitted_total = 0
_consumed_total = 0


def add_inflight(n: int) -> None:
    """Mark ``n`` rollouts as entering flight (generation started)."""
    global _inflight, _submitted_total
    with _lock:
        _inflight += n
        _submitted_total += n


def remove_inflight(n: int) -> None:
    """Mark ``n`` rollouts as leaving flight (consumed or filtered out)."""
    global _inflight, _consumed_total
    with _lock:
        _inflight -= n
        _consumed_total += n


def get_inflight() -> int:
    """Current number of in-flight rollouts."""
    return _inflight


def inflight_snapshot() -> dict:
    """Snapshot of the in-flight counters, for per-step inference tracing."""
    return {
        "inflight_rollouts": _inflight,
        "submitted_total": _submitted_total,
        "consumed_total": _consumed_total,
    }


def reset_inflight() -> None:
    """Reset all counters (e.g. when (re)creating the rollout generator)."""
    global _inflight, _submitted_total, _consumed_total
    with _lock:
        _inflight = 0
        _submitted_total = 0
        _consumed_total = 0
