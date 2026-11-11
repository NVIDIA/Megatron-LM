# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Step-time histograms and step-mix counters by step class, for the /metrics endpoint.

The histogram records only steps the engine already times (logging steps). A step is classed by
the forward it launched: ``prefill`` if this rank's batch has a prefill; otherwise
``decode_clean`` or ``decode_interfered`` by whether any rank in the expert-parallel group has a
prefill that step (from the EP consensus), or ``decode`` when the step had no such consensus.

The step-mix counters (:class:`StepMixCounter`) cover every step at no GPU cost: they split
``prefill`` into ``prefill_only`` (no decode request in this rank's batch) and ``mixed``
(prefill and decode requests together), and time each step by host wall clock between
consecutive steps, which under steady load approximates its GPU time.
"""

import time
from typing import Dict, List, Optional

# Upper bounds (ms) of the histogram buckets, powers of two; a final +Inf bucket is implied.
STEP_TIME_BUCKETS_MS: List[float] = [2.0**k for k in range(-2, 11)]  # 0.25 .. 1024


def classify_step(local_prefill: bool, ep_any_prefill: Optional[bool]) -> str:
    """Step class from this rank's batch and the EP group's (None: no EP consensus)."""
    if local_prefill:
        return "prefill"
    if ep_any_prefill is None:
        return "decode"
    return "decode_interfered" if ep_any_prefill else "decode_clean"


def classify_step_mix(
    local_prefill: bool, local_decode_requests: int, ep_any_prefill: Optional[bool]
) -> str:
    """Finer step class than :func:`classify_step`: splits a local prefill into
    ``prefill_only`` and ``mixed`` by whether decode requests share the batch."""
    if local_prefill:
        return "mixed" if local_decode_requests > 0 else "prefill_only"
    return classify_step(False, ep_any_prefill)


class StepMixCounter:
    """Lifetime per-class counts of steps, their tokens and host wall time (only grows)."""

    # A gap longer than this between steps is the engine idling (no forward to run), not a
    # step; such a step is counted but not timed.
    MAX_STEP_GAP_S = 2.0

    def __init__(self):
        self._steps: Dict[str, int] = {}
        self._tokens: Dict[str, int] = {}
        self._wall_ms: Dict[str, float] = {}
        self._last_step_end: Optional[float] = None

    def observe(self, step_class: str, tokens: int, now: Optional[float] = None) -> None:
        """Record one step of step_class with `tokens` input tokens that ended at `now`
        (perf_counter seconds). Its wall time is the gap since the previous step ended; the
        first step only starts the clock."""
        now = time.perf_counter() if now is None else now
        self._steps[step_class] = self._steps.get(step_class, 0) + 1
        self._tokens[step_class] = self._tokens.get(step_class, 0) + int(tokens)
        gap = None if self._last_step_end is None else now - self._last_step_end
        if gap is not None and gap <= self.MAX_STEP_GAP_S:
            self._wall_ms[step_class] = (
                self._wall_ms.get(step_class, 0.0) + gap * 1e3
            )
        self._last_step_end = now

    def idle(self) -> None:
        """Restart the clock after the engine idled, so idle time is not charged to a step."""
        self._last_step_end = None

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """{class: {"steps", "tokens", "wall_ms"}}."""
        return {
            step_class: {
                "steps": steps,
                "tokens": self._tokens[step_class],
                "wall_ms": self._wall_ms.get(step_class, 0.0),
            }
            for step_class, steps in self._steps.items()
        }


class StepTimeHistogram:
    """Lifetime per-class step-time histograms (counts only grow, like the other counters)."""

    def __init__(self):
        self._counts: Dict[str, List[int]] = {}
        self._sums: Dict[str, float] = {}

    def observe(self, step_class: str, ms: float) -> None:
        """Record one step of step_class that took ms milliseconds."""
        counts = self._counts.setdefault(step_class, [0] * (len(STEP_TIME_BUCKETS_MS) + 1))
        bucket = next(
            (i for i, le in enumerate(STEP_TIME_BUCKETS_MS) if ms <= le), len(STEP_TIME_BUCKETS_MS)
        )
        counts[bucket] += 1
        self._sums[step_class] = self._sums.get(step_class, 0.0) + ms

    def snapshot(self) -> Dict[str, Dict]:
        """{class: {"buckets": cumulative counts per bound (last is +Inf), "sum", "count"}}."""
        out = {}
        for step_class, counts in self._counts.items():
            cumulative, total = [], 0
            for c in counts:
                total += c
                cumulative.append(total)
            out[step_class] = {
                "buckets": cumulative,
                "sum": self._sums[step_class],
                "count": total,
            }
        return out
