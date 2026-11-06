# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Step-time histograms by step class, for the /metrics endpoint.

Only steps the engine already times (logging steps) are recorded. A step is classed by the
forward it launched: ``prefill`` if this rank's batch has a prefill; otherwise ``decode_clean``
or ``decode_interfered`` by whether any rank in the expert-parallel group has a prefill that
step (from the EP consensus), or ``decode`` when the step had no such consensus.
"""

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
