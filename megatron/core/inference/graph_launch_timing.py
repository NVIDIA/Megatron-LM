# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Host-side cost of CUDA graph replay, measured without a profiler.

`GAP-DECOMP-S17` found `cudaGraphLaunch` accounting for ~707 us/step of exposed GPU
idle at ~191 us per call, which would make it the single largest item in the 1.645 ms
idle budget. That number is not trustworthy on its own: both traces were captured with
``--cuda-graph-trace=node``, and node-level tracing makes the driver do per-node work
inside the launch, which can inflate a normally-cheap graph launch by an order of
magnitude. A profiler cannot settle a question about profiler overhead.

This module measures it with ``time.perf_counter_ns`` around
``torch.cuda.CUDAGraph.replay`` and no profiler attached. The host cost of a replay is
pure launch submission -- it does not wait for the GPU -- so wall time around the call
is exactly the quantity of interest.

Enable with ``MCORE_GRAPH_LAUNCH_TIMING=1``. Reports median rather than mean because
the first replays after capture are not representative.
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import atexit
import os
import sys
import time
from typing import Dict, List

ENABLED: bool = os.environ.get("MCORE_GRAPH_LAUNCH_TIMING", "0") == "1"

# Skip the first replays: allocator warmup and first-touch page faults land there.
SKIP: int = int(os.environ.get("MCORE_GRAPH_LAUNCH_SKIP", "20"))
# Report periodically, not only at exit: the benchmark harness SIGTERMs the server,
# which skips atexit handlers, so an exit-only report usually prints nothing.
REPORT_EVERY: int = int(os.environ.get("MCORE_GRAPH_LAUNCH_REPORT_EVERY", "2000"))

_samples: Dict[int, List[int]] = {}
_calls = 0
_installed = False


def install() -> None:
    """Wrap ``CUDAGraph.replay`` to record host wall time per call, keyed by graph."""
    global _installed
    if _installed or not ENABLED:
        return
    import torch

    orig = torch.cuda.CUDAGraph.replay

    def timed_replay(self, *args, **kwargs):
        global _calls
        t0 = time.perf_counter_ns()
        out = orig(self, *args, **kwargs)
        dt = time.perf_counter_ns() - t0
        _samples.setdefault(id(self), []).append(dt)
        _calls += 1
        if REPORT_EVERY > 0 and _calls % REPORT_EVERY == 0:
            report()
        return out

    torch.cuda.CUDAGraph.replay = timed_replay
    atexit.register(report)
    _installed = True
    print(
        f"[GRAPH_LAUNCH] host timing installed (skip {SKIP}, report every {REPORT_EVERY})",
        file=sys.stderr,
        flush=True,
    )


def report() -> None:
    """Print per-graph and aggregate host replay cost."""
    if not ENABLED or not _samples:
        return
    lines = ["[GRAPH_LAUNCH] host cost of CUDAGraph.replay (no profiler attached)"]
    total_median_us = 0.0
    total_calls = 0
    for i, (gid, xs) in enumerate(sorted(_samples.items(), key=lambda kv: -len(kv[1]))):
        ys = sorted(xs[SKIP:]) or sorted(xs)
        med = ys[len(ys) // 2] / 1000.0
        p90 = ys[int(len(ys) * 0.9)] / 1000.0
        total_median_us += med
        total_calls += 1
        lines.append(f"[GRAPH_LAUNCH]   graph#{i}: n={len(xs)} median={med:.2f}us p90={p90:.2f}us")
    lines.append(
        f"[GRAPH_LAUNCH] distinct graphs={total_calls} "
        f"sum-of-medians={total_median_us:.1f}us "
        f"(compare: profiled trace claimed ~191us/launch x4 = 766us/step)"
    )
    print("\n".join(lines), file=sys.stderr, flush=True)
