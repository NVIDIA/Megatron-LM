# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Component ablation for attributing the decode block's per-layer latency floor.

A batch-size sweep showed the transformer block is latency-bound, not
compute-bound: 32x the tokens costs only 1.70x the block time, leaving a
batch-independent floor of ~105 us/layer that is 55% of the whole decode step.
Closing the remaining gap to vLLM means removing part of that floor, and nothing
measured so far says what it is made of. Two attempts to infer the composition
from per-kernel coefficients had to be retracted, so this measures it instead.

The method: delete one component from the captured graph and read the change in
block time from ``step_gpu_timing``. Because the block is latency-bound, the
delta is a *chain* delta -- it includes the serialization the component imposed
on its neighbours, not just its own device time -- which is the quantity that
actually matters here. It is deliberately not comparable to a kernel duration
from a profiler.

Ablation happens **only while the CUDA graph is being captured**, which is the
mechanism that makes this safe:

* Warmup runs eagerly, before capture, so every real kernel runs at least once
  and every preallocated buffer holds plausible values with sane magnitudes.
* Capture takes the ablated branch, so all replays skip the component.
* The branch is therefore resolved once, at capture, and no replay contains
  data-dependent control flow.

Correctness is intentionally abandoned under ablation -- only timing is wanted --
but the *shapes* and the *token routing distribution* must survive, or the
measurement silently changes the work done by the components not being ablated.
Two rules keep that from happening:

1. Substitute zeros, never uninitialized memory. Zeros keep the residual stream
   finite, so the next layer's router still sees sane logits and still spreads
   tokens across experts. Garbage or NaN would collapse routing onto a few
   experts and change grouped-GEMM tile balance.
2. Cache the zero buffers by shape. A fresh ``torch.zeros`` per replay would add
   a multi-megabyte fill to the very measurement it is meant to isolate.

All gates default off and are read once at import.

Known limitation, ``ABLATE_NVLS_DISPATCH`` / ``ABLATE_NVLS_COMBINE``: prefill runs
the real collective eagerly while replays skip it, which desynchronizes the
symmetric-memory barrier sequence across ranks. Both arms hang the benchmark
(observed: 1500 s timeout), so their block times are measured in a degraded state
and must not be quoted. Fixing this means suppressing the collective in warmup as
well, so every rank agrees on the sequence. The other gates are unaffected --
they touch no cross-rank state.
"""

import os
from typing import Dict, Optional, Tuple

# Individual components. Each removes one item from the per-layer chain.
ABLATE_NVLS_DISPATCH: bool = os.environ.get("MCORE_ABLATE_NVLS_DISPATCH", "0") == "1"
ABLATE_NVLS_COMBINE: bool = os.environ.get("MCORE_ABLATE_NVLS_COMBINE", "0") == "1"
ABLATE_EXPERT_GEMM: bool = os.environ.get("MCORE_ABLATE_EXPERT_GEMM", "0") == "1"
ABLATE_MOE_SUM: bool = os.environ.get("MCORE_ABLATE_MOE_SUM", "0") == "1"
ABLATE_ATTN_CORE: bool = os.environ.get("MCORE_ABLATE_ATTN_CORE", "0") == "1"

ANY_ABLATION: bool = any(
    (
        ABLATE_NVLS_DISPATCH,
        ABLATE_NVLS_COMBINE,
        ABLATE_EXPERT_GEMM,
        ABLATE_MOE_SUM,
        ABLATE_ATTN_CORE,
    )
)

_zeros_cache: Dict[Tuple, "object"] = {}


def capturing() -> bool:
    """True while the current stream is capturing a CUDA graph.

    Gating on capture rather than on a step counter is what lets warmup populate
    real buffers before any component is removed, and it guarantees the ablated
    branch is baked into the graph exactly once.
    """
    if not ANY_ABLATION:
        return False
    import torch

    return torch.cuda.is_current_stream_capturing()


_hits: Dict[str, int] = {}


def hit(site: str) -> bool:
    """Record that ``site`` took its ablated branch; log the first occurrence.

    Without this a misplaced gate is indistinguishable from a gate that is on but
    never reached, and the run would look like a clean measurement of nothing.
    Always returns True so it can be used inline in the branch condition.
    """
    n = _hits.get(site, 0) + 1
    _hits[site] = n
    if n == 1:
        import sys

        print(f"[FLOOR_ABLATION] {site}: ablated branch taken", file=sys.stderr, flush=True)
    return True


def hits() -> Dict[str, int]:
    """Per-site ablated-branch counts, for end-of-run reporting."""
    return dict(_hits)


def zeros(shape, dtype, device):
    """A cached all-zero tensor, allocated and filled at most once per signature.

    Callers must treat the result as read-only and must not alias it into a
    buffer that something else writes, or two ablation sites would share state.
    """
    import torch

    key = (tuple(shape), dtype, str(device))
    t = _zeros_cache.get(key)
    if t is None:
        t = torch.zeros(*shape, dtype=dtype, device=device)
        _zeros_cache[key] = t
    return t


def zeros_like_cached(t):
    """``zeros`` keyed off an existing tensor's shape/dtype/device."""
    return zeros(t.shape, t.dtype, t.device)


_stale_cache: Dict[str, "object"] = {}


def remember(key: str, value):
    """Record a value observed on the real path, for a later ablated replay."""
    _stale_cache.setdefault(key, value)


def recall(key: str):
    """Value previously passed to ``remember``, or None if the real path never ran."""
    return _stale_cache.get(key)


_announced = False


def announce() -> Optional[str]:
    """Log which components are ablated, once, so a log can never be misread as a
    clean run. Returns the description, or None when nothing is ablated."""
    global _announced
    if not ANY_ABLATION:
        return None
    on = [
        name
        for name, flag in (
            ("nvls_dispatch", ABLATE_NVLS_DISPATCH),
            ("nvls_combine", ABLATE_NVLS_COMBINE),
            ("expert_gemm", ABLATE_EXPERT_GEMM),
            ("moe_sum", ABLATE_MOE_SUM),
            ("attn_core", ABLATE_ATTN_CORE),
        )
        if flag
    ]
    desc = "+".join(on)
    if not _announced:
        import sys

        print(
            f"[FLOOR_ABLATION] ACTIVE: {desc} -- output is numerically INVALID, "
            "timing only",
            file=sys.stderr,
            flush=True,
        )
        _announced = True
    return desc
