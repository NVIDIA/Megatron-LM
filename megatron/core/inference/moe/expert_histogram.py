# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-expert token-count histogram, for diagnosing EP rank load imbalance.

In-situ timing showed the barrier absorbing a static load imbalance: per layer, rank 0
spends 63.0 us in the expert GEMM and 28.2 us in the collectives while rank 1 spends 56.8
and 34.5 -- both summing to 91.2. The rank with less expert work waits correspondingly
longer at the barrier, so the critical path is set by the *busiest* rank and balancing the
experts across ranks is worth the difference between the max and the mean (~4.85 us/layer,
~233 us/step).

Acting on that needs to know which experts are hot. Experts are assigned to ranks in
contiguous blocks (rank r owns ``[r*E/W, (r+1)*E/W)``), so if popularity correlates with
index at all, the blocks are unbalanced. This module counts assignments per expert so a
balanced permutation can be computed offline.

The accumulate runs *inside* the CUDA graph -- Python executes only at capture, but the
recorded scatter-add replays every step and keeps accumulating into a fixed buffer, which
is exactly the desired behaviour. The dump has to happen from the host, so it is driven
from ``step_gpu_timing``'s periodic report rather than from here.

Counts are cumulative from capture, not per step. Only ratios between experts are
meaningful; do not read an absolute rate out of them.

Env-gated (``MCORE_EXPERT_HISTOGRAM``); default off, zero cost when off.
"""

# Reports go to stderr on purpose: this runs inside a multi-rank inference server
# whose logger configuration is the caller's, and a diagnostic must not depend on it.
# pylint: disable=bad-builtin

import os
import sys
from typing import Optional

ENABLED: bool = os.environ.get("MCORE_EXPERT_HISTOGRAM", "0") == "1"

_counts: Optional["object"] = None
_num_experts: int = 0


def record(indices) -> None:
    """Accumulate one selection tensor's expert assignments. No-op unless gated on.

    ``indices`` is ``[tokens, topk]`` of expert ids, possibly containing the ``-1``
    CUDA-graph padding sentinel, which is dropped by clamping into a scratch bin.
    """
    if not ENABLED:
        return
    import torch

    global _counts, _num_experts
    if _counts is None:
        return
    flat = indices.reshape(-1)
    # Sentinel rows land in the extra trailing bin, which the dump discards. Clamping is
    # cheaper than a boolean mask and keeps this to a single kernel.
    safe = torch.where(flat < 0, _num_experts, flat)
    _counts.scatter_add_(0, safe.to(torch.int64), torch.ones_like(safe, dtype=_counts.dtype))


def configure(num_experts: int, device) -> None:
    """Allocate the histogram once, before capture."""
    if not ENABLED:
        return
    import torch

    global _counts, _num_experts
    if _counts is not None:
        return
    _num_experts = num_experts
    # One extra bin absorbs the padding sentinel.
    _counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)


def dump() -> None:
    """Print the histogram and the per-rank totals implied by contiguous assignment."""
    if not ENABLED or _counts is None:
        return
    import torch

    c = _counts[:_num_experts].tolist()
    total = sum(c)
    if total == 0:
        return
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    world = int(os.environ.get("WORLD_SIZE", "4"))
    per = _num_experts // world
    loads = [sum(c[r * per : (r + 1) * per]) for r in range(world)]
    mean = total / world
    print(
        f"[EXPHIST] total {total} assignments over {_num_experts} experts\n"
        f"[EXPHIST] contiguous per-rank loads: "
        + "  ".join(f"r{r}:{l} ({100 * l / mean - 100:+.1f}%)" for r, l in enumerate(loads))
        + f"\n[EXPHIST] counts: {','.join(str(x) for x in c)}",
        file=sys.stderr,
        flush=True,
    )
