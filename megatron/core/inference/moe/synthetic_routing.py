# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed synthetic routing, to price EP load imbalance without moving expert weights.

The prize: expert popularity is uneven, experts are assigned to EP ranks in contiguous
blocks, and every MoE layer ends in a barrier, so the step is set by the busiest rank. A
127M-assignment histogram puts the contiguous blocks at +6.9%/-2.9%/+2.1%/-6.1% of the
mean, and in-situ timing shows the barrier absorbing exactly that (rank 0: 63.0 us GEMM +
28.2 us collectives; rank 1: 56.8 + 34.5; both 91.2). An equal-cardinality greedy
partition would take the worst rank to +0.1%, predicting ~2.8%.

Measuring that by permuting the router's gating weight **does not work, and the failure is
instructive**: it leaves the expert weights in place, so the numerics go bad, the logits
degenerate, routing collapses onto a handful of experts, and the run exceeds the per-rank
token capacity and hangs. The probe destroys the distribution it is trying to impose --
any measurement that corrupts numerics cannot rely on data-dependent routing.

So make routing not depend on the numerics at all. This module substitutes a fixed,
precomputed index tensor for the router's selection, with a chosen per-rank load split.
Both arms then have equally invalid numerics and differ *only* in the load distribution:

* ``skew``     reproduces the measured contiguous-block imbalance (+6.9/-2.9/+2.1/-6.1).
* ``balanced`` splits assignments evenly across the four rank blocks.

The difference in step time between those two arms is the prize, measured directly and
without redistributing a single expert weight.

The indices are generated once on the host and live in a fixed-address buffer, so CUDA
graph replay is safe and every step routes identically. That is a departure from reality
(real routing varies per step) but it is the point: it holds everything except the load
split constant.

Env: ``MCORE_SYNTH_ROUTING=skew|balanced``; default off.
"""

import os
import sys
from typing import Dict, Optional, Tuple

MODE: str = os.environ.get("MCORE_SYNTH_ROUTING", "")
ENABLED: bool = MODE in ("skew", "balanced")

# Measured contiguous-block shares, as fractions of the mean.
_SKEW: Tuple[float, ...] = (1.069, 0.971, 1.021, 0.939)

_cache: Dict[Tuple[int, int, int], "object"] = {}
_announced: bool = False


def _rank_shares(world: int) -> Tuple[float, ...]:
    if MODE == "balanced":
        return tuple(1.0 for _ in range(world))
    s = _SKEW[:world]
    return tuple(x * world / sum(s) for x in s)


def build_indices(num_tokens: int, topk: int, num_experts: int, world: int, device):
    """A fixed ``[num_tokens, topk]`` index tensor with the configured per-rank split.

    Assignments are dealt to rank blocks in proportion to the configured shares and spread
    uniformly *within* a block, so only the between-rank split differs across modes. The
    topk slots of a token are forced onto distinct experts, matching the real selection's
    contract, which the dispatcher's counting relies on.
    """
    key = (num_tokens, topk, num_experts)
    if key in _cache:
        return _cache[key]

    import torch

    per_rank = num_experts // world
    shares = _rank_shares(world)
    total = num_tokens * topk
    quota = [int(round(total * s / world)) for s in shares]
    quota[-1] = total - sum(quota[:-1])

    g = torch.Generator().manual_seed(1234)
    idx = torch.empty(num_tokens, topk, dtype=torch.int64)
    pool = []
    for r, q in enumerate(quota):
        base = r * per_rank
        pool.append(base + torch.randint(0, per_rank, (q,), generator=g))
    pool = torch.cat(pool)[torch.randperm(total, generator=g)]

    # Deal into rows, repairing duplicates within a row by walking the pool.
    flat = pool.tolist()
    cursor = 0
    for t in range(num_tokens):
        seen = set()
        for k in range(topk):
            v = flat[cursor % total]
            cursor += 1
            tries = 0
            while v in seen and tries < 4 * topk:
                v = flat[cursor % total]
                cursor += 1
                tries += 1
            seen.add(v)
            idx[t, k] = v
    out = idx.to(device)
    _cache[key] = out

    global _announced
    if not _announced and int(os.environ.get("RANK", "0")) == 0:
        _announced = True
        counts = [
            int(((out >= r * per_rank) & (out < (r + 1) * per_rank)).sum()) for r in range(world)
        ]
        mean = sum(counts) / world
        print(
            f"[SYNTHROUTE] mode={MODE} per-rank assignments: "
            + "  ".join(f"r{r}:{c} ({100 * c / mean - 100:+.1f}%)" for r, c in enumerate(counts))
            + "  (numerics intentionally invalid; timing measurement only)",
            file=sys.stderr,
            flush=True,
        )
    return out


def maybe_override(probs, indices, num_experts: int, world: int):
    """Replace ``indices`` with the fixed synthetic pattern. Probs are left alone."""
    if not ENABLED:
        return probs, indices
    fixed = build_indices(indices.shape[0], indices.shape[1], num_experts, world, indices.device)
    indices.copy_(fixed)
    return probs, indices
