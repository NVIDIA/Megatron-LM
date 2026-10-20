# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank-level load: what a dropless EP layer actually waits on.

For dropless MoE with expert parallelism, uneven expert routing becomes uneven work across EP
ranks, and the busiest rank sets the step time. Per-expert statistics alone cannot describe
that: the same hot experts can sit on one rank or be spread across several. This module defines
the rank-level quantities the benchmark reports and constructs routing that realizes a chosen
rank-load vector exactly.

Definitions
-----------
rank load
    ``L_r``: token-expert assignments whose expert lives on EP rank ``r``, counted after top-k
    expansion. Experts are placed contiguously, as in Megatron: rank ``r`` owns global experts
    ``[r * E / R, (r + 1) * E / R)``.
maxvio
    ``max_r L_r / mean_r L_r``. Exactly 1 for a balanced load; ``maxvio - 1`` is the relative
    overload of the busiest rank.
concentration
    ``1 - H(p) / log R`` with ``p_r = L_r / sum(L)``: 0 for a uniform load, 1 when one rank
    receives everything. It separates loads that share max/mean, e.g. ``[40, 20, 20, 20, 0]``
    (0.172) and ``[40, 15, 15, 15, 15]`` (0.065). Undefined for ``R == 1``.
hot-rank persistence
    Over consecutive steps, the fraction of transitions in which the busiest rank stays the
    same. 1 for a static hotspot, 0 when it moves every step.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import torch

# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def rank_loads(expert_counts: Sequence[int], ep_size: int) -> list[int]:
    """Per-EP-rank assignment counts from global per-expert counts (contiguous placement)."""
    num_experts = len(expert_counts)
    if ep_size < 1 or num_experts % ep_size != 0:
        raise ValueError(f"{num_experts} experts cannot be placed evenly on {ep_size} ranks")
    per_rank = num_experts // ep_size
    return [int(sum(expert_counts[r * per_rank:(r + 1) * per_rank])) for r in range(ep_size)]


def maxvio(loads: Sequence[int]) -> float | None:
    """max/mean; None (never a fake epsilon) when there is no load."""
    total = sum(loads)
    if not loads or total == 0:
        return None
    return max(loads) / (total / len(loads))


def concentration(loads: Sequence[int]) -> float | None:
    """1 - H(p)/log R. None when R == 1 (no distribution to speak of) or there is no load."""
    total = sum(loads)
    if len(loads) < 2 or total == 0:
        return None
    entropy = -sum((c / total) * math.log(c / total) for c in loads if c > 0)
    # Clamp the rounding residue of an exactly uniform load to 0.
    return max(0.0, 1.0 - entropy / math.log(len(loads)))


def hot_rank_persistence(per_step_loads: Sequence[Sequence[int]]) -> float | None:
    """Fraction of consecutive steps whose busiest rank is unchanged; None below two steps.

    Ties pick the lowest rank index, so a balanced step never counts as a moved hotspot
    merely because ``max`` is not unique.
    """
    if len(per_step_loads) < 2:
        return None
    hot = [max(range(len(loads)), key=lambda r: (loads[r], -r)) for loads in per_step_loads]
    same = sum(1 for a, b in zip(hot, hot[1:]) if a == b)
    return same / (len(hot) - 1)


def rank_load_summary(loads: Sequence[int]) -> dict[str, Any]:
    return {
        "rank_loads": [int(c) for c in loads],
        "maxvio": maxvio(loads),
        "concentration": concentration(loads),
    }


# --------------------------------------------------------------------------------------
# rank-target workloads: constructed for a given layout
# --------------------------------------------------------------------------------------

RANK_TARGET_WORKLOADS = (
    "R0_balanced",
    "R1_hot_rank_spread",
    "R2_hot_rank_concentrated",
)

RANK_TARGET_NOTES = {
    "R0_balanced": "every EP rank receives the same load (maxvio 1, concentration 0)",
    "R1_hot_rank_spread": "one rank at the target maxvio; the remaining load spread evenly over "
    "every other rank",
    "R2_hot_rank_concentrated": "same busiest-rank load as R1, but the remaining load sits at "
    "the mean on as few ranks as possible, leaving the rest empty: equal maxvio, higher "
    "concentration (needs EP >= 3 to differ from R1)",
}

SCHEDULES = ("static", "rotate", "random")


def _split_evenly(total: int, parts: int) -> list[int]:
    """`total` over `parts` slots, the remainder going to the lowest indices."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def target_rank_loads(workload: str, ep_size: int, total: int, hot_maxvio: float) -> list[int]:
    """Integer rank-load vector for rank 0 as the hot rank; sums to `total` exactly.

    R1 and R2 share the hot rank's integer load, so their realized maxvio is identical, not
    merely close.
    """
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    if workload == "R0_balanced" or ep_size == 1:
        return _split_evenly(total, ep_size)
    if not 1.0 <= hot_maxvio <= ep_size:
        raise ValueError(f"hot_maxvio {hot_maxvio} must lie in [1, ep_size={ep_size}]")
    mean = total / ep_size
    hot = min(total, int(round(hot_maxvio * mean)))
    rest = total - hot
    if workload == "R1_hot_rank_spread":
        return [hot] + _split_evenly(rest, ep_size - 1)
    if workload == "R2_hot_rank_concentrated":
        # Fill the other ranks at the (rounded-down) mean until the remainder runs out.
        cap = max(1, int(mean))
        loads = [hot]
        for _ in range(ep_size - 1):
            take = min(cap, rest)
            loads.append(take)
            rest -= take
        loads[1] += rest  # rounding residue, if any, stays on the first non-hot rank
        return loads
    raise ValueError(f"unknown rank-target workload {workload}")


def schedule_permutation(schedule: str, period: int, ep_size: int, step: int,
                         seed: int) -> list[int]:
    """Which rank carries load slot `i` at `step`: loads[perm[i]] = target[i].

    Only the assignment of loads to ranks changes with the step; the per-step load multiset
    (and therefore maxvio and concentration) is the same at every step. That separates the
    temporal dimension from the severity of the imbalance.
    """
    if schedule == "static":
        shift = 0
        return [(i + shift) % ep_size for i in range(ep_size)]
    if schedule == "rotate":
        if period < 1:
            raise ValueError("rotate needs a period >= 1")
        shift = step // period
        return [(i + shift) % ep_size for i in range(ep_size)]
    if schedule == "random":
        g = torch.Generator(device="cpu").manual_seed(
            int(hashlib.sha256(f"schedule|{seed}|{step}".encode()).hexdigest()[:8], 16)
        )
        return torch.randperm(ep_size, generator=g).tolist()
    raise ValueError(f"unknown schedule {schedule}")


def min_steps_for_schedule(schedule: str, period: int) -> int:
    """Smallest --steps at which the schedule is actually exercised."""
    if schedule == "static":
        return 1
    if schedule == "rotate":
        return period + 1
    return 2


def expert_counts_for_rank_loads(loads: Sequence[int], num_experts: int, tokens: int,
                                 topk: int) -> list[int]:
    """Spread each rank's load evenly over its local experts; checks realizability.

    A row holds `topk` distinct experts, so one expert can take at most `tokens` assignments
    and one rank at most `tokens * min(local_experts, topk)`.
    """
    ep_size = len(loads)
    if num_experts % ep_size != 0:
        raise ValueError(f"{num_experts} experts cannot be placed evenly on {ep_size} ranks")
    local = num_experts // ep_size
    if sum(loads) != tokens * topk:
        raise ValueError(f"rank loads sum to {sum(loads)}, expected tokens*topk={tokens * topk}")
    counts: list[int] = []
    for r, load in enumerate(loads):
        cap = tokens * min(local, topk)
        if load > cap:
            raise ValueError(
                f"rank {r} load {load} exceeds {cap}: with {local} local experts and "
                f"topk {topk}, {tokens} rows cannot send it that many distinct assignments"
            )
        counts.extend(_split_evenly(load, local))
    return counts


def routing_from_expert_counts(counts: Sequence[int], tokens: int, topk: int,
                               g: torch.Generator) -> torch.Tensor:
    """[tokens, topk] distinct expert ids whose per-expert histogram equals `counts` exactly.

    Slots are laid out expert-major and slot j goes to row j % tokens. Each expert's slots are
    contiguous and number at most `tokens`, so they land on distinct rows; every row receives
    exactly `topk` slots. Rows are then shuffled, which keeps every count.
    """
    if any(c < 0 or c > tokens for c in counts):
        raise ValueError("every expert count must lie in [0, tokens]")
    if sum(counts) != tokens * topk:
        raise ValueError(f"counts sum to {sum(counts)}, expected {tokens * topk}")
    slots = [e for e, c in enumerate(counts) for _ in range(c)]
    rows: list[list[int]] = [[] for _ in range(tokens)]
    for j, e in enumerate(slots):
        rows[j % tokens].append(e)
    ids = torch.tensor([sorted(r) for r in rows], dtype=torch.int64)
    return ids[torch.randperm(tokens, generator=g)]
