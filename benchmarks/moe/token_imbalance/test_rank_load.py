# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the rank-level load metrics and the rank_target workloads.

CPU only. The rank_target workloads claim to realize a chosen per-rank load EXACTLY at a given
layout, and to vary severity, concentration and temporal persistence one at a time; each claim
is asserted here on the generated routing itself, not on a summary of it.
"""

from __future__ import annotations

import math

import pytest
import torch

import benchmark as B
import rank_load as RL

TOKENS = 128


def _realized_loads(workload, num_experts, topk, step, ep, **kw):
    ids, _ = B.routing_for_step(workload, num_experts, topk, TOKENS, step, 5, ep, **kw)
    hist = B.routing_map_from_ids(ids, num_experts).sum(dim=0).tolist()
    return ids, RL.rank_loads(hist, ep)


# ---------------------------------------------------------------------------- metrics


def test_rank_loads_follow_contiguous_expert_placement():
    assert RL.rank_loads([1, 2, 3, 4, 5, 6, 7, 8], 4) == [3, 7, 11, 15]
    assert RL.rank_loads([1, 2, 3, 4], 1) == [10]
    with pytest.raises(ValueError):
        RL.rank_loads([1, 2, 3], 2)


def test_maxvio_is_one_when_balanced_and_null_without_load():
    assert RL.maxvio([5, 5, 5]) == 1.0
    assert RL.maxvio([40, 20, 20, 20, 0]) == pytest.approx(2.0)
    assert RL.maxvio([0, 0]) is None


def test_concentration_separates_loads_with_equal_maxvio():
    """The review's example: same total, same max/mean, different concentration."""
    a, b = [40, 20, 20, 20, 0], [40, 15, 15, 15, 15]
    assert RL.maxvio(a) == RL.maxvio(b)
    ca, cb = RL.concentration(a), RL.concentration(b)
    assert ca == pytest.approx(1 - (-(0.4 * math.log(0.4) + 3 * 0.2 * math.log(0.2))) / math.log(5))
    assert ca == pytest.approx(0.172, abs=1e-3) and cb == pytest.approx(0.065, abs=1e-3)
    assert ca > cb


def test_concentration_bounds():
    assert RL.concentration([7, 7, 7, 7]) == 0.0
    assert RL.concentration([12, 0, 0, 0]) == pytest.approx(1.0)
    assert RL.concentration([12]) is None
    assert RL.concentration([0, 0]) is None


def test_hot_rank_persistence():
    static = [[4, 1, 1], [4, 1, 1], [4, 1, 1]]
    moving = [[4, 1, 1], [1, 4, 1], [1, 1, 4]]
    assert RL.hot_rank_persistence(static) == 1.0
    assert RL.hot_rank_persistence(moving) == 0.0
    assert RL.hot_rank_persistence([[2, 2], [2, 2]]) == 1.0, "a tie is not a moved hotspot"
    assert RL.hot_rank_persistence(static[:1]) is None


# --------------------------------------------------------------------------- targets


def test_target_loads_reproduce_the_review_example():
    assert RL.target_rank_loads("R1_hot_rank_spread", 5, 100, 2.0) == [40, 15, 15, 15, 15]
    assert RL.target_rank_loads("R2_hot_rank_concentrated", 5, 100, 2.0) == [40, 20, 20, 20, 0]
    assert RL.target_rank_loads("R0_balanced", 5, 100, 2.0) == [20] * 5


@pytest.mark.parametrize("ep", [3, 4, 8])
@pytest.mark.parametrize("hot_maxvio", [1.5, 2.0, 3.0])
def test_r1_and_r2_share_maxvio_and_differ_in_concentration(ep, hot_maxvio):
    total = TOKENS * 2
    r1 = RL.target_rank_loads("R1_hot_rank_spread", ep, total, hot_maxvio)
    r2 = RL.target_rank_loads("R2_hot_rank_concentrated", ep, total, hot_maxvio)
    assert sum(r1) == sum(r2) == total
    assert max(r1) == max(r2) == r1[0] == r2[0], "the hot rank's load must be identical"
    assert RL.maxvio(r1) == RL.maxvio(r2)
    if hot_maxvio < ep:
        assert RL.concentration(r2) > RL.concentration(r1)
    else:  # maxvio == ep puts the whole load on one rank: nothing left to arrange
        assert r1 == r2


def test_r1_and_r2_coincide_at_two_ranks():
    """With two ranks the hot rank's share fixes the other's; concentration has no freedom."""
    total = TOKENS * 2
    assert RL.target_rank_loads("R1_hot_rank_spread", 2, total, 1.5) == \
        RL.target_rank_loads("R2_hot_rank_concentrated", 2, total, 1.5)


def test_hot_maxvio_outside_the_feasible_range_is_rejected():
    with pytest.raises(ValueError):
        RL.target_rank_loads("R1_hot_rank_spread", 4, 256, 0.5)
    with pytest.raises(ValueError):
        RL.target_rank_loads("R1_hot_rank_spread", 4, 256, 5.0)


# ------------------------------------------------------------------------- schedules


@pytest.mark.parametrize("schedule", RL.SCHEDULES)
def test_schedules_only_move_loads_between_ranks(schedule):
    """Per-step load multiset is constant: time is varied without varying severity."""
    base = RL.target_rank_loads("R2_hot_rank_concentrated", 4, 256, 2.0)
    for step in range(8):
        loads = B.target_loads_for_step("R2_hot_rank_concentrated", 4, 256, step, 9,
                                        hot_maxvio=2.0, schedule=schedule, period=3)
        assert sorted(loads) == sorted(base)


def test_rotate_moves_the_hotspot_every_period():
    def hot(step):
        loads = B.target_loads_for_step("R1_hot_rank_spread", 4, 256, step, 9, hot_maxvio=2.0,
                                        schedule="rotate", period=3)
        return loads.index(max(loads))

    assert [hot(s) for s in range(7)] == [0, 0, 0, 1, 1, 1, 2]


def test_schedule_steps_needed():
    assert RL.min_steps_for_schedule("static", 3) == 1
    assert RL.min_steps_for_schedule("rotate", 3) == 4
    assert RL.min_steps_for_schedule("random", 3) == 2
    assert B.min_steps("R1_hot_rank_spread", "rotate", 2) == 3


# ------------------------------------------------------------------------ realization


def test_routing_realizes_expert_counts_exactly():
    g = torch.Generator().manual_seed(0)
    counts = [128, 0, 64, 64, 96, 32, 16, 112]
    ids = RL.routing_from_expert_counts(counts, TOKENS, 4, g)
    assert ids.shape == (TOKENS, 4)
    assert all(len(set(r.tolist())) == 4 for r in ids)
    assert B.routing_map_from_ids(ids, 8).sum(dim=0).tolist() == counts


def test_unrealizable_counts_are_rejected():
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        RL.routing_from_expert_counts([TOKENS + 1, TOKENS - 1], TOKENS, 1, g)
    with pytest.raises(ValueError, match="exceeds"):
        # one local expert per rank and topk 2: a rank can take at most TOKENS assignments
        RL.expert_counts_for_rank_loads([TOKENS + 2, TOKENS - 2], 2, TOKENS, 2)


@pytest.mark.parametrize("workload", RL.RANK_TARGET_WORKLOADS)
@pytest.mark.parametrize("ep, num_experts", [(1, 8), (2, 8), (4, 8), (4, 16), (8, 32)])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("schedule", RL.SCHEDULES)
def test_rank_target_workloads_hit_their_target_exactly(workload, ep, num_experts, topk,
                                                        schedule):
    hot_maxvio = min(2.0, float(ep))
    kw = dict(hot_maxvio=hot_maxvio, schedule=schedule, period=2)
    for step in range(5):
        ids, loads = _realized_loads(workload, num_experts, topk, step, ep, **kw)
        assert ids.shape == (TOKENS, topk)
        assert all(len(set(r.tolist())) == topk for r in ids), "repeated expert in a row"
        target = B.target_loads_for_step(workload, ep, TOKENS * topk, step, 5, **kw)
        assert loads == target, f"step {step}: realized {loads} != target {target}"


def test_rank_target_workloads_depend_on_the_layout_by_design():
    a, _ = B.routing_for_step("R1_hot_rank_spread", 8, 2, TOKENS, 0, 5, 2)
    b, _ = B.routing_for_step("R1_hot_rank_spread", 8, 2, TOKENS, 0, 5, 4)
    assert not torch.equal(a, b)


def test_rank_level_summary_over_steps():
    counts = [
        [6, 2, 2, 2],   # rank loads at EP=2: [8, 4]
        [2, 2, 6, 2],   # [4, 8]: the hotspot moved
        [2, 2, 6, 2],   # [4, 8]: it stayed
    ]
    s = B.rank_level_summary(counts, 2)
    assert s["rank_loads_per_step"] == [[8, 4], [4, 8], [4, 8]]
    assert s["maxvio_per_step"] == pytest.approx([4 / 3] * 3)
    assert s["hot_rank_persistence"] == 0.5
    assert s["maxvio_max"] == pytest.approx(4 / 3)


def test_fingerprint_covers_rank_target_parameters():
    base = dict(hot_maxvio=2.0, schedule="rotate", period=2)
    a = B.workload_fingerprint("R1_hot_rank_spread", 8, 2, TOKENS, 3, 1, 4, **base)
    b = B.workload_fingerprint("R1_hot_rank_spread", 8, 2, TOKENS, 3, 1, 4,
                               **{**base, "hot_maxvio": 3.0})
    c = B.workload_fingerprint("R1_hot_rank_spread", 8, 2, TOKENS, 3, 1, 4,
                               **{**base, "schedule": "random"})
    assert len({a, b, c}) == 3
