# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the token-imbalance workload generator and its invariants.

CPU only. These are the properties the whole benchmark rests on: if the generator lets a
malformed workload through, every measurement taken with it is meaningless, so they are tested
directly rather than only through the end-to-end arms.
"""

from __future__ import annotations

import benchmark as B
import pytest
import rank_load as RL
import torch

NUM_EXPERTS = 8
TOKENS = 128


@pytest.mark.parametrize("workload", B.WORKLOADS)
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_every_workload_satisfies_the_routing_contract(workload, topk):
    """Exactly `topk` DISTINCT experts per row, all ids in range."""
    ids, gates = B.routing_for_step(workload, NUM_EXPERTS, topk, TOKENS, 0, 1234, 2)
    assert ids.shape == (TOKENS, topk)
    assert int(ids.min()) >= 0 and int(ids.max()) < NUM_EXPERTS
    for row in ids:
        assert len(set(row.tolist())) == topk, f"{workload}: repeated expert in a top-k row"


@pytest.mark.parametrize("workload", B.WORKLOADS)
def test_gates_are_a_valid_distribution(workload):
    _, gates = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 7, 2)
    assert bool(torch.isfinite(gates).all())
    assert bool((gates >= 0).all())
    torch.testing.assert_close(gates.sum(dim=1), torch.ones(TOKENS))


@pytest.mark.parametrize("workload", B.TRACE_WORKLOADS)
def test_generator_is_deterministic_and_layout_independent(workload):
    """Same (workload, seed, step) and same result at any EP size.

    This is what lets one logical trace be compared across execution methods, so it is
    asserted for every trace_replay workload rather than assumed. rank_target workloads are
    built for a layout and are tested in test_rank_load.py instead.
    """
    a_ids, a_gates = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 99, 1)
    b_ids, b_gates = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 99, 1)
    c_ids, _ = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 99, 4)
    assert torch.equal(a_ids, b_ids) and torch.equal(a_gates, b_gates)
    assert torch.equal(a_ids, c_ids), f"{workload}: EP size changed the logical workload"


@pytest.mark.parametrize("workload", [w for w in B.WORKLOADS if w != "W0_balanced"])
def test_a_different_seed_or_step_changes_the_workload(workload):
    a_ids, _ = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 1, 2)
    b_ids, _ = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 2, 2)
    assert not torch.equal(a_ids, b_ids), f"{workload}: seed had no effect"


@pytest.mark.parametrize("workload", ["W6_rotating_hotspot", "W7_burst"])
def test_a_different_step_changes_the_workload(workload):
    """Only the per-step workloads are expected to vary with the step.

    W0_balanced is a pure round-robin and W4_random_balanced draws from the full expert set,
    so neither is a function of the step; asserting otherwise would be a false requirement.
    """
    a_ids, _ = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 0, 1, 2)
    c_ids, _ = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, 1, 1, 2)
    assert not torch.equal(a_ids, c_ids), f"{workload}: step had no effect"


def test_unknown_workload_is_rejected():
    with pytest.raises(ValueError, match="unknown workload"):
        B.routing_for_step("W9_does_not_exist", NUM_EXPERTS, 2, TOKENS, 0, 1, 2)


def test_routing_map_matches_the_ids():
    ids, _ = B.routing_for_step("W1_hot_expert_set", NUM_EXPERTS, 2, TOKENS, 0, 5, 2)
    rmap = B.routing_map_from_ids(ids, NUM_EXPERTS)
    assert rmap.shape == (TOKENS, NUM_EXPERTS)
    assert rmap.dtype == torch.bool
    assert int(rmap.sum()) == TOKENS * 2
    for row in range(TOKENS):
        assert sorted(rmap[row].nonzero().flatten().tolist()) == sorted(ids[row].tolist())


def test_fingerprint_is_stable_and_seed_sensitive():
    a = B.workload_fingerprint("W6_rotating_hotspot", NUM_EXPERTS, 2, TOKENS, 4, 11, 2)
    b = B.workload_fingerprint("W6_rotating_hotspot", NUM_EXPERTS, 2, TOKENS, 4, 11, 2)
    c = B.workload_fingerprint("W6_rotating_hotspot", NUM_EXPERTS, 2, TOKENS, 4, 12, 2)
    d = B.workload_fingerprint("W6_rotating_hotspot", NUM_EXPERTS, 2, TOKENS, 4, 11, 4)
    assert a == b
    assert a != c, "seed must affect the checksum"
    assert a != d, "EP size must affect the checksum (it changes the stored tensors)"


def test_probs_matrix_matches_the_gates():
    ids, gates = B.routing_for_step("W3_spread_hot_experts", NUM_EXPERTS, 2, TOKENS, 0, 3, 2)
    full = B.probs_matrix_from_gates(ids, gates, NUM_EXPERTS, torch.float32)
    assert full.shape == (TOKENS, NUM_EXPERTS)
    # exactly topk non-zero entries per row: unrouted entries are zero, not merely small
    assert int((full > 0).sum()) == TOKENS * 2
    for row in range(TOKENS):
        assert float(full[row].sum()) == pytest.approx(float(gates[row].sum()), abs=1e-6)
        assert int((full[row] > 0).sum()) == ids.shape[1]


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([1, 1, 1, 1], 1.0),
        # max=2, mean=1, population std=sqrt(0.5) -> CV = sqrt(0.5)
        ([2, 1, 1, 0], 2.0),
        ([0, 0, 0, 0], None),
    ],
)
def test_imbalance_metrics(counts, expected):
    m = B.imbalance_metrics(counts)
    if expected is None:
        assert m["max_over_mean"] is None
        assert m["reason"] == "zero_assignments"
    else:
        assert m["max_over_mean"] == pytest.approx(expected)
        assert m["std_definition"] == "population"


def test_per_step_workloads_are_non_empty_at_every_step():
    """W6/W7 vary per step; an empty step would make the arm meaningless."""
    on_step = ("W6_rotating_hotspot", "W7_burst")
    for workload in on_step:
        for step in range(6):
            ids, gates = B.routing_for_step(workload, NUM_EXPERTS, 2, TOKENS, step, 4, 4)
            assert ids.shape[0] == TOKENS, f"{workload} step {step}: empty workload"
            assert bool(torch.isfinite(gates).all())


def _histogram(ids: torch.Tensor) -> list[int]:
    return B.routing_map_from_ids(ids, NUM_EXPERTS).sum(dim=0).tolist()


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_w3_keeps_w1_expert_heat_and_moves_only_placement(topk):
    """W3 is W1 relabeled: identical per-expert counts as a multiset, different rank loads."""
    w1, _ = B.routing_for_step("W1_hot_expert_set", NUM_EXPERTS, topk, TOKENS, 0, 21, 2)
    w3, _ = B.routing_for_step("W3_spread_hot_experts", NUM_EXPERTS, topk, TOKENS, 0, 21, 2)
    h1, h3 = _histogram(w1), _histogram(w3)
    assert sorted(h1) == sorted(h3), "W3 must not change per-expert heat"
    for ep in (2, 4):
        assert RL.rank_loads(h1, ep) != RL.rank_loads(h3, ep), f"EP={ep}: placement unchanged"
        assert RL.maxvio(RL.rank_loads(h3, ep)) < RL.maxvio(RL.rank_loads(h1, ep))


def test_w1_puts_its_hot_block_on_rank_0():
    ids, _ = B.routing_for_step("W1_hot_expert_set", NUM_EXPERTS, 2, TOKENS, 0, 3, 4)
    loads = RL.rank_loads(_histogram(ids), 4)
    assert loads[0] == max(loads) and loads[0] > 2 * (sum(loads) / 4)


def test_w5_leaves_the_lower_half_of_the_experts_empty():
    ids, _ = B.routing_for_step("W5_zero_expert_rank", NUM_EXPERTS, 2, TOKENS, 0, 3, 2)
    assert RL.rank_loads(_histogram(ids), 2) == [0, TOKENS * 2]


@pytest.mark.parametrize(
    "workload, need", [("W0_balanced", 1), ("W6_rotating_hotspot", 4), ("W7_burst", 5)]
)
def test_per_step_trace_workloads_declare_the_steps_they_need(workload, need):
    """With fewer steps a timed iteration replays only the first step(s) of the pattern."""
    assert B.min_steps(workload) == need


def test_w6_hotspot_moves_after_three_steps():
    def hot_rank(step):
        ids, _ = B.routing_for_step("W6_rotating_hotspot", NUM_EXPERTS, 2, TOKENS, step, 3, 4)
        loads = RL.rank_loads(_histogram(ids), 4)
        return loads.index(max(loads))

    assert hot_rank(0) == hot_rank(1) == hot_rank(2)
    assert hot_rank(3) != hot_rank(0)


def test_w7_bursts_only_at_step_3():
    def maxvio(step):
        ids, _ = B.routing_for_step("W7_burst", NUM_EXPERTS, 2, TOKENS, step, 3, 4)
        return RL.maxvio(RL.rank_loads(_histogram(ids), 4))

    assert maxvio(3) > max(maxvio(0), maxvio(1), maxvio(2), maxvio(4))


def test_retired_workload_ids_are_rejected():
    """v1's W2_hot_rank was the same construction as W1 and W4_source_ragged did not vary the
    source token count; both ids are gone rather than silently aliased."""
    for retired in ("W2_hot_rank", "W4_source_ragged"):
        with pytest.raises(ValueError, match="unknown workload"):
            B.routing_for_step(retired, NUM_EXPERTS, 2, TOKENS, 0, 1, 2)
