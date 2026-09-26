# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the token-imbalance workload generator and its invariants.

CPU only. These are the properties the whole benchmark rests on: if the generator lets a
malformed workload through, every measurement taken with it is meaningless, so they are tested
directly rather than only through the end-to-end arms.
"""

from __future__ import annotations

import pytest
import torch

import benchmark as B

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


@pytest.mark.parametrize("workload", B.WORKLOADS)
def test_generator_is_deterministic_and_layout_independent(workload):
    """Same (workload, seed, step) and same result at any EP size.

    This is what lets one logical workload be compared across execution methods, so it is
    asserted for every workload rather than assumed.
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

    W0_balanced is a pure round-robin and W4_source_ragged draws from the full expert set, so
    neither is a function of the step; asserting otherwise would be a false requirement.
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
