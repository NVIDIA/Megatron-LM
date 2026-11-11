# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for the token-imbalance benchmark's arm registry and adapter boundary.

GPU qualification of an arm is a separate finite run (``compare_arms.py``); these tests pin
the plumbing that makes the comparison fair: every arm shares one configuration except for
its own method, unavailable arms say why, and the route and numerical checks are exact.
"""

import sys
from pathlib import Path

import pytest
import torch

# The benchmark is a set of scripts, not a package: import its modules from their directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks/moe/token_imbalance"))
import arms as A  # noqa: E402
from arm_adapter import canonical_routes, numerical_error  # noqa: E402
from compare_arms import parse_args, spread  # noqa: E402

GEOMETRY = A.Geometry(ep=4, experts=16, topk=4, tokens=256, hidden=256, ffn=512)
ALL_OK = {
    token: (True, "")
    for token in (A.HYBRIDEP, A.HYBRIDEP_DENSE_ROUTING, A.MOE_SCHEDULER, A.TE_OP_FUSER, A.ULTRAEP)
}


@pytest.mark.parametrize(
    "kwargs", [{"ep": 1}, {"ep": 65}, {"experts": 7}, {"topk": 9}, {"tokens": 0}, {"hidden": 127}]
)
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        A.Geometry(**kwargs).validate()


def test_full_and_compact_routes_preserve_gate_gradients():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    ids = torch.tensor([[3, 1], [2, 0]])
    dense = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)
    full_ids, full = canonical_routes(scores, dense, 2)
    compact_ids, compact = canonical_routes(scores.gather(1, ids), ids, 2)
    assert torch.equal(full_ids, compact_ids)
    torch.testing.assert_close(full, compact, rtol=0, atol=0)
    full_grad = torch.autograd.grad(full.sum(), scores, retain_graph=True)[0]
    compact_grad = torch.autograd.grad(compact.sum(), scores)[0]
    torch.testing.assert_close(full_grad, compact_grad, rtol=0, atol=0)


def test_invalid_dense_topk_rejected():
    with pytest.raises(AssertionError):
        canonical_routes(torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool), 2)


def test_numerical_gate_checks_zeros_nonfinite_and_relative_error():
    assert numerical_error(torch.zeros(2), torch.zeros(2))["relative_l2"] == 0
    assert numerical_error(torch.ones(2), torch.zeros(2))["relative_l2"] == float("inf")
    assert numerical_error(torch.tensor([float("nan")]), torch.ones(1))["relative_l2"] == float(
        "inf"
    )
    assert numerical_error(torch.full((2,), 1.01), torch.ones(2))["relative_l2"] == pytest.approx(
        0.01, abs=1e-6
    )


def test_registry_has_dev_and_scheduler_arms():
    assert set(A.DEFAULT_ARMS) == {"alltoall", "hybridep", "moonep"}
    for planner in ("echo", "eplb", "moon_ep", "ultra_ep"):
        arm = A.ARMS[f"scheduler_{planner}"]
        assert arm.family == "moe_scheduler"
        assert A.MOE_SCHEDULER in arm.requires and A.TE_OP_FUSER in arm.requires
    assert A.ULTRAEP in A.ARMS["scheduler_ultra_ep"].requires


@pytest.mark.parametrize("name", sorted(A.ARMS))
def test_every_arm_keeps_the_shared_configuration(name):
    """Only the arm's method differs: geometry, precision and experts are identical."""
    shared = A.base_config(GEOMETRY)
    config = A.config_for(A.ARMS[name], GEOMETRY)
    assert {k: config[k] for k in shared} == shared
    assert config["num_moe_experts"] == GEOMETRY.experts
    assert config["moe_enable_routing_replay"] is True


def test_arm_methods_are_selected_by_their_overrides():
    config = {name: A.config_for(arm, GEOMETRY) for name, arm in A.ARMS.items()}
    assert config["alltoall"]["moe_token_dispatcher_type"] == "alltoall"
    assert config["hybridep"]["moe_flex_dispatcher_backend"] == "hybridep"
    assert "moe_virtual_expert_load_balance" not in config["hybridep"]
    assert config["moonep"]["moe_virtual_expert_load_balance"] is True
    assert config["scheduler_echo"]["moe_scheduler_planner_type"] == "echo"
    assert config["scheduler_echo"]["moe_scheduler_num_idle_experts"] == GEOMETRY.ep
    # MoonEP's planner needs one replica slot per logical expert (#8500 config validation).
    assert config["scheduler_moon_ep"]["moe_scheduler_num_idle_experts"] == GEOMETRY.experts
    for name in ("scheduler_echo", "scheduler_eplb", "scheduler_ultra_ep"):
        assert config[name]["moe_scheduler_num_idle_experts"] % GEOMETRY.ep == 0


def test_te_op_fuser_applies_to_every_arm_alike():
    for arm in A.ARMS.values():
        assert A.config_for(arm, GEOMETRY, te_op_fuser=True)["use_transformer_engine_op_fuser"]
        assert not A.config_for(arm, GEOMETRY)["use_transformer_engine_op_fuser"]


def test_an_arm_cannot_override_shared_settings():
    rogue = A.Arm(name="rogue", family="dispatcher", overrides={"moe_router_topk": 1})
    with pytest.raises(ValueError, match="shared settings"):
        A.config_for(rogue, GEOMETRY)


def test_arm_validation_and_registration():
    with pytest.raises(ValueError):
        A.Arm(name="x", family="unknown", overrides={})
    with pytest.raises(ValueError):
        A.Arm(name="x", family="moe_scheduler", overrides={})
    with pytest.raises(ValueError):
        A.Arm(name="x", family="dispatcher", overrides={}, idle_experts="per_rank")
    with pytest.raises(ValueError):
        A.register(A.Arm(name="alltoall", family="dispatcher", overrides={}))
    assert A.register(A.ARMS["alltoall"]) is A.ARMS["alltoall"]
    with pytest.raises(ValueError):
        A.config_for(A.ARMS["scheduler_echo"], GEOMETRY, scheduler_transport="nvshmem")


def test_resolve_and_rotated_order():
    arms = A.resolve("alltoall, hybridep,moonep")
    assert [a.name for a in arms] == ["alltoall", "hybridep", "moonep"]
    for bad in ("", "alltoall,alltoall", "nope"):
        with pytest.raises(ValueError):
            A.resolve(bad)
    orders = [[a.name for a in A.run_order(arms, i)] for i in range(3)]
    assert [o[0] for o in orders] == ["alltoall", "hybridep", "moonep"]
    assert all(sorted(o) == sorted(orders[0]) for o in orders)
    assert A.run_order(arms, 2, "fixed") == arms
    with pytest.raises(ValueError):
        A.run_order(arms, 0, "random")


def test_availability_reports_every_unmet_requirement():
    probes = dict(ALL_OK)
    probes[A.MOE_SCHEDULER] = (False, "no scheduler")
    probes[A.ULTRAEP] = (False, "no ultra_ep")
    ok, reasons = A.availability(A.ARMS["scheduler_ultra_ep"], probes)
    assert not ok and len(reasons) == 2
    assert A.availability(A.ARMS["moonep"], probes) == (True, [])
    assert A.availability(A.ARMS["alltoall"], {}) == (True, [])
    ok, reasons = A.availability(A.ARMS["hybridep"], {})
    assert not ok and "not probed" in reasons[0]
    table = A.availability_table(A.resolve("alltoall,scheduler_echo"), probes)
    assert [row["available"] for row in table] == [True, False]


def test_probes_on_this_checkout():
    """Without #8500 the scheduler arms must report themselves unavailable, with the reason."""
    probes = A.probe_requirements(te_op_fuser=False)
    assert set(probes) == set(ALL_OK)
    assert probes[A.TE_OP_FUSER][0] is False
    has_scheduler = "moe_enable_scheduler" in A._transformer_config_fields()
    assert probes[A.MOE_SCHEDULER][0] is has_scheduler
    if not has_scheduler:
        ok, reasons = A.availability(A.ARMS["scheduler_eplb"], probes)
        assert not ok and any("moe_enable_scheduler" in r for r in reasons)


def test_compare_arms_arguments():
    args = parse_args(["--out", "x", "--arms", "alltoall,moonep", "--workloads", "W0_balanced"])
    assert [a.name for a in args.arm_list] == ["alltoall", "moonep"]
    assert (args.warmup, args.iters, args.order) == (2, 5, "rotate")
    assert args.te_op_fuser is True
    assert parse_args(["--out", "x", "--no-te-op-fuser"]).te_op_fuser is False
    for bad in (
        ["--arms", "alltoall,bogus"],
        ["--workloads", "W2"],
        ["--workloads", "W7_burst", "--steps", "2"],
        ["--iters", "0"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["--out", "x", *bad])


def test_spread_summarizes_median_and_range():
    s = spread([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["n"] == 5 and s["median_ms"] == 3.0
    assert (s["min_ms"], s["max_ms"]) == (1.0, 5.0)
    assert s["p10_ms"] == pytest.approx(1.4) and s["p90_ms"] == pytest.approx(4.6)
