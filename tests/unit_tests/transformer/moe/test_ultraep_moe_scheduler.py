# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""UltraEP planner: contracts against a fake Manager, then real UltraEP on four GPUs."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import replica_expert_runtime
from megatron.core.transformer.moe import ultraep_moe_scheduler as ultraep
from megatron.core.transformer.moe.moe_scheduler import MoEScheduler, SchedulerContext
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = pytest.mark.launch_on_gb200

EP, LOCAL, SLOTS = 2, 2, 1  # physical layout per rank: [home, home, replica]


class FakeManager:
    """Round-robin rerouting over a settable placement, like UltraEP's single slot."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.destroyed = 0
        self.replicas = (0, 0)
        FakeManager.instances.append(self)

    def _p2l(self):
        homes = torch.arange(EP * LOCAL).view(EP, LOCAL)
        replicas = torch.tensor(self.replicas).view(EP, SLOTS)
        return torch.cat([homes, replicas], 1).flatten().int().cuda()

    def update_placement(self, slot, routing_map):
        assert slot == 0 and routing_map.dtype == torch.bool
        self.p2l = self._p2l()
        return self.p2l, None, None

    def reroute(self, slot, probs, routing_map):
        assert slot == 0
        physical = probs.new_zeros(probs.size(0), self.p2l.numel())
        routes = torch.zeros_like(physical, dtype=torch.bool)
        for token, expert in routing_map.nonzero().tolist():
            instances = (self.p2l == expert).nonzero().flatten().tolist()
            column = instances[token % len(instances)]
            routes[token, column] = True
            physical[token, column] = probs[token, expert]
        return physical, routes

    def destroy(self):
        self.destroyed += 1


@pytest.fixture
def fake_ultraep(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("The UltraEP planner requires CUDA.")
    FakeManager.instances = []
    monkeypatch.setitem(sys.modules, "ultra_ep", SimpleNamespace(Manager=FakeManager))
    monkeypatch.setattr(ultraep.dist, "get_world_size", lambda group: EP)
    monkeypatch.setattr(ultraep, "_managers", {})
    return FakeManager


def _context():
    return SchedulerContext(
        layer_number=1,
        num_logical_experts=EP * LOCAL,
        num_local_experts=LOCAL,
        local_expert_indices=tuple(range(LOCAL)),
        ep_size=EP,
        ep_rank=0,
        router_topk=2,
        training=True,
    )


def test_scheduler_factory_builds_ultraep_planner():
    group = object()
    scheduler = MoEScheduler.from_config(
        SimpleNamespace(
            moe_scheduler_planner_type="ultra_ep",
            moe_scheduler_expert_dispatcher_type="replica_peer_tma",
            moe_scheduler_num_idle_experts=EP * SLOTS,
            num_moe_experts=EP * LOCAL,
            expert_model_parallel_size=EP,
        ),
        SimpleNamespace(ep=group),
    )

    assert isinstance(scheduler.planner, ultraep.UltraEPLoadPlanner)
    assert scheduler.planner.group is group


def test_planner_returns_replica_sources_and_snapshot_backward(fake_ultraep):
    planner, context = ultraep.UltraEPLoadPlanner(EP * SLOTS, object()), _context()
    routing_map = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.bool, device="cuda"
    )
    probs = (torch.rand(4, 4, device="cuda") * routing_map).requires_grad_()
    manager = ultraep._get_manager(planner.group, LOCAL, SLOTS)
    manager.replicas = (0, 3)

    home, sources, placement = planner.update_placement(probs, routing_map, context)
    assert home is None and sources.tolist() == [[0], [3]]
    routes, physical = planner.reroute(probs, routing_map, placement, context)
    assert routes.shape == (4, EP * (LOCAL + SLOTS))
    assert torch.equal(routes.sum(1), routing_map.sum(1))

    # A later layer overwrites the single slot; backward must still invert this placement.
    manager.replicas = (1, 2)
    planner.update_placement(probs, routing_map, context)
    upstream = torch.linspace(0.1, 0.9, physical.numel(), device="cuda").view_as(physical)
    (physical * upstream).sum().backward()

    p2l = [0, 1, 0, 2, 3, 3]
    expected = torch.zeros(4, 4, device="cuda")
    for token, column in routes.nonzero().tolist():
        expected[token, p2l[column]] += upstream[token, column]
    torch.testing.assert_close(probs.grad, expected, rtol=0, atol=0)
    assert manager.kwargs["num_local_redundant_experts"] == SLOTS


def test_planner_requires_matching_reroute(fake_ultraep):
    planner, context = ultraep.UltraEPLoadPlanner(EP * SLOTS, object()), _context()
    routing_map = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    probs = torch.rand(2, 4, device="cuda")
    _, _, placement = planner.update_placement(probs, routing_map, context)
    with pytest.raises(RuntimeError, match="rerouted before planning again"):
        planner.update_placement(probs, routing_map, context)
    with pytest.raises(ValueError, match="matching pending placement"):
        planner.reroute(probs, routing_map, object(), context)
    planner.reroute(probs, routing_map, placement, context)
    with pytest.raises(ValueError, match="FP32 probs"):
        planner.update_placement(probs.bfloat16(), routing_map, context)


def test_one_manager_per_group_and_collective_finalize(fake_ultraep, monkeypatch):
    group, other = object(), object()
    first = ultraep._get_manager(group, LOCAL, SLOTS)
    assert ultraep._get_manager(group, LOCAL, SLOTS) is first
    with pytest.raises(ValueError, match="share expert counts"):
        ultraep._get_manager(group, LOCAL, SLOTS + 1)
    second = ultraep._get_manager(other, LOCAL, SLOTS)
    assert second is not first and first.kwargs["is_train"] is False

    # The replica-runtime teardown releases planner Managers before process groups go away.
    monkeypatch.setattr(replica_expert_runtime, "_replica_expert_runtimes", [])
    monkeypatch.setattr(replica_expert_runtime, "finalize_replica_weight_transports", lambda: None)
    replica_expert_runtime.finalize_replica_expert_runtimes()
    assert (first.destroyed, second.destroyed) == (1, 1)
    ultraep.finalize_ultraep_planner_managers()
    assert (first.destroyed, second.destroyed) == (1, 1)


def test_missing_ultraep_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultra_ep", None)
    monkeypatch.setattr(ultraep, "_managers", {})
    with pytest.raises(ImportError, match="requires UltraEP"):
        ultraep._get_manager(object(), LOCAL, SLOTS)


# Real UltraEP on four GPUs.


@pytest.fixture
def real_ultraep(distributed_world):
    pytest.importorskip("ultra_ep")
    yield distributed_world
    ultraep.finalize_ultraep_planner_managers()


def test_real_planner_and_probability_backward_replay(real_ultraep):
    """Replay hot-expert placement and nonuniform probability gradients with live EP."""
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    group = real_ultraep
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    experts, tokens = size * 4, 1025
    context = SchedulerContext(
        layer_number=1,
        num_logical_experts=experts,
        num_local_experts=4,
        local_expert_indices=tuple(range(4 * rank, 4 * rank + 4)),
        ep_size=size,
        ep_rank=rank,
        router_topk=2,
        training=True,
    )
    planner = ultraep.UltraEPLoadPlanner(size, group)
    ids = torch.stack(
        [
            torch.zeros(tokens, device="cuda", dtype=torch.long),
            1 + (torch.arange(tokens, device="cuda") + rank) % (experts - 1),
        ],
        dim=1,
    )
    routes = torch.zeros(tokens, experts, device="cuda", dtype=torch.bool).scatter(1, ids, True)
    probabilities = (
        torch.linspace(0.01, 0.99, tokens * experts, device="cuda")
        .view(tokens, experts)
        .requires_grad_()
    )

    def replay(probs):
        _, copies, placement = planner.update_placement(probs, routes, context)
        mapping, physical = planner.reroute(probs, routes, placement, context)
        upstream = torch.linspace(0.1, 0.9, physical.numel(), device="cuda").view_as(physical)
        return copies, mapping, physical, (physical * upstream).sum()

    outputs, gradients = assert_replays_bit_exact(
        replay, (probabilities,), contention=True, what="ultra_ep placement/reroute"
    )
    assert torch.any(outputs["out[0]"] == 0), "The hot expert 0 was not replicated"
    assert any(t.numel() and torch.any(t != 0) for t in gradients.values())


@pytest.mark.parametrize("backend", ["peer_tma", "nccl"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_real_layer_matches_baseline_and_replays(real_ultraep, monkeypatch, backend, precision):
    """Real router/HybridEP/grouped experts and Adam: match the scheduler-off layer, replay exactly."""
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
    from tests.unit_tests.transformer.moe import test_scheduler_bf16_layer_parity as parity
    from tests.unit_tests.transformer.moe.scheduler_precision_parity_utils import run_leg

    if dist.get_world_size() != 4:
        pytest.skip("The production-layer recipe requires four GPUs")
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    generator = torch.Generator(device="cuda").manual_seed(840 + dist.get_rank())
    inputs = [
        [
            torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    upstream = [
        [torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator) * 0.01 for x in xs]
        for xs in inputs
    ]
    case = ("ultra_ep", backend, "shared", "reuse")
    try:
        reference, _ = run_leg(precision, pg, inputs, upstream)
        actual, _ = run_leg(precision, pg, inputs, upstream, case)
        assert reference.keys() == actual.keys()
        failures = []
        for name in reference:
            if name.startswith("initial"):
                if not torch.equal(reference[name], actual[name]):
                    failures.append((name, "initialization differs"))
                continue
            metrics = parity._metrics(reference[name], actual[name])
            if not metrics["finite"] or metrics["relative_l2"] > (
                0.025 if "/master_delta/" in name else 0.01
            ):
                failures.append((name, metrics))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
        assert_replays_bit_exact(
            lambda: run_leg(precision, pg, inputs, upstream, case)[0],
            (),
            backward=False,
            contention=True,
            what=f"{precision}/ultra_ep/{backend}",
        )
    finally:
        Utils.destroy_model_parallel()
