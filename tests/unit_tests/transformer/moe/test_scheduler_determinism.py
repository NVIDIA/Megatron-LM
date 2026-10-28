# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-rank bit-exact replay of scheduler planning, transport and training.

Import the determinism harness at execution time: its environment pinning must
not affect unrelated tests merely because the GB200 bucket collects this file.
Replay is distinct from baseline parity, which is checked by the other suites.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.echo_moe_scheduler import EchoLoadPlanner
from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner
from megatron.core.transformer.moe.moe_scheduler import SchedulerContext
from megatron.core.transformer.moe.moonep_moe_scheduler import (
    MoonEPLoadPlanner,
    finalize_moonep_planner_workspaces,
)
from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport
from megatron.core.transformer.moe.replica_peer_tma_transport import PeerTmaTransport
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaWeightLayout,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_replica_storage as storage
from tests.unit_tests.transformer.moe.scheduler_precision_parity_utils import run_leg
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize(
    "planner_name,compact",
    [("echo", False), ("eplb", False), ("moon_ep", False), ("moon_ep", True)],
)
def test_scheduler_planner_and_probability_backward_replay(
    distributed_world, planner_name, compact
):
    """Replay hot-expert placement and nonuniform probability gradients with live EP."""
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    group = distributed_world
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
        pg_collection=SimpleNamespace(ep=group),
    )
    factories = {
        "echo": lambda: EchoLoadPlanner(size, require_triton=True),
        "eplb": lambda: EPLBLoadPlanner(num_redundant_experts=size),
        "moon_ep": MoonEPLoadPlanner,
    }
    planner = factories[planner_name]()
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

    if compact:
        routes = ids
        probabilities = probabilities.detach().gather(1, ids).requires_grad_()

    def replay(probs):
        _, copies, placement = planner.update_placement(probs, routes, context)
        mapping, physical = planner.reroute(probs, routes, placement, context)
        if compact:
            assert mapping.shape == physical.shape == ids.shape
            assert physical is probs, "Compact rerouting must retain probability storage/autograd"
        # Distinct upstream values catch a stable forward with an incorrect inverse map.
        upstream = torch.linspace(0.1, 0.9, physical.numel(), device="cuda").view_as(physical)
        return copies, mapping, physical, (physical * upstream).sum()

    try:
        outputs, gradients = assert_replays_bit_exact(
            replay, (probabilities,), contention=True, what=f"{planner_name} placement/reroute"
        )
        assert gradients, "Probability backward was not exercised"
        assert any(t.numel() and torch.any(t != 0) for t in gradients.values())
        assert outputs, "Planner produced no replay observations"
    finally:
        finalize_moonep_planner_workspaces()


@pytest.mark.parametrize("backend", [PeerTmaTransport, NcclP2PTransport], ids=["peer_tma", "nccl"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("cancellation", [False, True])
def test_scheduler_replica_transport_replay(distributed_world, backend, precision, cancellation):
    """Reuse storage and replay weights, FP8 scales, and additive owner gradients."""
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    cfg = storage.config(distributed_world, "shared", "reuse", precision)
    transport = backend(cfg)
    table = storage.placement(cfg)
    # All ranks contribute to the same expert, stressing ordered reduction.
    table = replace(table, slot_to_expert=torch.zeros_like(table.slot_to_expert))
    plan = transport.prepare_plan(table)
    rank = dist.get_rank(cfg.group)
    layouts = (
        [ReplicaWeightLayout.ROWWISE, ReplicaWeightLayout.COLUMNWISE]
        if precision == "mxfp8"
        else [ReplicaWeightLayout.PLAIN]
    )
    sources = [storage.sources(cfg, layout, i * 5) for i, layout in enumerate(layouts)]
    native = tuple(torch.empty((2, *shape), device=cfg.device) for shape in cfg.member_shapes)
    destinations = tuple(
        ReplicaGradDestination(
            tuple(t), torch.tensor([x.data_ptr() for x in t], device=cfg.device, dtype=torch.int64)
        )
        for t in native
    )

    def replay():
        observed = {}
        for i, source in enumerate(sources):
            handle = transport.start_weight_sync(sources=source, plan=plan)
            transport.wait_weight_sync(handle)
            for projection in range(2):
                weights, grads = transport.projection_views(projection)
                for slot, weight in enumerate(weights):
                    if precision == "mxfp8":
                        value, scale = weight[2:4] if i else weight[:2]
                        observed[f"scale/{i}/{projection}/{slot}"] = scale.clone()
                        assert torch.all(scale == 120 + i * 5)
                    else:
                        value = weight
                    assert torch.all(value == projection * 20 + i * 5)
                    observed[f"weight/{i}/{projection}/{slot}"] = value.clone()
                native[projection].fill_(0.125)
                # Large/small opposite-sign contributions make changes in the
                # accumulation order observable; exact fractions alone cannot.
                contributions = (16777216.0, 1.0, -16777216.0, 0.25)
                grads.fill_(contributions[rank % 4] if cancellation else (rank + 1) / 8)
                handle = transport.start_grad_reduce(
                    native_grads=destinations, plan=plan, projections=(projection,)
                )
                transport.wait_grad_reduce(handle)
                expected = torch.full_like(native[projection], 0.125)
                if rank == 0:
                    expected[0] += cfg.world_size * (cfg.world_size + 1) / 8
                if not cancellation:
                    torch.testing.assert_close(native[projection], expected, rtol=0, atol=0)
                assert torch.isfinite(native[projection]).all()
                observed[f"gradient/{i}/{projection}"] = native[projection].clone()
        return observed

    try:
        assert_replays_bit_exact(replay, (), backward=False, contention=True, what=backend.__name__)
    finally:
        transport.destroy()


@pytest.mark.parametrize(
    "planner,backend",
    [
        ("echo", "peer_tma"),
        ("echo", "nccl"),
        ("moon_ep", "peer_tma"),
        ("moon_ep", "nccl"),
        ("eplb", "nccl"),
    ],
)
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_scheduler_layer_optimizer_replay(
    distributed_world, monkeypatch, planner, backend, precision
):
    """Replay real router/HybridEP/grouped experts, accumulation and two Adam updates."""
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    if dist.get_world_size() != 4:
        pytest.skip("The production-layer replay recipe requires four GPUs")
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

    def replay():
        observed, _ = run_leg(
            precision,
            pg,
            inputs,
            upstream,
            (planner, backend, "shared", "reuse"),
            routing="learned",
            trace_routing=True,
        )
        assert any("master_delta" in key for key in observed)
        assert any("wgrad_accum" in key for key in observed)
        assert any("routing_probs" in key for key in observed)
        return observed

    try:
        assert_replays_bit_exact(
            replay, (), backward=False, contention=True, what=f"{precision}/{planner}/{backend}"
        )
    finally:
        finalize_moonep_planner_workspaces()
        Utils.destroy_model_parallel()
