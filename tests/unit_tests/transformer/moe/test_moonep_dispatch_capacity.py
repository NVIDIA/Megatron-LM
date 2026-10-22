# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Planner-bound HybridEP capacity: padding, fallback, overflow and layer parity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.enums import Fp8Recipe
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import token_dispatcher as dispatch_module
from megatron.core.transformer.moe.moe_scheduler import MoEPlacementResult, MoEScheduler
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe import test_scheduler_bf16_layer_parity as parity
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = [pytest.mark.launch_on_gb200, pytest.mark.experimental]


def _manager(monkeypatch, factor=None, alignment=256):
    # Exercise the real metadata/dispatch methods without loading the GPU backend.
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_expert_rank_capacity_factor=factor,
        moe_hybridep_pad_variable_tokens=False,
        moe_router_topk=2,
        moe_flex_dispatcher_num_sms=4,
        moe_hybridep_num_blocks_permute=None,
        moe_hybridep_num_blocks_unpermute=None,
        moe_hybridep_num_sms_preprocessing=4,
        moe_permute_fusion_into_hybridep=False,
        fp8=None,
        fp4=None,
    )
    manager.num_experts = 16
    manager.num_local_experts = 4
    manager.group = None
    manager.router_topk = 2
    manager.capacity_factor = None
    manager.use_dense_routing_for_bool_map = False
    manager.dense_routing_supported = False
    manager.drop_and_pad = False
    manager.pad_multiple = None
    manager.moe_expert_rank_capacity_factor = factor
    manager.max_dispatch_routes_per_rank = None
    manager.max_dispatch_padded_tokens_per_rank = None
    manager.over_budget = torch.zeros(1, dtype=torch.bool)
    monkeypatch.setattr(dispatch_module, "get_align_size_for_quantization", lambda _: alignment)
    return manager


@pytest.mark.parametrize("alignment", [1, 32, 128, 256])
def test_padding_bound_covers_uneven_expert_segments(monkeypatch, alignment):
    manager = _manager(monkeypatch, alignment=alignment)
    generator = torch.Generator().manual_seed(10)
    for num_tokens in [1, 127, 128, 257, 8192]:
        routes = num_tokens * 2
        manager.max_dispatch_routes_per_rank = routes
        manager.setup_metadata(
            torch.zeros(num_tokens, 16, dtype=torch.bool), torch.zeros(num_tokens, 16)
        )
        capacity = manager.num_permuted_tokens
        assert manager._using_planner_capacity
        # Independent random integer partitions, including empty experts and
        # a fully concentrated rank, check the actual per-expert padding demand.
        partitions = [[routes, 0, 0, 0]]
        for _ in range(20):
            cuts = sorted(torch.randint(routes + 1, (3,), generator=generator).tolist())
            edges = [0, *cuts, routes]
            partitions.append([b - a for a, b in zip(edges, edges[1:])])
        for counts in partitions:
            demand = sum(((n + alignment - 1) // alignment) * alignment for n in counts)
            assert demand <= capacity
        assert capacity % alignment == 0


def test_missing_guarantee_and_explicit_retry_remain_dynamic(monkeypatch):
    routing = torch.zeros(256, 16, dtype=torch.bool)
    probs = torch.zeros(256, 16)
    manager = _manager(monkeypatch)
    manager.max_dispatch_routes_per_rank = 512
    manager.setup_metadata(routing, probs)
    assert manager.num_permuted_tokens is not None
    manager.max_dispatch_routes_per_rank = None
    manager.max_dispatch_padded_tokens_per_rank = None
    manager.setup_metadata(routing, probs)
    assert manager.num_permuted_tokens is None
    assert not manager._using_planner_capacity

    manager = _manager(monkeypatch, factor=1.0)
    manager.max_dispatch_routes_per_rank = 512
    manager.setup_metadata(routing, probs)
    assert manager.num_permuted_tokens == 512  # Explicit policy retains retry handling.
    assert not manager._using_planner_capacity
    manager.moe_expert_rank_capacity_factor = None  # PagedStashRunner fallback.
    manager.setup_metadata(routing, probs)
    assert manager.num_permuted_tokens is None
    assert not manager._using_planner_capacity


@pytest.mark.parametrize("overflow", [False, True])
def test_capacity_reaches_backend_and_overflow_fails_closed(monkeypatch, overflow):
    manager = _manager(monkeypatch)
    manager.max_dispatch_routes_per_rank = 512
    manager.setup_metadata(torch.zeros(256, 16, dtype=torch.bool), torch.zeros(256, 16))
    seen = []

    def dispatch(**kwargs):
        seen.append(kwargs["num_permuted_tokens"])
        return (
            kwargs["x"],
            kwargs["probs"],
            None,
            torch.tensor([256] * 4),
            (torch.tensor([int(overflow)]),),
        )

    monkeypatch.setattr(dispatch_module, "hybrid_ep_dispatch", dispatch)
    if overflow:
        with pytest.raises(RuntimeError, match="capacity guarantee"):
            manager.dispatch(torch.zeros(256, 8))
    else:
        manager.dispatch(torch.zeros(256, 8))
    assert seen == [1536]  # 512 routes + four independently padded segments.
    assert manager.over_budget.item() == overflow


def test_planner_failure_clears_previous_bound():
    def fail(*args, **kwargs):
        raise ValueError("invalid placement input")

    scheduler = MoEScheduler(
        planner=SimpleNamespace(update_placement=fail), expert_dispatch=SimpleNamespace()
    )
    scheduler.max_dispatch_routes_per_rank = 1024
    routing, probs = torch.zeros(4, 8, dtype=torch.bool), torch.zeros(4, 8)
    with pytest.raises(ValueError, match="invalid placement input"):
        scheduler.schedule(probs, routing, None, None)
    assert scheduler.max_dispatch_routes_per_rank is None
    assert MoEPlacementResult().max_dispatch_routes_per_rank is None


def test_real_hybridep_quantization_padding(distributed_world):
    """Fixed capacity includes independently rounded expert segments, not just routes."""
    assert dist.get_world_size() == 4
    config = replace(
        parity._config("replica_peer_tma"),
        fp8="e4m3",
        fp8_recipe=Fp8Recipe.mxfp8,
        fp8_param=True,
        moe_router_padding_for_quantization=True,
    )
    # Two destinations per token, four runtime experts per rank. Uneven expert
    # segments force padding while the rank totals remain exactly N * topk.
    for n in [128, 384]:
        ids = torch.arange(n, device="cuda")
        experts = torch.stack((ids % 16, (ids + 4) % 16), dim=1)
        routing = torch.zeros(n, 16, dtype=torch.bool, device="cuda").scatter_(1, experts, True)
        generator = torch.Generator(device="cuda").manual_seed(27 + dist.get_rank())
        initial = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        outputs = []
        for bound in [None, n * 2]:
            manager = _HybridEPManager(dist.group.WORLD, 4, 16, config)
            manager.max_dispatch_routes_per_rank = bound
            probs = routing.float().mul(0.5).requires_grad_()
            x = initial.detach().clone().requires_grad_()
            manager.setup_metadata(routing, probs)
            dispatched = manager.dispatch(x)
            padded_demand = int(manager.tokens_per_expert.sum().item())
            if bound is not None:
                assert manager._using_planner_capacity
                assert isinstance(manager.num_permuted_tokens, int)
                assert manager.num_permuted_tokens >= padded_demand > n * 2
            y = manager.combine(dispatched * manager.dispatched_probs[:, None].to(dispatched.dtype))
            y.float().square().sum().backward()
            assert not manager.over_budget.item()
            outputs.append((y.detach().clone(), x.grad.clone(), probs.grad.clone()))
        for reference, actual in zip(*outputs):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["replica_nccl", "replica_peer_tma"])
@pytest.mark.parametrize("microbatches", [1, 4])
def test_moonep_dynamic_vs_bounded_layer(distributed_world, monkeypatch, backend, microbatches):
    """Identical planner/weights/inputs: only the dispatch capacity policy changes."""
    assert dist.get_world_size() == 4
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    original_config = parity._config
    monkeypatch.setattr(
        parity,
        "_config",
        lambda backend, capacity=None: replace(
            original_config(backend, capacity),
            moe_scheduler_planner_type="moon_ep",
            moe_router_mock_maxvio=4,
        ),
    )
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    generator = torch.Generator(device="cuda").manual_seed(9000 + dist.get_rank())
    # The second shape crosses an alignment boundary; all ranks use the same
    # physical shape. Distinct backward inputs exercise accumulation/reuse.
    inputs = [
        torch.randn(
            256 + (i % 2) * 128, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for i in range(microbatches)
    ]
    upstream = [
        torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator) * 0.01
        for x in inputs
    ]
    original_set = dispatch_module.MoEFlexTokenDispatcher.set_dispatch_route_bound
    observed = []

    def record_bound(dispatcher, bound, **kwargs):
        observed.append(kwargs["padded_tokens"])
        return original_set(dispatcher, bound, **kwargs)

    try:
        with monkeypatch.context() as control:
            control.setattr(
                dispatch_module.MoEFlexTokenDispatcher,
                "set_dispatch_route_bound",
                lambda dispatcher, bound, **kwargs: original_set(dispatcher, None),
            )
            initial, reference, reference_dispatch = parity._run(
                backend, pg, None, inputs, upstream
            )
        monkeypatch.setattr(
            dispatch_module.MoEFlexTokenDispatcher, "set_dispatch_route_bound", record_bound
        )
        _, actual, actual_dispatch = parity._run(backend, pg, initial, inputs, upstream)
        from megatron.core.transformer.moe.moonep_moe_scheduler import padded_rank_capacity

        assert observed == [padded_rank_capacity(x.shape[0] * 2, 8, 4, 256) for x in inputs]
        assert not any(d["overflow"] for d in reference_dispatch + actual_dispatch)
        failures = []
        for name in reference:
            metrics = parity._metrics(reference[name], actual[name])
            if (
                not metrics["finite"]
                or metrics["cosine"] < 0.999
                or metrics["tensor_similarity"] < 0.999
            ):
                failures.append((name, metrics))
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, failures)
        assert not any(reports), reports
    finally:
        Utils.destroy_model_parallel()


def test_block_capacity_is_not_padded_twice(monkeypatch):
    manager = _manager(monkeypatch)
    manager.max_dispatch_routes_per_rank = 768
    manager.max_dispatch_padded_tokens_per_rank = 768
    manager.setup_metadata(torch.zeros(256, 16, dtype=torch.bool), torch.zeros(256, 16))
    assert manager.num_permuted_tokens == 768
    assert manager._using_planner_capacity
