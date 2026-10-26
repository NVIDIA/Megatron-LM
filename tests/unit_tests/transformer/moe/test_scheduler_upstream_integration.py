# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ownership and routing contracts for the scheduler port onto upstream MoonEP."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.home_expert_training import prepare_home_expert_checkpoint_load
from megatron.core.transformer.moe.moe_scheduler import MoEScheduler, SchedulerContext
from megatron.core.transformer.moe.moe_scheduler_plan import MoonEPPlanner, VirtualExpertPlan
from megatron.core.transformer.moe.moonep_execution_runtime import MoonEPExecutionRuntime
from megatron.core.transformer.moe.moonep_moe_scheduler import (
    MoonEPLoadPlanner,
    MoonEPPlacementResult,
    padded_rank_capacity,
)
from megatron.core.transformer.moe.upstream_moon_expert_dispatch import (
    UpstreamMoonEPExpertDispatch,
    UpstreamMoonEPPlanner,
)
from megatron.core.transformer.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from tests.unit_tests.transformer.moe.test_home_only_expert_dispatch import _config
from tests.unit_tests.transformer.moe.test_scheduler_release_contracts import _checkpoint_args

pytestmark = pytest.mark.launch_on_gb200


def test_legacy_alias_and_capacity_share_upstream_runtime():
    assert VirtualExpertLoadBalancer is MoonEPExecutionRuntime
    assert VirtualExpertLoadBalancer.storages is MoonEPExecutionRuntime.storages
    runtime = object.__new__(MoonEPExecutionRuntime)
    runtime.ep_size, runtime.num_owned_experts, runtime.router_topk = 4, 2, 2
    for alignment in (1, 32, 128, 256):
        runtime._alignment = alignment
        for tokens in (0, 1, 13, 1025):
            capacity = runtime._compute_rank_capacity(tokens)
            assert capacity == padded_rank_capacity(tokens * 2, 8, 4, alignment)
            assert capacity == MoonEPPlanner.rank_capacity(tokens * 2, 8, 4, alignment)
            assert capacity % alignment == 0
            assert capacity >= tokens * 2


def test_scheduler_adapter_has_no_parameters_or_second_autograd_boundary():
    class Runtime:
        def bind_experts(self, experts):
            self.experts = experts

    runtime = Runtime()
    scheduler = MoEScheduler(UpstreamMoonEPPlanner(), UpstreamMoonEPExpertDispatch(runtime))
    experts = torch.nn.Linear(2, 2)
    scheduler.bind_experts(experts)
    assert runtime.experts is experts
    assert not list(scheduler.parameters()) and not scheduler.state_dict()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    for hook in (
        scheduler.wrap_layer_input,
        scheduler.before_token_dispatch,
        scheduler.after_token_dispatch,
        scheduler.before_token_combine,
        scheduler.after_token_combine,
    ):
        assert hook(x) is x
    scheduler.step()
    with pytest.raises(RuntimeError, match="owned by compact"):
        scheduler.expert_dispatch.dispatch(experts, None, None)


def test_mutually_exclusive_runtime_flags():
    with pytest.raises(ValueError, match="exactly one expert execution runtime"):
        _config(moe_virtual_expert_load_balance=True)


def test_compact_moon_reroute_retains_probability_graph_and_per_forward_ids():
    context = SchedulerContext(1, 4, 2, (0, 1), 2, 0, 2, True)
    planner = MoonEPLoadPlanner(2)
    ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64)
    probabilities = torch.tensor([[0.2, 0.8], [0.3, 0.7]], requires_grad=True)
    physical = torch.tensor([[0, 7], [1, 4]], dtype=torch.int16)
    placement = MoonEPPlacementResult(topk_indices=ids, virtual_experts=physical)
    mapped, scores = planner.reroute(probabilities, ids, placement, context)
    assert mapped is physical and scores is probabilities
    dy = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    (scores * dy).sum().backward()
    torch.testing.assert_close(probabilities.grad, dy, rtol=0, atol=0)
    first = VirtualExpertPlan(physical, torch.tensor([[3, -1], [0, -1]]))
    second = VirtualExpertPlan(physical.clone(), first.experts_to_copy.clone())
    first.started.add(1)
    first.push_in_flight = True
    assert not second.started and not second.push_in_flight
    assert first.ready is second.ready is None


def test_fresh_home_load_resets_authoritative_train_state(monkeypatch):
    from megatron.training import global_vars

    state = SimpleNamespace(
        iteration=40,
        consumed_train_samples=100,
        consumed_valid_samples=10,
        skipped_train_samples=3,
        num_floating_point_operations_so_far=1234.0,
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_TRAIN_STATE", state)
    target, source = _checkpoint_args(True), _checkpoint_args(False)
    assert prepare_home_expert_checkpoint_load(target, [], source)
    assert all(value == 0 for value in vars(state).values())


@pytest.mark.parametrize("fused", [False, True])
def test_scheduler_moon_router_compact_scores_match_dense_gradients(monkeypatch, fused):
    """Exercise the real router-to-planner contract, including TE's dense scores."""
    from dataclasses import replace

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import router as router_module
    from megatron.core.transformer.moe.router import TopKRouter
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.moe.test_virtual_experts import (
        _virtual_expert_hybridep_config,
    )

    monkeypatch.setattr(router_module, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=2)
    config = _virtual_expert_hybridep_config(
        moe_virtual_expert_load_balance=False,
        moe_enable_scheduler=True,
        moe_scheduler_planner_type="moon_ep",
        moe_scheduler_num_idle_experts=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_router_score_function="softmax",
        moe_router_fusion=fused,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        compact = TopKRouter(config, pg).cuda()
        dense = TopKRouter(
            replace(
                config,
                moe_enable_scheduler=False,
                moe_token_dispatcher_type="alltoall",
                moe_router_fusion=False,
            ),
            pg,
        ).cuda()
        torch.manual_seed(123)
        logits = torch.randn(16, 8, device="cuda", requires_grad=True)
        reference = logits.detach().clone().requires_grad_()
        scores, ids = compact.routing(logits.view(8, 2, 8))
        expected, mask = dense.routing(reference.view(8, 2, 8))
        assert ids.dtype == torch.int64 and scores.shape == ids.shape == (16, 2)
        context = SchedulerContext(1, 8, 4, (0, 1, 2, 3), 2, 0, 2, True)
        MoonEPLoadPlanner(4)._validate_inputs(scores, ids, context)
        torch.testing.assert_close(scores, expected.gather(1, ids))
        assert mask.gather(1, ids).all()
        dy = torch.randn_like(expected)
        (scores * dy.gather(1, ids)).sum().backward()
        (expected * dy).sum().backward()
        torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-4, atol=1e-6)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp_size", [1, 2])
def test_compact_dispatch_metadata_expansion_preserves_scores_and_gradients(tp_size):
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher.tp_size, dispatcher.ep_size, dispatcher.num_local_experts = tp_size, 2, 4
    ids = torch.tensor([[0, 5], [7, 2]], dtype=torch.int16)
    scores = torch.tensor([[0.2, 0.8], [0.3, 0.7]], requires_grad=True)
    routes, expanded = dispatcher._initialize_metadata(ids, scores)
    assert routes.shape == (2, 2 * tp_size)
    assert expanded.shape == (2, 2 * tp_size, 4)
    flat = expanded.reshape(2, -1)
    torch.testing.assert_close(flat.gather(1, routes.long()), scores.repeat_interleave(tp_size, 1))
    expanded.sum().backward()
    torch.testing.assert_close(scores.grad, torch.full_like(scores, tp_size))
