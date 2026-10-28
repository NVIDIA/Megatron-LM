# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Zero-replica native execution, NCCL home migration and optimizer continuity."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner
from megatron.core.transformer.moe.home_only_expert_dispatch import HomeOnlyExpertDispatch
from megatron.core.transformer.moe.moe_scheduler import MoEScheduler, SchedulerContext
from megatron.core.transformer.moe.replica_nccl_transport import _NcclWorkspace
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.moe import test_home_expert_exchange as home_tests
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world, ep_group

pytestmark = pytest.mark.launch_on_gb200


def test_home_only_workspace_does_not_allocate_replica_storage(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Home-only workspace allocated replica or native-gradient storage")

    stream = object()
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "empty_like", unexpected)
    for weight_format in ("bf16", "mxfp8"):
        workspace = _NcclWorkspace(
            SimpleNamespace(
                weight_format=weight_format,
                member_shapes=((4, 3), (3, 4)),
                num_local_replica_slots=0,
                device=torch.device("cpu"),
            )
        )
        assert workspace._weights == workspace._columnwise_weights == ()
        assert workspace._replica_grads == ()
        assert not hasattr(workspace, "_native_grads") and workspace._scales == {}
        assert workspace._stream is stream


def _config(**overrides):
    args = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_ffn_hidden_size=128,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_enable_scheduler=True,
        moe_scheduler_planner_type="eplb",
        moe_scheduler_expert_dispatcher_type="replica_nccl",
        moe_scheduler_num_idle_experts=0,
        moe_scheduler_home_update_interval=1,
    )
    args.update(overrides)
    return TransformerConfig(**args)


@pytest.mark.parametrize("fp8", [False, True])
def test_home_only_config_and_factory(fp8):
    config = _config(**(dict(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True) if fp8 else {}))
    scheduler = MoEScheduler.from_config(config, SimpleNamespace(ep=object()))
    assert isinstance(scheduler.expert_dispatch, HomeOnlyExpertDispatch)
    assert scheduler.home_expert_dispatch is not None
    assert list(scheduler.expert_dispatch.parameters()) == []
    for overrides in (
        dict(moe_scheduler_home_update_interval=0),
        dict(moe_scheduler_planner_type="echo"),
        dict(moe_scheduler_expert_dispatcher_type="replica_peer_tma"),
    ):
        with pytest.raises(ValueError):
            _config(**overrides)


def test_native_backward_lifetime_and_gradient():
    dispatcher = HomeOnlyExpertDispatch(config=_config(), pg_collection=SimpleNamespace(ep=None))
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
    first = dispatcher.after_token_combine(x * weight)
    second = dispatcher.after_token_combine(x * weight * 2)
    with pytest.raises(RuntimeError, match="native forwards/backwards"):
        dispatcher.assert_idle()
    first.sum().backward()
    with pytest.raises(RuntimeError, match="native forwards/backwards"):
        dispatcher.assert_idle()
    second.sum().backward()
    dispatcher.assert_idle()
    torch.testing.assert_close(x.grad, weight.detach() * 3)
    torch.testing.assert_close(weight.grad, x.detach() * 3)
    with torch.no_grad():
        assert dispatcher.after_token_combine(x) is x
    assert dispatcher.wrap_layer_input(x) is x
    assert dispatcher.before_token_dispatch(x) is x
    assert dispatcher.after_token_dispatch(x) is x
    assert dispatcher.before_token_combine(x) is x
    with pytest.raises(RuntimeError, match="cannot dispatch"):
        dispatcher.dispatch(None, torch.empty(1, 0), None)


def test_home_only_planner_commit_checkpoint_and_routing(monkeypatch):
    context = SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=1,
        training=True,
        pg_collection=SimpleNamespace(ep=object()),
    )
    counts = torch.tensor([[8, 6, 2, 1], [8, 6, 2, 1]])
    monkeypatch.setattr(dist, 'all_reduce', lambda tensor, group: tensor.copy_(counts.sum(0)))
    planner = EPLBLoadPlanner(0, home_update_interval=1, num_logical_experts=4)
    routes = torch.eye(4, dtype=torch.bool)
    probs = routes.float().requires_grad_()
    home, replicas, placement = planner.update_placement(probs, routes, context)
    assert home is None and replicas is None
    planner.step()
    home, replicas, old_placement = planner.update_placement(probs, routes, context)
    assert home is not None and replicas is None
    assert planner.active_home_layout.flatten().tolist() == [0, 1, 2, 3]
    old_routes, _ = planner.reroute(probs, routes, old_placement, context)
    assert torch.equal(old_routes, routes)
    # Until the successful-step completion, a second forward uses old homes.
    _, _, pending_placement = planner.update_placement(probs, routes, context)
    assert torch.equal(planner.reroute(probs, routes, pending_placement, context)[0], routes)
    # A resume must retain the pending permutation even with a [EP, 0] layout.
    restored = EPLBLoadPlanner(0, home_update_interval=1, num_logical_experts=4)
    restored.load_state_dict(planner.state_dict())
    replay, replicas, _ = restored.update_placement(probs, routes, context)
    assert replicas is None
    assert torch.equal(replay.source_slots, home.source_slots)
    restored.step(replay.version)
    planner.step(home.version)
    torch.testing.assert_close(restored.active_home_layout, planner.active_home_layout)
    _, replicas, placement = planner.update_placement(probs, routes, context)
    assert replicas is None
    physical_routes, physical_probs = planner.reroute(probs, routes, placement, context)
    layout = planner.active_home_layout.flatten()
    assert not torch.equal(layout, torch.arange(4))
    torch.testing.assert_close(physical_routes, routes[:, layout])
    coefficients = torch.arange(4).float()
    (physical_probs * coefficients).sum().backward()
    torch.testing.assert_close(probs.grad, routes.float() * coefficients[torch.argsort(layout)])


class _NativeExperts(torch.nn.Module):
    """Independent small native linear experts with TE-compatible parameter names."""

    def __init__(self, logical_experts):
        super().__init__()
        for projection, shape in enumerate(((4, 3), (3, 4)), 1):
            linear = torch.nn.Module()
            linear.out_features, linear.in_features = shape
            for slot, logical in enumerate(logical_experts):
                values = torch.arange(12, device='cuda', dtype=torch.float32).reshape(shape)
                linear.register_parameter(
                    f'weight{slot}',
                    torch.nn.Parameter(((values + 1 + logical) / (40 * projection)).bfloat16()),
                )
            setattr(self, f'linear_fc{projection}', linear)
        self._replica_expert_runtime = None

    def forward(self, x, probs):
        outputs = []
        for slot in range(probs.shape[1]):
            hidden = F.silu(F.linear(x, self.linear_fc1.get_parameter(f'weight{slot}')))
            outputs.append(
                F.linear(hidden, self.linear_fc2.get_parameter(f'weight{slot}')).float()
                * probs[:, slot : slot + 1]
            )
        return torch.stack(outputs).sum(0)


def test_home_only_native_adam_and_rerouting(ep_group):
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    e = size * 2
    if size == 1:
        with pytest.raises(ValueError, match="expert_model_parallel_size > 1"):
            _config(num_moe_experts=e, expert_model_parallel_size=size)
        return
    config = _config(num_moe_experts=e, expert_model_parallel_size=size)
    scheduler = MoEScheduler.from_config(config, SimpleNamespace(ep=ep_group))
    native = _NativeExperts(range(rank * 2, rank * 2 + 2))
    reference = _NativeExperts(range(e))
    native_parameters = tuple(native.parameters())
    addresses = [p.data_ptr() for p in native_parameters]
    # Both allocation sources must remain unreachable on the home-only path.
    with (
        patch(
            'megatron.core.transformer.moe.replica_expert_runtime.ReplicaExpertRuntime',
            side_effect=AssertionError('replica runtime constructed'),
        ),
    ):
        scheduler.bind_experts(native)
    dispatcher = scheduler.expert_dispatch
    transport = dispatcher.transport
    assert native._replica_expert_runtime is None
    assert tuple(native.parameters()) == native_parameters
    assert transport._weights == transport._replica_grads == ()
    assert not hasattr(transport, "_native_grads")
    assert list(dispatcher.parameters()) == []
    for index in (0, 1):
        with pytest.raises(RuntimeError, match='Home-only'):
            transport.projection_views(index)
    assert not hasattr(transport, "native_projection_grad_view")
    optim = torch.optim.Adam(native.parameters(), lr=0.001)
    ref_optim = torch.optim.Adam(reference.parameters(), lr=0.001)
    scheduler.bind_optimizer(optim)
    context = SchedulerContext(
        layer_number=1,
        num_logical_experts=e,
        num_local_experts=2,
        local_expert_indices=(rank * 2, rank * 2 + 1),
        ep_size=size,
        ep_rank=rank,
        router_topk=1,
        training=True,
        pg_collection=SimpleNamespace(ep=ep_group),
    )
    ids = torch.tensor([0] * 5 + [1] * 4 + list(range(e)), device='cuda')
    routes = F.one_hot(ids, e).bool()
    x = (torch.arange(ids.numel() * 3, device='cuda').reshape(-1, 3).float() / 100).bfloat16()
    versions = []
    try:
        for step in range(4):
            optim.zero_grad()
            ref_optim.zero_grad()
            probs = routes.float().requires_grad_()
            ref_probs = routes.float().requires_grad_()
            physical_routes, physical_probs = scheduler.schedule(probs, routes, native, context)
            layout = scheduler.planner.active_home_layout.flatten().clone()
            assert torch.equal(physical_routes, routes[:, layout])
            local_output = dispatcher.after_token_combine(
                native(x, physical_probs[:, rank * 2 : rank * 2 + 2])
            )
            expected = reference(x, ref_probs)
            observed = local_output.detach().clone()
            dist.all_reduce(observed, group=ep_group)
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
            with pytest.raises(RuntimeError, match='native forwards/backwards'):
                dispatcher.assert_idle()
            local_output.sum().backward()
            expected.sum().backward()
            dispatcher.assert_idle()
            # Every native parameter gradient must match its current logical owner.
            for projection in (1, 2):
                for slot in range(2):
                    p = getattr(native, f'linear_fc{projection}').get_parameter(f'weight{slot}')
                    r = getattr(reference, f'linear_fc{projection}').get_parameter(
                        f'weight{layout[rank*2+slot].item()}'
                    )
                    torch.testing.assert_close(p.grad, r.grad, rtol=0, atol=0)
            optim.step()
            ref_optim.step()
            scheduler.prepare_home_exchange()
            scheduler.step()
            versions.append(scheduler.home_expert_dispatch._completed_version)
            layout = scheduler.planner.active_home_layout.flatten()
            for projection in (1, 2):
                for slot in range(2):
                    p = getattr(native, f'linear_fc{projection}').get_parameter(f'weight{slot}')
                    r = getattr(reference, f'linear_fc{projection}').get_parameter(
                        f'weight{layout[rank*2+slot].item()}'
                    )
                    torch.testing.assert_close(p, r, rtol=0, atol=0)
                    for key in ('exp_avg', 'exp_avg_sq', 'step'):
                        torch.testing.assert_close(
                            optim.state[p][key], ref_optim.state[r][key], rtol=0, atol=0
                        )
            assert [p.data_ptr() for p in native.parameters()] == addresses
        if size > 1:
            assert max(versions) > 0
    finally:
        dispatcher.destroy()


def test_zero_replica_adam_cycles_and_local_swaps(ep_group, monkeypatch):
    original = home_tests._transport
    from dataclasses import replace

    from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport

    def home_transport(group):
        # Obtain the existing test's metadata without constructing replica storage.
        with patch.object(home_tests, 'NcclP2PTransport', side_effect=lambda config: config):
            config = original(group)
        return NcclP2PTransport(replace(config, num_local_replica_slots=0))

    monkeypatch.setattr(home_tests, '_transport', home_transport)
    home_tests.test_deferred_home_cycles_preserve_adam_continuation(ep_group)
