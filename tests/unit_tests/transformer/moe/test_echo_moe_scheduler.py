# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.echo_moe_scheduler import EchoLoadPlanner
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
    MoEScheduler,
    SchedulerContext,
)
from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner
from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch
from megatron.core.transformer.moe.replica_weight_triton import _grad_arguments, _push_arguments
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def _patch_echo_count_gather(monkeypatch):
    import megatron.core.transformer.moe.echo_moe_scheduler as echo_moe_scheduler

    def fake_gather_from_sequence_parallel_region(tokens_per_expert, group=None):
        assert group is not None
        return torch.tensor(
            [4, 0, 0, 0, 0, 0, 1, 1], dtype=tokens_per_expert.dtype, device=tokens_per_expert.device
        )

    monkeypatch.setattr(
        echo_moe_scheduler.tensor_parallel,
        "gather_from_sequence_parallel_region",
        fake_gather_from_sequence_parallel_region,
    )


def _echo_context() -> SchedulerContext:
    return SchedulerContext(
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


def _replica_dispatcher(num_idle_experts: int = 4) -> ReplicaExpertDispatch:
    class _Group:
        def size(self):
            return 2

        def rank(self):
            return 0

    config = SimpleNamespace(
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_scheduler_num_idle_experts=num_idle_experts,
        moe_scheduler_expert_dispatcher_type="replica_peer_tma",
        grad_reduce_in_bf16=False,
        moe_flex_dispatcher_num_sms=None,
        moe_scheduler_replica_storage="shared",
        moe_scheduler_replica_mxfp8_storage="reuse",
    )
    return ReplicaExpertDispatch(config=config, pg_collection=SimpleNamespace(ep=_Group()))


def _hot_expert_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.zeros(4, 4)
    routing_map = torch.zeros(4, 4, dtype=torch.bool)
    routing_map[:, 0] = True
    probs[:, 0] = 1
    return probs, routing_map, routing_map.sum(dim=0)


def test_echo_planner_reroutes_hot_expert_tokens_to_echo_slot():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()
    planner = EchoLoadPlanner(2)
    home_placement, physical_to_logical_map, placement_result = planner.update_placement(
        probs, routing_map, context, tokens_per_expert=tokens_per_expert
    )
    physical_routing_map, physical_probs = planner.reroute(
        probs, routing_map, placement_result, context
    )

    assert physical_to_logical_map.tolist() == [[-1], [0]]
    assert physical_routing_map.shape == (4, 6)
    assert physical_routing_map[:, 0].sum().item() == 3
    assert physical_routing_map[:, 5].sum().item() == 1
    assert torch.equal(physical_probs.sum(dim=1), probs.sum(dim=1))


def test_replica_dispatch_supports_configured_physical_source_table():
    context = _echo_context()
    dispatcher = _replica_dispatcher()
    smaller_dispatcher = _replica_dispatcher(num_idle_experts=2)

    assert dispatcher.supports(torch.tensor([[2, 0], [1, -1]]), context)
    assert smaller_dispatcher.supports(torch.tensor([[2], [0]]), context)
    assert not smaller_dispatcher.supports(torch.tensor([[2, 0], [1, -1]]), context)
    assert not dispatcher.supports(torch.tensor([[2.0, 0.0], [1.0, -1.0]]), context)
    assert not dispatcher.supports(torch.tensor([0, 1, 2, 0, 2, 3, 1, -1]), context)
    assert not dispatcher.supports(torch.arange(10), context)


def test_replica_transport_specializes_home_experts_and_slots_independently():
    common = {
        "member_numels": (128 * 128, 128 * 128),
        "world_size": 4,
        "num_local_home_experts": 8,
        "num_local_replica_slots": 2,
        "num_sms": 16,
    }
    push = _push_arguments(mxfp8=False, **common)
    grad = _grad_arguments(grad_dtype=torch.float32, **common)
    fc1_grad = _grad_arguments(grad_dtype=torch.float32, projections=(0,), **common)
    fc2_grad = _grad_arguments(grad_dtype=torch.float32, projections=(1,), **common)

    for arguments in (push, grad):
        assert arguments["NUM_LOCAL_HOME_EXPERTS"] == 8
        assert arguments["NUM_LOCAL_REPLICA_SLOTS"] == 2
        assert arguments["PLAN_POW2"] == 8
    assert fc1_grad["TILE_BEGIN"] == grad["TILE_BEGIN"] == 0
    assert fc1_grad["TILE_END"] == fc2_grad["TILE_BEGIN"]
    assert fc2_grad["TILE_END"] == grad["TILE_END"]


def test_echo_planner_requires_ep_group_for_multi_ep():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = SchedulerContext(
        layer_number=1,
        num_logical_experts=4,
        num_local_experts=2,
        local_expert_indices=(0, 1),
        ep_size=2,
        ep_rank=0,
        router_topk=1,
        training=True,
    )

    with pytest.raises(ValueError, match="pg_collection.ep"):
        EchoLoadPlanner(2).update_placement(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        )


class _RecordingRuntimeWorkspace:
    def __init__(self):
        self.workspace_acquired = False
        self.workspace_events = []

    def acquire_workspace(self):
        assert not self.workspace_acquired
        self.workspace_acquired = True
        self.workspace_events.append("acquire")

    def release_workspace(self):
        assert self.workspace_acquired
        self.workspace_acquired = False
        self.workspace_events.append("release")


def test_replica_dispatch_lowers_placement_to_runtime_plan():
    context = _echo_context()
    physical_to_logical_map = torch.tensor([[2], [0]])
    dispatcher = _replica_dispatcher(num_idle_experts=2)

    class _Runtime(_RecordingRuntimeWorkspace):
        source_parameters = ()

        def __init__(self):
            super().__init__()
            self.last_plan = None
            self.started_plan = None

        def start_prefetch(self, plan):
            self.started_plan = plan

    runtime = _Runtime()
    dispatcher.runtime = runtime
    dispatcher.wrap_layer_input(torch.ones(1))
    dispatcher.dispatch(torch.nn.Identity(), physical_to_logical_map, context)

    assert runtime.last_plan is runtime.started_plan
    assert runtime.started_plan.virtual_experts is None
    assert runtime.started_plan.experts_to_copy.dtype == torch.int32
    assert runtime.started_plan.experts_to_copy.tolist() == [[2], [0]]

    dispatcher.after_token_combine(torch.ones(1))
    assert dispatcher._active_plan is None
    assert runtime.workspace_events == ["acquire", "release"]


def test_replica_dispatch_binds_transport_backed_runtime(monkeypatch):
    from megatron.core.transformer.moe import replica_expert_dispatch as replica_dispatch

    captured = {}
    runtime = object()
    transport = object()

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return runtime

    def fake_transport_factory(dispatcher_type, config):
        captured["transport_dispatcher_type"] = dispatcher_type
        captured["transport_config"] = config
        return transport

    class _Experts:
        def set_replica_expert_runtime(self, value):
            self.bound_runtime = value

    monkeypatch.setattr(replica_dispatch, "ReplicaExpertRuntime", fake_runtime)
    monkeypatch.setattr(replica_dispatch, "create_replica_weight_transport", fake_transport_factory)
    dispatcher = _replica_dispatcher()
    experts = _Experts()
    dispatcher.bind_experts(experts)

    assert dispatcher.runtime is runtime
    assert experts.bound_runtime is runtime
    assert captured["num_experts"] == 4
    assert captured["num_local_home_experts"] == 2
    assert captured["num_local_replica_slots"] == 2
    transport_config = object()
    assert captured["transport_factory"](transport_config) is transport
    assert captured["transport_dispatcher_type"] == "replica_peer_tma"
    assert captured["transport_config"] is transport_config
    assert captured["grad_dtype"] == torch.float32
    assert captured["replica_storage"] == "shared"
    assert captured["replica_mxfp8_storage"] == "reuse"


def test_replica_transport_factory_uses_expert_dispatcher_type(monkeypatch):
    from megatron.core.transformer.moe import replica_peer_tma_transport
    from megatron.core.transformer.moe.replica_weight_transport import (
        create_replica_weight_transport,
    )

    config = object()
    transport = object()
    monkeypatch.setattr(replica_peer_tma_transport, "PeerTmaTransport", lambda value: transport)

    assert create_replica_weight_transport("replica_peer_tma", config) is transport
    with pytest.raises(ValueError, match="Unsupported replica expert-dispatch transport"):
        create_replica_weight_transport("unsupported", config)


def test_replica_dispatch_preserves_runtime_backward_order():
    events = []
    parameter = torch.nn.Parameter(torch.ones(()))

    class _Runtime(_RecordingRuntimeWorkspace):
        source_parameters = (parameter,)

        def __init__(self):
            super().__init__()
            self.last_plan = None

        def start_prefetch(self, plan):
            del plan
            events.append("forward_prefetch")

        def start_backward_prefetch(self, plan):
            del plan
            self.acquire_workspace()
            events.append("backward_prefetch")

        def wait_prefetch_for_backward(self, plan):
            del plan
            events.append("wait_prefetch")

        def start_pending_grad_reduces(self, plan):
            del plan
            events.append("start_grad_reduce")

        def wait_grad_reduce(self, plan):
            del plan
            events.append("wait_grad_reduce")
            return (torch.ones_like(parameter),)

    class _BackwardMarker(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, label):
            ctx.label = label
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(ctx.label)
            return grad, None

    dispatcher = _replica_dispatcher()
    dispatcher.runtime = _Runtime()
    hidden = dispatcher.wrap_layer_input(torch.ones((), requires_grad=True))
    dispatcher.dispatch(torch.nn.Identity(), torch.tensor([[2, -1], [0, 1]]), _echo_context())
    hidden = dispatcher.before_token_dispatch(hidden)
    hidden = _BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = dispatcher.after_token_dispatch(hidden)
    hidden = _BackwardMarker.apply(hidden, "expert_backward")
    hidden = dispatcher.before_token_combine(hidden)
    hidden = _BackwardMarker.apply(hidden, "combine_backward")
    hidden = dispatcher.after_token_combine(hidden)
    hidden.backward()

    # Pending reductions run after token dispatch backward; FC2's optional
    # earlier launch belongs to the expert runtime, not this pending hook.
    assert events == [
        "forward_prefetch",
        "backward_prefetch",
        "combine_backward",
        "wait_prefetch",
        "expert_backward",
        "dispatch_backward",
        "start_grad_reduce",
        "wait_grad_reduce",
    ]
    assert not any(slot.in_use for slot in dispatcher._plan_slots)
    assert dispatcher.runtime.workspace_events == ["acquire", "release", "acquire", "release"]
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_echo_scheduler_runs_planner_and_dispatch_adapter():
    probs, routing_map, tokens_per_expert = _hot_expert_inputs()
    context = _echo_context()

    class _NoOpDispatch(ExpertDispatch):
        dispatcher_name = "test"

        def dispatch(self, experts, physical_to_logical_map, context):
            del experts, physical_to_logical_map, context

    scheduler = MoEScheduler(planner=EchoLoadPlanner(2), expert_dispatch=_NoOpDispatch())

    output_probs, output_routing_map = scheduler.schedule(
        probs, routing_map, torch.nn.Identity(), context, tokens_per_expert=tokens_per_expert
    )

    assert output_routing_map[:, 5].sum().item() == 1
    assert output_probs.shape == output_routing_map.shape


@pytest.mark.parametrize("backend", ["replica_peer_tma", "replica_nccl"])
def test_moe_scheduler_builds_echo_planner_with_replica_dispatch(backend):
    class _Group:
        def size(self):
            return 1

        def rank(self):
            return 0

    scheduler = MoEScheduler.from_config(
        _scheduler_config(
            moe_scheduler_num_idle_experts=2, moe_scheduler_expert_dispatcher_type=backend
        ),
        SimpleNamespace(ep=_Group()),
    )

    assert isinstance(scheduler.planner, EchoLoadPlanner)
    assert scheduler.planner.num_echo_experts == 2
    assert isinstance(scheduler.expert_dispatch, ReplicaExpertDispatch)
    assert scheduler.expert_dispatch.dispatcher_name == backend


@pytest.mark.parametrize("ep_size", [2, 4])
def test_moe_scheduler_builds_moonep_planner_with_replica_dispatch(ep_size):
    class _Group:
        def size(self):
            return ep_size

        def rank(self):
            return 0

    scheduler = MoEScheduler.from_config(
        _scheduler_config(
            moe_scheduler_planner_type="moon_ep",
            expert_model_parallel_size=ep_size,
            num_moe_experts=ep_size * 2,
            moe_scheduler_num_idle_experts=ep_size * 2,
        ),
        SimpleNamespace(ep=_Group()),
    )

    assert isinstance(scheduler.planner, MoonEPLoadPlanner)
    # The config counts slots over the EP group; the planner takes per-rank slots.
    assert scheduler.planner.num_redundant_experts == 2
    assert isinstance(scheduler.expert_dispatch, ReplicaExpertDispatch)


def _scheduler_config(**overrides) -> TransformerConfig:
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_moe_experts": 4,
        "moe_ffn_hidden_size": 128,
        "use_cpu_initialization": True,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "gated_linear_unit": True,
        "activation_func": torch.nn.functional.silu,
        "moe_router_topk": 1,
        "moe_router_pre_softmax": True,
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "moe_enable_scheduler": True,
        "expert_model_parallel_size": 2,
        "moe_scheduler_num_idle_experts": 4,
        "moe_scheduler_expert_dispatcher_type": "replica_peer_tma",
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_transformer_config_validates_moe_scheduler_requirements():
    config = _scheduler_config()

    assert config.moe_enable_scheduler
    assert config.grad_reduce_in_bf16 is False
    assert config.moe_scheduler_planner_type == "echo"
    assert config.moe_scheduler_expert_dispatcher_type == "replica_peer_tma"

    with pytest.raises(ValueError, match="moe_scheduler_num_idle_experts"):
        _scheduler_config(moe_scheduler_num_idle_experts=None)
    with pytest.raises(ValueError, match="dropless"):
        _scheduler_config(moe_expert_capacity_factor=1.0)
    with pytest.raises(ValueError, match="add_bias_linear"):
        _scheduler_config(add_bias_linear=True)
    with pytest.raises(ValueError, match="moe_scheduler_num_idle_experts to equal"):
        _scheduler_config(moe_scheduler_planner_type="moon_ep", moe_scheduler_num_idle_experts=2)
    with pytest.raises(ValueError, match="Unsupported moe_scheduler_expert_dispatcher_type"):
        _scheduler_config(moe_scheduler_expert_dispatcher_type="hybridep")

    replica_config = _scheduler_config(moe_scheduler_num_idle_experts=2)
    assert replica_config.moe_scheduler_num_idle_experts == 2
    eplb_config = _scheduler_config(
        moe_scheduler_planner_type="eplb", moe_scheduler_num_idle_experts=2
    )
    assert eplb_config.moe_scheduler_planner_type == "eplb"
    with pytest.raises(ValueError, match="at least one replica slot"):
        _scheduler_config(moe_scheduler_num_idle_experts=0)


def test_replica_scheduler_accepts_native_mxfp8_with_router_padding():
    config = _scheduler_config(
        fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
    )

    assert (config.fp8, config.fp8_recipe, config.fp8_param) == ("e4m3", "mxfp8", True)
    assert config.moe_router_padding_for_quantization


def test_nccl_replica_scheduler_accepts_mxfp8():
    config = _scheduler_config(
        moe_scheduler_expert_dispatcher_type="replica_nccl",
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        moe_router_padding_for_quantization=True,
    )
    assert config.fp8_recipe == "mxfp8"


@pytest.mark.parametrize(
    "impl,scopes",
    [("local", []), ("local", ["moe"]), ("transformer_engine", ["moe"]), ("full_iteration", [])],
)
def test_nccl_replica_scheduler_rejects_moe_capture(impl, scopes):
    with pytest.raises(ValueError, match="replica_nccl host schedules"):
        _scheduler_config(
            moe_scheduler_expert_dispatcher_type="replica_nccl",
            moe_expert_rank_capacity_factor=1.0,
            cuda_graph_impl=impl,
            cuda_graph_modules=scopes,
        )


@pytest.mark.parametrize("planner", ["echo", "moon_ep"])
def test_replica_token_dispatcher_config_preserves_outer_moe_capture(planner):
    config = _scheduler_config(
        moe_scheduler_planner_type=planner,
        moe_scheduler_num_idle_experts=4,
        moe_expert_rank_capacity_factor=1.0,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["attn", "moe"],
    )
    layer = SimpleNamespace(config=config, num_physical_experts=8)

    dispatcher_config = MoELayer._get_token_dispatcher_config(layer)

    assert dispatcher_config.num_moe_experts == 8
    assert not dispatcher_config.moe_enable_scheduler
    assert dispatcher_config.cuda_graph_impl == "none"
    assert dispatcher_config.cuda_graph_modules == []
    assert config.num_moe_experts == 4
    assert config.moe_enable_scheduler
    assert config.cuda_graph_impl == "transformer_engine"
    assert [scope.name for scope in config.cuda_graph_modules] == ["attn", "moe"]


def test_nccl_replica_scheduler_allows_attention_capture():
    config = _scheduler_config(
        moe_scheduler_expert_dispatcher_type="replica_nccl",
        cuda_graph_impl="local",
        cuda_graph_modules=["attn"],
    )
    assert config.moe_scheduler_expert_dispatcher_type == "replica_nccl"


@pytest.mark.parametrize(
    ("fp8", "fp8_recipe", "fp8_param"),
    [("e4m3", "mxfp8", False), ("e4m3", "tensorwise", True), ("hybrid", "mxfp8", True)],
)
@pytest.mark.parametrize("backend", ["replica_peer_tma", "replica_nccl"])
def test_replica_scheduler_rejects_unsupported_fp8_parameter_storage(
    fp8, fp8_recipe, fp8_param, backend
):
    with pytest.raises(ValueError, match="MXFP8 E4M3 with native FP8 parameters"):
        _scheduler_config(
            fp8=fp8,
            fp8_recipe=fp8_recipe,
            fp8_param=fp8_param,
            moe_scheduler_expert_dispatcher_type=backend,
        )


@pytest.mark.parametrize("scope", ["moe_router", "moe_preprocess"])
def test_replica_scheduler_rejects_partial_moe_cuda_graph_scopes(scope):
    with pytest.raises(AssertionError, match="whole moe CUDA graph scope only"):
        _scheduler_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


def test_moe_layer_scheduler_helper_uses_unified_scheduler_output():
    probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    routing_map = probs.bool()
    physical_probs = torch.zeros(3, 6)
    physical_routing_map = torch.zeros(3, 6, dtype=torch.bool)
    physical_probs[:, 5] = probs[:, 0]
    physical_routing_map[:, 5] = routing_map[:, 0]
    physical_probs[:, 1] = probs[:, 1]
    physical_routing_map[:, 1] = routing_map[:, 1]

    class _RecordingScheduler:
        max_dispatch_routes_per_rank = 4
        max_dispatch_padded_tokens_per_rank = 256

        def __init__(self):
            self.probs = None
            self.routing_map = None
            self.context = None
            self.experts = None
            self.tokens_per_expert = None

        def schedule(
            self, probs_arg, routing_map_arg, experts_arg, context_arg, *, tokens_per_expert=None
        ):
            self.probs = probs_arg
            self.routing_map = routing_map_arg
            self.context = context_arg
            self.experts = experts_arg
            self.tokens_per_expert = tokens_per_expert
            return physical_probs, physical_routing_map

    layer = object.__new__(MoELayer)
    layer.moe_scheduler = _RecordingScheduler()
    layer.experts = object()
    layer.token_dispatcher = Mock(spec=["set_dispatch_route_bound"])
    layer.num_logical_experts = 4
    layer.num_local_home_experts = 4
    layer.local_home_expert_indices = [0, 1, 2, 3]
    layer.ep_group = None
    layer.config = type("_Config", (), {"moe_router_topk": 1})()
    layer.layer_number = 7
    layer.training = True
    layer.pg_collection = object()
    layer._moe_scheduler_context_cache = {}

    hidden_states = torch.randn(3, 12)
    new_probs, new_routing_map = MoELayer._maybe_schedule_moe(
        layer, hidden_states, probs, routing_map
    )

    assert new_probs is physical_probs
    assert new_routing_map is physical_routing_map
    assert layer.moe_scheduler.probs is probs
    assert layer.moe_scheduler.routing_map is routing_map
    assert layer.moe_scheduler.tokens_per_expert is None
    assert layer.moe_scheduler.experts is layer.experts
    assert layer.moe_scheduler.context.layer_number == 7
    assert layer.moe_scheduler.context.num_logical_experts == 4
    assert layer.moe_scheduler.context.num_local_experts == 4
    layer.token_dispatcher.set_dispatch_route_bound.assert_called_once_with(4, padded_tokens=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "backend,mxfp8", [("replica_peer_tma", False), ("replica_nccl", False), ("replica_nccl", True)]
)
def test_moe_layer_auto_instantiates_scheduler_from_config(backend, mxfp8):
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    layer = None
    try:
        model_parallel_cuda_manual_seed(123)
        config = _scheduler_config(
            moe_scheduler_num_idle_experts=2,
            use_cpu_initialization=False,
            moe_scheduler_expert_dispatcher_type=backend,
            **(
                dict(
                    fp8="e4m3",
                    fp8_recipe="mxfp8",
                    fp8_param=True,
                    moe_router_padding_for_quantization=True,
                )
                if mxfp8
                else {}
            ),
        )
        # Replica runtime binding requires discrete TE grouped expert weights
        # and the operation fuser, matching the scheduler configuration.
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=config.num_moe_experts, moe_grouped_gemm=True, use_te_op_fuser=True
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)

        with get_fp8_context(config, is_init=True):
            layer = MoELayer(config, submodules)

        assert layer.num_logical_experts == 4
        assert layer.num_physical_experts == 6
        assert layer.num_local_home_experts == 2
        assert layer.num_local_experts == 3
        assert layer._token_dispatcher_config.num_moe_experts == 6
        assert layer.token_dispatcher.config.num_moe_experts == 6
        assert layer.config.num_moe_experts == 4
        assert layer.moe_scheduler is not None
        assert layer.home_expert_indices == [0, 1]
        assert layer.idle_expert_indices == [2]
        assert layer.experts.num_local_experts == 2
        assert layer.experts._replica_expert_runtime is not None
        assert layer.experts._replica_expert_runtime.weight_format == ("mxfp8" if mxfp8 else "bf16")
    finally:
        if layer is not None:
            layer.moe_scheduler.expert_dispatch.runtime.destroy()
        Utils.destroy_model_parallel()
