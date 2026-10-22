# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Learned routing, initialization, and logical-checkpoint release contracts."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner
from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
from megatron.core.transformer.moe.home_expert_state import (
    OptimizerHomeExpertState,
    validate_home_expert_optimizer_config,
)
from megatron.core.transformer.moe.home_expert_training import (
    canonicalize_home_experts_for_export,
    finish_home_expert_checkpoint_load,
    initialize_home_expert_optimizers,
    prepare_home_expert_checkpoint_load,
    validate_home_expert_checkpoint_args,
)
from megatron.core.transformer.moe.moe_scheduler import (
    ExpertDispatch,
    MoEScheduler,
    SchedulerContext,
    validate_scheduler_environment,
)
from megatron.core.transformer.moe.moonep_moe_scheduler import (
    MoonEPLoadPlanner,
    extract_semantic_routes,
)
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement
from tests.unit_tests.transformer.moe.test_echo_moe_scheduler import _scheduler_config
from tests.unit_tests.transformer.moe.test_home_expert_exchange import (
    _transport,
    shared_checkpoint_dir,
)
from tests.unit_tests.transformer.moe.test_mock_router import _router
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world, ep_group

pytestmark = pytest.mark.launch_on_gb200


def test_runtime_release_before_binding_preserves_native_compute():
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    native_ops, runtime = object(), object()
    experts = SimpleNamespace(_replica_expert_runtime=None, _fused_ops=native_ops)
    TEGroupedMLP.release_replica_expert_runtime(experts, runtime)
    assert experts._fused_ops is native_ops
    experts._replica_expert_runtime = runtime
    with pytest.raises(RuntimeError, match="another layer"):
        TEGroupedMLP.release_replica_expert_runtime(experts, object())
    assert experts._replica_expert_runtime is runtime
    TEGroupedMLP.release_replica_expert_runtime(experts, runtime)
    assert experts._replica_expert_runtime is None
    assert experts._fused_ops is None


@pytest.mark.parametrize("capacity", [None, 1.5])
@pytest.mark.parametrize("padding", [[False] * 12, [True, False, False] * 4, [True] * 12])
def test_learned_padding_preserves_topk_and_zero_gradients(capacity, padding):
    router = _router("moon_ep", moe_expert_rank_capacity_factor=capacity)
    router.mock_route_cache = None
    router.config.moe_router_mock_maxvio = None
    router.routing_type = "none"
    router.expert_bias = router.router_replay = None
    logits = torch.randn(12, 1, 8, device="cuda", requires_grad=True)
    mask = torch.tensor(padding, device="cuda")
    reference_probs, reference_routes = router.routing(logits)
    probs, routes = router.routing(logits, padding_mask=mask)
    assert (routes.sum(1) == router.topk).all()
    torch.testing.assert_close(routes, reference_routes)
    torch.testing.assert_close(probs[~mask], reference_probs[~mask])
    assert not probs[mask].any()
    _, ids = extract_semantic_routes(routes, probs, router.topk)
    assert ((ids >= 0) & (ids < 8)).all()
    (probs * torch.arange(8, device="cuda")).sum().backward()
    assert not logits.grad[mask].any()


def test_learned_padding_runs_distributed_moonep(distributed_world):
    group = distributed_world
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    experts = 2 * size
    router = _router(
        "moon_ep",
        num_moe_experts=experts,
        expert_model_parallel_size=size,
        moe_router_mock_maxvio=1.0,
    )
    router.mock_route_cache = None
    router.config.moe_router_mock_maxvio = None
    router.routing_type = "none"
    router.expert_bias = router.router_replay = None
    logits = torch.randn(32, 1, experts, device="cuda", requires_grad=True)
    # Same physical token count with different valid lengths, including an
    # entirely padded rank. The planner still sees a full top-k on every rank.
    mask = torch.arange(32, device="cuda") >= (0 if rank == 0 else 24 + rank % 8)
    probs, routes = router.routing(logits, padding_mask=mask)
    context = SchedulerContext(
        1,
        experts,
        2,
        (2 * rank, 2 * rank + 1),
        size,
        rank,
        2,
        True,
        pg_collection=SimpleNamespace(ep=group),
    )
    planner = MoonEPLoadPlanner(2)
    _, _, placement = planner.update_placement(probs, routes, context)
    output_routes, output_probs = planner.reroute(probs, routes, placement, context)
    assert (output_routes.sum(1) == 2).all()
    torch.testing.assert_close(output_probs.sum(1), probs.sum(1))
    assert not output_probs[mask].any()
    output_probs.sum().backward()
    assert not logits.grad[mask].any()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"moe_scheduler_expert_dispatcher_type": "replica_hybridep"}, "not implemented"),
        ({"expert_model_parallel_size": 128}, "at most 64"),
        ({"moe_scheduler_planner_type": "moon_ep", "num_moe_experts": 16384}, "8192"),
    ],
)
def test_algorithm_rejections_happen_in_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        _scheduler_config(**overrides)


@pytest.mark.parametrize(
    "option",
    [
        "optimizer_cpu_offload",
        "use_precision_aware_optimizer",
        "use_layer_wise_distributed_optimizer",
        "chunked_optimizer_state_offload",
        "offload_optimizer_states",
    ],
)
def test_optimizer_representation_rejected_before_allocation(option):
    with pytest.raises(ValueError, match="resident uncompressed"):
        validate_home_expert_optimizer_config(SimpleNamespace(**{option: True}))


@pytest.mark.parametrize("precision_aware", [False, True])
def test_home_optimizer_validation_uses_resolved_training_arguments(monkeypatch, precision_aware):
    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setattr(sys, "argv", ["test_scheduler_release_contracts.py"])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.moe_scheduler_home_update_interval = 1
    # Defaults are applied inside validate_args, after the former early check.
    args.use_precision_aware_optimizer = None
    defaults = {"use_precision_aware_optimizer": precision_aware}
    if precision_aware:
        with pytest.raises(ValueError, match="resident uncompressed"):
            validate_args(args, defaults)
    else:
        validated = validate_args(args, defaults)
        assert validated.use_precision_aware_optimizer is False


def test_environment_rejects_missing_cuda_before_group_or_storage(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        validate_scheduler_environment(_scheduler_config(), None)


def test_unsupported_optimizer_algorithm_rejected_before_allocation():
    with pytest.raises(ValueError, match="Adam and SGD"):
        validate_home_expert_optimizer_config(SimpleNamespace(optimizer="muon"))


def _checkpoint_args(home=False):
    return SimpleNamespace(
        ckpt_format="torch_dist",
        moe_enable_scheduler=home,
        moe_scheduler_planner_type="eplb" if home else "echo",
        moe_scheduler_home_update_interval=1 if home else 0,
        moe_scheduler_num_idle_experts=2 if home else None,
        num_experts=4,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        world_size=2,
        finetune=False,
        consumed_train_samples=99,
        consumed_valid_samples=5,
        skipped_train_samples=3,
    )


class _BoundDispatch(ExpertDispatch):
    def __init__(self, runtime=None):
        super().__init__()
        self.runtime = runtime

    def bind_home_optimizer(self, home, optimizer):
        home.bind_state_adapter(
            OptimizerHomeExpertState(
                optimizer, self.runtime.source_parameters, self.runtime.transport.config
            )
        )

    def dispatch(self, experts, expert_placement, context):
        raise AssertionError("Checkpoint initialization/export must not run replica dispatch.")


def _planner_model(planner, runtime=None):
    model = torch.nn.Module()
    home = HomeExpertDispatch()
    if runtime is not None:
        home.bind_transport(runtime.transport)
    model.moe_scheduler = MoEScheduler(planner, _BoundDispatch(runtime), home)
    model.config = SimpleNamespace(moe_enable_scheduler=True, moe_scheduler_home_update_interval=1)
    return model


def test_logical_checkpoint_starts_fresh_and_waits_for_planner(monkeypatch):
    planner = EPLBLoadPlanner(2, 1)
    model = _planner_model(planner)
    target, source = _checkpoint_args(True), _checkpoint_args(False)
    assert prepare_home_expert_checkpoint_load(target, [model], source)
    assert target.finetune
    assert (
        target.consumed_train_samples
        == target.consumed_valid_samples
        == target.skipped_train_samples
        == 0
    )
    assert model.state_dict() == {}  # No unavailable planner object in the load template.
    model.load_state_dict({}, strict=True)
    # The generic training loader can retry for legacy TE extra-state keys.
    model.load_state_dict({}, strict=False)
    finish_home_expert_checkpoint_load([model])
    assert not planner._loading_logical_checkpoint
    assert "moe_scheduler.planner._planner_state" in model.state_dict()
    assert planner._steps == 0
    assert not model.moe_scheduler.home_expert_dispatch.has_pending
    context = SchedulerContext(1, 4, 2, (0, 1), 2, 0, 1, True)
    counts = torch.tensor([[100, 80, 3, 1], [0, 0, 0, 0]])
    monkeypatch.setattr(planner, "_get_count_matrix", lambda *a, **kw: counts)
    routes = torch.eye(4, dtype=torch.bool)
    first, _, _ = planner.update_placement(routes.float(), routes, context)
    assert first is None
    torch.testing.assert_close(planner.active_home_layout, torch.tensor([[0, 1], [2, 3]]))
    model.moe_scheduler.step()
    proposal, _, _ = planner.update_placement(routes.float(), routes, context)
    assert proposal is not None
    # Producing a proposal still leaves committed ownership untouched.
    torch.testing.assert_close(planner.active_home_layout, torch.tensor([[0, 1], [2, 3]]))


def test_missing_planner_state_is_not_treated_as_logical_initialization():
    planner = EPLBLoadPlanner(2, 1)
    model = _planner_model(planner)
    assert not prepare_home_expert_checkpoint_load(
        _checkpoint_args(True), [model], _checkpoint_args(True)
    )
    with pytest.raises(RuntimeError, match="Missing required"):
        model.load_state_dict({}, strict=False)


def test_distributed_logical_checkpoint_initialization(distributed_world, shared_checkpoint_dir):
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    rank, size = dist.get_rank(), dist.get_world_size()
    expected = torch.full((2, 3), float(rank + 1), device="cuda")
    path = shared_checkpoint_dir / "logical-initialization"
    if rank == 0:
        path.mkdir()
    dist.barrier()

    def weight_shard(tensor):
        return ShardedTensor.from_rank_offsets("weight", tensor, (0, rank, size))

    source, target = _checkpoint_args(False), _checkpoint_args(True)
    for args in (source, target):
        args.world_size = args.expert_model_parallel_size = size
        args.num_experts = 2 * size
    dist_checkpointing.save({"args": source, "model": {"weight": weight_shard(expected)}}, path)
    common = dist_checkpointing.load_common_state_dict(path)
    planner = EPLBLoadPlanner(size, 1)
    model = _planner_model(planner)
    model.weight = torch.nn.Parameter(torch.zeros_like(expected))
    assert prepare_home_expert_checkpoint_load(target, [model], common["args"])
    template = {"weight": weight_shard(model.weight)}
    template.update(planner.sharded_state_dict(prefix="moe_scheduler.planner."))
    loaded = dist_checkpointing.load({"model": template}, path)
    model.load_state_dict(loaded["model"], strict=True)
    finish_home_expert_checkpoint_load([model])
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    assert not planner.active_home_layout.numel()
    assert not model.moe_scheduler.home_expert_dispatch.has_pending


@pytest.mark.parametrize("home_source", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "world_size",
        "expert_model_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ],
)
def test_home_loading_rejects_topology_changes(home_source, field):
    target, source = _checkpoint_args(True), _checkpoint_args(home_source)
    setattr(target, field, getattr(source, field) * 2)
    with pytest.raises(ValueError, match=f"unchanged {field}"):
        validate_home_expert_checkpoint_args(target, source)


def test_export_restores_logical_weights_and_adam_state(ep_group, tmp_path):
    transport = _transport(ep_group)
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    device = transport.config.device
    count = 2 * size
    reference = [
        torch.nn.Parameter(torch.full(shape, float(expert + 10 * projection + 1), device=device))
        for projection, shape in enumerate(transport.config.member_shapes)
        for expert in range(count)
    ]
    physical = [
        torch.nn.Parameter(reference[p * count + rank * 2 + slot].detach().clone())
        for p in range(2)
        for slot in range(2)
    ]
    ref_optim = torch.optim.Adam(reference, lr=0.01)
    optim = torch.optim.Adam(physical, lr=0.01)
    for p in reference + physical:
        p.grad = p.detach().square()
    ref_optim.step()
    optim.step()
    runtime = SimpleNamespace(source_parameters=tuple(physical), transport=transport)
    planner = EPLBLoadPlanner(size, 1)
    layer = _planner_model(planner, runtime)
    layer.weights = torch.nn.ParameterList(physical)
    initialize_home_expert_optimizers([layer], optim)
    home = layer.moe_scheduler.home_expert_dispatch
    permutation = torch.arange(count, device=device).roll(1).reshape(size, 2)
    try:
        home.dispatch(None, HomeExpertPlacement(permutation, 1), None)
        home.step()
        planner.active_home_layout = permutation.clone()
        # Export ignores a not-yet-committed proposal restored from a checkpoint.
        planner._pending_home_layout = permutation.flip(1)
        planner._pending_version = 2
        canonicalize_home_experts_for_export([layer], optim)
        assert layer.moe_scheduler is None
        assert not layer.config.moe_enable_scheduler
        assert all("planner" not in key and "scheduler" not in key for key in layer.state_dict())
        for p in range(2):
            for slot in range(2):
                actual, expected = physical[p * 2 + slot], reference[p * count + rank * 2 + slot]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(
                        optim.state[actual][key], ref_optim.state[expected][key], rtol=0, atol=0
                    )
        # An ordinary scheduler-free model can load the exported parameter keys.
        plain = torch.nn.Module()
        plain.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty_like(p)) for p in physical]
        )
        path = tmp_path / f"logical-rank-{dist.get_rank()}.pt"
        torch.save({"model": layer.state_dict(), "optimizer": optim.state_dict()}, path)
        restored = torch.load(path, weights_only=True)
        plain.load_state_dict(restored["model"], strict=True)
        plain_optim = torch.optim.Adam(plain.parameters(), lr=0.01)
        plain_optim.load_state_dict(restored["optimizer"])
        for p in reference + list(plain.parameters()):
            p.grad = p.detach() * 0.3
        ref_optim.step()
        plain_optim.step()
        for p in range(2):
            for slot in range(2):
                torch.testing.assert_close(
                    plain.weights[p * 2 + slot],
                    reference[p * count + rank * 2 + slot],
                    rtol=0,
                    atol=0,
                )
    finally:
        transport.destroy()


def test_failed_installation_blocks_checkpoint_and_export():
    model = _planner_model(EPLBLoadPlanner(2, 1))
    # Reproduce the fatal state set by a failed install; consumers must use the
    # dispatcher contract rather than interpreting its internal flag.
    model.moe_scheduler.home_expert_dispatch._failed = True
    with pytest.raises(RuntimeError, match="installation failed"):
        model.state_dict()
    with pytest.raises(RuntimeError, match="installation failed"):
        model.moe_scheduler.canonicalize_home_for_export()


def test_scheduler_logging_uses_module_logger(caplog, monkeypatch):
    import logging

    from megatron.core.transformer.moe import moe_scheduler

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with caplog.at_level(logging.INFO, logger=moe_scheduler.__name__):
        moe_scheduler._rank0_info("scheduler diagnostic")
    assert [(record.name, record.message) for record in caplog.records] == [
        (moe_scheduler.__name__, "scheduler diagnostic")
    ]


@pytest.mark.parametrize("successful", [False, True])
def test_home_boundary_releases_hold_and_refreshes_only_after_success(successful, monkeypatch):
    from megatron.core.optimizer.optimizer import FP32Optimizer
    from megatron.core.transformer.moe.home_expert_training import finish_home_expert_step

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = FP32Optimizer(
        torch.optim.SGD([parameter], lr=0.1), SimpleNamespace(), lambda _: None
    )
    events = []
    optimizer.defer_param_sync_for_state_update(True)
    monkeypatch.setattr(optimizer, "refresh_model_weights", lambda: events.append("refresh"))
    scheduler = SimpleNamespace(step=lambda: events.append("migrate"))
    finish_home_expert_step([scheduler], optimizer, successful)
    assert events == (["migrate", "refresh"] if successful else [])
    assert not optimizer.param_sync_deferred_for_state_update


@pytest.mark.parametrize("planner", ["echo", "eplb", "moon_ep"])
def test_documented_moe_capture_capacity_boundary(planner):
    options = dict(
        moe_scheduler_planner_type=planner,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["moe"],
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        _scheduler_config(**options)
    config = _scheduler_config(**options, moe_expert_rank_capacity_factor=1.0)
    assert config.moe_expert_rank_capacity_factor == 1.0


def test_optimizer_binding_preserves_master_identity_and_rejects_unowned_parameter():
    from megatron.core.optimizer.optimizer import FP32Optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    parameter = torch.nn.Parameter(torch.ones(3, device="cuda"))
    unowned = torch.nn.Parameter(torch.zeros_like(parameter))
    inner = torch.optim.Adam([parameter])
    optimizer = FP32Optimizer(inner, OptimizerConfig(), lambda _: None)
    binding = optimizer.get_parameter_state_binding(parameter)
    assert binding.optimizer is inner
    assert binding.main is parameter and binding.start == 0 and binding.group is None
    assert optimizer.get_parameter_state_binding(unowned) is None
    optimizer.refresh_model_weights(force=True)
    assert inner.param_groups[0]["params"][0] is parameter
