# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU boundary regressions; real TE/NCCL parity is covered by the GPU suites."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.eplb_moe_scheduler import (
    EPLBHomeOnlyPlacementResult,
    EPLBLoadPlanner,
)
from megatron.core.transformer.moe.replica_expert_runtime import (
    _DirectionalBinding,
    _ReplicaProjection,
    _wrap_mxfp8,
)
from tests.unit_tests.transformer.moe.test_eplb_moe_scheduler import _context
from tests.unit_tests.transformer.moe.test_home_only_expert_dispatch import _config

pytestmark = pytest.mark.launch_on_gb200


def test_home_binding_preserves_lazy_optimizer_state():
    from megatron.core.transformer.moe.home_expert_state import OptimizerHomeExpertState

    parameters = [torch.nn.Parameter(torch.ones(2, 3)) for _ in range(4)]
    optimizer = torch.optim.Adam(parameters)
    config = SimpleNamespace(
        num_local_home_experts=2,
        member_shapes=((2, 3), (2, 3)),
        device=torch.device("cpu"),
        group=None,
    )
    adapter = OptimizerHomeExpertState(optimizer, parameters, config)
    adapter.prepare_components()
    assert len(optimizer.state) == 0, "Inspection must not initialize empty Adam state entries"
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    states = {p: dict(optimizer.state[p]) for p in parameters}
    adapter.prepare_components()
    for parameter, state in states.items():
        assert optimizer.state[parameter].keys() == state.keys()
        assert all(optimizer.state[parameter][key] is value for key, value in state.items())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"expert_model_parallel_size": 1}, "expert_model_parallel_size > 1"),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "full activation",
        ),
        ({"moe_scheduler_replica_exchange_buffer_mb": 0}, "must be positive"),
    ],
)
def test_scheduler_rejects_unsupported_memory_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_mxfp8_wrapper_cardinality_and_native_metadata(monkeypatch, slots):
    # Constructor spy checks cardinality/aliasing only, never claims GEMM parity.
    module = ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    module.MXFP8Tensor = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sources = tuple(
        SimpleNamespace(
            dtype=torch.bfloat16,
            _fp8_dtype="E4M3",
            _quantizer=object(),
            _with_gemm_swizzled_scales=True,
        )
        for _ in range(2)
    )
    spec = SimpleNamespace(member_shape=(4, 4), source_tensors=sources)
    views = tuple(tuple(torch.empty(4, dtype=torch.uint8) for _ in range(4)) for _ in range(slots))
    replicas = _wrap_mxfp8(spec, views, torch.device("cpu"))
    assert len(replicas) == slots
    for wrapper, view in zip(replicas, views):
        assert wrapper.rowwise_data is view[0]
        assert wrapper.columnwise_scale_inv is view[3]
    native = _wrap_mxfp8(spec, (views[0],) * 2, torch.device("cpu"), native=True)
    assert len(native) == 2
    assert all(wrapper.quantizer is source._quantizer for wrapper, source in zip(native, sources))


def _projection():
    parameters = tuple(torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16)) for _ in range(2))
    for p in parameters:
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        p.grad_added_to_main_grad = False
    binding = _DirectionalBinding(
        torch.empty(2, dtype=torch.int64),
        None,
        host_pointer_table=torch.empty(1, 2, dtype=torch.int64),
    )
    return _ReplicaProjection(
        "FC1",
        torch.device("cpu"),
        "bf16",
        parameters,
        None,
        parameters,
        binding,
        binding,
        torch.empty(2, dtype=torch.int64),
        (2, 3),
        None,
        None,
        (torch.ones(2, 3, dtype=torch.bfloat16),),
        torch.zeros(1, 2, 3),
        (),
    )


@pytest.mark.parametrize("fsdp", [False, True])
def test_native_gradient_binding_uses_owner_storage(monkeypatch, fsdp):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    projection = _projection()
    for p in projection.parameters:
        p.__fsdp_param__ = fsdp
        p.overwrite_main_grad = fsdp
    projection.prepare_runtime_parameters(torch.float32)
    for runtime, owner in zip(projection.runtime_parameters, projection.parameters):
        assert runtime.main_grad.data_ptr() == owner.main_grad.data_ptr()
        assert runtime.overwrite_main_grad == fsdp
    # FSDP may change its unsharded grad bucket; DDP must retain fixed pointers.
    for p in projection.parameters:
        p.main_grad = torch.full_like(p.main_grad, 7)
    if not fsdp:
        with pytest.raises(RuntimeError, match="storage changed"):
            projection.prepare_runtime_parameters(torch.float32)
    else:
        projection.prepare_runtime_parameters(torch.float32)
        for runtime, owner in zip(projection.runtime_parameters, projection.parameters):
            runtime.main_grad.add_(3)
            torch.testing.assert_close(owner.main_grad, torch.full_like(owner.main_grad, 10))


def test_native_gradient_binding_rejects_missing_full_owner():
    projection = _projection()
    projection.parameters[0].main_grad = None
    with pytest.raises(ValueError, match="owner-provided full-sized main_grad"):
        projection.prepare_runtime_parameters(torch.float32)
    assert projection.native_grad == ()


def test_fsdp_rebinds_retired_storage_with_reused_addresses(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    projection = _projection()
    for owner in projection.parameters:
        owner.__fsdp_param__ = True
        owner.overwrite_main_grad = True
    projection.prepare_runtime_parameters(torch.float32)
    leaves = projection.runtime_parameters
    for runtime, owner in zip(leaves, projection.parameters):
        # Model freed runtime views while the allocator reuses cached addresses
        # for the live owner buckets. Pointer tables already describe the owner.
        runtime.data = torch.empty(0, dtype=owner.dtype)
        runtime.main_grad = torch.empty(0, dtype=torch.float32)
    projection.prepare_runtime_parameters(torch.float32)
    assert projection.runtime_parameters is leaves
    for runtime, owner in zip(leaves, projection.parameters):
        assert runtime.data_ptr() == owner.data_ptr()
        runtime.main_grad.add_(3)
        torch.testing.assert_close(owner.main_grad, torch.full_like(owner.main_grad, 3))


@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_home_only_route_matches_column_reference_without_gather(monkeypatch, tokens):
    context = _context(ep_group=object())
    planner = EPLBLoadPlanner(0, home_update_interval=10, num_logical_experts=4)

    def unexpected(*args, **kwargs):
        raise AssertionError("Home-only steady state must not gather counts")

    monkeypatch.setattr(planner, "_get_count_matrix", unexpected)
    generator = torch.Generator().manual_seed(28)
    routes = torch.rand(tokens, 4, generator=generator) > 0.5
    probs = torch.rand(tokens, 4, generator=generator, requires_grad=True)
    planner.active_home_layout = torch.tensor([[2, 0], [3, 1]])
    planner._load_history = torch.zeros(4, dtype=torch.float64)
    _, replica, result = planner.update_placement(probs, routes, context)
    assert replica is None and isinstance(result, EPLBHomeOnlyPlacementResult)
    actual_routes, actual_probs = planner.reroute(probs, routes, result, context)
    columns = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(actual_routes, routes[:, columns])
    torch.testing.assert_close(actual_probs, probs.masked_fill(~routes, 0)[:, columns])
    (actual_probs * torch.arange(4)).sum().backward()
    torch.testing.assert_close(probs.grad, routes.float() * torch.argsort(columns))
    saved = planner.get_extra_state()
    restored = EPLBLoadPlanner(0, home_update_interval=10, num_logical_experts=4)
    restored.set_extra_state(saved)
    torch.testing.assert_close(restored._load_history, planner._load_history)
    saved.pop("load_history_scope")
    restored.set_extra_state(saved)
    torch.testing.assert_close(restored._load_history, planner._load_history / 2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bounded_replica_pack_and_reduce_cpu_oracle(monkeypatch, dtype):
    """Run production tiled pack/reduce against an independent global-table oracle.

    Only CUDA stream/completion and the wire are simulated. This checks the
    algorithm and allocation bound, not NCCL ordering or device kernels.
    """
    from contextlib import nullcontext

    from megatron.core.transformer.moe.replica_nccl_transport import (
        NcclP2PTransport,
        _compile_schedule,
        _ReplicaChunkWorkspace,
    )
    from megatron.core.transformer.moe.replica_weight_transport import (
        ReplicaGradDestination,
        ReplicaWeightLayout,
        ReplicaWeightSource,
    )

    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    table = [[2, 0, 2, -1], [0, 3, 4, 0], [-1, 1, 5, 2], [-1, -1, -1, -1]]
    shapes = ((3, 7), (2, 5))
    weights = [
        [torch.arange(a * b, dtype=torch.bfloat16).reshape(a, b) + e * 32 for e in range(8)]
        for a, b in shapes
    ]
    replicas = [
        [torch.full((4, a, b), rank + 1, dtype=dtype) for a, b in shapes] for rank in range(4)
    ]
    for rank in range(4):
        schedule = _compile_schedule(table, 2, rank)
        transport = object.__new__(NcclP2PTransport)
        transport.config = SimpleNamespace(device=torch.device("cpu"), num_local_home_experts=2)
        transport.config.grad_dtype = dtype
        transport._mxfp8 = False
        transport._weights = tuple(
            torch.full((4, *shape), -17, dtype=torch.bfloat16) for shape in shapes
        )
        transport._replica_grads = replicas[rank]
        transport._numels = tuple(a * b for a, b in shapes)
        transport._replica_chunks = _ReplicaChunkWorkspace(torch.device("cpu"), 96)
        transport._stream = stream
        transport._schedule = lambda plan: schedule
        transport._finish = lambda *refs: None
        scratch = transport._replica_chunks
        assert (
            sum(
                t.numel() * t.element_size()
                for t in (scratch.send, scratch.receive, scratch.accumulate)
            )
            <= 96
        )

        def wire(reverse):
            component, offset = 0, 0

            def exchange(sends, *, receives):
                nonlocal component, offset
                routes_out = schedule.receives if reverse else schedule.sends
                routes_in = schedule.sends if reverse else schedule.receives
                wire_dtype = dtype if reverse else torch.bfloat16
                width = 32 // (schedule.max_transfer_slots * wire_dtype.itemsize)
                if reverse:
                    width = min(width, 8 // 2)
                count = min(width, transport._numels[component] - offset)
                for route in routes_out:
                    indices = route.replica_slots if reverse else route.home_indices
                    for row, index in zip(sends[route.peer], indices):
                        expected = (
                            replicas[rank][component][index]
                            if reverse
                            else weights[component][rank * 2 + index]
                        )
                        torch.testing.assert_close(
                            row, expected.flatten()[offset : offset + count], rtol=0, atol=0
                        )
                for route in routes_in:
                    for row, home, slot in zip(
                        receives[route.peer], route.home_indices, route.replica_slots
                    ):
                        source = (
                            replicas[route.peer][component][slot]
                            if reverse
                            else weights[component][route.peer * 2 + home]
                        )
                        row.copy_(source.flatten()[offset : offset + count])
                assert sum(t.numel() * t.element_size() for t in sends.values()) <= 32
                assert sum(t.numel() * t.element_size() for t in receives.values()) <= 32
                offset += count
                if offset == transport._numels[component]:
                    component, offset = component + 1, 0
                return receives, ()

            return exchange

        transport._exchange = wire(False)
        sources = tuple(
            ReplicaWeightSource(
                tuple(part[rank * 2 : rank * 2 + 2]), None, ReplicaWeightLayout.PLAIN
            )
            for part in weights
        )
        transport.start_weight_sync(sources=sources, plan=None)
        for p in range(2):
            for slot, expert in enumerate(table[rank]):
                expected = (
                    weights[p][expert] if expert >= 0 else torch.full_like(weights[p][0], -17)
                )
                torch.testing.assert_close(transport._weights[p][slot], expected, rtol=0, atol=0)
        native = tuple(
            ReplicaGradDestination(tuple(torch.full(shape, 256, dtype=dtype) for _ in range(2)))
            for shape in shapes
        )
        transport._exchange = wire(True)
        transport.start_grad_reduce(native_grads=native, plan=None, projections=(1, 0))
        for p in range(2):
            for index, actual in enumerate(native[p].tensors):
                expected = 256 + sum(
                    r + 1
                    for r, row in enumerate(table)
                    for expert in row
                    if expert == rank * 2 + index
                )
                torch.testing.assert_close(
                    actual, torch.full_like(actual, expected), rtol=0, atol=0
                )


def test_home_metadata_still_rejects_changed_options_and_steps():
    """Control-plane consolidation must continue checking live optimizer state."""
    from megatron.core.transformer.moe.home_expert_state import OptimizerHomeExpertState

    parameters = tuple(torch.nn.Parameter(torch.ones(2, 3)) for _ in range(4))
    optimizer = torch.optim.Adam([{"params": [p]} for p in parameters], lr=0.01)
    config = SimpleNamespace(
        device=torch.device("cpu"),
        group=None,
        num_local_home_experts=2,
        member_shapes=((2, 3), (2, 3)),
    )
    state = OptimizerHomeExpertState(optimizer, parameters, config)
    sum(p.square().sum() for p in parameters).backward()
    optimizer.step()
    components = state.prepare_components()
    assert len(components) == 6  # masters plus Adam's two moments, for both projections.
    optimizer.param_groups[0]["lr"] = 0.02
    with pytest.raises(ValueError, match="identical optimizer options"):
        state.prepare_components()
    optimizer.param_groups[0]["lr"] = 0.01
    optimizer.state[parameters[0]]["step"].add_(1)
    with pytest.raises(ValueError, match="common per-parameter optimizer step"):
        state.prepare_components()


@pytest.mark.parametrize(
    "main_dtype,comm_dtype,sharding,expert_dp_size,bf16_reduce,accepted",
    [
        (None, None, "optim_grads_params", 2, False, False),
        (torch.float32, None, "optim_grads_params", 2, False, True),
        (None, torch.float32, "optim_grads_params", 2, False, True),
        (torch.float32, torch.bfloat16, "optim_grads_params", 2, False, False),
        (torch.float32, torch.bfloat16, "optim_grads_params", 1, False, True),
        (None, torch.float32, "optim_grads_params", 1, False, False),
        (torch.float32, torch.bfloat16, "optim", 2, False, True),
        (None, torch.float32, "optim", 2, False, False),
        (None, None, "optim_grads_params", 2, True, True),
        (torch.bfloat16, torch.bfloat16, "optim_grads_params", 2, True, True),
        (torch.float32, None, "optim_grads_params", 2, True, False),
        (torch.float32, torch.bfloat16, "optim_grads_params", 2, True, True),
    ],
)
def test_fsdp_replica_gradient_policy_validated_at_initialization(
    main_dtype, comm_dtype, sharding, expert_dp_size, bf16_reduce, accepted
):
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _validate_moe_scheduler_fsdp

    config = _config(moe_scheduler_num_idle_experts=2, grad_reduce_in_bf16=bf16_reduce)
    ddp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy="optim",
        expert_data_parallel_sharding_strategy=sharding,
        megatron_fsdp_main_grads_dtype=main_dtype,
        megatron_fsdp_grad_comm_dtype=comm_dtype,
    )
    if accepted:
        _validate_moe_scheduler_fsdp(
            config, ddp_config, fsdp_version=1, expert_fsdp_size=expert_dp_size
        )
    else:
        with pytest.raises(ValueError, match="owner main_grad dtype.*Megatron-FSDP exposes"):
            _validate_moe_scheduler_fsdp(
                config, ddp_config, fsdp_version=1, expert_fsdp_size=expert_dp_size
            )
    assert ddp_config.megatron_fsdp_main_grads_dtype == main_dtype
    assert ddp_config.megatron_fsdp_grad_comm_dtype == comm_dtype


def test_fsdp_scheduler_support_boundaries():
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _validate_moe_scheduler_fsdp

    ddp_config = DistributedDataParallelConfig()
    # Zero replicas retain the native FSDP dtype policy, including its BF16 default.
    _validate_moe_scheduler_fsdp(_config(), ddp_config, fsdp_version=1, expert_fsdp_size=2)
    for slots in (0, 2):
        with pytest.raises(ValueError, match="FSDP v1 only"):
            _validate_moe_scheduler_fsdp(
                _config(moe_scheduler_num_idle_experts=slots), ddp_config, fsdp_version=2
            )
    _validate_moe_scheduler_fsdp(_config(moe_enable_scheduler=False), ddp_config, fsdp_version=2)


def test_scheduler_router_retains_boolean_logical_routes():
    from megatron.core.transformer.moe.router import TopKRouter

    router = SimpleNamespace(config=_config(moe_router_fusion=True))
    assert TopKRouter._dense_route_indices_dtype(router) is None


def test_scheduler_rejects_megakernel_bypass():
    with pytest.raises(ValueError, match="moe_megakernel_backend=None"):
        _config(moe_megakernel_backend="mok")


def test_fsdp_wrappers_reject_before_buffer_allocation(monkeypatch):
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter as adapter

    monkeypatch.setattr(adapter, "HAVE_MEGATRON_FSDP", True)
    monkeypatch.setattr(adapter, "has_config_logger_enabled", lambda config: False)
    monkeypatch.setattr(adapter, "log_single_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        adapter.FullyShardedDataParallelV1,
        "_init_dist_index",
        lambda *args: SimpleNamespace(
            get_fsdp_group=lambda **kwargs: SimpleNamespace(size=lambda: 2)
        ),
    )

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("FSDP allocated buffers before validating scheduler ownership")

    monkeypatch.setattr(adapter, "MegatronFSDP", unexpected_allocation)
    config = _config(moe_scheduler_num_idle_experts=2)
    with pytest.raises(ValueError, match="owner main_grad dtype"):
        adapter.FullyShardedDataParallelV1(
            config, DistributedDataParallelConfig(), torch.nn.Linear(2, 2)
        )
    with pytest.raises(ValueError, match="FSDP v1 only"):
        adapter.FullyShardedDataParallelV2(
            config, DistributedDataParallelConfig(), torch.nn.Linear(2, 2)
        )
