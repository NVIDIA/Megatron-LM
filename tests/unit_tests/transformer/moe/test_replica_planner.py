# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for deterministic replica planning and HybridEP routing."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.replica_planner import (
    ReplicaCuTeDSLWeightBridge,
    ReplicaPlan,
    _collect_replica_projection_specs,
    _CuTeDSLReplicaProjection,
    _DirectionalBinding,
    _WeightDirection,
    map_replica_plan_to_hybridep,
    start_replica_grad_reduce_after_expert_backward,
    start_replica_weight_prefetch_before_combine_backward,
    wait_replica_grad_reduce_after_dispatch_backward,
    wait_replica_weight_prefetch_before_expert_backward,
)


def test_replica_hybridep_rank_layout_requires_equal_shapes(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import _validate_replica_rank_layout

    group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_gather_object(outputs, value, group):
        outputs[:] = [value, (value[0] + 1, value[1])]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    with pytest.raises(ValueError, match="equal local token counts"):
        _validate_replica_rank_layout(
            group, num_tokens=8, hidden_dim=16, backend_name="Replica-HybridEP"
        )


@pytest.mark.parametrize(
    ("grad_reduce_in_bf16", "expected_grad_dtype"), [(False, torch.float32), (True, torch.bfloat16)]
)
def test_replica_hybridep_binds_the_cutedsl_bridge(
    monkeypatch, grad_reduce_in_bf16, expected_grad_dtype
):
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import _ReplicaHybridEPManager

    captured = {}
    bridge = object()

    def fake_bridge(**kwargs):
        captured.update(kwargs)
        return bridge

    class FakeExperts:
        def set_replica_weight_bridge(self, value):
            self.bound_bridge = value

    monkeypatch.setattr(token_dispatcher, "ReplicaCuTeDSLWeightBridge", fake_bridge)
    manager = _ReplicaHybridEPManager.__new__(_ReplicaHybridEPManager)
    manager.group = object()
    manager.semantic_num_experts = 8
    manager.num_owned_experts = 2
    manager.config = SimpleNamespace(
        moe_flex_dispatcher_num_sms=12,
        moe_hybridep_num_blocks_permute=7,
        moe_hybridep_num_blocks_unpermute=5,
        moe_hybridep_num_sms_preprocessing=9,
        grad_reduce_in_bf16=grad_reduce_in_bf16,
    )
    experts = FakeExperts()
    manager.bind_experts(experts)

    assert manager._bridge is bridge
    assert experts.bound_bridge is bridge
    assert captured["num_experts"] == 8
    assert captured["num_local_experts"] == 2
    assert captured["grad_dtype"] == expected_grad_dtype


def test_replica_hybridep_metadata_uses_routing_map_for_zero_probability_routes():
    """A selected zero-probability expert must not be replaced by a tied zero."""
    from megatron.core.transformer.moe.token_dispatcher import _ReplicaHybridEPManager

    manager = _ReplicaHybridEPManager.__new__(_ReplicaHybridEPManager)
    manager.semantic_num_experts = 4
    manager.router_topk = 2
    routing_map = torch.tensor([[False, True, False, True], [True, False, True, False]])
    probs = torch.tensor([[0.0, 0.75, 0.0, 0.0], [0.6, 0.0, 0.4, 0.0]], requires_grad=True)

    manager.setup_metadata(routing_map, probs)

    actual_routes = [set(row) for row in manager.semantic_token_indices.tolist()]
    assert actual_routes == [{1, 3}, {0, 2}]
    torch.testing.assert_close(manager.semantic_token_probs.sum(dim=-1), torch.tensor([0.75, 1.0]))
    manager.semantic_token_probs.sum().backward()
    torch.testing.assert_close(probs.grad, routing_map.to(probs.dtype))


def test_replica_hybridep_expands_virtual_routes_for_hybridep():
    plan = ReplicaPlan(
        virtual_experts=torch.tensor([[1, 6], [3, 4]], dtype=torch.int64),
        experts_to_copy=torch.empty((0,), dtype=torch.int32),
    )
    probs = torch.tensor([[0.75, 0.25], [0.6, 0.4]], requires_grad=True)
    routing_map, dense_probs = map_replica_plan_to_hybridep(plan, probs, num_experts=8)

    assert routing_map.shape == (2, 8)
    assert dense_probs.shape == (2, 8)
    assert routing_map[0, 1] and routing_map[0, 6]
    assert dense_probs[1, 3] == 0.6
    dense_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_replica_hybridep_rank_capacity_includes_per_expert_padding():
    from megatron.core.transformer.moe.token_dispatcher import _get_replica_hybridep_rank_capacity

    common = {"num_tokens": 8192, "router_topk": 22, "num_runtime_experts": 64}
    assert (
        _get_replica_hybridep_rank_capacity(**common, capacity_factor=1.0, alignment=256) == 196608
    )
    assert (
        _get_replica_hybridep_rank_capacity(**common, capacity_factor=2.0, alignment=256) == 360448
    )
    assert _get_replica_hybridep_rank_capacity(**common, capacity_factor=1.0, alignment=0) == 180224


def test_replica_weight_bridge_rejects_single_grouped_source_weights():
    packed_linear = SimpleNamespace(in_features=16, out_features=32, single_grouped_weight=True)
    experts = SimpleNamespace(linear_fc1=packed_linear, linear_fc2=packed_linear)

    with pytest.raises(ValueError, match="moe_single_grouped_weight must be False"):
        _collect_replica_projection_specs(
            experts, num_local_experts=2, backend_name="Replica-CuTeDSL"
        )


def test_replica_async_collectives_span_transport_backward():
    events = []
    plan = object()

    class FakeBridge:
        source_parameters = (
            torch.nn.Parameter(torch.ones(())),
            torch.nn.Parameter(torch.ones(())),
        )

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def start_prefetch(self, current_plan, direction=_WeightDirection.FORWARD):
            assert current_plan is plan and direction is _WeightDirection.BACKWARD
            events.append("start_prefetch")

        def wait_prefetch(self, current_plan):
            assert current_plan is plan
            events.append("wait_prefetch")

        def start_grad_reduce(self, current_plan):
            assert current_plan is plan
            events.append("start_grad_reduce")

        def wait_grad_reduce(self, current_plan):
            assert current_plan is plan
            events.append("wait_grad_reduce")
            return self.source_grads

    class BackwardMarker(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, label):
            ctx.label = label
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(ctx.label)
            return grad, None

    bridge = FakeBridge()
    hidden = torch.ones((), requires_grad=True)
    hidden = wait_replica_grad_reduce_after_dispatch_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = start_replica_grad_reduce_after_expert_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = wait_replica_weight_prefetch_before_expert_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = start_replica_weight_prefetch_before_combine_backward(hidden, bridge, plan)
    hidden.backward()

    assert events == [
        "start_prefetch",
        "combine_backward",
        "wait_prefetch",
        "expert_backward",
        "start_grad_reduce",
        "dispatch_backward",
        "wait_grad_reduce",
    ]
    for index, parameter in enumerate(bridge.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )
        assert parameter.grad.data_ptr() != bridge.source_grads[index].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replica_fused_wgrad_handoff_preserves_fp32():
    """Accumulate an FP32 replica wgrad before returning a BF16 dummy."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16, device=device))
    parameter.main_grad = torch.zeros(4, dtype=torch.float32, device=device)
    parameter.grad_added_to_main_grad = False
    external_wgrad = torch.full((4,), 1.0001, dtype=torch.float32, device=device)
    plan = object()

    class FakeBridge:
        source_parameters = (parameter,)

        @staticmethod
        def wait_grad_reduce(current_plan):
            assert current_plan is plan
            return (external_wgrad,)

    hidden = torch.ones((), device=device, requires_grad=True)
    wait_replica_grad_reduce_after_dispatch_backward(hidden, FakeBridge(), plan).backward()

    torch.testing.assert_close(parameter.main_grad, external_wgrad, rtol=0, atol=0)
    assert parameter.grad_added_to_main_grad
    assert parameter.grad.dtype == torch.bfloat16


def test_replica_planner_slots_preserve_two_outstanding_forwards(monkeypatch):
    """Repeated MTP forwards must retain disjoint plans until their own backwards."""
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import _ReplicaHybridEPManager

    allocated = []

    def fake_allocate(**kwargs):
        workspace = SimpleNamespace(
            virtual_experts=torch.empty(
                (kwargs["num_tokens"], kwargs["router_topk"]), dtype=torch.int64
            ),
            experts_to_copy=torch.empty(
                (kwargs["ep_size"], kwargs["num_experts"] // kwargs["ep_size"]),
                dtype=torch.int32,
            ),
        )
        allocated.append(workspace)
        return workspace

    planner_calls = 0

    def fake_plan_replica_routes(indices, counts, group, workspace, on_placement_ready):
        nonlocal planner_calls
        del indices, counts, group
        planner_calls += 1
        workspace.virtual_experts.fill_(planner_calls)
        workspace.experts_to_copy.fill_(planner_calls)
        plan = ReplicaPlan(workspace.virtual_experts, workspace.experts_to_copy)
        on_placement_ready(plan)
        return plan

    class FakeBridge:
        source_parameters = ()

        def __init__(self):
            self.last_plan = None
            self.prefetched = []
            self.reduced = []

        def start_prefetch(self, plan):
            self.prefetched.append(plan)

        def wait_grad_reduce(self, plan):
            self.reduced.append(plan)
            return ()

    monkeypatch.setattr(
        token_dispatcher.ReplicaPlannerWorkspace,
        "allocate",
        classmethod(lambda cls, **kwargs: fake_allocate(**kwargs)),
    )
    monkeypatch.setattr(token_dispatcher, "plan_replica_routes", fake_plan_replica_routes)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    manager = _ReplicaHybridEPManager.__new__(_ReplicaHybridEPManager)
    manager._replica_backend_name = "Replica-HybridEP"
    manager.semantic_num_experts = 2
    manager.semantic_token_indices = torch.zeros((2, 1), dtype=torch.int32)
    manager.semantic_tokens_per_expert = torch.tensor([2, 0], dtype=torch.int32)
    manager.semantic_token_probs = torch.ones((2, 1), requires_grad=True)
    manager.group = object()
    manager._bridge = FakeBridge()
    manager._planner_num_tokens = 2
    manager._replica_plan_slots = []
    manager._active_replica_plan_slot = None
    manager._plan = None

    retained_outputs = []
    retained_plans = []
    retained_snapshots = []
    for depth in range(2):
        manager.semantic_token_indices.fill_(depth)
        plan = manager._prepare_replica_plan(torch.ones((2, 4)))
        retained_plans.append(plan)
        retained_snapshots.append((plan.virtual_experts.clone(), plan.experts_to_copy.clone()))
        retained_outputs.append(
            manager._wrap_replica_dispatch_input(torch.ones((), requires_grad=True))
        )
        manager._finish_replica_plan()

    assert len(allocated) == 2
    assert all(slot.in_use for slot in manager._replica_plan_slots)
    assert allocated[0] is not allocated[1]
    for plan, (expected_routes, expected_copies) in zip(retained_plans, retained_snapshots):
        torch.testing.assert_close(plan.virtual_experts, expected_routes)
        torch.testing.assert_close(plan.experts_to_copy, expected_copies)

    torch.stack(retained_outputs).sum().backward()

    assert not any(slot.in_use for slot in manager._replica_plan_slots)
    assert {id(plan) for plan in manager._bridge.reduced} == {
        id(plan) for plan in retained_plans
    }


def _set_main_grad(parameter, dtype=torch.float32):
    parameter.main_grad = torch.zeros(parameter.shape, dtype=dtype, device=parameter.device)
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = True


def _test_directional_binding(device, num_experts, components=0):
    return _DirectionalBinding(
        torch.empty(num_experts, dtype=torch.int64, device=device),
        (
            torch.empty(num_experts, dtype=torch.int64, device=device)
            if components == 2
            else None
        ),
        host_pointer_table=(
            torch.empty((components, num_experts), dtype=torch.int64, pin_memory=True)
            if components
            else None
        ),
    )


def _test_projection(
    *,
    parameters,
    source_tensors,
    virtual_weight,
    virtual_grad,
    member_shape,
    forward,
    backward,
    weight_format="bf16",
    scale_shape=None,
    gtp_native_grad=None,
    runtime_parameters=None,
):
    device = source_tensors[0].device
    native_grad = (
        gtp_native_grad
        if gtp_native_grad is not None
        else torch.empty(
            (len(parameters), *member_shape), dtype=virtual_grad.dtype, device=device
        )
    )
    return _CuTeDSLReplicaProjection(
        name="test projection",
        device=device,
        weight_format=weight_format,
        parameters=parameters,
        gtp_leader=parameters[0] if gtp_native_grad is not None else None,
        source_tensors=source_tensors,
        forward=forward,
        backward=backward,
        native_grad_bases=torch.empty(len(parameters), dtype=torch.int64, device=device),
        member_shape=member_shape,
        member_numel=member_shape[0] * member_shape[1],
        rowwise_scale_shape=scale_shape,
        columnwise_scale_shape=scale_shape,
        virtual_weight=virtual_weight,
        virtual_grad=virtual_grad,
        native_grad=native_grad,
        runtime_parameters=runtime_parameters,
        runtime_bound=runtime_parameters is not None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) > 1,
    reason="Process-local CUDA graph probe must run outside distributed parity tests",
)
def test_replica_native_grad_table_ignores_transient_gtp_buffers_during_capture():
    """Keep replica reduction bound to native staging when GTP scratch changes."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 2
    member_shape = (4, 4)
    source_parameters = tuple(
        torch.nn.Parameter(torch.empty(member_shape, dtype=torch.bfloat16, device=device))
        for _ in range(num_local_experts)
    )
    source_tensors = tuple(parameter.data for parameter in source_parameters)
    native_grad = torch.empty(
        (num_local_experts, *member_shape), dtype=torch.float32, device=device
    )
    virtual_weight = tuple(
        torch.empty(member_shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_local_experts)
    )
    virtual_grad = torch.empty_like(native_grad)
    runtime_parameters = []
    for weight, grad in zip(
        source_tensors + virtual_weight, tuple(native_grad) + tuple(virtual_grad)
    ):
        runtime_parameter = torch.nn.Parameter(weight, requires_grad=True)
        runtime_parameter.main_grad = grad
        runtime_parameter.grad_added_to_main_grad = True
        runtime_parameter.overwrite_main_grad = True
        runtime_parameters.append(runtime_parameter)

    forward = _test_directional_binding(device, num_local_experts, components=1)
    backward = _test_directional_binding(device, num_local_experts, components=1)
    projection = _test_projection(
        parameters=source_parameters,
        source_tensors=source_tensors,
        gtp_native_grad=native_grad,
        virtual_weight=virtual_weight,
        virtual_grad=virtual_grad,
        member_shape=member_shape,
        forward=forward,
        backward=backward,
        runtime_parameters=tuple(runtime_parameters),
    )
    bridge = ReplicaCuTeDSLWeightBridge.__new__(ReplicaCuTeDSLWeightBridge)
    bridge.device = device
    bridge.num_local_experts = num_local_experts
    bridge.projections = [projection]
    bridge.workspace = SimpleNamespace(grad_dtype=torch.float32)
    bridge.prepare_runtime_parameters()

    stable_ptrs = tuple(grad.data_ptr() for grad in native_grad)
    assert projection.native_grad_ptrs == stable_ptrs
    # Model the eager/captured GTP buffers that previously replaced the stable
    # destination table. The bridge must not consult them during forward capture.
    projection.gtp_leader.gtp_wgrad_tensors = tuple(
        torch.empty_like(grad) for grad in native_grad
    )
    capture_probe = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bridge.prepare_runtime_parameters()
        capture_probe.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)

    assert projection.native_grad_ptrs == stable_ptrs
    assert tuple(projection.native_grad_bases.cpu().tolist()) == stable_ptrs
    torch.testing.assert_close(capture_probe, torch.ones_like(capture_probe), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) > 1,
    reason="Process-local CUDA graph probe must run outside distributed parity tests",
)
def test_replica_gtp_bf16_weights_bind_directional_stable_buffers():
    """Bind separate stable GTP forward/backward buffers without packing or copies."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 2
    member_shape = (4, 4)
    forward_weights = tuple(
        torch.empty(member_shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_local_experts)
    )
    backward_weights = tuple(
        torch.empty(member_shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_local_experts)
    )
    runtime_parameters = tuple(
        torch.nn.Parameter(torch.empty_like(weight)) for weight in forward_weights
    )
    forward = _test_directional_binding(device, num_local_experts, components=1)
    backward = _test_directional_binding(device, num_local_experts, components=1)
    projection = _test_projection(
        parameters=runtime_parameters,
        gtp_native_grad=torch.empty(
            (num_local_experts, *member_shape), dtype=torch.float32, device=device
        ),
        member_shape=member_shape,
        source_tensors=forward_weights,
        forward=forward,
        backward=backward,
        virtual_weight=(),
        virtual_grad=torch.empty(
            (0, *member_shape), dtype=torch.float32, device=device
        ),
        runtime_parameters=runtime_parameters,
    )

    runtime_ids = tuple(id(parameter) for parameter in runtime_parameters)
    capture_probe = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        projection.bind_materialized_weights(forward_weights, _WeightDirection.FORWARD)
        capture_probe.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)

    forward_ptrs = tuple(weight.data_ptr() for weight in forward_weights)
    assert forward.source_ptrs == tuple((ptr,) for ptr in forward_ptrs)
    assert tuple(forward.data_bases.cpu().tolist()) == forward_ptrs
    assert tuple(parameter.data_ptr() for parameter in runtime_parameters) == forward_ptrs

    projection.bind_materialized_weights(backward_weights, _WeightDirection.BACKWARD)
    backward_ptrs = tuple(weight.data_ptr() for weight in backward_weights)
    assert backward.source_ptrs == tuple((ptr,) for ptr in backward_ptrs)
    assert tuple(backward.data_bases.cpu().tolist()) == backward_ptrs
    assert tuple(parameter.data_ptr() for parameter in runtime_parameters) == backward_ptrs
    assert tuple(id(parameter) for parameter in runtime_parameters) == runtime_ids

    projection.bind_materialized_weights(forward_weights, _WeightDirection.FORWARD)
    assert tuple(parameter.data_ptr() for parameter in runtime_parameters) == forward_ptrs
    replacement = list(forward_weights)
    replacement[0] = torch.empty_like(replacement[0])
    with pytest.raises(RuntimeError, match="forward all-gather storage changed"):
        projection.bind_materialized_weights(
            tuple(replacement), _WeightDirection.FORWARD
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Replica weights require CUDA")
def test_replica_bf16_runtime_over_64_experts_remains_discrete():
    """Build more than 64 runtime experts without an internal GroupedTensor."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 33
    member_shape = (4, 4)
    parameters = tuple(
        torch.nn.Parameter(torch.empty(member_shape, dtype=torch.bfloat16, device=device))
        for _ in range(num_local_experts)
    )
    for parameter in parameters:
        _set_main_grad(parameter)
    virtual_weight = tuple(
        torch.empty(member_shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_local_experts)
    )
    virtual_grad = torch.empty(
        (num_local_experts, *member_shape), dtype=torch.float32, device=device
    )
    binding = _test_directional_binding(device, num_local_experts)
    projection = _test_projection(
        parameters=parameters,
        source_tensors=tuple(parameter.data for parameter in parameters),
        virtual_weight=virtual_weight,
        virtual_grad=virtual_grad,
        member_shape=member_shape,
        forward=binding,
        backward=binding,
    )
    bridge = ReplicaCuTeDSLWeightBridge.__new__(ReplicaCuTeDSLWeightBridge)
    bridge.device = device
    bridge.num_local_experts = num_local_experts
    bridge.num_runtime_experts = 2 * num_local_experts
    bridge.projections = [projection]
    bridge.workspace = SimpleNamespace(grad_dtype=torch.float32)

    bridge.prepare_runtime_parameters()

    assert len(projection.runtime_parameters) == 66
    assert all(
        isinstance(parameter, torch.nn.Parameter) for parameter in projection.runtime_parameters
    )
    assert tuple(parameter.data_ptr() for parameter in projection.runtime_parameters) == tuple(
        weight.data_ptr()
        for weight in tuple(parameter.data for parameter in parameters) + virtual_weight
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) > 1,
    reason="Process-local CUDA graph probe must run outside distributed parity tests",
)
def test_replica_gtp_mxfp8_weights_bind_directly_during_capture():
    """Alias static GTP gather storage without capturing full-weight DtoD copies."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 2
    member_shape = (4, 4)
    scale_shape = (2, 2)

    def make_weight():
        return SimpleNamespace(
            shape=member_shape,
            device=device,
            _rowwise_data=torch.empty(member_shape, dtype=torch.uint8, device=device),
            _rowwise_scale_inv=torch.empty(scale_shape, dtype=torch.uint8, device=device),
            _columnwise_data=torch.empty(member_shape, dtype=torch.uint8, device=device),
            _columnwise_scale_inv=torch.empty(scale_shape, dtype=torch.uint8, device=device),
        )

    destinations = tuple(make_weight() for _ in range(num_local_experts))
    materialized = tuple(make_weight() for _ in range(num_local_experts))
    runtime_parameters = tuple(make_weight() for _ in range(num_local_experts))
    forward = _test_directional_binding(device, num_local_experts, components=2)
    backward = _test_directional_binding(device, num_local_experts, components=2)
    projection = _test_projection(
        parameters=destinations,
        source_tensors=destinations,
        virtual_weight=(),
        virtual_grad=torch.empty(
            (0, *member_shape), dtype=torch.float32, device=device
        ),
        member_shape=member_shape,
        forward=forward,
        backward=backward,
        weight_format="mxfp8",
        scale_shape=scale_shape,
        gtp_native_grad=torch.empty(
            (num_local_experts, *member_shape), dtype=torch.float32, device=device
        ),
        runtime_parameters=runtime_parameters,
    )
    projection.source_storage_ptrs = tuple(_weight_storage_ptrs(w) for w in destinations)

    capture_probe = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        projection.bind_materialized_weights(materialized, _WeightDirection.FORWARD)
        capture_probe.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)

    rowwise_ptrs = tuple(
        (weight._rowwise_data.data_ptr(), weight._rowwise_scale_inv.data_ptr())
        for weight in materialized
    )
    assert forward.source_ptrs == rowwise_ptrs
    assert tuple(forward.data_bases.cpu().tolist()) == tuple(ptrs[0] for ptrs in rowwise_ptrs)
    assert tuple(forward.scale_bases.cpu().tolist()) == tuple(ptrs[1] for ptrs in rowwise_ptrs)
    for destination, runtime_parameter, source in zip(
        destinations, runtime_parameters, materialized
    ):
        assert destination._rowwise_data is source._rowwise_data
        assert destination._rowwise_scale_inv is source._rowwise_scale_inv
        assert runtime_parameter._rowwise_data is source._rowwise_data
        assert runtime_parameter._rowwise_scale_inv is source._rowwise_scale_inv
    torch.testing.assert_close(capture_probe, torch.ones_like(capture_probe), rtol=0, atol=0)

    # Repeated materialization is validation-only: it retains the same aliases
    # and does not enqueue another pointer-table update or weight copy.
    projection.bind_materialized_weights(materialized, _WeightDirection.FORWARD)
    replacement = list(materialized)
    replacement[0] = make_weight()
    with pytest.raises(RuntimeError, match="all-gather storage changed"):
        projection.bind_materialized_weights(
            tuple(replacement), _WeightDirection.FORWARD
        )


def _set_main_grads(layer, dtype=torch.float32):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        parameters = tuple(linear.get_parameter(f"weight{i}") for i in range(linear.num_gemms))
        for parameter in parameters:
            _set_main_grad(parameter, dtype)
    if layer.config.moe_latent_size is not None:
        _set_main_grad(layer.fc1_latent_proj.weight, dtype)
        _set_main_grad(layer.fc2_latent_proj.weight, dtype)


def _stack_linear_main_grad(linear):
    return torch.stack(
        tuple(
            linear.get_parameter(f"weight{i}").main_grad.detach() for i in range(linear.num_gemms)
        )
    )


def _weight_storage_ptrs(weight):
    if hasattr(weight, "_rowwise_data"):
        return (
            weight._rowwise_data.data_ptr(),
            weight._rowwise_scale_inv.data_ptr(),
            weight._columnwise_data.data_ptr(),
            weight._columnwise_scale_inv.data_ptr(),
        )
    return (weight.data_ptr(),)


def _assert_replica_mxfp8_prefetch_exact(bridge, orientation):
    """Check every active virtual MXFP8 component against its owning rank."""
    assert orientation in ("rowwise", "columnwise")
    component_names = (
        ("_rowwise_data", "_rowwise_scale_inv")
        if orientation == "rowwise"
        else ("_columnwise_data", "_columnwise_scale_inv")
    )
    local_errors = []
    for projection_index, projection in enumerate(bridge.projections):
        for component_name in component_names:
            local_sources = torch.stack(
                tuple(getattr(source, component_name) for source in projection.source_tensors)
            )
            gathered_sources = [torch.empty_like(local_sources) for _ in range(bridge.world_size)]
            torch.distributed.all_gather(gathered_sources, local_sources, group=bridge.group)
            for slot, global_expert in enumerate(
                bridge.last_plan.experts_to_copy[bridge.rank].tolist()
            ):
                if global_expert < 0:
                    continue
                owner_rank = global_expert // bridge.num_local_experts
                owner_expert = global_expert % bridge.num_local_experts
                actual = getattr(projection.virtual_weight[slot], component_name)
                expected = gathered_sources[owner_rank][owner_expert]
                if not torch.equal(actual, expected):
                    local_errors.append(
                        f"projection={projection_index} component={component_name} "
                        f"slot={slot} expert={global_expert}"
                    )
    any_error = torch.tensor(int(bool(local_errors)), dtype=torch.int32, device=bridge.device)
    torch.distributed.all_reduce(any_error, op=torch.distributed.ReduceOp.MAX, group=bridge.group)
    assert not any_error.item(), f"rank {bridge.rank} {orientation} MXFP8 prefetch mismatch: " + (
        ", ".join(local_errors) if local_errors else "reported by another rank"
    )


def _run_replica_hybridep_full_layer_parity(
    monkeypatch,
    backend,
    activation_func,
    gated_linear_unit,
    weighted_squared_relu,
    glu_interleave,
    moe_latent_size,
    mxfp8=False,
    gtp=False,
    grad_dtype=torch.float32,
    reference_dispatcher="alltoall",
    verify_hybridep_contract=False,
):
    """Compare full expert/router fwd+bwd and grouped main_grads on 4 NVLink GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("replica_hybridep distributed coverage requires a 4-rank torchrun launch")

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    expert_model_parallel_size = 2 if gtp else 4
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        expert_gtp_remat_size=2 if gtp else 1,
    )
    if gtp:
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        # This test isolates the bridge's explicit materialize-before-exchange
        # dependency. The production-script test covers linked async GTP chains.
        update_gtp_config(
            weight_prefetch=False,
            async_reduction=False,
            reduce_scatter_with_fp32_accumulation=(grad_dtype == torch.bfloat16),
        )
    torch.manual_seed(1234)

    common = {
        "num_layers": 1,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "num_attention_heads": 8,
        "num_moe_experts": 4,
        "expert_model_parallel_size": expert_model_parallel_size,
        "expert_tensor_parallel_size": 1,
        "expert_tensor_parallel_num_weight_shards": 2 if gtp else 1,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": False,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_cpu_initialization": False,
        "activation_func": activation_func,
        "gated_linear_unit": gated_linear_unit,
        "use_fused_weighted_squared_relu": weighted_squared_relu,
        "moe_mlp_glu_interleave_size": glu_interleave,
        "moe_latent_size": moe_latent_size,
    }
    if mxfp8:
        common.update(
            fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
        )
    if reference_dispatcher == "alltoall":
        reference_config = TransformerConfig(**common, moe_token_dispatcher_type="alltoall")
    elif reference_dispatcher == "hybridep":
        reference_config = TransformerConfig(
            **common, moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep"
        )
    else:
        raise ValueError(f"Unsupported reference dispatcher {reference_dispatcher!r}.")
    backend_config = {}
    if backend == "replica_hybridep":
        bf16_grad_reduce = grad_dtype == torch.bfloat16
        backend_config.update(
            grad_reduce_in_bf16=bf16_grad_reduce,
            ddp_reduce_scatter_with_fp32_accumulation=bf16_grad_reduce,
            gtp_remat_reduce_scatter_with_fp32_accumulation=bf16_grad_reduce and gtp,
        )
    replica_config = TransformerConfig(
        **common,
        **backend_config,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend=backend,
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:
        if mxfp8:
            from transformer_engine.common.recipe import MXFP8BlockScaling
            from transformer_engine.pytorch import fp8_model_init

            with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                ref_layer = MoELayer(reference_config, submodules).cuda()
            with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                replica_layer = MoELayer(replica_config, submodules).cuda()
        else:
            ref_layer = MoELayer(reference_config, submodules).cuda()
            replica_layer = MoELayer(replica_config, submodules).cuda()
        for layer in (ref_layer, replica_layer):
            assert not layer.experts.linear_fc1.single_grouped_weight
            assert not layer.experts.linear_fc2.single_grouped_weight
        if mxfp8 and moe_latent_size is not None:
            # In production DDP exposes an MXFP8 parameter's main-grad buffer
            # through its distributed-weight wrapper. This focused MoELayer
            # test has no DDP wrapper, so let the two ordinary latent linears
            # return wgrads through autograd; expert wgrads remain fused and
            # exercise replica reduction.
            for layer in (ref_layer, replica_layer):
                layer.fc1_latent_proj.fuse_wgrad_accumulation = False
                layer.fc2_latent_proj.fuse_wgrad_accumulation = False
        replica_layer.load_state_dict(ref_layer.state_dict())
        assert replica_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer, grad_dtype)
        _set_main_grads(replica_layer, grad_dtype)
        if backend.startswith("replica_"):
            bridge = replica_layer.token_dispatcher._comm_manager._bridge
            assert bridge.workspace.grad_arena.dtype == grad_dtype
            if mxfp8:
                assert bridge.workspace.weight_arena is not None
                assert bridge.workspace.weight_handle is not None
                for projection in bridge.projections:
                    for virtual in projection.virtual_weight:
                        assert virtual._rowwise_data.data_ptr() == virtual._columnwise_data.data_ptr()
            if gtp and mxfp8:
                # GTP gather outputs replace these aliases before execution; no
                # second full native row/column staging allocation is retained.
                for projection in bridge.projections:
                    if projection.gtp_leader is None:
                        # Bare MoELayer construction does not install the DDP-time
                        # GTP remat wrapper exercised by the full recipe.
                        continue
                    for source, virtual in zip(
                        projection.source_tensors, projection.virtual_weight
                    ):
                        assert source is not virtual
                        assert source._rowwise_data.data_ptr() == virtual._rowwise_data.data_ptr()
                        assert (
                            source._columnwise_data.data_ptr()
                            == virtual._columnwise_data.data_ptr()
                        )
            assert all(
                projection.virtual_grad.dtype == grad_dtype for projection in bridge.projections
            )
            bridge.prepare_runtime_parameters()
            if mxfp8:
                reference_specs, _ = _collect_replica_projection_specs(
                    ref_layer.experts,
                    num_local_experts=bridge.num_local_experts,
                    backend_name="test-alltoall",
                )
                for reference, replica in zip(reference_specs, bridge.projections):
                    for reference_weight, replica_weight in zip(
                        reference.source_tensors, replica.source_tensors
                    ):
                        for component_name in (
                            "_rowwise_data",
                            "_rowwise_scale_inv",
                            "_columnwise_data",
                            "_columnwise_scale_inv",
                        ):
                            getattr(replica_weight, component_name).copy_(
                                getattr(reference_weight, component_name)
                            )
                            torch.testing.assert_close(
                                getattr(replica_weight, component_name),
                                getattr(reference_weight, component_name),
                                rtol=0,
                                atol=0,
                            )
            if mxfp8:
                with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                    second_layer = MoELayer(replica_config, submodules).cuda()
            else:
                second_layer = MoELayer(replica_config, submodules).cuda()
            second_bridge = second_layer.token_dispatcher._comm_manager._bridge
            assert second_bridge.workspace is bridge.workspace
            second_bridge.destroy()
            del second_layer
            for projection, runtime_weights in zip(
                bridge.projections, (bridge.runtime_fc1_weights, bridge.runtime_fc2_weights)
            ):
                assert len(runtime_weights) == bridge.num_runtime_experts
                native_weights = projection.source_tensors
                assert projection.native_grad is not None
                native_grads = tuple(projection.native_grad)
                for index, runtime_weight in enumerate(runtime_weights):
                    if index < bridge.num_local_experts:
                        expected_weight = native_weights[index]
                        expected_grad = native_grads[index]
                    else:
                        slot = index - bridge.num_local_experts
                        expected_weight = projection.virtual_weight[slot]
                        expected_grad = projection.virtual_grad[slot]
                    assert _weight_storage_ptrs(runtime_weight) == _weight_storage_ptrs(
                        expected_weight
                    )
                    assert runtime_weight.main_grad.data_ptr() == expected_grad.data_ptr()

        torch.manual_seed(1234)
        test_input = torch.randn(2, 4, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer, *, replica_bridge=None):
            hidden = test_input.detach().clone().requires_grad_(True)
            output, _ = layer(hidden)
            if replica_bridge is not None and mxfp8:
                _assert_replica_mxfp8_prefetch_exact(replica_bridge, "rowwise")
            output.float().sum().backward()
            if replica_bridge is not None:
                for projection in replica_bridge.projections:
                    if projection.gtp_leader is not None:
                        for parameter in projection.parameters:
                            parameter.grad = None
                        continue
                    assert all(parameter.grad is not None for parameter in projection.parameters)
                    for parameter in projection.parameters:
                        assert parameter.grad_added_to_main_grad
                        parameter.grad = None
                assert all(
                    runtime_parameter.grad is None
                    for projection in replica_bridge.projections
                    for runtime_parameter in projection.runtime_parameters
                )
            if replica_bridge is not None and mxfp8:
                _assert_replica_mxfp8_prefetch_exact(replica_bridge, "columnwise")
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                _stack_linear_main_grad(layer.experts.linear_fc1),
                _stack_linear_main_grad(layer.experts.linear_fc2),
            ]
            if layer.config.moe_latent_size is not None:
                latent_grads = []
                for latent_projection in (layer.fc1_latent_proj, layer.fc2_latent_proj):
                    gradient = (
                        latent_projection.weight.main_grad
                        if latent_projection.fuse_wgrad_accumulation
                        else latent_projection.weight.grad
                    )
                    latent_grads.append(gradient.detach().clone())
                values.extend(latent_grads)
            return values

        ref_values = run(ref_layer)
        if reference_dispatcher == "hybridep":
            # HybridEP owns one process-global buffer for a fixed local-expert
            # count. Baseline and replica layouts use N and 2N respectively,
            # so reinitialize it between the two sequential comparisons.
            torch.cuda.synchronize()
            torch.distributed.barrier()
            fused_a2a.reset_hybrid_ep_buffer()
            torch.distributed.barrier()
        replica_values = run(replica_layer, replica_bridge=bridge)
        if backend == "replica_hybridep":
            manager = replica_layer.token_dispatcher._comm_manager
            assert manager.moe_expert_rank_capacity_factor == 1.0
            assert not manager.over_budget.item()
            if mxfp8:
                # This input has far fewer than 256 routes per runtime expert, so the
                # comparison below exercises padding-heavy execution. Matching input,
                # router, and expert-weight gradients proves padding is gradient-neutral.
                assert torch.all(manager.tokens_per_expert % 256 == 0)
                num_routes = test_input.shape[0] * test_input.shape[1] * 2
                num_dispatched = manager.tokens_per_expert.sum().item()
                assert num_dispatched > num_routes
                dispatched_probs = manager.dispatched_probs[:num_dispatched]
                assert torch.count_nonzero(dispatched_probs).item() == num_routes
        if reference_dispatcher == "hybridep":
            active_replica = torch.any(bridge.last_plan.experts_to_copy >= 0).to(torch.int32)
            torch.distributed.all_reduce(active_replica, op=torch.distributed.ReduceOp.MAX)
            assert active_replica.item(), "HybridEP parity must exercise an active replica"
        if backend.startswith("replica_") and not mxfp8:
            for projection in bridge.projections:
                torch.testing.assert_close(
                    projection.virtual_grad,
                    torch.zeros_like(projection.virtual_grad),
                    rtol=0,
                    atol=0,
                )
        value_names = ["output", "input grad", "router grad", "FC1 main_grad", "FC2 main_grad"]
        if moe_latent_size is not None:
            value_names.extend(["latent FC1 main_grad", "latent FC2 main_grad"])
        for value_name, actual, expected in zip(value_names, replica_values, ref_values):
            if verify_hybridep_contract and value_name not in ("FC1 main_grad", "FC2 main_grad"):
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"{value_name} must be bitwise equal: {msg}",
                )
                continue
            if verify_hybridep_contract:
                # A replicated expert's wgrad is accumulated from independently
                # rounded FP32 partials. This changes addition order but not the
                # mathematical gradient.
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-7,
                    atol=2e-6,
                    msg=lambda msg: f"{value_name} reduction-order bound: {msg}",
                )
                continue
            if mxfp8:
                # Replica placement changes the token population of each MX
                # quantization block. Per-element absolute tolerances are not
                # stable under those different block boundaries. Bound
                # execution-level noise while the raw weights and scales
                # remain byte-exact above; these limits are far below the
                # corruption produced by a wrong expert or scale mapping.
                if "main_grad" in value_name:
                    atol = 32.0
                elif value_name == "router grad":
                    atol = 16.0
                else:
                    atol = 0.75
                torch.testing.assert_close(
                    actual, expected, rtol=0.2, atol=atol, msg=lambda msg: f"{value_name}: {msg}"
                )
                continue
            # Different dispatchers may change BF16 reduction ordering even
            # when replica planning leaves the mathematical result unchanged.
            rtol = 2e-2
            atol = 2e-2
            torch.testing.assert_close(
                actual, expected, rtol=rtol, atol=atol, msg=lambda msg: f"{value_name}: {msg}"
            )
        if (
            backend.startswith("replica_")
            and not mxfp8
            and not gtp
            and not verify_hybridep_contract
        ):
            # Weight transfer runs at the eager prefetch boundary outside the
            # configured expert graph. Capture consumers of the stable native-
            # then-virtual wrappers, refresh virtual slots from changed native
            # weights, and prove replay observes unchanged wrapper addresses.
            plan = bridge.last_plan
            bridge.start_prefetch(plan)
            bridge.wait_prefetch(plan)
            runtime_weight_tuples = (bridge.runtime_fc1_weights, bridge.runtime_fc2_weights)
            stable_pointers = tuple(
                tuple(weight.data_ptr() for weight in runtime_weights)
                for runtime_weights in runtime_weight_tuples
            )
            captured_weights = tuple(
                torch.empty(
                    (bridge.num_runtime_experts, *projection.member_shape),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for projection in bridge.projections
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for captured, runtime_weights in zip(captured_weights, runtime_weight_tuples):
                    for index, runtime_weight in enumerate(runtime_weights):
                        captured[index].copy_(runtime_weight)
            before_source_update = tuple(
                torch.stack(tuple(runtime_weight.detach() for runtime_weight in runtime_weights))
                for runtime_weights in runtime_weight_tuples
            )
            for projection in bridge.projections:
                for parameter in projection.parameters:
                    rowwise_data = getattr(parameter, "rowwise_data", parameter.data)
                    rowwise_data.add_(1000)
            for runtime_weights, previous in zip(runtime_weight_tuples, before_source_update):
                current = torch.stack(
                    tuple(runtime_weight.detach() for runtime_weight in runtime_weights)
                )
                torch.testing.assert_close(
                    current[: bridge.num_local_experts],
                    previous[: bridge.num_local_experts] + 1000,
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"rank {bridge.rank} native alias update check: {msg}",
                )
                torch.testing.assert_close(
                    current[bridge.num_local_experts :],
                    previous[bridge.num_local_experts :],
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"rank {bridge.rank} virtual isolation check: {msg}",
                )
            # The owner-push writes directly into peer virtual arenas. Ensure every
            # rank has validated the old contents before any rank starts the refresh.
            torch.distributed.barrier(group=bridge.group)
            plan_snapshot = plan.experts_to_copy.clone()
            # The normal training path may cache this exact plan in the shared
            # workspace.  Evict that cache here so this check exercises an
            # actual refresh from the modified optimizer-owned weights.
            bridge.workspace.resident_bridge = None
            bridge.workspace.resident_plan = None
            bridge.start_prefetch(plan)
            bridge.wait_prefetch(plan)
            torch.testing.assert_close(
                plan.experts_to_copy,
                plan_snapshot,
                rtol=0,
                atol=0,
                msg=lambda msg: f"rank {bridge.rank} plan stability check: {msg}",
            )
            after_prefetch = tuple(
                torch.stack(tuple(runtime_weight.detach() for runtime_weight in runtime_weights))
                for runtime_weights in runtime_weight_tuples
            )
            graph.replay()
            torch.cuda.synchronize()
            active_slots = plan.experts_to_copy[bridge.rank] >= 0
            for pointers, captured, previous, current, runtime_weights in zip(
                stable_pointers,
                captured_weights,
                before_source_update,
                after_prefetch,
                runtime_weight_tuples,
            ):
                torch.testing.assert_close(captured, current, rtol=0, atol=0)
                if torch.any(active_slots):
                    assert torch.any(
                        current[bridge.num_local_experts :][active_slots]
                        != previous[bridge.num_local_experts :][active_slots]
                    )
                torch.testing.assert_close(
                    current[bridge.num_local_experts :][~active_slots],
                    previous[bridge.num_local_experts :][~active_slots],
                    rtol=0,
                    atol=0,
                )
                assert tuple(weight.data_ptr() for weight in runtime_weights) == pointers
    finally:
        # Replica bridges own CUDA work that can reference the HybridEP execution
        # context. Finalize them first, then destroy the process-global buffer in
        # lockstep across ranks.
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if gtp:
            update_gtp_config(
                weight_prefetch=True,
                async_reduction=True,
                reduce_scatter_with_fp32_accumulation=False,
            )


def _run_replica_hybridep_repeated_mtp_parity(monkeypatch):
    """Compare a two-depth shared MTP block against ordinary HybridEP end to end."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("replica_hybridep repeated-MTP coverage requires a 4-rank torchrun launch")

    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_mtp_block_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)

    common = {
        "num_layers": 1,
        "hidden_size": 128,
        "ffn_hidden_size": 256,
        "moe_ffn_hidden_size": 256,
        "num_attention_heads": 8,
        "kv_channels": 16,
        "num_moe_experts": 4,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": False,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_cpu_initialization": False,
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "attention_backend": AttnBackend.unfused,
        "mtp_num_layers": 2,
        "mtp_use_repeated_layer": True,
        "mtp_loss_scaling_factor": 0.1,
        "calculate_per_token_loss": True,
    }
    reference_config = TransformerConfig(
        **common,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    replica_config = TransformerConfig(
        **common,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="replica_hybridep",
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    )

    def build_model(config):
        return GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=8,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=False,
            mtp_block_spec=get_gpt_mtp_block_spec(
                config, layer_spec, use_transformer_engine=True
            ),
        ).cuda()

    def initialize_main_grads(model):
        for parameter in model.parameters():
            _set_main_grad(parameter)
            # Ordinary Megatron DDP zeroes persistent main-grad buffers and TE
            # accumulates every tied-layer use into them.  ``overwrite=True``
            # is only appropriate for a single-use synthetic forward: when a
            # repeated MTP layer has two outstanding autograd contexts, both
            # contexts would otherwise overwrite the same buffer in backward.
            del parameter.overwrite_main_grad

    def snapshot_gradients(model):
        snapshot = {}
        for name, parameter in model.named_parameters():
            accumulated_in_main_grad = bool(
                getattr(parameter, "grad_added_to_main_grad", False)
            )
            snapshot[name] = (
                (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None and not accumulated_in_main_grad
                    else None
                ),
                parameter.main_grad.detach().clone(),
                accumulated_in_main_grad,
            )
        return snapshot

    batch = 2
    sequence = 8
    generator = torch.Generator(device="cuda").manual_seed(5678)
    input_ids = torch.randint(
        0, 128, (batch, sequence), generator=generator, device="cuda"
    )
    labels = torch.randint(0, 128, (batch, sequence), generator=generator, device="cuda")
    position_ids = torch.arange(sequence, device="cuda").unsqueeze(0).expand(batch, -1)
    loss_mask = torch.ones((batch, sequence), device="cuda")

    reference_model = None
    replica_model = None
    try:
        reference_model = build_model(reference_config)
        replica_model = build_model(replica_config)
        replica_model.load_state_dict(reference_model.state_dict())
        assert replica_model.state_dict().keys() == reference_model.state_dict().keys()
        initialize_main_grads(reference_model)
        initialize_main_grads(replica_model)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reference_loss = reference_model(
                input_ids,
                position_ids,
                attention_mask=None,
                labels=labels,
                loss_mask=loss_mask,
            )
        reference_loss.sum().backward()
        reference_gradients = snapshot_gradients(reference_model)

        # HybridEP's baseline and replica layouts use N and 2N local runtime
        # experts. Reinitialize its process-global transport buffer between them.
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.distributed.barrier()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            replica_loss = replica_model(
                input_ids,
                position_ids,
                attention_mask=None,
                labels=labels,
                loss_mask=loss_mask,
            )

        mtp_moe_layers = [
            module
            for module in replica_model.mtp.layers[0].mtp_model_layer.modules()
            if isinstance(module, MoELayer)
        ]
        assert len(mtp_moe_layers) == 1, "repeated MTP test must exercise one shared MoE layer"
        manager = mtp_moe_layers[0].token_dispatcher._comm_manager
        assert len(manager._replica_plan_slots) == 2
        assert all(slot.in_use and slot.plan is not None for slot in manager._replica_plan_slots)
        assert (
            manager._replica_plan_slots[0].workspace
            is not manager._replica_plan_slots[1].workspace
        )
        assert (
            manager._replica_plan_slots[0].plan.virtual_experts.data_ptr()
            != manager._replica_plan_slots[1].plan.virtual_experts.data_ptr()
        )
        active_replica = torch.stack(
            [torch.any(slot.plan.experts_to_copy >= 0) for slot in manager._replica_plan_slots]
        ).any()
        torch.distributed.all_reduce(active_replica, op=torch.distributed.ReduceOp.MAX)
        assert active_replica.item(), "repeated MTP parity must exercise an active replica"

        replica_loss.sum().backward()
        replica_gradients = snapshot_gradients(replica_model)
        assert not any(slot.in_use for slot in manager._replica_plan_slots)

        torch.testing.assert_close(
            replica_loss,
            reference_loss,
            rtol=0,
            atol=0,
            msg=lambda msg: f"repeated MTP loss must be bitwise equal: {msg}",
        )
        assert replica_gradients.keys() == reference_gradients.keys()
        for name in replica_gradients:
            replica_grad, replica_main_grad, replica_uses_main_grad = replica_gradients[name]
            reference_grad, reference_main_grad, reference_uses_main_grad = reference_gradients[
                name
            ]
            assert replica_uses_main_grad == reference_uses_main_grad, name
            assert (replica_grad is None) == (reference_grad is None), name
            if replica_grad is not None:
                assert torch.isfinite(replica_grad).all(), name
                assert torch.isfinite(reference_grad).all(), name
                torch.testing.assert_close(
                    replica_grad,
                    reference_grad,
                    rtol=0,
                    atol=0,
                    msg=lambda msg, name=name: f"{name} autograd gradient: {msg}",
                )
            if ".experts." in name:
                # Replica wgrads sum independently rounded FP32 partials in a
                # different order while retaining the same mathematical result.
                rtol, atol = 2e-7, 2e-6
            else:
                rtol, atol = 0, 0
            assert torch.isfinite(replica_main_grad).all(), name
            assert torch.isfinite(reference_main_grad).all(), name
            torch.testing.assert_close(
                replica_main_grad,
                reference_main_grad,
                rtol=rtol,
                atol=atol,
                msg=lambda msg, name=name: f"{name} main_grad: {msg}",
            )
    finally:
        del reference_model, replica_model
        Utils.destroy_model_parallel()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        fused_a2a.reset_hybrid_ep_buffer()
        torch.cuda.synchronize()
        torch.distributed.barrier()


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_full_layer_gradients_match_alltoall(monkeypatch):
    """Check replica-planned HybridEP output and all training gradients."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch, "replica_hybridep", F.silu, True, False, None, None
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_gtp_gradients_match_alltoall(monkeypatch):
    """Check discrete GTP-sharded experts through replica weight and grad exchange."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch, "replica_hybridep", F.silu, True, False, None, None, gtp=True
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
@pytest.mark.parametrize("gtp", [False, True], ids=["ep4", "ep2-gtp2"])
def test_replica_hybridep_bf16_gradients_match_alltoall(monkeypatch, gtp):
    """Reduce replica gradients over BF16 transport with one local FP32 sum."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        gtp=gtp,
        grad_dtype=torch.bfloat16,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_mxfp8_gradients_match_alltoall(monkeypatch):
    """Check native MXFP8 replica outputs and gradients against alltoall."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch, "replica_hybridep", F.silu, True, False, None, None, mxfp8=True
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
@pytest.mark.parametrize("mxfp8", [False, True])
def test_replica_hybridep_full_layer_squared_relu_matches_alltoall(monkeypatch, mxfp8):
    """Check the squared-ReLU configuration used by main_cg_debug_int.sh."""
    try:
        from transformer_engine.pytorch.ops import ScaledSReLU  # noqa: F401
    except ImportError:
        pytest.skip("Transformer Engine ScaledSReLU is required")
    _run_replica_hybridep_full_layer_parity(
        monkeypatch, "replica_hybridep", squared_relu, False, True, None, 640, mxfp8=mxfp8
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_gtp_mxfp8_squared_relu_matches_alltoall(monkeypatch):
    """Cover the MXFP8, BF16-grad, GTP, and latent-MoE production combination."""
    try:
        from transformer_engine.pytorch.ops import ScaledSReLU  # noqa: F401
    except ImportError:
        pytest.skip("Transformer Engine ScaledSReLU is required")
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        squared_relu,
        False,
        True,
        None,
        640,
        mxfp8=True,
        gtp=True,
        grad_dtype=torch.bfloat16,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_bf16_semantics_match_hybridep(monkeypatch):
    """Require bitwise semantics and tightly bounded expert wgrad reduction noise."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        reference_dispatcher="hybridep",
        verify_hybridep_contract=True,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_repeated_mtp_semantics_match_hybridep(monkeypatch):
    """Preserve two-depth tied-layer MTP loss and every model gradient."""
    _run_replica_hybridep_repeated_mtp_parity(monkeypatch)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_mxfp8_matches_hybridep(monkeypatch):
    """Bound MX execution noise while requiring byte-exact weight transport."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        mxfp8=True,
        reference_dispatcher="hybridep",
    )
