# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for replica route planning and the weight bridge.

Every test here runs in one process on one GPU, so a plain
``pytest tests/unit_tests/transformer/moe/test_replica_planner.py`` runs the
whole file. The planner kernels take a gathered histogram as an argument rather
than performing the collective themselves, so a complete expert-parallel
group's placement is reproducible here without a distributed launch.

Cross-rank transport lives in ``test_replica_weight_triton.py`` and end-to-end
gradient parity in ``test_replica_hybridep.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.experts import _ReplicaFC2WgradStore
from megatron.core.transformer.moe.replica_planner import (
    ReplicaPlan,
    ReplicaPlannerWorkspace,
    ReplicaWeightBridge,
    _collect_replica_projection_specs,
    _DirectionalBinding,
    _ReplicaProjection,
    _WeightDirection,
    extract_semantic_routes,
    map_replica_plan_to_hybridep,
    start_replica_grad_reduce_after_dispatch_backward,
    start_replica_weight_prefetch_before_layer_backward,
    wait_replica_grad_reduce_at_layer_input,
    wait_replica_weight_prefetch_before_expert_backward,
)
from megatron.core.transformer.moe.replica_weight_triton import (
    launch_replica_placement,
    launch_replica_route_mapping,
    launch_replica_route_ranking,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# One fixed planner shape drives every placement test: four ranks owning two
# experts each, sixteen tokens picking three distinct experts.
EP_SIZE = 4
NUM_LOCAL_EXPERTS = 2
NUM_EXPERTS = EP_SIZE * NUM_LOCAL_EXPERTS
NUM_TOKENS = 16
ROUTER_TOPK = 3
NUM_ROUTES = NUM_TOKENS * ROUTER_TOPK


def _routes_for_skew(skew: str) -> torch.Tensor:
    """Return ``[ep_size, num_tokens, router_topk]`` semantic routes for one load skew."""
    weights = torch.ones(NUM_EXPERTS)
    if skew == "hot_expert":
        weights[0] = 20.0
    elif skew == "hot_rank":
        weights[:NUM_LOCAL_EXPERTS] = 12.0
    elif skew == "two_ranks_own_everything":
        # Only ranks 0 and 1 hold routed experts, so ranks 2 and 3 must receive
        # a full capacity each from a single sender: the tightest slot fit the
        # single-sender migration rule allows.
        weights[ROUTER_TOPK:] = 0.0
    elif skew != "balanced":
        raise ValueError(f"Unknown skew {skew!r}.")
    generator = torch.Generator().manual_seed(1234)
    return torch.stack(
        [
            torch.stack(
                [
                    torch.multinomial(weights, ROUTER_TOPK, replacement=False, generator=generator)
                    for _ in range(NUM_TOKENS)
                ]
            )
            .sort(dim=1)
            .values.to(torch.int32)
            for _ in range(EP_SIZE)
        ]
    )


def _histogram(routes: torch.Tensor) -> torch.Tensor:
    """Return the ``[ep_size, num_experts]`` route histogram of every rank."""
    return torch.stack(
        [torch.bincount(rank.reshape(-1), minlength=NUM_EXPERTS) for rank in routes]
    ).to(torch.int32)


def _plan_locally(
    gathered_counts: torch.Tensor, routes: torch.Tensor | None, source_rank: int, device
) -> ReplicaPlannerWorkspace:
    """Run placement (and optionally ranking plus mapping) for one source rank."""
    workspace = ReplicaPlannerWorkspace.allocate(
        num_tokens=NUM_TOKENS,
        router_topk=ROUTER_TOPK,
        num_experts=NUM_EXPERTS,
        ep_size=EP_SIZE,
        device=device,
    )
    workspace.gathered_counts.copy_(gathered_counts)
    if routes is not None:
        launch_replica_route_ranking(
            routes.reshape(-1).to(device),
            workspace.sort_route_metadata,
            workspace.sort_partition_counts,
            workspace.sort_grid_sync,
            num_experts=NUM_EXPERTS,
            num_routes=NUM_ROUTES,
        )
    launch_replica_placement(
        workspace.gathered_counts,
        workspace.balance,
        workspace.allocation,
        workspace.destination_boundaries,
        workspace.experts_to_copy,
        workspace.expert_replica_slots,
        workspace.placement_grid_sync,
        rank_route_capacity=NUM_ROUTES,
        source_rank=source_rank,
        ep_size=EP_SIZE,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL_EXPERTS,
    )
    if routes is not None:
        launch_replica_route_mapping(
            workspace.sort_route_metadata,
            workspace.sort_partition_counts,
            workspace.destination_boundaries,
            workspace.expert_replica_slots,
            workspace.virtual_experts,
            ep_size=EP_SIZE,
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_LOCAL_EXPERTS,
            num_routes=NUM_ROUTES,
        )
    torch.cuda.synchronize(device)
    return workspace


@requires_cuda
@pytest.mark.parametrize("skew", ["balanced", "hot_expert", "hot_rank", "two_ranks_own_everything"])
def test_replica_placement_balances_every_destination(skew):
    """Equalize route load across ranks without over-subscribing replica slots."""
    device = torch.device("cuda", torch.cuda.current_device())
    counts = _histogram(_routes_for_skew(skew))
    workspace = _plan_locally(counts.to(device), None, source_rank=0, device=device)

    allocation = workspace.allocation.cpu()
    experts_to_copy = workspace.experts_to_copy.cpu()
    global_per_expert = counts.sum(0).to(torch.int32)

    # Every rank executes exactly its own route capacity. The dispatcher's
    # dropless rank capacity is derived from this equality, so a placement that
    # merely reduced the imbalance would silently drop real routes.
    torch.testing.assert_close(
        allocation.sum(0).to(torch.int32),
        torch.full((EP_SIZE,), NUM_ROUTES, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    # Migration moves routes; it never invents or loses them.
    torch.testing.assert_close(allocation.sum(1).to(torch.int32), global_per_expert, rtol=0, atol=0)
    assert (allocation >= 0).all()
    torch.testing.assert_close(
        workspace.balance.cpu(),
        global_per_expert.view(EP_SIZE, NUM_LOCAL_EXPERTS).sum(1).to(torch.int32) - NUM_ROUTES,
        rtol=0,
        atol=0,
    )

    for destination in range(EP_SIZE):
        slots = experts_to_copy[destination].tolist()
        migrated = {
            expert
            for expert in range(NUM_EXPERTS)
            if expert // NUM_LOCAL_EXPERTS != destination and allocation[expert, destination] > 0
        }
        filled = [expert for expert in slots if expert >= 0]
        # A remote expert that receives routes but owns no replica slot would
        # execute against a stale slot's weights. The single-sender rule bounds
        # this set by num_local_experts; the device-side trap covers the rest.
        assert set(filled) == migrated, f"rank {destination} slots {slots} vs migrated {migrated}"
        assert len(filled) == len(set(filled))
        assert all(expert // NUM_LOCAL_EXPERTS != destination for expert in filled)

    # Placement is replayed independently on every rank and must agree exactly.
    repeated = _plan_locally(counts.to(device), None, source_rank=EP_SIZE - 1, device=device)
    for field in ("balance", "allocation", "experts_to_copy", "expert_replica_slots"):
        torch.testing.assert_close(
            getattr(repeated, field).cpu(), getattr(workspace, field).cpu(), rtol=0, atol=0
        )


@requires_cuda
@pytest.mark.parametrize("skew", ["hot_expert", "two_ranks_own_everything"])
def test_replica_planner_maps_every_route_to_the_expert_it_selected(skew):
    """Decode each virtual route back to the semantic expert and destination it was given."""
    device = torch.device("cuda", torch.cuda.current_device())
    routes = _routes_for_skew(skew)
    counts = _histogram(routes).to(device)

    allocation = None
    experts_to_copy = None
    observed = torch.zeros((NUM_EXPERTS, EP_SIZE), dtype=torch.int32)
    for source_rank in range(EP_SIZE):
        workspace = _plan_locally(counts, routes[source_rank], source_rank, device)
        if allocation is None:
            allocation = workspace.allocation.cpu()
            experts_to_copy = workspace.experts_to_copy.cpu()
        virtual_experts = workspace.virtual_experts.cpu().reshape(-1).tolist()
        for route, virtual in zip(routes[source_rank].reshape(-1).tolist(), virtual_experts):
            destination, runtime_local = divmod(virtual, 2 * NUM_LOCAL_EXPERTS)
            if runtime_local < NUM_LOCAL_EXPERTS:
                assert (destination, runtime_local) == divmod(
                    route, NUM_LOCAL_EXPERTS
                ), f"route to expert {route} became native id {virtual}"
            else:
                slot = runtime_local - NUM_LOCAL_EXPERTS
                assert int(experts_to_copy[destination][slot]) == route, (
                    f"route to expert {route} became replica slot {slot} on rank "
                    f"{destination}, which holds expert "
                    f"{int(experts_to_copy[destination][slot])}"
                )
            observed[route, destination] += 1

    # The allocation names how many routes each destination owes each expert;
    # the mapped routes must reproduce it exactly, not merely in aggregate.
    torch.testing.assert_close(observed, allocation, rtol=0, atol=0)


@requires_cuda
def test_replica_semantic_routes_follow_the_routing_map():
    """The routing map is authoritative: a selected zero-probability route survives."""
    routing_map = torch.tensor(
        [[False, True, False, True], [True, False, True, False]], device="cuda"
    )
    probs = torch.tensor(
        [[0.0, 0.75, 0.0, 0.0], [0.6, 0.0, 0.4, 0.0]], device="cuda", requires_grad=True
    )

    token_probs, token_indices, tokens_per_expert = extract_semantic_routes(
        routing_map, probs, router_topk=2
    )

    torch.testing.assert_close(
        token_indices, torch.tensor([[1, 3], [0, 2]], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        tokens_per_expert, torch.tensor([1, 1, 1, 1], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(token_probs.sum(dim=-1), torch.tensor([0.75, 1.0], device="cuda"))
    token_probs.sum().backward()
    torch.testing.assert_close(probs.grad, routing_map.to(probs.dtype))


def test_replica_plan_expands_to_dense_hybridep_inputs():
    """Scatter compact virtual routes into HybridEP's dense map without losing gradients."""
    plan = ReplicaPlan(
        virtual_experts=torch.tensor([[1, 6], [3, 4]], dtype=torch.int64),
        experts_to_copy=torch.empty((0,), dtype=torch.int32),
    )
    probs = torch.tensor([[0.75, 0.25], [0.6, 0.4]], requires_grad=True)

    routing_map, dense_probs = map_replica_plan_to_hybridep(plan, probs, num_experts=8)

    assert routing_map.shape == (2, 8) and dense_probs.shape == (2, 8)
    assert routing_map[0, 1] and routing_map[0, 6]
    assert dense_probs[1, 3] == 0.6
    assert routing_map.sum() == 4
    dense_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))


def test_replica_rank_capacity_includes_per_expert_padding():
    """Size the static dropless capacity for HybridEP's per-segment alignment."""
    from megatron.core.transformer.moe.token_dispatcher import _get_replica_hybridep_rank_capacity

    common = {"num_tokens": 8192, "router_topk": 22, "num_runtime_experts": 64}
    assert (
        _get_replica_hybridep_rank_capacity(**common, capacity_factor=1.0, alignment=256) == 196608
    )
    assert (
        _get_replica_hybridep_rank_capacity(**common, capacity_factor=2.0, alignment=256) == 360448
    )
    assert _get_replica_hybridep_rank_capacity(**common, capacity_factor=1.0, alignment=0) == 180224


def test_replica_backward_hooks_span_the_transport_window():
    """Order the four transport hooks across one layer's backward."""
    events = []
    plan = object()

    class FakeBridge:
        source_parameters = (torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(())))

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def start_prefetch(self, current_plan, direction=_WeightDirection.FORWARD):
            assert current_plan is plan and direction is _WeightDirection.BACKWARD
            events.append("start_prefetch")

        def wait_prefetch_for_backward(self, current_plan):
            assert current_plan is plan
            events.append("wait_prefetch")

        def start_pending_grad_reduces(self, current_plan):
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
    hidden = wait_replica_grad_reduce_at_layer_input(hidden, bridge, SimpleNamespace(plan=plan))
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = start_replica_grad_reduce_after_dispatch_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = wait_replica_weight_prefetch_before_expert_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_projection_backward")
    hidden = start_replica_weight_prefetch_before_layer_backward(hidden, bridge, plan)
    hidden.backward()

    assert events == [
        "start_prefetch",
        "latent_up_projection_backward",
        "combine_backward",
        "wait_prefetch",
        "expert_backward",
        "dispatch_backward",
        "start_grad_reduce",
        "router_and_shared_expert_backward",
        "wait_grad_reduce",
    ]
    # The reduction's output is accumulated into the optimizer parameters
    # rather than aliased into them.
    for index, parameter in enumerate(bridge.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )
        assert parameter.grad.data_ptr() != bridge.source_grads[index].data_ptr()


def test_replica_fc2_reduction_starts_from_the_wgrad_store_and_fc1_after_dispatch():
    """FC2 reduces behind its own wgrad GEMM; only FC1 waits for dispatch backward."""
    plan = object()
    started = []
    bridge = ReplicaWeightBridge.__new__(ReplicaWeightBridge)
    bridge._grad_reduce_plan = None
    bridge._grad_reduce_started = set()
    bridge._backward_plan = plan

    def record(current_plan, projection):
        assert current_plan is plan
        started.append(projection)
        bridge._grad_reduce_plan = current_plan
        bridge._grad_reduce_started.add(projection)

    bridge.start_grad_reduce = record

    # TE's delayed-wgrad protocol hands the GEMM to the store instead of
    # launching it; the store runs it and starts FC2's reduction right behind.
    gemm_calls = []
    store = _ReplicaFC2WgradStore(bridge)
    assert store.delay_wgrad_compute() and store.context is None
    store.put(["x", "dy", "out"], lambda *tensors: gemm_calls.append(tensors))
    assert gemm_calls == [("x", "dy", "out")]
    assert started == [1]

    bridge.start_pending_grad_reduces(plan)
    assert started == [1, 0]


@requires_cuda
def test_replica_fused_wgrad_handoff_preserves_fp32():
    """Accumulate the FP32 replica wgrad into main_grad and return a BF16 dummy."""
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
    wait_replica_grad_reduce_at_layer_input(
        hidden, FakeBridge(), SimpleNamespace(plan=plan)
    ).backward()

    torch.testing.assert_close(parameter.main_grad, external_wgrad, rtol=0, atol=0)
    assert parameter.grad_added_to_main_grad
    assert parameter.grad.dtype == torch.bfloat16


MEMBER_SHAPE = (4, 4)
SCALE_SHAPE = (2, 2)


def _mxfp8_member(device):
    """Build one MXFP8 weight carrier with the four component tensors TE exposes."""
    return SimpleNamespace(
        shape=MEMBER_SHAPE,
        device=device,
        _rowwise_data=torch.empty(MEMBER_SHAPE, dtype=torch.uint8, device=device),
        _rowwise_scale_inv=torch.empty(SCALE_SHAPE, dtype=torch.uint8, device=device),
        _columnwise_data=torch.empty(MEMBER_SHAPE, dtype=torch.uint8, device=device),
        _columnwise_scale_inv=torch.empty(SCALE_SHAPE, dtype=torch.uint8, device=device),
    )


def _gtp_projection(weight_format, device, num_local_experts=2):
    """Build a GTP-sharded projection whose weights arrive from an all-gather."""
    components = 1 if weight_format == "bf16" else 2
    if weight_format == "bf16":

        def make():
            return torch.empty(MEMBER_SHAPE, dtype=torch.bfloat16, device=device)

    else:

        def make():
            return _mxfp8_member(device)

    destinations = tuple(make() for _ in range(num_local_experts))
    parameters = (
        tuple(torch.nn.Parameter(weight) for weight in destinations)
        if weight_format == "bf16"
        else destinations
    )
    runtime_parameters = (
        tuple(torch.nn.Parameter(torch.empty_like(weight)) for weight in destinations)
        if weight_format == "bf16"
        else tuple(make() for _ in range(num_local_experts))
    )
    bindings = tuple(
        _DirectionalBinding(
            torch.empty(num_local_experts, dtype=torch.int64, device=device),
            (
                torch.empty(num_local_experts, dtype=torch.int64, device=device)
                if components == 2
                else None
            ),
            host_pointer_table=torch.empty(
                (components, num_local_experts), dtype=torch.int64, pin_memory=True
            ),
        )
        for _ in range(2)
    )
    return _ReplicaProjection(
        name="test projection",
        device=device,
        weight_format=weight_format,
        parameters=parameters,
        gtp_leader=parameters[0],
        source_tensors=destinations,
        forward=bindings[0],
        backward=bindings[1],
        native_grad_bases=torch.empty(num_local_experts, dtype=torch.int64, device=device),
        member_shape=MEMBER_SHAPE,
        rowwise_scale_shape=SCALE_SHAPE,
        columnwise_scale_shape=SCALE_SHAPE,
        virtual_weight=(),
        virtual_grad=torch.empty((0, *MEMBER_SHAPE), dtype=torch.float32, device=device),
        native_grad=torch.empty(
            (num_local_experts, *MEMBER_SHAPE), dtype=torch.float32, device=device
        ),
        runtime_parameters=runtime_parameters,
    ), tuple(make() for _ in range(num_local_experts))


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert pointers the push reads for one direction."""
    if weight_format == "bf16":
        return tuple(weight.data_ptr() for weight in weights)
    field = "_rowwise_data" if direction is _WeightDirection.FORWARD else "_columnwise_data"
    return tuple(getattr(weight, field).data_ptr() for weight in weights)


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_replica_projection_binds_gtp_weights_during_capture(weight_format):
    """Bind each GTP gather's storage into the pointer tables without a payload copy."""
    device = torch.device("cuda", torch.cuda.current_device())
    projection, backward_weights = _gtp_projection(weight_format, device)
    forward_weights = tuple(projection.source_tensors)
    runtime_ids = tuple(id(parameter) for parameter in projection.runtime_parameters)

    # The first forward binds under CUDA-graph capture: it may only enqueue an
    # async pointer-table update from pinned staging, never a device sync.
    capture_probe = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        projection.bind_materialized_weights(forward_weights, _WeightDirection.FORWARD)
        capture_probe.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(capture_probe, torch.ones_like(capture_probe), rtol=0, atol=0)

    forward_ptrs = _data_ptrs(forward_weights, weight_format, _WeightDirection.FORWARD)
    assert tuple(projection.forward.data_bases.cpu().tolist()) == forward_ptrs

    # Forward and backward hold separate gathers; binding one must not disturb
    # the other's table.
    projection.bind_materialized_weights(backward_weights, _WeightDirection.BACKWARD)
    backward_ptrs = _data_ptrs(backward_weights, weight_format, _WeightDirection.BACKWARD)
    assert tuple(projection.backward.data_bases.cpu().tolist()) == backward_ptrs
    assert tuple(projection.forward.data_bases.cpu().tolist()) == forward_ptrs
    if weight_format == "mxfp8":
        assert tuple(projection.forward.scale_bases.cpu().tolist()) == tuple(
            weight._rowwise_scale_inv.data_ptr() for weight in forward_weights
        )
        assert tuple(projection.backward.scale_bases.cpu().tolist()) == tuple(
            weight._columnwise_scale_inv.data_ptr() for weight in backward_weights
        )

    # The runtime parameters TE executes against are created once and rebound in
    # place, so a captured expert GEMM keeps observing the same wrappers.
    assert tuple(id(parameter) for parameter in projection.runtime_parameters) == runtime_ids
    if weight_format == "bf16":
        assert (
            tuple(parameter.data_ptr() for parameter in projection.runtime_parameters)
            == backward_ptrs
        )
    else:
        for runtime_parameter, source in zip(projection.runtime_parameters, backward_weights):
            assert runtime_parameter._columnwise_data is source._columnwise_data
            assert runtime_parameter._columnwise_scale_inv is source._columnwise_scale_inv

    # Rebinding the same gather is validation-only; a moved gather is fatal
    # because the captured push would read a freed address.
    projection.bind_materialized_weights(forward_weights, _WeightDirection.FORWARD)
    moved = list(forward_weights)
    moved[0] = _gtp_projection(weight_format, device)[1][0]
    with pytest.raises(RuntimeError, match="all-gather storage of test projection changed"):
        projection.bind_materialized_weights(tuple(moved), _WeightDirection.FORWARD)


@requires_cuda
def test_replica_runtime_parameters_keep_a_stable_grad_table():
    """Keep the reduction's destination table pinned to the native wgrad staging."""
    device = torch.device("cuda", torch.cuda.current_device())
    projection, _ = _gtp_projection("bf16", device)
    projection.bind_materialized_weights(tuple(projection.source_tensors), _WeightDirection.FORWARD)
    for runtime_parameter, grad in zip(projection.runtime_parameters, projection.native_grad):
        runtime_parameter.main_grad = grad
    bridge = ReplicaWeightBridge.__new__(ReplicaWeightBridge)
    bridge.device = device
    bridge.num_local_experts = len(projection.parameters)
    bridge.projections = [projection]
    bridge.workspace = SimpleNamespace(grad_dtype=torch.float32)

    bridge.prepare_runtime_parameters()
    stable_ptrs = tuple(grad.data_ptr() for grad in projection.native_grad)
    assert projection.native_grad_ptrs == stable_ptrs
    assert tuple(projection.native_grad_bases.cpu().tolist()) == stable_ptrs

    # Every later forward re-validates rather than rebinding, and stays capture-safe.
    capture_probe = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bridge.prepare_runtime_parameters()
        capture_probe.add_(1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert tuple(projection.native_grad_bases.cpu().tolist()) == stable_ptrs
    torch.testing.assert_close(capture_probe, torch.ones_like(capture_probe), rtol=0, atol=0)

    projection.native_grad = torch.empty_like(projection.native_grad)
    with pytest.raises(RuntimeError, match="native-grad storage changed"):
        bridge.prepare_runtime_parameters()


def test_replica_bridge_rejects_single_grouped_source_weights():
    """A packed grouped weight has no per-expert address for the push to read."""
    packed_linear = SimpleNamespace(in_features=16, out_features=32, single_grouped_weight=True)
    experts = SimpleNamespace(linear_fc1=packed_linear, linear_fc2=packed_linear)

    with pytest.raises(ValueError, match="moe_single_grouped_weight must be False"):
        _collect_replica_projection_specs(
            experts, num_local_experts=2, backend_name="Replica-HybridEP"
        )
