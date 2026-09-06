# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for virtual-expert route planning and the weight bridge.

Every test here runs in one process on one GPU, so a plain ``pytest
tests/unit_tests/transformer/moe/test_virtual_expert_planner.py`` runs the whole file.
The planner kernel can take a gathered histogram instead of exchanging
one, so a complete expert-parallel group's placement is reproducible here
without a distributed launch.

Cross-rank transport lives in ``test_virtual_expert_triton.py`` and end-to-end
gradient parity in ``test_virtual_expert_hybridep.py``.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.experts import _VirtualExpertFC2WgradStore
from megatron.core.transformer.moe.token_dispatcher import _VirtualExpertHybridEPManager
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    BACKWARD,
    FORWARD,
    VirtualExpertLoadBalancer,
    VirtualExpertPlan,
    VirtualExpertPlannerWorkspace,
    VirtualExpertWeightBridge,
    _VirtualExpertBackwardHook,
    _VirtualExpertProjection,
    _VirtualExpertWaitGradReduce,
    plan_virtual_expert_routes,
)

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# One fixed planner shape drives every placement test: four ranks owning two
# experts each, sixteen tokens picking three distinct experts.
EP_SIZE = 4
NUM_LOCAL_EXPERTS = 2
NUM_EXPERTS = EP_SIZE * NUM_LOCAL_EXPERTS
NUM_TOKENS = 16
ROUTER_TOPK = 3
NUM_ROUTES = NUM_TOKENS * ROUTER_TOPK


def _routes_for_skew(skew: str, num_tokens: int = NUM_TOKENS) -> torch.Tensor:
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
                    for _ in range(num_tokens)
                ]
            )
            for _ in range(EP_SIZE)
        ]
    )


def _histogram(routes: torch.Tensor) -> torch.Tensor:
    """Return the ``[ep_size, num_experts]`` route histogram of every rank."""
    return torch.stack(
        [torch.bincount(rank.reshape(-1), minlength=NUM_EXPERTS) for rank in routes]
    ).to(torch.int32)


def _reference_map_routes(
    routes: torch.Tensor, workspace: VirtualExpertPlannerWorkspace
) -> torch.Tensor:
    """Torch oracle for the planner's route mapping: stable sort by expert, one global array of
    segment ends, then the runtime id of every route."""
    num_local_experts, ep_size = workspace.num_local_experts, workspace.ep_size
    flat = routes.reshape(-1)
    experts, order = torch.sort(flat, stable=True)
    tokens_per_expert = torch.bincount(flat, minlength=workspace.num_experts)
    bucket_start = torch.cumsum(tokens_per_expert, 0) - tokens_per_expert
    boundaries = (
        workspace.destination_boundaries[:, :ep_size]
        .clamp(min=0)
        .minimum(tokens_per_expert[:, None])
    )
    ends = (bucket_start[:, None] + boundaries).reshape(-1)
    positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    destination = torch.searchsorted(ends, positions, right=True) - experts * ep_size
    slot = workspace.virtual_expert_slots.view(-1)[experts * ep_size + destination]
    runtime_local = torch.where(
        destination == experts // num_local_experts,
        experts % num_local_experts,
        num_local_experts + slot,
    )
    virtual = torch.empty_like(positions)
    virtual[order] = destination * (2 * num_local_experts) + runtime_local
    return virtual.view(routes.shape)


def _plan_locally(
    gathered_counts: torch.Tensor,
    routes: torch.Tensor,
    source_rank: int,
    device,
    probs: torch.Tensor | None = None,
) -> tuple[VirtualExpertPlannerWorkspace, VirtualExpertPlan, torch.Tensor]:
    """Run the planner for one source rank of a complete EP group in this process: the window
    is filled with every rank's histogram instead of exchanged."""
    workspace = VirtualExpertPlannerWorkspace.scratch(
        NUM_EXPERTS, EP_SIZE, device, rank=source_rank
    )
    workspace.gathered_counts.copy_(gathered_counts)
    routes = routes.to(device=device, dtype=torch.int64)
    if probs is None:
        probs = torch.rand(routes.shape, device=device)
    plan, runtime_probs = plan_virtual_expert_routes(routes, probs, workspace, exchange=False)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        workspace.tokens_per_expert, gathered_counts[source_rank].contiguous(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.virtual_experts.long(), _reference_map_routes(routes, workspace), rtol=0, atol=0
    )
    return workspace, plan, runtime_probs


@requires_cuda
@pytest.mark.parametrize("skew", ["balanced", "hot_expert", "hot_rank", "two_ranks_own_everything"])
def test_virtual_expert_placement_balances_every_destination(skew):
    """Equalize route load across ranks without over-subscribing virtual-expert slots."""
    device = torch.device("cuda", torch.cuda.current_device())
    routes = _routes_for_skew(skew)
    counts = _histogram(routes)
    workspace, plan, _ = _plan_locally(counts.to(device), routes[0], source_rank=0, device=device)

    allocation = workspace.allocation.cpu()
    experts_to_copy = plan.experts_to_copy.cpu()
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
        # A remote expert that receives routes but owns no virtual-expert slot would
        # execute against a stale slot's weights. The single-sender rule bounds
        # this set by num_local_experts; the device-side trap covers the rest.
        assert set(filled) == migrated, f"rank {destination} slots {slots} vs migrated {migrated}"
        assert len(filled) == len(set(filled))
        assert all(expert // NUM_LOCAL_EXPERTS != destination for expert in filled)

    # Placement is replayed independently on every rank and must agree exactly.
    repeated, repeated_plan, _ = _plan_locally(
        counts.to(device), routes[EP_SIZE - 1], source_rank=EP_SIZE - 1, device=device
    )
    for field in ("balance", "allocation"):
        torch.testing.assert_close(
            getattr(repeated, field).cpu(), getattr(workspace, field).cpu(), rtol=0, atol=0
        )
    torch.testing.assert_close(repeated_plan.experts_to_copy.cpu(), experts_to_copy, rtol=0, atol=0)
    # The slot table is written only for assigned slots; check exactly those.
    for destination, slots in enumerate(experts_to_copy.tolist()):
        for slot, expert in enumerate(slots):
            if expert >= 0:
                assert int(workspace.virtual_expert_slots[expert, destination]) == slot
                assert int(repeated.virtual_expert_slots[expert, destination]) == slot


@requires_cuda
@pytest.mark.parametrize("skew", ["hot_expert", "two_ranks_own_everything"])
def test_virtual_expert_planner_maps_every_route_to_the_expert_it_selected(skew):
    """Decode each virtual route back to the semantic expert and destination it was given."""
    device = torch.device("cuda", torch.cuda.current_device())
    routes = _routes_for_skew(skew)
    counts = _histogram(routes).to(device)

    allocation = None
    experts_to_copy = None
    observed = torch.zeros((NUM_EXPERTS, EP_SIZE), dtype=torch.int32)
    for source_rank in range(EP_SIZE):
        workspace, plan, _ = _plan_locally(counts, routes[source_rank], source_rank, device)
        if allocation is None:
            allocation = workspace.allocation.cpu()
            experts_to_copy = plan.experts_to_copy.cpu()
        virtual_experts = plan.virtual_experts.cpu().reshape(-1).tolist()
        for route, virtual in zip(routes[source_rank].reshape(-1).tolist(), virtual_experts):
            destination, runtime_local = divmod(virtual, 2 * NUM_LOCAL_EXPERTS)
            if runtime_local < NUM_LOCAL_EXPERTS:
                assert (destination, runtime_local) == divmod(
                    route, NUM_LOCAL_EXPERTS
                ), f"route to expert {route} became native id {virtual}"
            else:
                slot = runtime_local - NUM_LOCAL_EXPERTS
                assert int(experts_to_copy[destination][slot]) == route, (
                    f"route to expert {route} became virtual-expert slot {slot} on rank "
                    f"{destination}, which holds expert "
                    f"{int(experts_to_copy[destination][slot])}"
                )
            observed[route, destination] += 1

    # The allocation names how many routes each destination owes each expert;
    # the mapped routes must reproduce it exactly, not merely in aggregate.
    torch.testing.assert_close(observed, allocation, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("skew", ["balanced", "hot_expert", "two_ranks_own_everything"])
@pytest.mark.parametrize("num_tokens", [NUM_TOKENS, 1000])
def test_virtual_expert_planner_writes_hybridep_inputs_with_probability_gradients(skew, num_tokens):
    """The planner's dense runtime probabilities match a torch scatter of the router's compact
    probabilities at the mapped runtime ids, and their gradient flows back to those entries."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cuda").manual_seed(4321)
    routes = _routes_for_skew(skew, num_tokens).to(device)
    counts = _histogram(routes)
    for source_rank in (0, EP_SIZE - 1):
        probs = torch.rand((num_tokens, ROUTER_TOPK), device=device, generator=generator)
        probs = probs.requires_grad_(True)
        workspace, plan, runtime_probs = _plan_locally(
            counts, routes[source_rank], source_rank, device, probs
        )
        expected = torch.zeros((num_tokens, 2 * NUM_EXPERTS), device=device)
        expected = expected.scatter(1, plan.virtual_experts.long(), probs)
        torch.testing.assert_close(runtime_probs, expected, rtol=0, atol=0)
        assert runtime_probs.requires_grad and not plan.virtual_experts.requires_grad
        assert plan.virtual_experts.dtype == torch.int16

        grad = torch.rand(runtime_probs.shape, device=device, generator=generator)
        (actual_grad,) = torch.autograd.grad(runtime_probs, probs, grad)
        (expected_grad,) = torch.autograd.grad(expected, probs, grad)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_virtual_expert_rank_capacity_includes_per_expert_padding():
    """Size the static dropless capacity for HybridEP's per-segment alignment."""
    common = {"num_tokens": 8192, "router_topk": 22, "num_runtime_experts": 64}
    assert (
        VirtualExpertLoadBalancer._get_rank_capacity(**common, capacity_factor=1.0, alignment=256)
        == 196608
    )
    assert (
        VirtualExpertLoadBalancer._get_rank_capacity(**common, capacity_factor=2.0, alignment=256)
        == 360448
    )
    assert (
        VirtualExpertLoadBalancer._get_rank_capacity(**common, capacity_factor=1.0, alignment=0)
        == 180224
    )


def test_virtual_expert_backward_hooks_span_the_transport_window():
    """Order the four transport hooks across one layer's backward, as the dispatcher places them."""
    events = []
    plan = object()

    class FakeBridge:
        source_parameters = (torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(())))

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def start_prefetch(self, current_plan, direction=FORWARD):
            assert current_plan is plan and direction == BACKWARD
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
    hidden = _VirtualExpertWaitGradReduce.apply(
        hidden, *bridge.source_parameters, bridge, SimpleNamespace(plan=plan)
    )
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, functools.partial(bridge.start_pending_grad_reduces, plan)
    )
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, functools.partial(bridge.wait_prefetch_for_backward, plan)
    )
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_projection_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, functools.partial(bridge.start_prefetch, plan, BACKWARD)
    )
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
    # Without fused accumulation the reduction's output is copied, never aliased.
    for index, parameter in enumerate(bridge.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )
        assert parameter.grad.data_ptr() != bridge.source_grads[index].data_ptr()


def test_virtual_expert_fc2_reduction_starts_from_the_wgrad_store_and_fc1_after_dispatch():
    """FC2 reduces behind its own wgrad GEMM; only FC1 waits for dispatch backward."""
    plan = object()
    started = []
    bridge = VirtualExpertWeightBridge.__new__(VirtualExpertWeightBridge)
    bridge._backward_plan = plan
    bridge._reduced = set()

    def record(projection):
        started.append(projection)
        bridge._reduced.add(projection)

    bridge.start_grad_reduce = record

    # TE's delayed-wgrad protocol hands the GEMM to the store instead of
    # launching it; the store runs it and starts FC2's reduction right behind.
    gemm_calls = []
    store = _VirtualExpertFC2WgradStore(bridge)
    assert store.delay_wgrad_compute() and store.context is None
    store.put(["x", "dy", "out"], lambda *tensors: gemm_calls.append(tensors))
    assert gemm_calls == [("x", "dy", "out")]
    assert started == [1]

    bridge.start_pending_grad_reduces(plan)
    assert started == [1, 0]
    with pytest.raises(RuntimeError, match="another plan"):
        bridge.start_pending_grad_reduces(object())


@requires_cuda
def test_virtual_expert_fused_wgrad_handoff_preserves_fp32():
    """Accumulate the FP32 virtual-expert wgrad into main_grad and return a BF16 dummy."""
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
    _VirtualExpertWaitGradReduce.apply(
        hidden, parameter, FakeBridge(), SimpleNamespace(plan=plan)
    ).backward()

    torch.testing.assert_close(parameter.main_grad, external_wgrad, rtol=0, atol=0)
    assert parameter.grad_added_to_main_grad
    assert parameter.grad.dtype == torch.bfloat16


MEMBER_SHAPE = (128, 128)


def _mxfp8(tensor):
    """Quantize a BF16 tensor into an MXFP8 tensor holding both GEMM orientations."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(DType.kFloat8E4M3)(tensor)


def _gtp_projection(weight_format, device, num_local_experts=2):
    """Build a GTP-sharded projection whose full weights arrive from a fake all-gather."""
    mxfp8 = weight_format == "mxfp8"

    def make():
        weight = torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device)
        return _mxfp8(weight) if mxfp8 else weight

    gathers = {FORWARD: [make() for _ in range(num_local_experts)]}
    gathers[BACKWARD] = [make() for _ in range(num_local_experts)]
    leader = make()
    leader.is_gtp_weight_remat = True
    if mxfp8:
        leader._gtp_gather_quantizer = leader._quantizer
    leader.materialize_group_for_forward = lambda: gathers[FORWARD]
    leader.materialize_group_for_backward = lambda: gathers[BACKWARD]
    parameters = (leader, *(make() for _ in range(num_local_experts - 1)))
    # GTP's wgrad protocol: each group member hands out fresh full-size scratch per backward.
    leader._weights = [
        SimpleNamespace(
            get_wgrad_tensor=lambda: torch.zeros(MEMBER_SHAPE, dtype=torch.float32, device=device)
        )
        for _ in range(num_local_experts)
    ]

    numel = MEMBER_SHAPE[0] * MEMBER_SHAPE[1]
    dtype = torch.uint8 if mxfp8 else torch.bfloat16
    slots = torch.zeros((num_local_experts, *MEMBER_SHAPE), dtype=dtype, device=device)
    scales = (
        torch.zeros((num_local_experts, numel // 32), dtype=torch.uint8, device=device)
        if mxfp8
        else None
    )
    workspace = SimpleNamespace(
        mxfp8=mxfp8,
        member_shapes=(MEMBER_SHAPE,),
        grad_dtype=torch.float32,
        slot_views=lambda index: (slots, scales),
        grad_slots=lambda index: torch.zeros(
            (num_local_experts, *MEMBER_SHAPE), dtype=torch.float32, device=device
        ),
        native_staging=lambda index: pytest.fail("GTP projections must not allocate staging"),
    )
    return _VirtualExpertProjection("test projection", parameters, workspace, 0), gathers


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert data pointers the push reads for one direction."""
    if weight_format == "bf16":
        return [weight.data_ptr() for weight in weights]
    field = "_rowwise_data" if direction == FORWARD else "_columnwise_data"
    return [getattr(weight, field).data_ptr() for weight in weights]


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_projection_binds_gtp_gathers_into_its_pointer_tables(weight_format):
    """Point the push tables and the runtime parameters at each direction's GTP gather."""
    device = torch.device("cuda", torch.cuda.current_device())
    projection, gathers = _gtp_projection(weight_format, device)
    runtime_ids = [id(parameter) for parameter in projection.runtime_parameters]
    natives = projection.runtime_parameters[:2]

    projection.prepare(FORWARD)
    torch.cuda.synchronize(device)
    forward_ptrs = _data_ptrs(gathers[FORWARD], weight_format, FORWARD)
    assert projection.tables[FORWARD][0].tolist() == forward_ptrs
    assert _data_ptrs(natives, weight_format, FORWARD) == forward_ptrs

    # Forward and backward hold separate gathers; binding one leaves the other's table.
    projection.prepare(BACKWARD)
    torch.cuda.synchronize(device)
    assert projection.tables[BACKWARD][0].tolist() == _data_ptrs(
        gathers[BACKWARD], weight_format, BACKWARD
    )
    assert projection.tables[FORWARD][0].tolist() == forward_ptrs
    if weight_format == "mxfp8":
        assert projection.tables[FORWARD][1].tolist() == [
            weight._rowwise_scale_inv.data_ptr() for weight in gathers[FORWARD]
        ]
        assert projection.tables[BACKWARD][1].tolist() == [
            weight._columnwise_scale_inv.data_ptr() for weight in gathers[BACKWARD]
        ]
        for parameter, forward, backward in zip(natives, gathers[FORWARD], gathers[BACKWARD]):
            assert parameter._rowwise_data is forward._rowwise_data
            assert parameter._columnwise_data is backward._columnwise_data

    # A gather landing in a new buffer rebinds; the TE ops keep the same parameter objects.
    gathers[FORWARD][0] = _gtp_projection(weight_format, device)[1][FORWARD][0]
    projection.prepare(FORWARD)
    torch.cuda.synchronize(device)
    assert projection.tables[FORWARD][0].tolist() == _data_ptrs(
        gathers[FORWARD], weight_format, FORWARD
    )
    assert _data_ptrs(natives, weight_format, FORWARD) == _data_ptrs(
        gathers[FORWARD], weight_format, FORWARD
    )
    assert [id(parameter) for parameter in projection.runtime_parameters] == runtime_ids
    # Every runtime parameter carries the overwrite flag; the virtual slots accumulate into the
    # arena, the natives into per-backward GTP scratch that is bound below.
    for parameter in projection.runtime_parameters:
        assert parameter.overwrite_main_grad and parameter.main_grad is not None
    # TE's forward only checks for a main_grad; GTP natives get an empty placeholder until the
    # backward binds the layer's scratch.
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    assert all(parameter.main_grad.numel() > 0 for parameter in projection.runtime_parameters[2:])


@requires_cuda
def test_virtual_expert_gtp_projection_writes_wgrads_into_gtp_scratch():
    """Under GTP the natives' ``main_grad`` and the reduction's pointer table point at the
    layer's GTP wgrad scratch, which is handed to the reduce-scatter without a copy."""
    device = torch.device("cuda", torch.cuda.current_device())
    projection, _ = _gtp_projection("bf16", device)
    natives = projection.runtime_parameters[:2]

    with pytest.raises(RuntimeError, match="no GTP wgrad scratch"):
        projection.take_wgrads()

    projection.bind_wgrad_scratch()
    torch.cuda.synchronize(device)
    scratch = projection.wgrad_scratch
    assert len(scratch) == 2 and all(grad.shape == MEMBER_SHAPE for grad in scratch)
    assert [parameter.main_grad.data_ptr() for parameter in natives] == [
        grad.data_ptr() for grad in scratch
    ]
    assert projection.native_grad_bases.tolist() == [grad.data_ptr() for grad in scratch]
    with pytest.raises(RuntimeError, match="already bound"):
        projection.bind_wgrad_scratch()

    # The hand-off releases the binding so the next backward gets fresh scratch.
    assert projection.take_wgrads() == tuple(scratch)
    assert projection.wgrad_scratch is None
    projection.bind_wgrad_scratch()
    torch.cuda.synchronize(device)
    assert projection.native_grad_bases.tolist() != [grad.data_ptr() for grad in scratch]
    assert [
        parameter.main_grad.data_ptr() for parameter in natives
    ] == projection.native_grad_bases.tolist()
