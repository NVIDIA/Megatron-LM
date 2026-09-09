# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for the virtual-expert weight bridge, its hooks and the config rules.

Every test here runs in one process on one GPU, so a plain ``pytest
tests/unit_tests/transformer/moe/test_virtual_expert_planner.py`` runs the whole file.

The planner kernel exchanges histograms between ranks itself, so its placement and route
mapping are covered by the four-rank kernel tier in ``test_virtual_expert_triton.py``;
end-to-end gradient parity lives in ``test_virtual_expert_hybridep.py``.
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
    # GTP's protocol: a peek reads the gathered buffers, the consume (materialize) hands out the
    # same buffers and moves the prefetch chain along. Record which one the bridge calls.
    leader.calls = []

    def protocol(name, direction):
        def call():
            leader.calls.append((name, direction))
            return gathers[direction]

        return call

    leader.peek_group_for_forward = protocol("peek", FORWARD)
    leader.peek_group_for_backward = protocol("peek", BACKWARD)
    leader.materialize_group_for_forward = protocol("consume", FORWARD)
    leader.materialize_group_for_backward = protocol("consume", BACKWARD)
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
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_projection_peeks_for_the_push_and_consumes_at_the_gemm(weight_format):
    """The push reads GTP's gathered weights through the non-consuming peek; the consume, GTP's
    real chain step, happens separately and must find the buffers the push read."""
    device = torch.device("cuda", torch.cuda.current_device())
    projection, gathers = _gtp_projection(weight_format, device)
    leader = projection.gtp_leader

    for direction in (FORWARD, BACKWARD):
        leader.calls.clear()
        projection.prepare(direction)
        assert leader.calls == [("peek", direction)]
        bound = projection.bound[direction]
        projection.consume(direction)
        assert leader.calls == [("peek", direction), ("consume", direction)]
        # The consume validates against the bound pointers and rebinds nothing.
        assert projection.bound[direction] == bound

    # A consume that hands out other buffers than the peeked ones is an error, not a rebind:
    # the push has already copied the peeked bytes to the peers.
    gathers[FORWARD][0] = _gtp_projection(weight_format, device)[1][FORWARD][0]
    with pytest.raises(RuntimeError, match="other buffers"):
        projection.consume(FORWARD)


def test_virtual_expert_bridge_consumes_gtp_weights_in_gemm_order():
    """Forward consumes FC1 then FC2 at the expert GEMMs; the backward hook waits for the push,
    then consumes FC2 before FC1 (the expert backward's order) and binds the wgrad scratch."""
    plan = object()
    events = []

    class FakeProjection:
        def __init__(self, name):
            self.name = name

        def consume(self, direction):
            events.append(("consume", self.name, direction))

        def bind_wgrad_scratch(self):
            events.append(("bind", self.name))

    bridge = VirtualExpertWeightBridge.__new__(VirtualExpertWeightBridge)
    bridge.projections = [FakeProjection("FC1"), FakeProjection("FC2")]
    bridge.wait_prefetch = lambda current_plan: events.append(("wait", current_plan))
    bridge._backward_plan = None

    bridge.consume(FORWARD)
    assert events == [("consume", "FC1", FORWARD), ("consume", "FC2", FORWARD)]

    events.clear()
    bridge.wait_prefetch_for_backward(plan)
    assert events == [
        ("wait", plan),
        ("consume", "FC2", BACKWARD),
        ("bind", "FC2"),
        ("consume", "FC1", BACKWARD),
        ("bind", "FC1"),
    ]
    assert bridge._backward_plan is plan


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
