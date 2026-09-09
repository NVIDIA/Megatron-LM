# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for the virtual-expert load balancer's host side: runtime parameters
and pointer tables, the backward hooks and the GTP protocol calls, plus the config rules.

Every test here runs in one process on one GPU, so a plain ``pytest
tests/unit_tests/transformer/moe/test_virtual_expert_planner.py`` runs the whole file.

The planner kernel exchanges histograms between ranks itself, so its placement and route
mapping are covered by the four-rank kernel tier in ``test_virtual_expert_triton.py``;
end-to-end gradient parity lives in ``test_virtual_expert_hybridep.py``.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.experts import _VirtualExpertFC2WgradStore
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    _GRAD,
    BACKWARD,
    FORWARD,
    VirtualExpertLoadBalancer,
    _PassTemporaries,
    _VirtualExpertBackwardHook,
    _VirtualExpertFCLayer,
    _VirtualExpertWaitGradReduce,
    _weak_hook,
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


def test_virtual_expert_hooks_do_not_keep_their_owner_alive():
    """A hook captured by an autograd context must not close a cycle through the owner."""

    class Owner:
        def method(self, value):
            return value

    owner = Owner()
    hook = _weak_hook(owner.method, 7)
    assert hook() == 7
    alive = weakref.ref(owner)
    del owner
    assert alive() is None


def test_virtual_expert_backward_hooks_span_the_transport_window():
    """Order the four transport hooks across one layer's backward, as the dispatcher places them."""
    events = []
    temporaries = _PassTemporaries()

    class FakeLoadBalancer:
        source_parameters = (torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(())))

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def _start_backward(self, current):
            assert current is temporaries
            events.append("start_weight_push")

        def _prepare_expert_backward(self):
            events.append("wait_weight_push")

        def _start_pending_grad_reduces(self):
            events.append("start_grad_reduce")

        def _finish_grad_reduce(self):
            events.append("finish_grad_reduce")
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

    load_balancer = FakeLoadBalancer()
    hidden = torch.ones((), requires_grad=True)
    hidden = _VirtualExpertWaitGradReduce.apply(
        hidden, *load_balancer.source_parameters, weakref.ref(load_balancer)
    )
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, _weak_hook(load_balancer._start_pending_grad_reduces)
    )
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, _weak_hook(load_balancer._prepare_expert_backward)
    )
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_fc_layer_backward")
    hidden = _VirtualExpertBackwardHook.apply(
        hidden, _weak_hook(load_balancer._start_backward, temporaries)
    )
    hidden.backward()

    assert events == [
        "start_weight_push",
        "latent_up_fc_layer_backward",
        "combine_backward",
        "wait_weight_push",
        "expert_backward",
        "dispatch_backward",
        "start_grad_reduce",
        "router_and_shared_expert_backward",
        "finish_grad_reduce",
    ]
    # Without fused accumulation the reduction's output is copied, never aliased.
    for index, parameter in enumerate(load_balancer.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )
        assert parameter.grad.data_ptr() != load_balancer.source_grads[index].data_ptr()


def test_virtual_expert_fc2_reduction_starts_from_the_wgrad_store_and_fc1_after_dispatch():
    """FC2 reduces behind its own wgrad GEMM; only FC1 waits for dispatch backward."""
    started = []
    load_balancer = VirtualExpertLoadBalancer.__new__(VirtualExpertLoadBalancer)
    load_balancer._temporaries = _PassTemporaries(plan=object(), scratch=(None, None))

    def record(fc_layer):
        started.append(fc_layer)
        load_balancer._temporaries.started.add(fc_layer)

    load_balancer._start_grad_reduce = record

    # TE's delayed-wgrad protocol hands the GEMM to the store instead of
    # launching it; the store runs it and starts FC2's reduction right behind.
    gemm_calls = []
    store = _VirtualExpertFC2WgradStore(load_balancer)
    assert store.delay_wgrad_compute() and store.context is None
    store.put(["x", "dy", "out"], lambda *tensors: gemm_calls.append(tensors))
    assert gemm_calls == [("x", "dy", "out")]
    assert started == [1]

    load_balancer._start_pending_grad_reduces()
    assert started == [1, 0]


@requires_cuda
def test_virtual_expert_fused_wgrad_handoff_preserves_fp32():
    """Accumulate the FP32 virtual-expert wgrad into main_grad and return a BF16 dummy."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16, device=device))
    parameter.main_grad = torch.zeros(4, dtype=torch.float32, device=device)
    parameter.grad_added_to_main_grad = False
    external_wgrad = torch.full((4,), 1.0001, dtype=torch.float32, device=device)

    class FakeLoadBalancer:
        source_parameters = (parameter,)

        @staticmethod
        def _finish_grad_reduce():
            return (external_wgrad,)

    load_balancer = FakeLoadBalancer()
    hidden = torch.ones((), device=device, requires_grad=True)
    _VirtualExpertWaitGradReduce.apply(hidden, parameter, weakref.ref(load_balancer)).backward()

    torch.testing.assert_close(parameter.main_grad, external_wgrad, rtol=0, atol=0)
    assert parameter.grad_added_to_main_grad
    assert parameter.grad.dtype == torch.bfloat16


MEMBER_SHAPE = (128, 128)


def _mxfp8(tensor):
    """Quantize a BF16 tensor into an MXFP8 tensor holding both GEMM orientations."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(DType.kFloat8E4M3)(tensor)


def _fake_workspace(mxfp8, device, num_local_experts, native_staging):
    """The workspace surface a FC layer consumes, without symmetric memory or a group."""
    numel = MEMBER_SHAPE[0] * MEMBER_SHAPE[1]
    dtype = torch.uint8 if mxfp8 else torch.bfloat16
    slots = torch.zeros((num_local_experts, *MEMBER_SHAPE), dtype=dtype, device=device)
    scales = (
        torch.zeros((num_local_experts, numel // 32), dtype=torch.uint8, device=device)
        if mxfp8
        else None
    )
    return SimpleNamespace(
        mxfp8=mxfp8,
        member_shapes=(MEMBER_SHAPE,),
        grad_dtype=torch.float32,
        slot_views=lambda index: (slots, scales),
        grad_slots=lambda index: torch.zeros(
            (num_local_experts, *MEMBER_SHAPE), dtype=torch.float32, device=device
        ),
        native_staging=native_staging,
    )


def _count_uploads(fc_layer):
    """Count the pointer-table writes a FC layer issues from now on."""
    uploads = []
    upload = fc_layer._upload

    def counting_upload(table, rows):
        uploads.append(table)
        upload(table, rows)

    fc_layer._upload = counting_upload
    return uploads


def _gtp_fc_layer(weight_format, device, num_local_experts=2):
    """Build a GTP-sharded FC layer whose full weights arrive from a fake all-gather."""
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
    # same buffers and moves the prefetch chain along. Record which one the FC layer calls.
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
    # GTP's wgrad protocol: each group member hands out full-size scratch from a pool; the test
    # swaps a member's ``scratch`` to model the pool handing out another buffer.
    leader._weights = [
        SimpleNamespace(scratch=torch.zeros(MEMBER_SHAPE, dtype=torch.float32, device=device))
        for _ in range(num_local_experts)
    ]
    for weight in leader._weights:
        weight.get_wgrad_tensor = lambda weight=weight: weight.scratch

    workspace = _fake_workspace(
        mxfp8,
        device,
        num_local_experts,
        native_staging=lambda index: pytest.fail("GTP fc_layers must not allocate staging"),
    )
    return _VirtualExpertFCLayer("test fc_layer", parameters, workspace, 0), gathers


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert data pointers the push reads for one direction."""
    if weight_format == "bf16":
        return [weight.data_ptr() for weight in weights]
    field = "_rowwise_data" if direction == FORWARD else "_columnwise_data"
    return [getattr(weight, field).data_ptr() for weight in weights]


@requires_cuda
def test_virtual_expert_plain_fc_layer_binds_its_tables_once():
    """Plain parameters and their staging never move: every table is written at construction
    and a later ``prepare`` only checks the canary, the first expert's pointers."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(3)
    )
    staging = torch.zeros((3, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    workspace = _fake_workspace(False, device, 3, native_staging=lambda index: staging)
    fc_layer = _VirtualExpertFCLayer("plain fc_layer", parameters, workspace, 0)
    torch.cuda.synchronize(device)
    assert fc_layer.gtp_leader is None
    for direction in (FORWARD, BACKWARD):
        assert fc_layer.tables[direction][0].tolist() == _data_ptrs(parameters, "bf16", FORWARD)
    assert fc_layer.native_grad_bases.tolist() == [grad.data_ptr() for grad in staging]
    assert [parameter.main_grad.data_ptr() for parameter in fc_layer.runtime_parameters[:3]] == [
        grad.data_ptr() for grad in staging
    ]
    assert fc_layer.acquire_wgrad_scratch() is None

    uploads = _count_uploads(fc_layer)
    for _ in range(3):
        fc_layer.prepare(FORWARD)
        fc_layer.prepare(BACKWARD)
        fc_layer.consume(FORWARD)
    assert uploads == []

    # A parameter reload moves every expert's storage at once; the canary catches it.
    for parameter in parameters:
        parameter.data = torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device)
    fc_layer.prepare(FORWARD)
    torch.cuda.synchronize(device)
    assert uploads == [FORWARD]
    assert fc_layer.tables[FORWARD][0].tolist() == _data_ptrs(parameters, "bf16", FORWARD)
    assert _data_ptrs(fc_layer.runtime_parameters[:3], "bf16", FORWARD) == _data_ptrs(
        parameters, "bf16", FORWARD
    )


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_fc_layer_binds_gtp_gathers_into_its_pointer_tables(weight_format):
    """Point the push tables and the runtime parameters at each direction's GTP gather."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    runtime_ids = [id(parameter) for parameter in fc_layer.runtime_parameters]
    natives = fc_layer.runtime_parameters[:2]
    uploads = _count_uploads(fc_layer)

    fc_layer.prepare(FORWARD)
    torch.cuda.synchronize(device)
    forward_ptrs = _data_ptrs(gathers[FORWARD], weight_format, FORWARD)
    assert fc_layer.tables[FORWARD][0].tolist() == forward_ptrs
    assert _data_ptrs(natives, weight_format, FORWARD) == forward_ptrs

    # Forward and backward hold separate gathers; binding one leaves the other's table.
    fc_layer.prepare(BACKWARD)
    torch.cuda.synchronize(device)
    assert fc_layer.tables[BACKWARD][0].tolist() == _data_ptrs(
        gathers[BACKWARD], weight_format, BACKWARD
    )
    assert fc_layer.tables[FORWARD][0].tolist() == forward_ptrs
    if weight_format == "mxfp8":
        assert fc_layer.tables[FORWARD][1].tolist() == [
            weight._rowwise_scale_inv.data_ptr() for weight in gathers[FORWARD]
        ]
        assert fc_layer.tables[BACKWARD][1].tolist() == [
            weight._columnwise_scale_inv.data_ptr() for weight in gathers[BACKWARD]
        ]
        for parameter, forward, backward in zip(natives, gathers[FORWARD], gathers[BACKWARD]):
            assert parameter._rowwise_data is forward._rowwise_data
            assert parameter._columnwise_data is backward._columnwise_data
    # GTP gathers land in per-ticket buffers that stay put: later pushes write nothing.
    fc_layer.prepare(FORWARD)
    fc_layer.prepare(BACKWARD)
    assert uploads == [FORWARD, BACKWARD]

    # A gather landing in a new buffer rebinds when the canary (the first expert) sees it; the
    # TE ops keep the same parameter objects. Buffers only ever move together, so a lone other
    # expert moving is outside the guard.
    gathers[FORWARD][1] = _gtp_fc_layer(weight_format, device)[1][FORWARD][1]
    fc_layer.prepare(FORWARD)
    assert uploads == [FORWARD, BACKWARD]
    gathers[FORWARD][0] = _gtp_fc_layer(weight_format, device)[1][FORWARD][0]
    fc_layer.prepare(FORWARD)
    torch.cuda.synchronize(device)
    assert uploads == [FORWARD, BACKWARD, FORWARD]
    assert fc_layer.tables[FORWARD][0].tolist() == _data_ptrs(
        gathers[FORWARD], weight_format, FORWARD
    )
    assert _data_ptrs(natives, weight_format, FORWARD) == _data_ptrs(
        gathers[FORWARD], weight_format, FORWARD
    )
    assert [id(parameter) for parameter in fc_layer.runtime_parameters] == runtime_ids
    # Every runtime parameter carries the overwrite flag; the virtual slots accumulate into the
    # arena, the natives into per-backward GTP scratch that is acquired below.
    for parameter in fc_layer.runtime_parameters:
        assert parameter.overwrite_main_grad and parameter.main_grad is not None
    # TE's forward only checks for a main_grad; GTP natives get an empty placeholder until the
    # backward acquires the layer's scratch.
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    assert all(parameter.main_grad.numel() > 0 for parameter in fc_layer.runtime_parameters[2:])


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_fc_layer_peeks_for_the_push_and_consumes_at_the_gemm(weight_format):
    """The push reads GTP's gathered weights through the non-consuming peek; the consume, GTP's
    real chain step, happens separately and must find the buffers the push read."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    leader = fc_layer.gtp_leader

    for direction in (FORWARD, BACKWARD):
        leader.calls.clear()
        fc_layer.prepare(direction)
        assert leader.calls == [("peek", direction)]
        bound = fc_layer.bound[direction]
        fc_layer.consume(direction)
        assert leader.calls == [("peek", direction), ("consume", direction)]
        # The consume validates against the bound pointers and rebinds nothing.
        assert fc_layer.bound[direction] == bound

    # A consume that hands out other buffers than the peeked ones is an error, not a rebind:
    # the push has already copied the peeked bytes to the peers.
    gathers[FORWARD][0] = _gtp_fc_layer(weight_format, device)[1][FORWARD][0]
    with pytest.raises(RuntimeError, match="other buffers"):
        fc_layer.consume(FORWARD)


def test_virtual_expert_backward_consumes_gtp_weights_in_gemm_order():
    """Forward consumes FC1 then FC2 at the expert GEMMs; the layer-output hook takes the pass
    back and starts its push, then the combine hook waits for it and consumes FC2 before FC1
    (the expert backward's order), acquiring each FC layer's wgrad scratch."""
    events = []

    class FakeFCLayer:
        def __init__(self, name, index):
            self.name, self.index = name, index

        def consume(self, direction):
            events.append(("consume", self.name, direction))

        def acquire_wgrad_scratch(self):
            events.append(("acquire", self.name))
            return f"{self.name} scratch"

    load_balancer = VirtualExpertLoadBalancer.__new__(VirtualExpertLoadBalancer)
    load_balancer.fc_layers = [FakeFCLayer("FC1", 0), FakeFCLayer("FC2", 1)]
    load_balancer._wait_weight_push = lambda: events.append("wait")
    load_balancer._start_weight_push = lambda direction: events.append(("push", direction))
    temporaries = load_balancer._temporaries = _PassTemporaries(plan=object())

    load_balancer.prepare_expert_forward()
    assert events == ["wait", ("consume", "FC1", FORWARD), ("consume", "FC2", FORWARD)]

    # The layer output closes the forward; its backward hook hands the pass back.
    events.clear()
    load_balancer._temporaries = None
    load_balancer._start_backward(temporaries)
    assert load_balancer._temporaries is temporaries and events == [("push", BACKWARD)]
    with pytest.raises(RuntimeError, match="outstanding"):
        load_balancer._start_backward(_PassTemporaries())

    events.clear()
    load_balancer._prepare_expert_backward()
    assert events == [
        "wait",
        ("consume", "FC2", BACKWARD),
        ("acquire", "FC2"),
        ("consume", "FC1", BACKWARD),
        ("acquire", "FC1"),
    ]
    assert temporaries.scratch == ("FC1 scratch", "FC2 scratch") and temporaries.started == set()


@requires_cuda
def test_virtual_expert_gtp_fc_layer_writes_wgrads_into_gtp_scratch():
    """Under GTP the natives' ``main_grad`` and the reduction's pointer table point at the scratch
    acquired for the backward, which GTP's reduce-scatter reads without a copy; the release
    returns the natives to the placeholder so an unacquired backward fails."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, _ = _gtp_fc_layer("bf16", device)
    natives = fc_layer.runtime_parameters[:2]
    uploads = _count_uploads(fc_layer)

    scratch = fc_layer.acquire_wgrad_scratch()
    torch.cuda.synchronize(device)
    assert len(scratch) == 2 and all(grad.shape == MEMBER_SHAPE for grad in scratch)
    pointers = [grad.data_ptr() for grad in scratch]
    assert [parameter.main_grad.data_ptr() for parameter in natives] == pointers
    assert fc_layer.native_grad_bases.tolist() == pointers
    assert uploads == [_GRAD]

    fc_layer.release_wgrad_scratch()
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    # GTP's LIFO pool usually hands the same buffers back: nothing to write.
    assert fc_layer.acquire_wgrad_scratch() == scratch
    assert uploads == [_GRAD]
    fc_layer.release_wgrad_scratch()

    # One buffer changing is enough to rewrite the table; the pool recycles them one by one.
    fc_layer.gtp_leader._weights[1].scratch = torch.zeros(
        MEMBER_SHAPE, dtype=torch.float32, device=device
    )
    fresh = fc_layer.acquire_wgrad_scratch()
    torch.cuda.synchronize(device)
    assert uploads == [_GRAD, _GRAD]
    assert fresh[0] is scratch[0] and fresh[1] is not scratch[1]
    assert fc_layer.native_grad_bases.tolist() == [grad.data_ptr() for grad in fresh]
    assert [parameter.main_grad.data_ptr() for parameter in natives] == [
        grad.data_ptr() for grad in fresh
    ]
