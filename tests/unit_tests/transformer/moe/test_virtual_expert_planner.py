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
    BACKWARD,
    FORWARD,
    VirtualExpertLoadBalancer,
    _FCLayerPointerTables,
    _make_native_parameters,
    _make_virtual_parameters,
    _PassTemporaries,
    _VirtualExpertHook,
    _weak_hook,
)

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _load_balancer(fc_layers=()):
    """A balancer with only the state the process-local paths read; no transport, no arenas."""
    load_balancer = VirtualExpertLoadBalancer.__new__(VirtualExpertLoadBalancer)
    load_balancer.fc_layers = list(fc_layers)
    load_balancer._temporaries = None
    return load_balancer


def test_virtual_expert_rank_capacity_includes_per_expert_padding():
    """Size the static dropless capacity for HybridEP's per-segment alignment."""
    load_balancer = _load_balancer()
    load_balancer.router_topk, load_balancer.num_owned_experts = 22, 32
    load_balancer.config = SimpleNamespace(moe_expert_rank_capacity_factor=1.0)
    load_balancer._alignment = 256
    assert load_balancer._compute_rank_capacity(8192) == 196608
    load_balancer.config.moe_expert_rank_capacity_factor = 2.0
    assert load_balancer._compute_rank_capacity(8192) == 360448
    load_balancer.config.moe_expert_rank_capacity_factor = 1.0
    load_balancer._alignment = 0
    assert load_balancer._compute_rank_capacity(8192) == 180224


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
    """Order the four transport hooks across one layer's backward, as the dispatcher places them,
    and deliver what the finish returns as the source parameters' gradients."""
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
    hidden = _VirtualExpertHook.apply(
        hidden, _weak_hook(load_balancer._finish_grad_reduce), *load_balancer.source_parameters
    )
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, _weak_hook(load_balancer._start_pending_grad_reduces))
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, _weak_hook(load_balancer._prepare_expert_backward))
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_fc_layer_backward")
    hidden = _VirtualExpertHook.apply(
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
    for index, parameter in enumerate(load_balancer.source_parameters):
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter, index + 1), rtol=0, atol=0
        )


def test_virtual_expert_fc2_reduction_starts_from_the_wgrad_store_and_fc1_after_dispatch():
    """FC2 reduces behind its own wgrad GEMM; only FC1 waits for dispatch backward."""
    started = []
    load_balancer = _load_balancer()
    load_balancer._temporaries = _PassTemporaries(plan=object())

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

    # A reduction cannot start before the backward push was waited for (the expert backward's
    # preparation), nor outside a backward pass.
    load_balancer._temporaries = _PassTemporaries(plan=object(), push_in_flight=True)
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)
    load_balancer._temporaries = None
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)


MEMBER_SHAPE = (128, 128)


def _mxfp8(tensor):
    """Quantize a BF16 tensor into an MXFP8 tensor holding both GEMM orientations."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(DType.kFloat8E4M3)(tensor)


def _slot_parameters(mxfp8, device, num_local_experts, template):
    """Virtual-expert slot parameters over plain tensors, without symmetric memory or a group."""
    numel = MEMBER_SHAPE[0] * MEMBER_SHAPE[1]
    data = torch.zeros(
        (num_local_experts, *MEMBER_SHAPE),
        dtype=torch.uint8 if mxfp8 else torch.bfloat16,
        device=device,
    )
    scales = (
        torch.zeros((num_local_experts, numel // 32), dtype=torch.uint8, device=device)
        if mxfp8
        else None
    )
    grads = torch.zeros((num_local_experts, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    return _make_virtual_parameters(data, scales, grads, template)


def _build_fc_layer(name, parameters, template, staging, *, mxfp8=False):
    """Build a FC layer with its native and virtual runtime parameters; ``staging`` is the plain
    natives' wgrad staging (None under GTP)."""
    slots = _slot_parameters(mxfp8, parameters[0].device, len(parameters), template)
    natives = _make_native_parameters(parameters, slots, staging)
    return _FCLayerPointerTables(name, parameters, (*natives, *slots), staging, 0)


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
    # same buffers and moves the prefetch chain along. Record which one the balancer calls.
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

    fc_layer = _build_fc_layer("test fc_layer", parameters, leader, None, mxfp8=mxfp8)
    assert fc_layer.native_grads is None  # GTP natives write per-backward scratch, not staging
    return fc_layer, gathers


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert data pointers the push reads for one direction."""
    if weight_format == "bf16":
        return [weight.data_ptr() for weight in weights]
    field = "_rowwise_data" if direction == FORWARD else "_columnwise_data"
    return [getattr(weight, field).data_ptr() for weight in weights]


@requires_cuda
def test_virtual_expert_plain_fc_layer_binds_its_tables_once():
    """Plain parameters and their staging never move: the first weight table of a direction is
    written once, every later call checks that the pointers it was built from still hold."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(3)
    )
    staging = torch.zeros((3, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    fc_layer = _build_fc_layer("plain fc_layer", parameters, parameters[0], staging)
    load_balancer = _load_balancer([fc_layer])
    torch.cuda.synchronize(device)
    assert fc_layer.gtp_leader is None and fc_layer.native_grads is staging
    assert fc_layer.grad_table().tolist() == [grad.data_ptr() for grad in staging]
    assert [parameter.main_grad.data_ptr() for parameter in fc_layer.runtime_parameters[:3]] == [
        grad.data_ptr() for grad in staging
    ]
    # Plain sources are the parameters themselves, in both directions; GTP has nothing to consume.
    for direction in (FORWARD, BACKWARD):
        assert load_balancer._weight_sources(fc_layer, direction) is parameters
    load_balancer._consume_gtp_weights(fc_layer, FORWARD)

    # The first call of each direction writes its table; every later call returns the same
    # device tensor after validating.
    tables = {
        direction: fc_layer.weight_table(direction, parameters) for direction in (FORWARD, BACKWARD)
    }
    for _ in range(3):
        for direction in (FORWARD, BACKWARD):
            table = fc_layer.weight_table(direction, parameters)
            assert table is tables[direction]
            assert table[0].tolist() == _data_ptrs(parameters, "bf16", FORWARD)

    # The storage is static by contract: a moved parameter is an error, not a rebind.
    for parameter in parameters:
        parameter.data = torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="moved"):
        fc_layer.weight_table(FORWARD, parameters)


@requires_cuda
def test_virtual_expert_plain_fc_layer_hands_wgrads_to_the_source_parameters():
    """Accumulate the FP32 staging into ``main_grad`` and return a BF16 dummy for autograd; a
    parameter without a main_grad gets a copy of the staging, never an alias."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.ones(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(2)
    )
    parameters[0].main_grad = torch.zeros(MEMBER_SHAPE, dtype=torch.float32, device=device)
    parameters[0].grad_added_to_main_grad = False
    staging = torch.full((2, *MEMBER_SHAPE), 1.0001, dtype=torch.float32, device=device)
    fc_layer = _build_fc_layer("plain fc_layer", parameters, parameters[0], staging)

    grads = _load_balancer([fc_layer])._hand_off_wgrads(fc_layer)
    torch.testing.assert_close(parameters[0].main_grad, staging[0], rtol=0, atol=0)
    assert parameters[0].grad_added_to_main_grad
    assert grads[0].dtype == torch.bfloat16 and tuple(grads[0].shape) == MEMBER_SHAPE
    torch.testing.assert_close(grads[1], staging[1], rtol=0, atol=0)
    assert grads[1].data_ptr() != staging[1].data_ptr()


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_fc_layer_binds_gtp_gathers_into_its_pointer_tables(weight_format):
    """Point the push tables and the runtime parameters at each direction's GTP gather."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    load_balancer = _load_balancer([fc_layer])
    runtime_ids = [id(parameter) for parameter in fc_layer.runtime_parameters]
    natives = fc_layer.runtime_parameters[:2]

    def push_table(direction):
        return fc_layer.weight_table(direction, load_balancer._weight_sources(fc_layer, direction))

    forward_table = push_table(FORWARD)
    torch.cuda.synchronize(device)
    forward_ptrs = _data_ptrs(gathers[FORWARD], weight_format, FORWARD)
    assert forward_table[0].tolist() == forward_ptrs
    assert _data_ptrs(natives, weight_format, FORWARD) == forward_ptrs

    # Forward and backward hold separate gathers; binding one leaves the other's table.
    backward_table = push_table(BACKWARD)
    torch.cuda.synchronize(device)
    assert backward_table[0].tolist() == _data_ptrs(gathers[BACKWARD], weight_format, BACKWARD)
    assert forward_table[0].tolist() == forward_ptrs
    if weight_format == "mxfp8":
        assert forward_table[1].tolist() == [
            weight._rowwise_scale_inv.data_ptr() for weight in gathers[FORWARD]
        ]
        assert backward_table[1].tolist() == [
            weight._columnwise_scale_inv.data_ptr() for weight in gathers[BACKWARD]
        ]
        for parameter, forward, backward in zip(natives, gathers[FORWARD], gathers[BACKWARD]):
            assert parameter._rowwise_data is forward._rowwise_data
            assert parameter._columnwise_data is backward._columnwise_data
    # GTP gathers land in per-ticket buffers that stay put: later pushes write nothing.
    assert push_table(FORWARD) is forward_table and push_table(BACKWARD) is backward_table

    # Any gather landing in a new buffer is an error: the table records every expert's pointers.
    gathers[FORWARD][1] = _gtp_fc_layer(weight_format, device)[1][FORWARD][1]
    with pytest.raises(RuntimeError, match="moved"):
        push_table(FORWARD)
    assert forward_table[0].tolist() == forward_ptrs
    assert [id(parameter) for parameter in fc_layer.runtime_parameters] == runtime_ids
    # Every runtime parameter carries the overwrite flag; the virtual slots accumulate into the
    # arena, the natives into per-backward GTP scratch that a backward binds.
    for parameter in fc_layer.runtime_parameters:
        assert parameter.overwrite_main_grad and parameter.main_grad is not None
    # TE's forward only checks for a main_grad; GTP natives get an empty placeholder until the
    # backward binds the layer's scratch.
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    assert all(parameter.main_grad.numel() > 0 for parameter in fc_layer.runtime_parameters[2:])


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_balancer_peeks_for_the_push_and_consumes_at_the_gemm(weight_format):
    """The push reads GTP's gathered weights through the non-consuming peek; the consume, GTP's
    real chain step, happens separately and must find the buffers the push read."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    load_balancer = _load_balancer([fc_layer])
    leader = fc_layer.gtp_leader

    for direction in (FORWARD, BACKWARD):
        leader.calls.clear()
        table = fc_layer.weight_table(direction, load_balancer._weight_sources(fc_layer, direction))
        assert leader.calls == [("peek", direction)]
        load_balancer._consume_gtp_weights(fc_layer, direction)
        assert leader.calls == [("peek", direction), ("consume", direction)]
        # The consume validates against the bound pointers and rebinds nothing.
        assert fc_layer.weight_table(direction, gathers[direction]) is table

    # A consume that hands out other buffers than the peeked ones is an error, not a rebind:
    # the push has already copied the peeked bytes to the peers.
    gathers[FORWARD][0] = _gtp_fc_layer(weight_format, device)[1][FORWARD][0]
    with pytest.raises(RuntimeError, match="moved"):
        load_balancer._consume_gtp_weights(fc_layer, FORWARD)


def test_virtual_expert_backward_consumes_gtp_weights_in_gemm_order():
    """Forward consumes FC1 then FC2 at the expert GEMMs; the layer-output hook takes the pass
    back and starts its push, then the combine hook waits for it and consumes FC2 before FC1
    (the expert backward's order), binding each FC layer's GTP wgrad scratch."""
    events = []

    class FakeFCLayer:
        def __init__(self, name, index):
            self.name, self.index = name, index
            self.gtp_leader = SimpleNamespace(
                _weights=[SimpleNamespace(get_wgrad_tensor=lambda: f"{name} scratch")]
            )

        def weight_table(self, direction, sources):
            events.append(("consume", self.name, direction, sources))

        def bind_native_grads(self, grads):
            events.append(("bind", self.name, grads))

    load_balancer = _load_balancer([FakeFCLayer("FC1", 0), FakeFCLayer("FC2", 1)])
    load_balancer._wait_weight_push = lambda: events.append("wait")
    load_balancer._start_weight_push = lambda direction: events.append(("push", direction))
    load_balancer._weight_sources = lambda fc_layer, direction, peek=True: (
        "consumed" if not peek else "peeked"
    )
    temporaries = load_balancer._temporaries = _PassTemporaries(plan=object())

    load_balancer.prepare_expert_forward()
    assert events == [
        "wait",
        ("consume", "FC1", FORWARD, "consumed"),
        ("consume", "FC2", FORWARD, "consumed"),
    ]

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
        ("consume", "FC2", BACKWARD, "consumed"),
        ("bind", "FC2", ("FC2 scratch",)),
        ("consume", "FC1", BACKWARD, "consumed"),
        ("bind", "FC1", ("FC1 scratch",)),
    ]
    assert temporaries.started == set()

    # One wait per push: waiting for a push that was never started, or twice, is an error.
    assert not temporaries.push_in_flight
    with pytest.raises(RuntimeError, match="unwaited"):
        VirtualExpertLoadBalancer._wait_weight_push(load_balancer)


@requires_cuda
def test_virtual_expert_gtp_fc_layer_writes_wgrads_into_gtp_scratch():
    """Under GTP the natives' ``main_grad`` and the reduction's pointer table point at the scratch
    bound for the backward, which GTP's reduce-scatter reads without a copy; the hand-off gives
    that scratch to GTP's finalize and returns the natives to the placeholder, so an unbound
    backward fails."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, _ = _gtp_fc_layer("bf16", device)
    load_balancer = _load_balancer([fc_layer])
    natives = fc_layer.runtime_parameters[:2]
    finalized = []

    def finalize_group_grads(scratch):
        finalized.append([id(grad) for grad in scratch])
        return ["FC grad 0", "FC grad 1"]

    fc_layer.gtp_leader.finalize_group_grads = finalize_group_grads

    def bind_pool_scratch():
        fc_layer.bind_native_grads(
            tuple(weight.get_wgrad_tensor() for weight in fc_layer.gtp_leader._weights)
        )
        return fc_layer.native_grads

    scratch = bind_pool_scratch()
    torch.cuda.synchronize(device)
    assert len(scratch) == 2 and all(tuple(grad.shape) == MEMBER_SHAPE for grad in scratch)
    pointers = [grad.data_ptr() for grad in scratch]
    assert [parameter.main_grad.data_ptr() for parameter in natives] == pointers
    table = fc_layer.grad_table()
    assert table.tolist() == pointers
    with pytest.raises(RuntimeError, match="twice"):
        bind_pool_scratch()

    # The hand-off is GTP's protocol call over the bound scratch; its result is what autograd
    # delivers to the source parameters.
    assert load_balancer._hand_off_wgrads(fc_layer) == ("FC grad 0", "FC grad 1")
    assert finalized == [[id(grad) for grad in scratch]]
    assert fc_layer.native_grads is None
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    with pytest.raises(RuntimeError, match="no GTP wgrad scratch"):
        load_balancer._hand_off_wgrads(fc_layer)

    # GTP's LIFO pool usually hands the same buffers back: nothing to write.
    assert all(a is b for a, b in zip(bind_pool_scratch(), scratch))
    assert fc_layer.grad_table().data_ptr() == table.data_ptr() and table.tolist() == pointers
    load_balancer._hand_off_wgrads(fc_layer)

    # One buffer changing is enough to rewrite the table; the pool recycles them one by one.
    fc_layer.gtp_leader._weights[1].scratch = torch.zeros(
        MEMBER_SHAPE, dtype=torch.float32, device=device
    )
    fresh = bind_pool_scratch()
    torch.cuda.synchronize(device)
    # The table is rewritten in place, so the reduction keeps reading one device address.
    assert fc_layer.grad_table().data_ptr() == table.data_ptr()
    assert table.tolist() == [g.data_ptr() for g in fresh]
    assert fresh[0] is scratch[0] and fresh[1] is not scratch[1]
    assert [parameter.main_grad.data_ptr() for parameter in natives] == [
        grad.data_ptr() for grad in fresh
    ]
