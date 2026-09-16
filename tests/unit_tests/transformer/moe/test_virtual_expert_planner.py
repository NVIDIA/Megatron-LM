# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local coverage for the virtual-expert load balancer's host side: runtime parameters
and pointer tables, the backward hooks and the GTP protocol calls, plus the config rules.

Every test here runs in one process on one GPU, so a plain ``pytest
tests/unit_tests/transformer/moe/test_virtual_expert_planner.py`` runs the whole file.

The planner kernel exchanges histograms between ranks itself, so its placement and route
mapping are covered by the four-rank kernel tier in ``test_virtual_expert_triton.py``;
end-to-end gradient parity lives in ``test_virtual_expert_hybridep.py``.
"""

import os
import weakref
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.distributed_weight import (
    finalize_weight_grads,
    materialize_weight_for_backward,
    materialize_weight_for_forward,
    weight_grad_buffers,
)

from megatron.core.transformer.moe.experts import _VirtualExpertFC2WgradStore
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    VirtualExpertPlan,
    WeightDirection,
    _VirtualExpertHook,
    _VirtualExperts,
    _VirtualExpertStorage,
)
from tests.unit_tests.transformer.test_transformer_config import _virtual_expert_hybridep_config

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module", autouse=True)
def _select_rank_device():
    """Select the rank's GPU before these tests populate TE's device-implicit caches."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _load_balancer(virtual_experts=None):
    """A balancer with only the state the process-local paths read; no transport, no arenas."""
    load_balancer = VirtualExpertLoadBalancer.__new__(VirtualExpertLoadBalancer)
    load_balancer.virtual_experts = virtual_experts
    load_balancer._plan = None
    return load_balancer


@pytest.mark.parametrize(
    ("ep_size", "num_experts", "num_local_experts", "topk"),
    [
        (1, 2, 2, 2),
        (4, 4, 1, 2),  # A valid topology that differs from the configured EP size.
        (2, 4, 2, 2),  # A different global expert count.
        (2, 2, 2, 2),  # Incorrect local expert ownership.
        (2, 2, 1, 1),  # A different top-k.
    ],
)
def test_virtual_expert_init_rejects_layout_before_cuda(
    monkeypatch, ep_size, num_experts, num_local_experts, topk
):
    """Validate actual process-group size and expert ownership before touching CUDA."""
    config = _virtual_expert_hybridep_config()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: ep_size)
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("invalid layout reached CUDA")
    )
    with pytest.raises(ValueError, match="runtime layout must match TransformerConfig"):
        VirtualExpertLoadBalancer().initialize_virtual_expert_load_balancer(
            group=object(),
            num_local_experts=num_local_experts,
            router_topk=topk,
            num_experts=num_experts,
            config=config,
        )


@requires_cuda
def test_virtual_expert_compact_api_requirement_preserves_plain_hybridep(monkeypatch):
    """Old HybridEP remains usable without virtual experts; virtual experts fail before launch."""
    from megatron.core.transformer.moe import moe_utils, token_dispatcher

    monkeypatch.setattr(moe_utils, "hybrid_ep_dense_topk_routing", lambda *_: False)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dense_topk_routing", lambda *_: False)
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", object())
    for virtual in (False, True):
        config = _virtual_expert_hybridep_config(moe_virtual_expert_load_balance=virtual)
        if virtual:
            with pytest.raises(AssertionError, match="compact top-k routing API"):
                token_dispatcher._HybridEPManager(object(), 1, 2, config)
        else:
            manager = token_dispatcher._HybridEPManager(object(), 1, 2, config)
            assert not manager._dense_topk_routing
            assert manager.dense_routing_metadata


@requires_cuda
@pytest.mark.parametrize(
    ("ep_size", "num_experts", "topk", "routing"),
    [
        (2, 2, 1, "none"),
        (33, 66, 2, "none"),
        (64, 8192, 32, "seq_aux_loss"),
        (64, 512, 10, "quantile_balancing"),
    ],
)
def test_virtual_expert_init_accepts_supported_limits(
    monkeypatch, ep_size, num_experts, topk, routing
):
    """Large positive HybridEP SM budgets are valid; transport caps its own budget later."""
    config = _virtual_expert_hybridep_config(
        expert_model_parallel_size=ep_size,
        num_moe_experts=num_experts,
        moe_router_topk=topk,
        moe_router_pre_softmax=True,
        moe_flex_dispatcher_num_sms=64,
        moe_router_load_balancing_type=routing,
        moe_router_fusion=routing != "quantile_balancing",
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: ep_size)
    manager = VirtualExpertLoadBalancer()
    manager.initialize_virtual_expert_load_balancer(
        group=object(),
        num_local_experts=num_experts // ep_size,
        router_topk=topk,
        num_experts=num_experts,
        config=config,
    )
    assert (manager.ep_size, manager.num_owned_experts, manager.router_topk) == (
        ep_size,
        num_experts // ep_size,
        topk,
    )


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


@pytest.mark.parametrize("num_tokens", [0, 7, 9])
def test_virtual_expert_runtime_rejects_changed_token_count(num_tokens):
    """A rejected resize leaves the initialized layer usable at its original token count."""
    manager = _load_balancer(virtual_experts=object())
    manager.num_tokens, manager.rank_capacity = 8, 64
    with pytest.raises(ValueError, match=f"fixed token count; expected 8, got {num_tokens}"):
        manager._runtime_init(torch.empty((num_tokens, 128), device="meta"))
    assert (manager.num_tokens, manager.rank_capacity) == (8, 64)
    # The flattened token count is fixed; sequence/batch dimensions may be reshaped.
    manager._runtime_init(torch.empty((2, 4, 128), device="meta"))
    assert (manager.num_tokens, manager.rank_capacity) == (8, 64)


def test_virtual_expert_hooks_do_not_keep_their_owner_alive():
    """A method captured by an autograd context must not close a cycle through its owner, and
    is called with its arguments when the gradient passes."""

    class Owner:
        def __init__(self):
            self.calls = []

        def method(self, value):
            self.calls.append(value)

    owner = Owner()
    hidden = torch.ones((), requires_grad=True)
    _VirtualExpertHook.apply(hidden, owner.method, (7,)).backward()
    assert owner.calls == [7]
    kept = _VirtualExpertHook.apply(hidden, owner.method, (8,))
    alive = weakref.ref(owner)
    del owner
    assert alive() is None and kept.requires_grad


def test_virtual_expert_backward_hooks_span_the_transport_window():
    """Order the four transport hooks across one layer's backward, as the dispatcher places them,
    and deliver what the finish returns as the source parameters' gradients."""
    events = []
    plan = VirtualExpertPlan(None, None)

    class FakeLoadBalancer:
        source_parameters = (torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(())))

        def __init__(self):
            self.source_grads = tuple(
                torch.full_like(parameter, index + 1)
                for index, parameter in enumerate(self.source_parameters)
            )

        def _start_backward(self, current):
            assert current is plan
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
        hidden, load_balancer._finish_grad_reduce, (), *load_balancer.source_parameters
    )
    hidden = BackwardMarker.apply(hidden, "router_and_shared_expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._start_pending_grad_reduces, ())
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._prepare_expert_backward, ())
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = BackwardMarker.apply(hidden, "latent_up_fc_layer_backward")
    hidden = _VirtualExpertHook.apply(hidden, load_balancer._start_backward, (plan,))
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
    load_balancer._plan = VirtualExpertPlan(None, None)

    def record(fc_layer):
        started.append(fc_layer)
        load_balancer._plan.started.add(fc_layer)

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
    load_balancer._plan = VirtualExpertPlan(None, None, push_in_flight=True)
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)

    load_balancer._plan = None
    with pytest.raises(RuntimeError, match="prepared backward"):
        VirtualExpertLoadBalancer._start_grad_reduce(load_balancer, 1)

    alive = weakref.ref(load_balancer)
    del load_balancer
    assert alive() is None, "TE's wgrad store must not retain the load balancer"


MEMBER_SHAPE = (128, 128)


def _mxfp8(tensor):
    """Quantize a BF16 tensor into an MXFP8 tensor holding both GEMM orientations."""
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    return MXFP8Quantizer(DType.kFloat8E4M3)(tensor)


def _fake_virtual_experts(mxfp8, device, num_local_experts, template, main_grads):
    """The slots object over plain tensors: no symmetric memory, no group."""
    numel = MEMBER_SHAPE[0] * MEMBER_SHAPE[1]

    virtual_experts = _VirtualExpertStorage.__new__(_VirtualExpertStorage)
    virtual_experts.weight_handle = virtual_experts.grad_handle = None
    virtual_experts.config = SimpleNamespace(
        mxfp8=mxfp8,
        member_shapes=(MEMBER_SHAPE,),
        gtp=(getattr(template, "is_gtp_weight_remat", False),),
        grad_dtype=main_grads.dtype if main_grads is not None else torch.float32,
        device=device,
    )
    virtual_experts.num_local_experts = num_local_experts
    virtual_experts._weight_sections = [
        num_local_experts * numel,
        num_local_experts * (numel // 32 if mxfp8 else 0),
    ]
    virtual_experts._grad_sections = [num_local_experts * numel]
    virtual_experts.weight_arena = torch.zeros(
        sum(virtual_experts._weight_sections),
        dtype=torch.uint8 if mxfp8 else torch.bfloat16,
        device=device,
    )
    virtual_experts.grad_arena = torch.zeros(
        sum(virtual_experts._grad_sections), dtype=virtual_experts.config.grad_dtype, device=device
    )
    virtual_experts.slot_weights = (virtual_experts._slot_parameters(0, template),)
    return virtual_experts


def _build_fc_layer(parameters, template, main_grads, *, mxfp8=False):
    """Build a FC layer with DDP main gradients, or deferred GTP buffers when None."""
    if main_grads is not None:
        for parameter, grad in zip(parameters, main_grads):
            parameter.main_grad = grad
    virtual_experts = _fake_virtual_experts(
        mxfp8, parameters[0].device, len(parameters), template, main_grads
    )
    return _VirtualExperts(virtual_experts, (parameters,))


def _gtp_fc_layer(weight_format, device, num_local_experts=2):
    """Build a GTP-sharded FC layer whose full weights arrive from a fake all-gather."""
    mxfp8 = weight_format == "mxfp8"

    def make():
        weight = torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device)
        return _mxfp8(weight) if mxfp8 else weight

    gathers = {WeightDirection.FORWARD: [make() for _ in range(num_local_experts)]}
    gathers[WeightDirection.BACKWARD] = [make() for _ in range(num_local_experts)]
    leader = make()
    leader.is_gtp_weight_remat = True
    leader.prefetch_initialized = (
        True  # Established chain; cold setup uses real GTP in test_gtp_peek.
    )
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

    leader.peek_group_for_forward = protocol("peek", WeightDirection.FORWARD)
    leader.peek_group_for_backward = protocol("peek", WeightDirection.BACKWARD)
    leader.materialize_group_for_forward = protocol("consume", WeightDirection.FORWARD)
    leader.materialize_group_for_backward = protocol("consume", WeightDirection.BACKWARD)
    parameters = (leader, *(make() for _ in range(num_local_experts - 1)))
    # GTP's wgrad protocol hands out persistent full-size scratch. Tests can move a buffer
    # deliberately to verify that the bridge rejects a broken storage invariant.
    leader._weights = [
        SimpleNamespace(scratch=torch.zeros(MEMBER_SHAPE, dtype=torch.float32, device=device))
        for _ in range(num_local_experts)
    ]
    for weight in leader._weights:
        weight.get_wgrad_tensor = lambda weight=weight, **kwargs: weight.scratch

    fc_layer = _build_fc_layer(parameters, leader, None, mxfp8=mxfp8)
    assert fc_layer.native_grads[0] is None  # GTP natives write per-backward scratch, not staging
    return fc_layer, gathers


def _weight_push(monkeypatch, fc_layer):
    """Exercise the real push binding and stream handoff, replacing only the transport kernel."""
    balancer = _load_balancer(fc_layer)
    balancer.device = fc_layer.config.device
    balancer._plan = VirtualExpertPlan(None, None)
    balancer.prefetch_done = torch.cuda.Event()
    balancer.weight_streams = (torch.cuda.Stream(),)
    monkeypatch.setattr(
        "megatron.core.transformer.moe.virtual_expert_load_balancer."
        "launch_virtual_expert_weight_prefetch",
        lambda *args, **kwargs: None,
    )

    def push(direction):
        balancer._start_weight_push(direction)
        balancer._wait_weight_push()
        return fc_layer._tables[0][direction][0]

    return balancer, push


def _data_ptrs(weights, weight_format, direction):
    """Return the per-expert data pointers the push reads for one direction."""
    if weight_format == "bf16":
        return [weight.data_ptr() for weight in weights]
    field = "_rowwise_data" if direction == WeightDirection.FORWARD else "_columnwise_data"
    return [getattr(weight, field).data_ptr() for weight in weights]


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_plain_fc_layer_binds_its_tables_once(weight_format):
    """Plain parameters and their main gradients never move: the first weight table of a direction is
    written once, every later call checks that the pointers it was built from still hold."""
    device = torch.device("cuda", torch.cuda.current_device())
    mxfp8 = weight_format == "mxfp8"
    weights = [torch.randn(MEMBER_SHAPE, dtype=torch.bfloat16, device=device) for _ in range(3)]
    parameters = tuple(torch.nn.Parameter(_mxfp8(w) if mxfp8 else w) for w in weights)
    main_grads = torch.zeros(
        (3, *MEMBER_SHAPE), dtype=torch.bfloat16 if mxfp8 else torch.float32, device=device
    )
    fc_layer = _build_fc_layer(parameters, parameters[0], main_grads, mxfp8=mxfp8)
    torch.cuda.synchronize(device)
    assert fc_layer.gtp_leaders[0] is None
    assert all(g is p.main_grad for g, p in zip(fc_layer.native_grads[0], parameters))
    assert not fc_layer._tables[0]
    grad_table = fc_layer.grad_table(0)
    assert grad_table.tolist() == [grad.data_ptr() for grad in main_grads]
    assert fc_layer.grad_table(0) is grad_table
    assert [parameter.main_grad.data_ptr() for parameter in fc_layer.runtime_weights[0][:3]] == [
        grad.data_ptr() for grad in main_grads
    ]
    assert materialize_weight_for_forward(fc_layer.runtime_weights[0]) == list(
        fc_layer.runtime_weights[0]
    )

    # The first call of each direction writes its table; every later call returns the same
    # device tensor after validating.
    tables = {direction: fc_layer.weight_tables(direction)[0] for direction in WeightDirection}
    runtime_ptrs = {d: fc_layer._ptrs(d, fc_layer.runtime_weights[0]) for d in WeightDirection}
    grad_ptrs = [p.main_grad.data_ptr() for p in fc_layer.runtime_weights[0]]
    for _ in range(3):
        for direction in WeightDirection:
            table = fc_layer.get_weight_table(0, direction, parameters)
            assert table is tables[direction]
            assert table[0].tolist() == _data_ptrs(parameters, weight_format, direction)
            assert fc_layer._ptrs(direction, fc_layer.runtime_weights[0]) == runtime_ptrs[direction]
            assert [p.main_grad.data_ptr() for p in fc_layer.runtime_weights[0]] == grad_ptrs

    # The storage is static by contract: a moved parameter is an error, not a rebind.
    if mxfp8:
        parameters[0]._rowwise_data = parameters[0]._rowwise_data.clone()
    else:
        parameters[0].data = parameters[0].data.clone()
    with pytest.raises(RuntimeError, match="moved"):
        fc_layer.get_weight_table(0, WeightDirection.FORWARD, parameters)


@requires_cuda
def test_virtual_expert_plain_fc_layer_hands_wgrads_to_the_source_parameters():
    """Preserve DDP accumulation and return BF16 dummies without adding direct gradients twice."""
    device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(
        torch.nn.Parameter(torch.ones(MEMBER_SHAPE, dtype=torch.bfloat16, device=device))
        for _ in range(2)
    )
    for parameter in parameters:
        parameter.main_grad = torch.full(MEMBER_SHAPE, 2.0, dtype=torch.float32, device=device)
        parameter.grad_added_to_main_grad = False
    contributions = torch.full((2, *MEMBER_SHAPE), 1.0001, dtype=torch.float32, device=device)
    storage = _fake_virtual_experts(False, device, len(parameters), parameters[0], contributions)
    fc_layer = _VirtualExperts(storage, (parameters,))

    expected = torch.full_like(contributions, 2.0)
    for _ in range(2):
        for target, contribution in zip(fc_layer.native_grads[0], contributions):
            target.add_(contribution)
        expected.add_(contributions)
        grads = fc_layer.hand_off_wgrads(0)
        for parameter, grad, accumulated in zip(parameters, grads, expected):
            torch.testing.assert_close(parameter.main_grad, accumulated, rtol=0, atol=0)
            assert parameter.grad_added_to_main_grad
            assert grad.dtype == torch.bfloat16 and tuple(grad.shape) == MEMBER_SHAPE


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_fc_layer_binds_gtp_gathers_into_its_pointer_tables(
    monkeypatch, weight_format
):
    """Push binds the actual gathered buffers; initialization never requests gather storage."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    _, push_table = _weight_push(monkeypatch, fc_layer)
    runtime_ids = [id(parameter) for parameter in fc_layer.runtime_weights[0]]
    natives = fc_layer.runtime_weights[0][:2]
    assert not fc_layer._tables[0] and fc_layer.gtp_leaders[0].calls == []
    for direction in WeightDirection:
        assert _data_ptrs(natives, weight_format, direction) != _data_ptrs(
            gathers[direction], weight_format, direction
        )

    forward_table = push_table(WeightDirection.FORWARD)
    torch.cuda.synchronize(device)
    forward_ptrs = _data_ptrs(
        gathers[WeightDirection.FORWARD], weight_format, WeightDirection.FORWARD
    )
    assert forward_table[0].tolist() == forward_ptrs
    assert _data_ptrs(natives, weight_format, WeightDirection.FORWARD) == forward_ptrs

    # Directions can use different buffers, including BF16. Each keeps its own static table.
    backward_table = push_table(WeightDirection.BACKWARD)
    torch.cuda.synchronize(device)
    assert backward_table[0].tolist() == _data_ptrs(
        gathers[WeightDirection.BACKWARD], weight_format, WeightDirection.BACKWARD
    )
    assert forward_table[0].tolist() == forward_ptrs
    if weight_format == "mxfp8":
        assert forward_table[1].tolist() == [
            weight._rowwise_scale_inv.data_ptr() for weight in gathers[WeightDirection.FORWARD]
        ]
        assert backward_table[1].tolist() == [
            weight._columnwise_scale_inv.data_ptr() for weight in gathers[WeightDirection.BACKWARD]
        ]
        for parameter, forward, backward in zip(
            natives, gathers[WeightDirection.FORWARD], gathers[WeightDirection.BACKWARD]
        ):
            assert parameter._rowwise_data is forward._rowwise_data
            assert parameter._columnwise_data is backward._columnwise_data
    assert backward_table is not forward_table
    # Cached calls retain the same tables and runtime storage in both directions.
    tables = {WeightDirection.FORWARD: forward_table, WeightDirection.BACKWARD: backward_table}
    for direction in (*WeightDirection, WeightDirection.FORWARD):
        assert push_table(direction) is tables[direction]
        assert _data_ptrs(natives, weight_format, direction) == _data_ptrs(
            gathers[direction], weight_format, direction
        )

    # Any gather landing in a new buffer is an error: the table records every expert's pointers.
    gathers[WeightDirection.FORWARD][1] = _gtp_fc_layer(weight_format, device)[1][
        WeightDirection.FORWARD
    ][1]
    with pytest.raises(RuntimeError, match="moved"):
        push_table(WeightDirection.FORWARD)
    assert forward_table[0].tolist() == forward_ptrs
    assert [id(parameter) for parameter in fc_layer.runtime_weights[0]] == runtime_ids
    # Every runtime parameter carries the overwrite flag; the virtual slots accumulate into the
    # arena, the natives into per-backward GTP scratch that a backward binds.
    for parameter in fc_layer.runtime_weights[0]:
        assert parameter.overwrite_main_grad and parameter.main_grad is not None
    # TE's forward only checks for a main_grad; GTP natives get an empty placeholder until the
    # backward binds the layer's scratch.
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    assert all(parameter.main_grad.numel() > 0 for parameter in fc_layer.runtime_weights[0][2:])


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
@pytest.mark.parametrize("bind_table_first", [False, True], ids=["first_table", "cached_table"])
def test_virtual_expert_rejects_moved_runtime_storage(monkeypatch, weight_format, bind_table_first):
    """Every table lookup checks the runtime binding, including before a table exists."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    _, push = _weight_push(monkeypatch, fc_layer)
    push(WeightDirection.FORWARD)
    if not bind_table_first:
        fc_layer._tables[0].clear()
    parameter = fc_layer.runtime_weights[0][0]
    if weight_format == "bf16":
        parameter.data = parameter.data.clone()
    else:
        parameter._rowwise_data = parameter._rowwise_data.clone()
    moved_pointer = _data_ptrs([parameter], weight_format, WeightDirection.FORWARD)
    with pytest.raises(RuntimeError, match="runtime storage moved"):
        fc_layer.get_weight_table(0, WeightDirection.FORWARD, gathers[WeightDirection.FORWARD])
    assert _data_ptrs([parameter], weight_format, WeightDirection.FORWARD) == moved_pointer


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_balancer_peeks_for_the_push_and_consumes_at_the_gemm(
    monkeypatch, weight_format
):
    """The push reads GTP's gathered weights through the non-consuming peek; the consume, GTP's
    real chain step, happens separately and must find the buffers the push read."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, gathers = _gtp_fc_layer(weight_format, device)
    _, push = _weight_push(monkeypatch, fc_layer)
    leader = fc_layer.gtp_leaders[0]

    for direction in WeightDirection:
        leader.calls.clear()
        table = push(direction)
        assert leader.calls == [("peek", direction)]
        materialize = (
            materialize_weight_for_forward
            if direction == WeightDirection.FORWARD
            else materialize_weight_for_backward
        )
        materialized = materialize(fc_layer.runtime_weights[0])
        assert leader.calls == [("peek", direction), ("consume", direction)]
        assert len(materialized) == 2 * len(gathers[direction])
        assert _data_ptrs(materialized[:2], weight_format, direction) == _data_ptrs(
            gathers[direction], weight_format, direction
        )
        # The consume validates against the bound pointers and rebinds nothing.
        assert fc_layer.get_weight_table(0, direction, gathers[direction]) is table
        if direction == WeightDirection.BACKWARD:
            fc_layer._bind_gtp_grads(0, None)

    # A consume that hands out other buffers than the peeked ones is an error, not a rebind:
    # the push has already copied the peeked bytes to the peers.
    gathers[WeightDirection.FORWARD][0] = _gtp_fc_layer(weight_format, device)[1][
        WeightDirection.FORWARD
    ][0]
    with pytest.raises(RuntimeError, match="moved"):
        materialize_weight_for_forward(fc_layer.runtime_weights[0])


def test_virtual_expert_hooks_leave_gtp_materialization_to_te():
    """VE hooks wait for transport but leave materialization and wgrad acquisition to TE."""
    events = []

    owner = _VirtualExperts.__new__(_VirtualExperts)
    owner.config = SimpleNamespace(gtp=(True, True))
    owner.materialize = lambda *args: pytest.fail("VE hook materialized GTP weights")
    owner._bind_gtp_grads = lambda *args: pytest.fail("VE hook acquired GTP wgrad buffers")
    load_balancer = _load_balancer(owner)
    load_balancer._wait_weight_push = lambda: events.append("wait")
    load_balancer._start_weight_push = lambda direction: events.append(("push", direction))
    plan = load_balancer._plan = VirtualExpertPlan(None, None)

    # The layer output closes the forward; its backward hook hands the pass back.
    events.clear()
    load_balancer._plan = None
    load_balancer._start_backward(plan)
    assert load_balancer._plan is plan and events == [("push", WeightDirection.BACKWARD)]
    with pytest.raises(RuntimeError, match="outstanding"):
        load_balancer._start_backward(VirtualExpertPlan(None, None))

    events.clear()
    load_balancer._prepare_expert_backward()
    assert events == ["wait"]
    assert plan.started == set()

    # One wait per push: waiting for a push that was never started, or twice, is an error.
    assert not plan.push_in_flight
    with pytest.raises(RuntimeError, match="unwaited"):
        VirtualExpertLoadBalancer._wait_weight_push(load_balancer, None, None)


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_te_protocol_uses_persistent_buffers_and_defers_gtp(weight_format):
    """TE gets the native ring and virtual arena buffers; GTP finalizes only after VE reduction."""
    device = torch.device("cuda", torch.cuda.current_device())
    owner, _ = _gtp_fc_layer(weight_format, device)
    weights = owner.runtime_weights[0]
    finalized = []

    def finalize(grads):
        finalized.append(tuple(g.data_ptr() for g in grads))
        return [None, None]

    owner.gtp_leaders[0].finalize_group_grads = finalize
    for _ in range(2):
        assert owner.native_grads[0] is None
        buffers = weight_grad_buffers(weights, MEMBER_SHAPE, torch.bfloat16, device)
        assert all(buffer.dtype == torch.float32 for buffer in buffers)
        assert all(buffer is weight.main_grad for buffer, weight in zip(buffers, weights))
        assert all(
            buffer is source.scratch
            for buffer, source in zip(buffers, owner.gtp_leaders[0]._weights)
        )
        pointers = tuple(g.data_ptr() for g in buffers[:2])
        completed = len(finalized)
        assert finalize_weight_grads(weights, buffers) == [None] * len(weights)
        assert len(finalized) == completed
        assert owner.hand_off_wgrads(0) == (None, None)
        assert finalized[-1] == pointers
        assert owner.native_grads[0] is None

    # TE may keep the runtime parameters alive, but their protocol must not retain the owner.
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None


@requires_cuda
def test_virtual_expert_gtp_fc_layer_writes_wgrads_into_gtp_scratch():
    """Under GTP the natives' ``main_grad`` and the reduction's pointer table point at the scratch
    bound for the backward, which GTP's reduce-scatter reads without a copy; the hand-off gives
    that scratch to GTP's finalize and returns the natives to the placeholder, so an unbound
    backward fails."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, _ = _gtp_fc_layer("bf16", device)
    natives = fc_layer.runtime_weights[0][:2]
    finalized = []

    def finalize_group_grads(scratch):
        finalized.append([id(grad) for grad in scratch])
        return ["FC grad 0", "FC grad 1"]

    fc_layer.gtp_leaders[0].finalize_group_grads = finalize_group_grads

    def bind_pool_scratch():
        fc_layer._bind_gtp_grads(
            0, tuple(weight.get_wgrad_tensor() for weight in fc_layer.gtp_leaders[0]._weights)
        )
        return fc_layer.native_grads[0]

    scratch = bind_pool_scratch()
    torch.cuda.synchronize(device)
    assert len(scratch) == 2 and all(tuple(grad.shape) == MEMBER_SHAPE for grad in scratch)
    pointers = [grad.data_ptr() for grad in scratch]
    assert [parameter.main_grad.data_ptr() for parameter in natives] == pointers
    assert "grad" not in fc_layer._tables[0]
    table = fc_layer.grad_table(0)
    assert table.tolist() == pointers
    with pytest.raises(RuntimeError, match="twice"):
        bind_pool_scratch()

    # The hand-off is GTP's protocol call over the bound scratch; its result is what autograd
    # delivers to the source parameters.
    assert fc_layer.hand_off_wgrads(0) == ("FC grad 0", "FC grad 1")
    assert finalized == [[id(grad) for grad in scratch]]
    assert fc_layer.native_grads[0] is None
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)
    with pytest.raises(RuntimeError, match="no GTP wgrad scratch"):
        fc_layer.hand_off_wgrads(0)

    # Persistent GTP scratch keeps the table unchanged on the next backward.
    assert all(a is b for a, b in zip(bind_pool_scratch(), scratch))
    assert fc_layer.grad_table(0).data_ptr() == table.data_ptr() and table.tolist() == pointers
    fc_layer.hand_off_wgrads(0)

    # Reject moved scratch before changing the pointer table or binding runtime destinations.
    fc_layer.gtp_leaders[0]._weights[1].scratch = torch.zeros(
        MEMBER_SHAPE, dtype=torch.float32, device=device
    )
    with pytest.raises(RuntimeError, match="grad: source or runtime storage moved"):
        bind_pool_scratch()
    assert fc_layer._tables[0]["grad"][0][0].data_ptr() == table.data_ptr()
    assert table.tolist() == pointers and fc_layer.native_grads[0] is None
    assert all(parameter.main_grad.numel() == 0 for parameter in natives)


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_unchanged_pointer_table_requires_no_upload(monkeypatch, weight_format):
    """Repeated bindings reuse every device table without allocating or copying pointers."""
    device = torch.device("cuda", torch.cuda.current_device())
    fc_layer, _ = _gtp_fc_layer(weight_format, device)
    _, push = _weight_push(monkeypatch, fc_layer)
    tables = {direction: push(direction) for direction in WeightDirection}
    grads = [weight.scratch for weight in fc_layer.gtp_leaders[0]._weights]
    grad_table = fc_layer.get_weight_table(0, "grad", grads)

    def unexpected_upload(*args, **kwargs):
        pytest.fail("unchanged pointer table allocated or copied a tensor")

    monkeypatch.setattr(torch, "tensor", unexpected_upload)
    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_upload)
    for _ in range(3):
        for direction in WeightDirection:
            assert push(direction) is tables[direction]
            materialize = (
                materialize_weight_for_forward
                if direction == WeightDirection.FORWARD
                else materialize_weight_for_backward
            )
            materialize(fc_layer.runtime_weights[0])
        assert fc_layer.get_weight_table(0, "grad", grads) is grad_table
        fc_layer._bind_gtp_grads(0, None)


@requires_cuda
@pytest.mark.parametrize("weight_format", ["bf16", "mxfp8"])
def test_virtual_expert_owners_share_slots_but_keep_native_bindings_separate(weight_format):
    """Two layers share transport memory, preserve their own weights/tables, and tear down once."""
    device = torch.device("cuda", torch.cuda.current_device())
    mxfp8 = weight_format == "mxfp8"
    weights = [
        torch.full(MEMBER_SHAPE, float(i + 1), dtype=torch.bfloat16, device=device)
        for i in range(2)
    ]
    parameters = [torch.nn.Parameter(_mxfp8(w) if mxfp8 else w) for w in weights]
    main_grads = torch.zeros((1, *MEMBER_SHAPE), dtype=torch.float32, device=device)
    first = _build_fc_layer((parameters[0],), parameters[0], main_grads, mxfp8=mxfp8)
    cls = first.storage
    parameters[1].main_grad = torch.zeros_like(parameters[0].main_grad)
    second = _VirtualExperts(cls, ((parameters[1],),))
    assert first.storage is second.storage
    assert first.runtime_weights[0][0] is not second.runtime_weights[0][0]
    assert first.runtime_weights[0][1] is second.runtime_weights[0][1]
    assert first.native_grads[0][0] is parameters[0].main_grad
    assert second.native_grads[0][0] is parameters[1].main_grad
    assert first.native_grads[0][0].data_ptr() != second.native_grads[0][0].data_ptr()
    for direction in WeightDirection:
        tables = [owner.weight_tables(direction)[0] for owner in (first, second)]
        assert tables[0] is not tables[1]
        for owner, parameter, table in zip((first, second), parameters, tables):
            expected = _data_ptrs([parameter], weight_format, direction)
            assert table[0].tolist() == expected
            assert _data_ptrs(owner.runtime_weights[0][:1], weight_format, direction) == expected
    arenas = [weakref.ref(cls.weight_arena), weakref.ref(cls.grad_arena)]
    cls.destroy()
    cls.destroy()  # Idempotent, even while layers retain runtime parameters.
    assert all(ref() is None for ref in arenas), "slot views retained their arena base"
    assert cls.weight_arena is None and cls.grad_arena is None
    assert cls.slot_weights == ()
    for owner in (first, second):
        slot = owner.runtime_weights[0][1]
        assert slot.main_grad is None
        assert (
            all(getattr(slot, name).numel() == 0 for name in ("_rowwise_data", "_columnwise_data"))
            if mxfp8
            else slot.numel() == 0
        )
    for parameter, expected in zip(parameters, weights):
        torch.testing.assert_close(
            parameter.dequantize() if mxfp8 else parameter, expected, rtol=0, atol=0
        )
    alive = weakref.ref(first)
    del first
    assert alive() is None, "shared storage must not retain per-layer owners"


@requires_cuda
def test_virtual_expert_grad_table_is_created_before_reduction_stream_wait(monkeypatch):
    """Binding main_grad does not upload; first reduce creates on the producer stream, then reuses."""
    device = torch.device("cuda", torch.cuda.current_device())
    owner, _ = _gtp_fc_layer("bf16", device)
    balancer = _load_balancer(owner)
    balancer.device = device
    balancer.grad_stream = torch.cuda.Stream()
    balancer.grad_reduce_done = torch.cuda.Event()
    producer = torch.cuda.Stream()
    grads = tuple(w.scratch for w in owner.gtp_leaders[0]._weights)
    owner._bind_gtp_grads(0, grads)
    assert not owner._tables[0]
    seen = []
    get_table = owner.get_weight_table

    def table_on_producer(*args):
        assert torch.cuda.current_stream() == producer
        return get_table(*args)

    def reduction_on_consumer(*args, native_grads, **kwargs):
        assert torch.cuda.current_stream() == balancer.grad_stream
        seen.append(native_grads[0].clone())

    monkeypatch.setattr(owner, "get_weight_table", table_on_producer)
    monkeypatch.setattr(
        "megatron.core.transformer.moe.virtual_expert_load_balancer.launch_virtual_expert_grad_reduce",
        reduction_on_consumer,
    )
    for _ in range(2):
        balancer._plan = VirtualExpertPlan(None, None)
        with torch.cuda.stream(producer):
            balancer._start_grad_reduce(0)
        balancer.grad_reduce_done.synchronize()
        assert seen[-1].tolist() == [g.data_ptr() for g in grads]
        monkeypatch.setattr(
            torch, "tensor", lambda *args, **kwargs: pytest.fail("repeated table upload")
        )
        monkeypatch.setattr(
            owner, "get_weight_table", lambda *args: pytest.fail("rechecking bound table")
        )
    assert len(seen) == 2


@requires_cuda
def test_virtual_expert_shared_allocation_failure_releases_storage():
    """A failed allocation releases its partial arena before it can enter the shared cache."""
    allocations = []

    class FailingStorage(_VirtualExpertStorage):
        def _allocate(self, group, config, templates):
            self.weight_arena = torch.empty(1, device=config.device)
            allocations.append(weakref.ref(self.weight_arena))
            raise RuntimeError("allocation failed")

    config = SimpleNamespace(device=torch.device("cuda", torch.cuda.current_device()))
    with pytest.raises(RuntimeError, match="allocation failed"):
        FailingStorage(None, config, (None,))
    assert allocations[0]() is None
