# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for grouped-expert CUDA graphs and asynchronous gather completion."""

import pytest
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import fp8_autocast

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp
from megatron.core import parallel_state as ps
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import (
    gtp_native_fp8_load_context,
    wrap_module_params_gtp,
)
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    create_cudagraphs,
    delete_cuda_graphs,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (
    _make_gtp_linear,
    _requires_mxfp8,
)


@pytest.fixture
def expert_group():
    """Give each test its own GTP4 x EGTP4 process groups."""
    if torch.distributed.get_world_size() != 4:
        pytest.skip("requires torchrun --nproc-per-node=4")
    torch.cuda.synchronize()
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(gtp_remat_size=4, expert_gtp_remat_size=4)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    try:
        yield ps.get_expert_gtp_weight_remat_group()
    finally:
        torch.cuda.synchronize()
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class _Block(torch.nn.Module):
    def __init__(self, group):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        for name, output in (("linear_fc1", 256), ("linear_fc2", 128)):
            op = te.ops.GroupedLinear(
                2, 128, output, bias=False, device="cuda", dtype=torch.bfloat16
            )
            op.gtp_remat_size = group.size()
            wrap_module_params_gtp(op, ["weight0", "weight1"], group, is_grouped=True)
            setattr(self.mlp.experts, name, op)
        # Immutable routing metadata, not a mutable model buffer. The runner restores
        # model buffers between warmup and capture; doing so would bump a saved tensor's version.
        self.splits = torch.tensor([32, 32], device="cuda", dtype=torch.int64)

    def forward(self, x):
        gate, value = self.mlp.experts.linear_fc1(x, self.splits).chunk(2, dim=-1)
        return x + self.mlp.experts.linear_fc2(F.silu(gate) * value, self.splits)


def _initialize_grads(model, grad_ready=None):
    """Allocate main_grad and register a DDP-style grad-ready hook that appends to ``grad_ready``."""
    params = list(model.parameters())
    for p in params:
        p.main_grad = torch.zeros(p.shape, device=p.device, dtype=torch.bfloat16)
        hook = (lambda p=p: grad_ready.append(p)) if grad_ready is not None else (lambda: None)
        # Retain AccumulateGrad, as DDP does, for local graph replay.
        p.register_grad_accum_hook(p.expand_as(p).grad_fn.next_functions[0][0], hook)
    return params


def _allow_retained_grad_nodes():
    if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
        torch.autograd.graph.set_override_stale_capture_stream(True)


@torch.no_grad()
def _copy_weights(params, values):
    for p, value in zip(params, values):
        p.copy_(value)


def _step(model, x, recipe=None):
    x.grad = None
    for p in model.parameters():
        p.main_grad.zero_()
        p.grad = None
        p.grad_added_to_main_grad = False
    with fp8_autocast(enabled=recipe is not None, fp8_recipe=recipe):
        y = model(x)
    # Local backward graphs can reuse forward output storage.
    saved_y = y.detach().clone()
    (y.float().square().mean() * (100 if recipe else 1)).backward()
    gtp.wait_for_gtp_grad_reduction_on_current_stream()
    return saved_y


def _snapshot(y, x, params):
    torch.cuda.synchronize()
    return y.clone(), x.grad.clone(), [p.main_grad.clone() for p in params]


def _assert_snapshot(actual, expected):
    y, dx, grads = actual
    ref_y, ref_dx, ref_grads = expected
    torch.testing.assert_close(y, ref_y, rtol=0.01, atol=0.01)
    for grad, ref in zip([dx, *grads], [ref_dx, *ref_grads]):
        torch.testing.assert_close(grad, ref, rtol=0.02, atol=2e-5)


def _native_model(group, recipe):
    config = TransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=256,
        moe_grouped_gemm=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        bias_dropout_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # Upstream prohibits interleaved GLU with GTP. Use the supported,
        # noninterleaved op-fuser path; this does not require a fused MLP kernel.
        gated_linear_unit=True,
        activation_func=F.silu,
        use_transformer_engine_op_fuser=True,
        moe_router_padding_for_quantization=True,
        moe_token_dispatcher_type="alltoall",
        fp8='e4m3',
        fp8_recipe=Fp8Recipe.mxfp8,
        fp8_param=True,
        gtp_weight_remat_size=group.size(),
        expert_gtp_weight_remat_size=group.size(),
        gradient_accumulation_fusion=True,
    )

    class FusedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.experts = TEGroupedMLP(
                2,
                config,
                GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            self.splits = torch.tensor([256, 256], device='cuda', dtype=torch.int64)
            self.probs = torch.ones(512, device='cuda', dtype=torch.float32)

        def forward(self, x):
            output, _ = self.mlp.experts(x, self.splits, self.probs)
            return x + output

    with te.quantized_model_init(enabled=True, recipe=recipe):
        model = torch.nn.Sequential(*[FusedBlock() for _ in range(4)]).cuda()
    return model


@pytest.mark.parametrize(
    "mxfp8",
    [False, pytest.param(True, marks=pytest.mark.launch_on_gb200)],
    ids=["bf16", "native-mxfp8"],
)
def test_grouped_full_iteration_replay(expert_group, monkeypatch, mxfp8):
    if mxfp8:
        _requires_mxfp8()
        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")
    recipe = MXFP8BlockScaling() if mxfp8 else None
    model = (
        _native_model(expert_group, recipe)
        if mxfp8
        else torch.nn.Sequential(*[_Block(expert_group) for _ in range(4)]).cuda()
    )
    gtp.classify_gtp_remat_chains(model, cuda_graph_impl="full_iteration")
    assert all(hasattr(p, "_gtp_prefetch_chain_name") for p in model.parameters())
    params = _initialize_grads(model)
    assert len(params) == 16
    if mxfp8:
        assert all(getattr(p, "_gtp_native_fp8", False) for p in params)
    masters = [
        torch.nn.Parameter(
            (gtp.dequantize_gtp_native_fp8(p) if mxfp8 else p.detach()).float().clone()
        )
        for p in params
    ]
    initial = [m.detach().clone() for m in masters]
    shape = (512, 256) if mxfp8 else (64, 128)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Keep AccumulateGrad alive across capture; this reference is load-bearing.
    input_grad_node = x.expand_as(x).grad_fn.next_functions[0][0]
    inputs = [torch.randn_like(x) for _ in range(6)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def sync_params():
        with gtp_native_fp8_load_context(model):
            _copy_weights(params, [m.to(torch.bfloat16) for m in masters])

    def trajectory(run, expected=None):
        optimizer = torch.optim.Adam(masters, lr=1e-3)
        snapshots = []
        for i, value in enumerate(inputs):
            _copy_weights([x], [value])
            actual = _snapshot(run(), x, params)
            if expected is not None:
                _assert_snapshot(actual, expected[i])
            snapshots.append(actual)
            for p, m in zip(params, masters):
                m.grad = p.main_grad.float() / expert_group.size()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            sync_params()
        return snapshots

    with torch.cuda.stream(stream):
        sync_params()
        for _ in range(3):
            _step(model, x, recipe)
        # Four blocks exercise alternating buffers and same-role links in both directions.
        roles = [
            [getattr(b.mlp.experts, role).weight0 for b in model]
            for role in ("linear_fc1", "linear_fc2")
        ]
        assert roles[0][0].chain_id != roles[1][0].chain_id
        for leaders in roles:
            assert all(
                gtp._chain_is_grouped(p.chain_id) and gtp._chain_is_graphed(p.chain_id)
                for p in leaders
            )
            for previous, current in zip(leaders, leaders[1:]):
                assert previous.next_w is current and current.prev_w is previous
        if mxfp8:
            assert all(
                b.mlp.experts._with_fused_impl and b.mlp.experts._fused_ops is not None
                for b in model
            )
        expected = trajectory(lambda: _step(model, x, recipe))
        assert any(not torch.equal(m, value) for m, value in zip(masters, initial))
        _copy_weights(masters, initial)
        sync_params()
        _allow_retained_grad_nodes()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                captured_y = _step(model, x, recipe)

            def replay():
                graph.replay()
                return captured_y

            trajectory(replay, expected)
        finally:
            graph.reset()
            torch.cuda.synchronize()


def _run_local_graph_replay(group):
    """Replay per-block local graphs against an eager reference of the same weight updates.

    Weight updates are published lazily, through the parameter-ready callback a sharding backend
    such as DDP with overlapped param gather installs: a replay that gathers a weight without
    first making it ready reads the previous step's value. Every step also checks that each
    weight's grad-ready hook fired exactly once.
    """
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        gtp_weight_remat_size=group.size(),
        num_moe_experts=2,
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True,
        cuda_graph_impl="local",
        cuda_graph_modules=["moe"],
        cuda_graph_warmup_steps=2,
    )

    class LocalBlock(MegatronModule):
        def __init__(self, index):
            super().__init__(config)
            self.unit = _Block(group)
            self.layer_number = index + 1
            self.is_first_layer = index == 0
            self.is_last_layer = index == 3

        def compute(self, hidden_states):
            return self.unit(hidden_states)

        def forward(self, hidden_states):
            return self.compute(hidden_states=hidden_states)

    model = torch.nn.Sequential(*[LocalBlock(i) for i in range(4)]).cuda()
    gtp.classify_gtp_remat_chains(model, cuda_graph_modules=["moe"], cuda_graph_impl="local")
    grad_ready = []
    params = _initialize_grads(model, grad_ready)
    inputs = [torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    initial = [p.detach().clone() for p in params]
    unpublished = {}

    def publish(index):
        if index in unpublished:
            params[index].detach().copy_(unpublished.pop(index))

    for index, p in enumerate(params):
        setattr(p, PARAM_READY_CALLBACK_ATTR, lambda index=index: publish(index))

    def step(x):
        x = x.detach().clone().requires_grad_()
        grad_ready.clear()
        snapshot = _snapshot(_step(model, x), x, params)
        assert sorted(map(id, grad_ready)) == sorted(map(id, params)), "grad-ready not once each"
        return snapshot

    def update(lazy):
        with torch.no_grad():
            for index, p in enumerate(params):
                # The sign flip makes a stale weight visible in every output and gradient.
                value = (p.main_grad * (0.01 / group.size()) - p).detach()
                if lazy:
                    unpublished[index] = value
                else:
                    p.copy_(value)

    reference = []
    for x in inputs:
        reference.append(step(x))
        update(lazy=False)
    _copy_weights(params, initial)
    for block in model:
        block.manager = CudaGraphManager(config, base_module=block, function_name="compute")
    step(inputs[0])  # Record the cross-block call order without updating weights.
    _allow_retained_grad_nodes()
    create_cudagraphs()
    try:
        for x, expected in zip(inputs, reference):
            _assert_snapshot(step(x), expected)
            update(lazy=True)
        assert any(block.manager.cudagraph_runners[0]._gtp_wgrad_ring_slots for block in model)
    finally:
        delete_cuda_graphs()


def test_grouped_local_graph_replay(expert_group):
    """Local graphs replay like eager, and still do for a model rebuilt after teardown."""
    _run_local_graph_replay(expert_group)
    # The documented in-process rebuild: tear down the graphs, reset GTP's chain state, rebuild.
    gtp.reset_gtp_state()
    _run_local_graph_replay(expert_group)


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("grouped", [False, True], ids=["dense", "grouped"])
def test_consumer_does_not_wait_for_later_gather(expert_group, captured, grouped):
    group = expert_group
    model = _Block(group) if grouped else _make_gtp_linear(128, 128, group)
    gtp.set_cuda_graph_modules([], cuda_graph_impl="full_iteration" if captured else "none")
    gtp.classify_gtp_chains(torch.nn.Sequential(model))
    param = model.mlp.experts.linear_fc2.weight0 if grouped else model.weight
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ready = torch.cuda.Event(enable_timing=True, external=True)
    tail = torch.cuda.Event(enable_timing=True, external=True)
    ag_stream = gtp.get_ag_stream(param.chain_id, group)

    def run():
        _, param._prefetch_handle = param._all_gather_weight(async_op=True, fwd=True)
        with torch.cuda.stream(ag_stream):
            # Stand in for unrelated work queued after this gather on the AG stream.
            torch.cuda._sleep(100_000_000)
            tail.record()
        param._wait_param_gather()
        param.ag_event.wait()
        ready.record()
        torch.cuda.current_stream().wait_stream(ag_stream)

    with torch.cuda.stream(stream):
        param._all_gather_weight(async_op=False, fwd=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph() if captured else None
    try:
        if captured:
            with torch.cuda.graph(graph, stream=stream):
                run()
            graph.replay()
        else:
            with torch.cuda.stream(stream):
                run()
        torch.cuda.synchronize()
    finally:
        if captured:
            graph.reset()
    # CUDA timestamps compare device dependencies, not Python launch timings. The
    # consumer must become ready before the unrelated tail, in eager AND replay.
    assert ready.elapsed_time(tail) > 1.0, "consumer incorrectly depends on later AG-stream work"


@pytest.mark.parametrize("recompute", [False, True])
def test_sync_fallback_records_completion(expert_group, monkeypatch, recompute):
    """A gather that falls back to synchronous (no Work handle) must still record its completion
    event, or the consumer waits on whatever the event recorded last."""
    group = expert_group
    layer = _make_gtp_linear(128, 128, group)
    param = layer.weight
    full = torch.empty(param._unsharded_shape_padded, device="cuda", dtype=torch.bfloat16)
    torch.distributed.all_gather_into_tensor(full, param.detach(), group=group)
    original_gather = gtp.gather_along_first_dim

    def synchronous_fallback(*args, **kwargs):
        kwargs["async_op"] = False
        result, handle = original_gather(*args, **kwargs)
        assert handle is None
        return result, None

    monkeypatch.setattr(gtp, "gather_along_first_dim", synchronous_fallback)
    event = param._recompute_ag_event if recompute else param.ag_event
    # The event's previous record stays pending behind a long kernel on an unrelated stream.
    stale = torch.cuda.Stream()
    with torch.cuda.stream(stale):
        torch.cuda._sleep(1_000_000_000)
        event.record()
    result, handle = param._all_gather_weight(async_op=True, fwd=True, recompute=recompute)
    assert handle.handle is None
    gtp.get_ag_stream(param.chain_id, group).synchronize()
    completed = event.query()
    torch.cuda.synchronize()
    assert completed, "the gather did not record its completion event"
    torch.testing.assert_close(result, full, atol=0, rtol=0)
