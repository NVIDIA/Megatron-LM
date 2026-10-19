# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for #6057's grouped-expert chains under CUDA graphs."""

import pytest
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8_autocast

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel.gtp_api import wrap_module_params_gtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


@pytest.fixture(autouse=True)
def isolate_grouped_graph_streams():
    """Do not join streams belonging to a previous test's destroyed process groups."""
    # Production keeps stream caches for a model's lifetime. These tests construct
    # and destroy independent models/groups; a full graph's final fence must not
    # join uncaptured streams left behind by an earlier eager or local-graph test.
    torch.cuda.synchronize()
    gtp._AG_STREAMS.clear()
    gtp._RS_STREAMS.clear()
    yield
    torch.cuda.synchronize()
    gtp._AG_STREAMS.clear()
    gtp._RS_STREAMS.clear()


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


def _worker_full_iteration(rank, world_size, port):
    del port
    group = torch.distributed.new_group(list(range(world_size)))
    torch.manual_seed(123)
    model = torch.nn.Sequential(*[_Block(group) for _ in range(4)]).cuda()
    gtp.set_cuda_graph_modules([], cuda_graph_impl="full_iteration")
    gtp.classify_gtp_chains(model)
    for param in model.parameters():
        param.main_grad = torch.zeros_like(param)
        # Mirror DDP's retained AccumulateGrad node across warmup and capture.
        param.register_grad_accum_hook(param.expand_as(param).grad_fn.next_functions[0][0], None)
        param.grad_added_to_main_grad = False
    static_x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Keep AccumulateGrad alive across capture; this strong reference is intentional.
    input_grad_node = static_x.expand_as(static_x).grad_fn.next_functions[0][0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def step():
        static_x.grad = None
        for param in model.parameters():
            param.main_grad.zero_()
            param.grad = None
        with fp8_autocast(enabled=False):
            out = model(static_x)
        loss = out.float().square().mean()
        loss.backward()
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        return out, loss

    with torch.cuda.stream(stream):
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Four blocks exercise alternating AG-buffer reuse in both directions.
        for role in ("linear_fc1", "linear_fc2"):
            leaders = [getattr(block.mlp.experts, role).weight0 for block in model]
            assert all("grouped" in p.chain_id and p.chain_id.endswith("_graphed") for p in leaders)
            for previous, current in zip(leaders, leaders[1:]):
                assert previous.next_w is current
                assert current.prev_w is previous
            assert leaders[0].chain_id != (
                model[0].mlp.experts.linear_fc2.weight0.chain_id
                if role == "linear_fc1"
                else model[0].mlp.experts.linear_fc1.weight0.chain_id
            )

        # Save an eager trajectory with changing inputs AND weights; constant-weight replay
        # alone would miss stale prefetch buffers and graph-owned gradients.
        initial = [param.detach().clone() for param in model.parameters()]
        inputs = [torch.randn_like(static_x) for _ in range(6)]
        expected = []
        with torch.no_grad():
            for param, value in zip(model.parameters(), initial):
                param.copy_(value)
        for x in inputs:
            with torch.no_grad():
                static_x.copy_(x)
            out, loss = step()
            torch.cuda.synchronize()
            expected.append(
                (
                    out.detach().clone(),
                    static_x.grad.clone(),
                    [param.main_grad.clone() for param in model.parameters()],
                )
            )
            with torch.no_grad():
                for param in model.parameters():
                    param.sub_(0.01 / world_size * param.main_grad)

        with torch.no_grad():
            for param, value in zip(model.parameters(), initial):
                param.copy_(value)
        torch.cuda.synchronize()
        if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
            torch.autograd.graph.set_override_stale_capture_stream(True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_out, captured_loss = step()
        for x, (ref_out, ref_dx, ref_grads) in zip(inputs, expected):
            with torch.no_grad():
                static_x.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_out, ref_out, rtol=0.01, atol=0.01)
            torch.testing.assert_close(static_x.grad, ref_dx, rtol=0.02, atol=2e-5)
            for param, ref_grad in zip(model.parameters(), ref_grads):
                torch.testing.assert_close(param.main_grad, ref_grad, rtol=0.02, atol=2e-5)
            with torch.no_grad():
                for param in model.parameters():
                    param.sub_(0.01 / world_size * param.main_grad)
        graph.reset()
        torch.cuda.synchronize()
    torch.distributed.destroy_process_group(group)


def test_grouped_full_iteration_replay():
    _run_distributed(_worker_full_iteration, 4)


def _worker_local_grouped(rank, world_size, port):
    del port
    from megatron.core import parallel_state as ps
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
    from megatron.core.transformer.transformer_config import TransformerConfig

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(gtp_remat_size=world_size, expert_gtp_remat_size=world_size)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    group = ps.get_expert_gtp_weight_remat_group()
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        gtp_weight_remat_size=world_size,
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
    gtp.set_cuda_graph_modules(["moe"], cuda_graph_impl="local")
    gtp.classify_gtp_chains(model)
    for param in model.parameters():
        param.main_grad = torch.zeros_like(param)
        param.grad_added_to_main_grad = False
        # Upstream records finalization only when a DDP readiness hook is registered.
        param.register_grad_accum_hook(
            param.expand_as(param).grad_fn.next_functions[0][0], lambda: None
        )
    inputs = [torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    initial = [p.detach().clone() for p in model.parameters()]

    def step(x):
        for p in model.parameters():
            p.main_grad.zero_()
            p.grad = None
            p.grad_added_to_main_grad = False
        x = x.detach().clone().requires_grad_()
        with fp8_autocast(enabled=False):
            out = model(x)
        # Local graphs share a memory pool across forward/backward. Preserve the
        # output before backward can reuse its storage; reading it afterward is invalid.
        saved_out = out.detach().clone()
        out.float().square().mean().backward()
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        torch.cuda.synchronize()
        return (saved_out, x.grad.clone(), [p.main_grad.clone() for p in model.parameters()])

    def update():
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(0.01 / world_size * p.main_grad)

    reference = []
    for x in inputs:
        reference.append(step(x))
        update()
    with torch.no_grad():
        for p, value in zip(model.parameters(), initial):
            p.copy_(value)
    for block in model:
        block.manager = CudaGraphManager(config, base_module=block, function_name="compute")
    step(inputs[0])  # Record the cross-block graph call order without an optimizer update.
    # Match the full-iteration runner's compatibility setting for retained
    # AccumulateGrad nodes on this PyTorch build.
    if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
        torch.autograd.graph.set_override_stale_capture_stream(True)
    create_cudagraphs()
    try:
        for x, (ref_y, ref_dx, ref_grads) in zip(inputs, reference):
            y, dx, grads = step(x)
            torch.testing.assert_close(y, ref_y, rtol=0.01, atol=0.01)
            torch.testing.assert_close(dx, ref_dx, rtol=0.02, atol=2e-5)
            for grad, ref_grad in zip(grads, ref_grads):
                torch.testing.assert_close(grad, ref_grad, rtol=0.02, atol=2e-5)
            update()
        runners = [block.manager.cudagraph_runners[0] for block in model]
        assert any(runner._gtp_wgrad_ring_slots for runner in runners)
        # Cross-graph RS finalizes the next block's gradients in the current
        # block's backward graph. The last forward block can have no hooks;
        # require exact coverage across the whole stack, not a nonempty plan per block.
        finalized_ids = [
            id(param)
            for runner in runners
            for _, params in runner._gtp_finalize_hook_plan
            for param in params
        ]
        assert sorted(finalized_ids) == sorted(id(param) for param in model.parameters())
    finally:
        torch.cuda.synchronize()
        for block in model:
            for runner in block.manager.cudagraph_runners:
                if runner.fwd_graph is not None:
                    runner.fwd_graph.reset()
                if runner.bwd_graph is not None:
                    runner.bwd_graph.reset()
        delete_cuda_graphs()
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def test_grouped_local_graph_replay():
    _run_distributed(_worker_local_grouped, 4)


def _worker_grouped_completion(rank, world_size, port, captured):
    del rank, port
    group = torch.distributed.new_group(list(range(world_size)))
    model = _Block(group)
    gtp.set_cuda_graph_modules([], cuda_graph_impl="full_iteration" if captured else "none")
    gtp.classify_gtp_chains(torch.nn.Sequential(model))
    param = model.mlp.experts.linear_fc2.weight0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ready = torch.cuda.Event(enable_timing=True, external=True)
    tail = torch.cuda.Event(enable_timing=True, external=True)
    ag_stream = gtp.get_ag_stream(param.chain_id, group)

    def run():
        _, handle = param._all_gather_weight(async_op=True, fwd=True)
        param._prefetch_handle = handle
        with torch.cuda.stream(ag_stream):
            # Stand in for the later, unrelated FC1 prefetch on the shared AG stream.
            torch.cuda._sleep(100_000_000)
            tail.record()
        param._wait_param_gather()
        param.ag_event.wait()
        ready.record()
        torch.cuda.current_stream().wait_stream(ag_stream)

    with torch.cuda.stream(stream):
        param._all_gather_weight(async_op=False, fwd=True)
    torch.cuda.synchronize()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        graph.replay()
    else:
        with torch.cuda.stream(stream):
            run()
    torch.cuda.synchronize()
    # CUDA timestamps compare device dependencies, not Python launch timings. The
    # consumer must become ready before the unrelated tail, in eager AND replay.
    assert ready.elapsed_time(tail) > 1.0, "consumer incorrectly depends on later AG-stream work"
    if captured:
        graph.reset()
    torch.distributed.destroy_process_group(group)


@pytest.mark.parametrize("captured", [False, True])
def test_grouped_consumer_does_not_wait_for_later_gather(captured):
    _run_distributed(_worker_grouped_completion, 4, captured)


def _worker_grouped_sync_fallback(rank, world_size, port, monkeypatch, recompute):
    del rank, port
    group = torch.distributed.new_group(list(range(world_size)))
    op = te.ops.GroupedLinear(1, 128, 128, bias=False, device="cuda", dtype=torch.bfloat16)
    op.gtp_remat_size = group.size()
    wrap_module_params_gtp(op, ["weight0"], group, is_grouped=True)
    param = op.weight0
    param.chain_id = "GTP_remat_grouped_fc1_ungraphed"
    original_gather = gtp.gather_along_first_dim
    calls = []

    def synchronous_fallback(*args, **kwargs):
        assert kwargs["async_op"] is True
        kwargs["async_op"] = False
        result, handle = original_gather(*args, **kwargs)
        assert handle is None
        calls.append(True)
        return result, None

    monkeypatch.setattr(gtp, "gather_along_first_dim", synchronous_fallback)
    try:
        result, handle = param._all_gather_weight(async_op=True, fwd=True, recompute=recompute)
        assert calls == [True]  # One local expert takes the non-coalesced gather path.
        assert handle.completion_recorded
        if recompute:
            param._recompute_prefetch_handle = handle
            param._wait_recompute_param_gather()
            event = param._recompute_ag_event
        else:
            param._prefetch_handle = handle
            param._wait_param_gather()
            event = param.ag_event
        event.wait()
        torch.cuda.synchronize()
        assert event.query()
        assert result[0].shape == (128, 128)
    finally:
        torch.distributed.destroy_process_group(group)


@pytest.mark.parametrize("recompute", [False, True])
def test_grouped_sync_fallback_records_completion(monkeypatch, recompute):
    _run_distributed(_worker_grouped_sync_fallback, 4, monkeypatch, recompute)


def _worker_mxfp8_op_fuser(rank, world_size, port):
    """Exercise op-fuser SwiGLU experts with native MXFP8 shards across Adam updates."""
    del rank, port
    from transformer_engine.common.recipe import MXFP8BlockScaling

    from megatron.core import parallel_state as ps
    from megatron.core.enums import Fp8Recipe
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.gtp_api import gtp_native_fp8_load_context
    from megatron.core.tensor_parallel.random import (
        initialize_rng_tracker,
        model_parallel_cuda_manual_seed,
    )
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(gtp_remat_size=world_size, expert_gtp_remat_size=world_size)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    recipe = MXFP8BlockScaling()
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
        gtp_weight_remat_size=world_size,
        expert_gtp_weight_remat_size=world_size,
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
                pg_collection=pg,
            )
            self.splits = torch.tensor([256, 256], device='cuda', dtype=torch.int64)
            self.probs = torch.ones(512, device='cuda', dtype=torch.float32)

        def forward(self, x):
            output, _ = self.mlp.experts(x, self.splits, self.probs)
            return x + output

    with te.quantized_model_init(enabled=True, recipe=recipe):
        model = torch.nn.Sequential(*[FusedBlock() for _ in range(4)]).cuda()
    gtp.set_cuda_graph_modules([], cuda_graph_impl='full_iteration')
    gtp.classify_gtp_chains(model)
    params = list(model.parameters())
    assert len(params) == 16
    assert all(getattr(p, '_gtp_native_fp8', False) for p in params)
    masters = [torch.nn.Parameter(gtp.dequantize_gtp_native_fp8(p).float().clone()) for p in params]
    initial = [m.detach().clone() for m in masters]
    for p in params:
        p.main_grad = torch.zeros(p.shape, device='cuda', dtype=torch.bfloat16)
        p.grad_added_to_main_grad = False
        p.register_grad_accum_hook(p.expand_as(p).grad_fn.next_functions[0][0], None)
    static_x = torch.randn(512, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    # Keep AccumulateGrad alive across capture; this strong reference is intentional.
    input_grad_node = static_x.expand_as(static_x).grad_fn.next_functions[0][0]
    inputs = [torch.randn_like(static_x) for _ in range(6)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def sync_params():
        with torch.no_grad(), gtp_native_fp8_load_context(model):
            for p, m in zip(params, masters):
                p.copy_(m.to(torch.bfloat16))

    def step():
        static_x.grad = None
        for p in params:
            p.main_grad.zero_()
            p.grad = None
            p.grad_added_to_main_grad = False
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output = model(static_x)
        loss = output.float().square().mean() * 100
        loss.backward()
        gtp.wait_for_gtp_grad_reduction_on_current_stream()
        return output

    def update(optimizer):
        for p, m in zip(params, masters):
            m.grad = p.main_grad.float() / world_size
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_params()

    with torch.cuda.stream(stream):
        sync_params()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        assert all(block.mlp.experts._with_fused_impl for block in model)
        assert all(block.mlp.experts._fused_ops is not None for block in model)
        optimizer = torch.optim.Adam(masters, lr=1e-3)
        expected = []
        for x in inputs:
            with torch.no_grad():
                static_x.copy_(x)
            output = step()
            torch.cuda.synchronize()
            expected.append(
                (
                    output.detach().clone(),
                    static_x.grad.clone(),
                    [p.main_grad.clone() for p in params],
                )
            )
            update(optimizer)
        assert any(not torch.equal(m, value) for m, value in zip(masters, initial))
        with torch.no_grad():
            for m, value in zip(masters, initial):
                m.copy_(value)
        sync_params()
        optimizer = torch.optim.Adam(masters, lr=1e-3)
        if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
            torch.autograd.graph.set_override_stale_capture_stream(True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = step()
        for x, (ref_output, ref_dx, ref_grads) in zip(inputs, expected):
            with torch.no_grad():
                static_x.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured_output, ref_output, rtol=0.01, atol=0.01)
            torch.testing.assert_close(static_x.grad, ref_dx, rtol=0.02, atol=2e-5)
            for p, ref_grad in zip(params, ref_grads):
                torch.testing.assert_close(p.main_grad, ref_grad, rtol=0.02, atol=2e-5)
            update(optimizer)
        graph.reset()
        torch.cuda.synchronize()
    ps.destroy_model_parallel()
    ps.initialize_model_parallel()


def test_mxfp8_op_fuser_full_iteration_replay(monkeypatch):
    _requires_mxfp8()
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")
    _run_distributed(_worker_mxfp8_op_fuser, 4)
