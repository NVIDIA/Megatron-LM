# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``moe_router`` CUDA-graph scope for megakernel (MOK) MoE layers.

MOK used to reject every MoE scope. The megakernel itself (dispatch, routed and shared experts,
combine) is still never captured, but the router and the static-shape route compaction in front
of it are: the graph ends with ``[hidden_states, router_weights, top_experts]`` and the
megakernel runs eagerly on those outputs. These tests cover

* the relaxed configuration check (what is accepted, what stays rejected, and why);
* the backend contract: the compact-route entry point hands the megakernel exactly what the
  dense path does;
* the layer plumbing end to end: a hybrid MoE layer whose router prefix is
  captured with Transformer Engine and replayed matches eager execution bit for bit
  (forward output, input gradient, and every parameter gradient). The megakernel is replaced
  by a deterministic toy behind the real backend ``forward`` so the test does not need the
  MOK extension or expert parallelism.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel.mok import backend as mok_backend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 256
EXPERTS = 8
TOPK = 2
SEQ = 128


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_moe_experts=EXPERTS,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_router_topk=TOPK,
        moe_router_score_function="softmax",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        gated_linear_unit=True,
        activation_func=F.silu,
        gradient_accumulation_fusion=True,
        moe_megakernel_backend="mok",
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        transformer_impl="transformer_engine",
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestMokCudaGraphScopeValidation:
    """The relaxed check: only ``moe_router``, only with the Transformer Engine implementation."""

    @pytest.mark.parametrize(
        "modules",
        [
            [CudaGraphModule.attn],
            [CudaGraphModule.attn, CudaGraphModule.moe_router],
            [CudaGraphModule.moe_router],
        ],
    )
    def test_accepted_scopes(self, modules):
        config = _config(cuda_graph_modules=modules)
        assert config.cuda_graph_modules == modules

    @pytest.mark.parametrize(
        "modules",
        [
            [CudaGraphModule.attn, CudaGraphModule.moe],
            [CudaGraphModule.attn, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        ],
    )
    def test_megakernel_scopes_stay_rejected(self, modules):
        # 'moe' would capture the megakernel; 'moe_preprocess' is the native dispatcher's.
        with pytest.raises(ValueError, match="MOK supports only the moe_router MoE scope"):
            _config(cuda_graph_modules=modules)

    def test_whole_layer_capture_stays_rejected(self):
        with pytest.raises(ValueError, match="whole-layer"):
            _config(cuda_graph_modules=[])

    @pytest.mark.parametrize("recompute", [["moe"], ["layernorm"]])
    def test_router_scope_rejects_unvalidated_recompute(self, recompute):
        with pytest.raises(ValueError, match="recompute_modules 'moe' or 'layernorm'"):
            _config(recompute_granularity="selective", recompute_modules=recompute)
        # Without the router scope the eager megakernel path keeps supporting MoE recompute.
        _config(
            cuda_graph_modules=[CudaGraphModule.attn],
            recompute_granularity="selective",
            recompute_modules=recompute,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestCompactRoutes:
    """``forward(..., routes_are_compact=True)`` is the dense path minus the compaction."""

    def test_compact_entry_matches_dense_entry(self, monkeypatch):
        module = _ToyMegakernel()
        captured = []

        def apply(*arguments):
            captured.append(arguments)
            return arguments[1]

        monkeypatch.setattr(mok_backend._MoKAutograd, "apply", apply)
        torch.manual_seed(0)
        tokens, experts = 64, 2
        logits = torch.randn(tokens, experts, device="cuda")
        probs = torch.softmax(logits, dim=-1)
        routing_map = torch.ones(tokens, experts, dtype=torch.bool, device="cuda")
        routing_map[::3, 1] = False  # some rows carry fewer than top-k routes
        x = torch.randn(tokens, 1, 8, device="cuda", dtype=torch.bfloat16)

        module(x, probs, routing_map)
        router_weights, top_experts = module.compact_routes(tokens, probs, routing_map)
        module(x, router_weights, top_experts, routes_are_compact=True)

        dense, compact = captured
        assert torch.equal(dense[2], compact[2]) and dense[2].dtype == torch.float32
        assert torch.equal(dense[3], compact[3]) and dense[3].dtype == torch.int32
        assert (compact[3][::3, 1] == -1).all()
        for a, b in zip(dense[4:], compact[4:]):
            assert a is b

    def test_compact_entry_validates_shapes(self, monkeypatch):
        module = _ToyMegakernel()
        x = torch.zeros(4, 1, 8, device="cuda", dtype=torch.bfloat16)
        weights = torch.zeros(4, 2, device="cuda", dtype=torch.float32)
        experts = torch.zeros(4, 2, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="compact MOK routes"):
            module(x, weights[:, :1], experts, routes_are_compact=True)
        with pytest.raises(ValueError, match="compact MOK routes"):
            module(x, weights, experts.long(), routes_are_compact=True)


class _ToyMegakernel(mok_backend.MoKMegakernel):
    """The real MOK backend ``forward``/``compact_routes`` around a deterministic toy kernel.

    ``_MoKAutograd.apply`` is replaced by ``_toy_apply`` (see the test): it mixes per-expert
    scales with the compact router weights through a one-hot matmul, so the router gradient and
    the expert-parameter gradient are both exercised and deterministic (no atomics).
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.topk = TOPK
        self.expert_scale = torch.nn.Parameter(
            1.0 + 0.1 * torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16)
        )
        self.shared_output_gate_weight = None
        self.shared_fc1_weight = None
        self.shared_fc2_weight = None

    @property
    def autograd_routed_parameters(self):
        return (self.expert_scale,)


def _toy_apply(module, x, router_weights, top_experts, gate, scale, *unused):
    del module, gate, unused
    valid = (top_experts >= 0).to(torch.float32)
    one_hot = F.one_hot(top_experts.clamp(min=0).long(), EXPERTS).to(torch.float32)
    mix = (one_hot * (router_weights * valid).unsqueeze(-1)).sum(dim=1)  # [tokens, experts]
    return (x.float() * (mix @ scale.float())).to(x.dtype)


def _moe_layer(config, monkeypatch):
    import megatron.core.transformer.moe.moe_layer as moe_layer_module
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
    from megatron.core.transformer.spec_utils import build_module

    toy = _ToyMegakernel()
    monkeypatch.setattr(moe_layer_module, "build_megakernel_backend", lambda **kwargs: toy)
    monkeypatch.setattr(mok_backend, "_MoKAutograd", SimpleNamespace(apply=_toy_apply))
    spec = hybrid_stack_spec.submodules.moe_layer
    layer = build_module(spec, config=config, layer_number=1)
    assert layer.mlp.megakernel_experts is toy
    convert_module_to_dtype_except_fp32_marked(layer, torch.bfloat16)
    return layer.cuda().train()


def _run(wrapper, x, grad_out):
    wrapper.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    out, _ = wrapper(hidden_states=x)
    out.backward(grad_out)
    grads = {
        name: p.grad.detach().clone()
        for name, p in wrapper.named_parameters()
        if p.grad is not None
    }
    return out.detach().clone(), x.grad.detach().clone(), grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestMokRouterCaptureMatchesEager:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def test_layer_is_graphable_with_router_prefix_only(self, monkeypatch):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        config = _config()
        layer = _moe_layer(config, monkeypatch)
        assert _layer_is_graphable(layer, config)
        covered = layer._get_submodules_under_cudagraphs()
        assert layer.mlp.router in covered
        # The shared expert is fused into the eager megakernel: not graph-owned.
        assert layer.mlp.shared_experts not in covered
        assert layer.mlp.megakernel_experts not in covered
        assert not layer._moe_cuda_graph_outputs_shared_expert()
        # attn-only scope: the MoE layer stays eager, as before.
        attn_only = _config(cuda_graph_modules=[CudaGraphModule.attn])
        assert not _layer_is_graphable(layer, attn_only)

    @pytest.mark.parametrize("load_balancing", ["aux_loss", "none"])
    def test_replay_matches_eager_bitwise(self, monkeypatch, load_balancing):
        from transformer_engine.pytorch import graph as te_graph

        from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start

        config = _config(moe_router_load_balancing_type=load_balancing)
        wrapper = _moe_layer(config, monkeypatch)
        calls = {"compact": 0}
        compact = wrapper.mlp.megakernel_experts.compact_routes

        def counting_compact(*args, **kwargs):
            calls["compact"] += 1
            return compact(*args, **kwargs)

        monkeypatch.setattr(wrapper.mlp.megakernel_experts, "compact_routes", counting_compact)

        torch.manual_seed(0)
        shape = (SEQ, 1, HIDDEN)
        xs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        grad_outs = [torch.randn_like(x) for x in xs]

        # Eager warmup (training runs cuda_graph_warmup_steps eager iterations first).
        _run(wrapper, torch.randn(shape, device="cuda", dtype=torch.bfloat16), xs[0].clone())

        # Capture the router prefix the way TECudaGraphHelper does (one chunk, one microbatch).
        sample_x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        _set_capture_start()
        try:
            graphs = te_graph.make_graphed_callables(
                (wrapper,),
                ((sample_x,),),
                sample_kwargs=[{}],
                allow_unused_input=True,
                num_warmup_iters=2,
                _order=[1, -1],
                _num_layers_per_chunk=[1],
                retain_graph_in_backward=False,
                fp8_enabled=False,
            )
        finally:
            _set_capture_end()
        wrapper.cuda_graphs = [graphs[0]]
        wrapper.cuda_graph_manual_hooks = []

        def eager(x, g):
            saved, wrapper.cuda_graphs = wrapper.cuda_graphs, []
            try:
                return _run(wrapper, x, g)
            finally:
                wrapper.cuda_graphs = saved

        refs = [eager(x, g) for x, g in zip(xs, grad_outs)]
        repeat = eager(xs[1], grad_outs[1])
        assert torch.equal(repeat[0], refs[1][0]) and torch.equal(repeat[1], refs[1][1])

        for x, g, (out_ref, dx_ref, grads_ref) in zip(xs, grad_outs, refs):
            before = calls["compact"]
            out, dx, grads = _run(wrapper, x, g)
            # Replay: the compaction ran inside the graph, not eagerly.
            assert calls["compact"] == before
            assert wrapper.mlp.cudagraph_tensor_store.is_empty()
            assert torch.equal(out, out_ref)
            assert torch.equal(dx, dx_ref)
            assert grads.keys() == grads_ref.keys()
            assert "mlp.router.weight" in grads
            assert any(name.endswith("expert_scale") for name in grads)
            for name, ref in grads_ref.items():
                assert torch.equal(grads[name], ref), f"{name} grad differs"
