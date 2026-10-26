# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``cuda_graph_capture_engram_host``: capture an Engram-hosting hybrid layer minus its lookup.

By default a layer that hosts Engram stays eager under every per-layer CUDA graph scope, because
the hashed lookup and its all-to-all have data-dependent shapes. With the opt-in option the
hybrid hyper-connection wrapper runs the lookup eagerly in front of the layer's graph and
captures the rest. These tests check the configuration contract, the capture eligibility and
grouping rules, and that the captured layer replays bit-identically to the default eager layer
(forward output, input gradient, and every parameter gradient, including the Engram module's),
with the Engram module called once per step and never during capture.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

STREAMS = 4
HIDDEN = 512
SEQ = 256


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        num_query_groups=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_attention_freq=[1],
        experimental_attention_variant="gdn",
        gdn_kernel_backend="fla",
        activation_func=F.silu,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        is_hybrid_model=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_connection_variant="gated_residual",
        hc_lowrank=64,
        transformer_impl="transformer_engine",
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn],
        engram_enabled=True,
        cuda_graph_capture_engram_host=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _ToyEngram(torch.nn.Module):
    """Stand-in for the Engram module: a parameterized n-stream residual and a call counter."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            0.1 * torch.randn(STREAMS * HIDDEN, device="cuda", dtype=torch.bfloat16)
        )
        self.calls = 0

    def forward(self, hidden_states, input_ids, packed_seq_params):
        del packed_seq_params
        assert input_ids is not None, "the wrapper must forward input_ids to the lookup"
        self.calls += 1
        return hidden_states * self.scale


def _gdn_wrapper(config, with_engram=True):
    from megatron.core.models.hybrid.hybrid_layer_specs import gated_residual_hybrid_stack_spec
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
        HyperConnectionHybridLayer,
    )
    from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
    from megatron.core.transformer.spec_utils import build_module

    spec = gated_residual_hybrid_stack_spec.submodules.gdn_layer
    layer = build_module(spec, config=config, layer_number=1)
    wrapper = HyperConnectionHybridLayer(config, layer)
    convert_module_to_dtype_except_fp32_marked(wrapper, torch.bfloat16)
    wrapper = wrapper.cuda().train()
    if with_engram:
        wrapper.inner_layer.engram = _ToyEngram()
    return wrapper


class TestEngramHostCaptureConfig:
    def test_option_is_off_by_default(self):
        assert TransformerConfig.__dataclass_fields__["cuda_graph_capture_engram_host"].default is (
            False
        )

    def test_requires_engram(self):
        with pytest.raises(ValueError, match="requires Engram to be enabled"):
            _config(engram_enabled=False)

    @pytest.mark.parametrize(
        "overrides",
        [dict(cuda_graph_impl="none"), dict(cuda_graph_modules=[CudaGraphModule.moe_router])],
    )
    def test_requires_te_attn_scope(self, overrides):
        with pytest.raises(ValueError, match="'attn' scope"):
            _config(**overrides)

    def test_rejects_unvalidated_recompute(self):
        with pytest.raises(ValueError, match="full recompute or the mhc selective recompute"):
            _config(recompute_granularity="selective", recompute_modules=["mhc"])


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestEngramHostCapture:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def test_eligibility_follows_the_option(self, monkeypatch):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        config = _config()
        wrapper = _gdn_wrapper(config)
        # The inner layer still declares itself uncapturable (GPT path, plain hybrid layers).
        assert not wrapper.inner_layer.is_cuda_graph_capturable()
        assert wrapper.is_cuda_graph_capturable()
        assert _layer_is_graphable(wrapper, config)

        monkeypatch.setattr(config, "cuda_graph_capture_engram_host", False)
        assert not wrapper.is_cuda_graph_capturable()
        assert not _layer_is_graphable(wrapper, config)
        monkeypatch.setattr(config, "cuda_graph_capture_engram_host", True)

        # The other submodules' own declarations are still honoured.
        attention = wrapper.inner_layer.self_attention
        monkeypatch.setattr(attention, "is_cuda_graph_capturable", lambda: False, raising=False)
        assert not wrapper.is_cuda_graph_capturable()

    def test_engram_host_never_becomes_a_group_tail(self, monkeypatch):
        config = _config()
        head = _gdn_wrapper(config, with_engram=False)
        tail = _gdn_wrapper(config, with_engram=False)
        # Pretend the tail is an MoE layer under moe_router partial capture.
        monkeypatch.setattr(tail, "_inner_is_partial_moe_capture", lambda: True)
        assert head._can_group_te_cuda_graph_with(tail)
        tail.inner_layer.engram = _ToyEngram()
        assert not head._can_group_te_cuda_graph_with(tail)
        # An Engram host may still head a group: its lookup runs before the whole graph.
        host = _gdn_wrapper(config)
        tail.inner_layer.engram = None
        assert host._can_group_te_cuda_graph_with(tail)

    def test_replay_matches_default_eager_bitwise(self, monkeypatch):
        from transformer_engine.pytorch import graph as te_graph

        from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start

        config = _config()
        wrapper = _gdn_wrapper(config)
        engram = wrapper.inner_layer.engram

        torch.manual_seed(0)
        shape = (SEQ, 1, STREAMS * HIDDEN)
        input_ids = torch.randint(0, 1000, (1, SEQ), device="cuda")
        xs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        grad_outs = [torch.randn_like(x) for x in xs]

        def run(x, g):
            wrapper.zero_grad(set_to_none=True)
            x = x.detach().clone().requires_grad_(True)
            out, _ = wrapper(hidden_states=x, input_ids=input_ids)
            out.backward(g)
            grads = {
                name: p.grad.detach().clone()
                for name, p in wrapper.named_parameters()
                if p.grad is not None
            }
            return out.detach().clone(), x.grad.detach().clone(), grads

        # Eager warmup iteration with a throwaway layer, as in training (warmup steps first).
        warm = _gdn_wrapper(config)
        warm_x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        warm(hidden_states=warm_x, input_ids=input_ids)[0].sum().backward()
        del warm

        sample_x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        calls_before_capture = engram.calls
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
        # The lookup is never part of the captured region (nor of TE's warmup passes).
        assert engram.calls == calls_before_capture
        wrapper.cuda_graphs = [graphs[0]]
        wrapper.cuda_graph_manual_hooks = []

        def default_eager(x, g):
            # The pre-existing behaviour: option off, layer eager, lookup inside forward.
            saved, wrapper.cuda_graphs = wrapper.cuda_graphs, []
            monkeypatch.setattr(config, "cuda_graph_capture_engram_host", False)
            try:
                return run(x, g)
            finally:
                monkeypatch.setattr(config, "cuda_graph_capture_engram_host", True)
                wrapper.cuda_graphs = saved

        refs = [default_eager(x, g) for x, g in zip(xs, grad_outs)]
        repeat = default_eager(xs[1], grad_outs[1])
        assert torch.equal(repeat[0], refs[1][0]) and torch.equal(repeat[1], refs[1][1])

        for x, g, (out_ref, dx_ref, grads_ref) in zip(xs, grad_outs, refs):
            calls = engram.calls
            out, dx, grads = run(x, g)
            assert engram.calls == calls + 1
            assert torch.equal(out, out_ref)
            assert torch.equal(dx, dx_ref)
            assert grads.keys() == grads_ref.keys()
            assert any(name.endswith("engram.scale") for name in grads)
            for name, ref in grads_ref.items():
                assert torch.equal(grads[name], ref), f"{name} grad differs"
