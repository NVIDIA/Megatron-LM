# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``attn``-scope CUDA graphs for gated-residual hybrid GDN layers.

Covers the per-layer capture eligibility that layers declare for themselves (an Engram host
or an uncapturable attention variant stays eager under every scope) and checks that a GDN
layer captured with static THD inputs replays bit-identically to eager execution on packings
that differ from the one it was captured with.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
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
# Static THD capacity of the captured graph.
TOKENS = 512
MAX_SEQS = 4


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
        thd_max_packed_sequences=MAX_SEQS,
        max_seqlen_per_dp_cp_rank=TOKENS,
        pad_packed_seq_alignment="max",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _gdn_wrapper(config):
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
    return wrapper.cuda()


@pytest.fixture
def fla_tensor_cache_disabled(monkeypatch):
    """Capture with THD inputs requires fla's tensor cache off (GDN refuses otherwise)."""
    import fla.utils

    import megatron.core.ssm.gated_delta_net.common as common

    monkeypatch.setattr(fla.utils, "FLA_DISABLE_TENSOR_CACHE", True, raising=False)
    monkeypatch.setattr(common, "FLA_DISABLE_TENSOR_CACHE", True)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("fla_tensor_cache_disabled")
class TestPerLayerCaptureEligibility:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # _layer_is_graphable reads only cuda_graph_modules from the config, so scope variants
    # use a stand-in rather than a TransformerConfig: the dense test layer would make a real
    # config reject the MoE scopes, and the gated-residual guard rejects whole-layer capture.
    @pytest.mark.parametrize(
        "modules",
        [
            [CudaGraphModule.attn],
            [CudaGraphModule.attn, CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
        ],
    )
    def test_gdn_layer_is_graphable_under_attn(self, modules):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        wrapper = _gdn_wrapper(_config())
        assert wrapper.is_cuda_graph_capturable()
        assert _layer_is_graphable(wrapper, SimpleNamespace(cuda_graph_modules=modules))
        # The MoE-only scopes leave an attention-type layer eager, as before.
        moe_only = SimpleNamespace(cuda_graph_modules=[CudaGraphModule.moe_router])
        assert not _layer_is_graphable(wrapper, moe_only)

    @pytest.mark.parametrize("modules", [[CudaGraphModule.attn], []])
    def test_engram_host_stays_eager_under_every_scope(self, modules):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        wrapper = _gdn_wrapper(_config())
        # Stand-in for the Engram module: only its presence matters to the predicate.
        wrapper.inner_layer.engram = torch.nn.Identity()
        assert not wrapper.inner_layer.is_cuda_graph_capturable()
        assert not wrapper.is_cuda_graph_capturable()
        assert not _layer_is_graphable(wrapper, SimpleNamespace(cuda_graph_modules=modules))

    def test_attention_variant_declaration_is_consulted(self, monkeypatch):
        from megatron.core.transformer.cuda_graphs import _layer_is_graphable

        config = _config()
        wrapper = _gdn_wrapper(config)
        attention = wrapper.inner_layer.self_attention
        monkeypatch.setattr(attention, "is_cuda_graph_capturable", lambda: False, raising=False)
        assert not _layer_is_graphable(wrapper, config)

    def test_qsa_is_capturable_with_and_without_context_parallelism(self):
        from types import SimpleNamespace

        from megatron.core.transformer.experimental_attention_variant.qsa import QSASelfAttention

        def capturable(cp_size):
            fake = SimpleNamespace(
                pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: cp_size))
            )
            return QSASelfAttention.is_cuda_graph_capturable(fake)

        assert capturable(1) is True
        # The allgather-CP selection is host-sync free (test_qsa_select_cp.py replays a CP2
        # capture bitwise against eager).
        assert capturable(2) is True

    def test_gdn_is_not_capturable_under_chunkwise_context_parallelism(self):
        from types import SimpleNamespace

        from megatron.core.ssm.gated_delta_net.common import _GDNBase

        def capturable(cp_size, mode):
            # Mirrors _GDNBase.__init__: chunkwise CP is active only with more than one CP rank.
            fake = SimpleNamespace(chunkwise_context_parallel=mode == "chunkwise" and cp_size > 1)
            return _GDNBase.is_cuda_graph_capturable(fake)

        assert capturable(1, "chunkwise") is True
        assert capturable(2, "headwise") is True
        # The chunkwise layout swap builds its all-to-all route on the host.
        assert capturable(2, "chunkwise") is False

    def test_cudnn_gdr_capture_requires_eager_warmup_iterations(self):
        with pytest.raises(ValueError, match="cuda_graph_warmup_steps > 0"):
            _config(gdn_kernel_backend="cudnn", cuda_graph_warmup_steps=0)
        # fla keeps no process-global per-shape state, and eager warmup is the default.
        _config(gdn_kernel_backend="fla", cuda_graph_warmup_steps=0)
        _config(gdn_kernel_backend="cudnn")

    def test_packed_replay_into_sbhd_captured_attention_is_refused(self):
        """Attention captured with SBHD inputs must not silently replay packed data."""
        config = _config(thd_max_packed_sequences=None)
        wrapper = _gdn_wrapper(config)
        assert not wrapper._is_thd_cuda_graph()
        cu = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
        params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        with pytest.raises(RuntimeError, match="thd-max-packed-sequences"):
            wrapper._decompose_packed_seq_params_to_kwargs({"packed_seq_params": params})

    def test_missing_padded_boundaries_fall_back_to_unpadded(self):
        config = _config()
        wrapper = _gdn_wrapper(config)
        assert wrapper._is_thd_cuda_graph()
        cu = torch.tensor([0, 100, TOKENS, TOKENS, TOKENS], dtype=torch.int32, device="cuda")
        kwargs = {
            "packed_seq_params": PackedSeqParams(
                qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu
            )
        }
        wrapper._decompose_packed_seq_params_to_kwargs(kwargs)
        assert kwargs["cu_seqlens_q_padded"] is cu and kwargs["cu_seqlens_kv_padded"] is cu


def _cu_seqlens(lengths):
    assert sum(lengths) == TOKENS and len(lengths) <= MAX_SEQS
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    cu += [TOKENS] * (MAX_SEQS + 1 - len(cu))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


# Every replay packing differs from the capture-time seed [0, T, T, ...]; several end
# mid-chunk (fla's chunk is 64 tokens) so the device-built chunk_indices table changes.
_REPLAY_PACKINGS = [[TOKENS], [256, 256], [100, 156, 256], [37, 200, 75, 200], [TOKENS - 1, 1]]


# Parameters whose gradients the streamed pre-GDR fusion accumulates with atomics.
_FUSED_ATOMIC_GRADS = ("conv1d.weight", "conv1d.bias", "A_log", "dt_bias")


class _PackedForward(torch.nn.Module):
    """Graph callable: the wrapper forward on a THD pack with a static boundary buffer."""

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, hidden_states, cu_seqlens):
        # Mirrors the wrapper's capture-path reconstruction: static max_seqlen and
        # pad_between_seqs, no host reads of the boundaries.
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=TOKENS,
            max_seqlen_kv=TOKENS,
            pad_between_seqs=True,
        )
        out, _ = self.wrapper(hidden_states, packed_seq_params=params)
        return out


def _run(fn, wrapper, x, cu, grad_out):
    wrapper.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    out = fn(x, cu)
    out.backward(grad_out)
    grads = {name: p.grad.detach().clone() for name, p in wrapper.named_parameters()}
    return out.detach().clone(), x.grad.detach().clone(), grads


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("fla_tensor_cache_disabled")
class TestGDNCaptureMatchesEager:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Capture needs a graph-safe RNG tracker (production requires --te-rng-tracker with
        # CUDA graphs); the default tracker's states are plain tensors TE cannot save.
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    # cuDNN GDR is not exercised here. Its capture itself replays bit-identically (checked
    # standalone, one capture per process), but cudnn-frontend keeps process-global per-shape
    # GDR state that does not reliably survive a released CUDA graph: once one captured graph
    # has been freed, the next GDR call with the same shapes in the same process can fault
    # with an illegal memory access. Training keeps its graphs for the whole run; a test
    # module that captures, frees and recaptures cannot.
    @pytest.mark.parametrize("backend", ["fla"])
    @pytest.mark.parametrize("fusion", [False, True])
    def test_replay_on_new_packings_is_bit_identical(self, backend, fusion):
        from transformer_engine.pytorch import make_graphed_callables

        if fusion:
            try:
                import megatron.core.fusions.fused_pre_gated_delta_rule  # noqa: F401
            except ImportError:
                pytest.skip("streamed pre-GDR fusion dependencies are missing")

        config = _config(gdn_kernel_backend=backend, gdn_pre_gated_delta_rule_fusion=fusion)
        wrapper = _gdn_wrapper(config)
        fn = _PackedForward(wrapper)

        torch.manual_seed(0)
        shape = (TOKENS, 1, STREAMS * HIDDEN)
        xs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in _REPLAY_PACKINGS]
        grad_outs = [torch.randn_like(x) for x in xs]

        # Captured with the static-input seed one full-length sequence + empty slots, i.e.
        # what TransformerLayer.get_layer_static_inputs builds. Capture comes before the eager
        # references: eager backward passes on the default stream would leave the parameters'
        # AccumulateGrad nodes bound to it, which capture on TE's side stream rejects.
        seed_cu = _cu_seqlens([TOKENS])
        # Training captures only after eager iterations (cuda_graph_warmup_steps > 0), which
        # build every per-shape kernel/plan cache first. Mirror them with a throwaway layer:
        # its fresh parameters keep any AccumulateGrad stream state away from the captured one.
        warm = _gdn_wrapper(config)
        warm_x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        _run(_PackedForward(warm), warm, warm_x, seed_cu, torch.randn_like(warm_x))
        del warm
        sample_x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        graphed = make_graphed_callables(fn, (sample_x, seed_cu), num_warmup_iters=2)

        eager = [
            _run(fn, wrapper, x, _cu_seqlens(lengths), g)
            for x, g, lengths in zip(xs, grad_outs, _REPLAY_PACKINGS)
        ]
        # Eager is deterministic run to run; the comparison below is exact.
        repeat = _run(fn, wrapper, xs[2], _cu_seqlens(_REPLAY_PACKINGS[2]), grad_outs[2])
        assert torch.equal(repeat[0], eager[2][0]) and torch.equal(repeat[1], eager[2][1])

        for (x, g, lengths), (out_ref, dx_ref, grads_ref) in zip(
            zip(xs, grad_outs, _REPLAY_PACKINGS), eager
        ):
            out, dx, grads = _run(graphed, wrapper, x, _cu_seqlens(lengths), g)
            assert torch.equal(out, out_ref), f"forward differs for packing {lengths}"
            assert torch.equal(dx, dx_ref), f"input grad differs for packing {lengths}"
            for name, ref in grads_ref.items():
                if fusion and name.endswith(_FUSED_ATOMIC_GRADS):
                    # The streamed pre-GDR fusion accumulates these weight gradients with
                    # tl.atomic_add, so they already vary run to run in eager mode.
                    torch.testing.assert_close(grads[name], ref, msg=f"{name} for {lengths}")
                else:
                    assert torch.equal(grads[name], ref), f"{name} grad differs for {lengths}"
