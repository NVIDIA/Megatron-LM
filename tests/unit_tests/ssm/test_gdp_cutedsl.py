# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the optional CuTeDSL GatedDeltaProduct chunk kernel."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gated_delta_product as gdp_module
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    GatedDeltaProductMixerSubmodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import causal_conv1d  # noqa: F401
    import einops  # noqa: F401
    import fla  # noqa: F401
    import gdp_attn  # noqa: F401
    import mamba_ssm  # noqa: F401

    HAVE_GDP_KERNELS = True
except ImportError:
    HAVE_GDP_KERNELS = False


class _IdentityProjection(torch.nn.Module):
    """Return a fixed projected tensor while preserving the mixer call contract."""

    def __init__(self, projected: torch.Tensor):
        super().__init__()
        self.projected = projected

    def forward(self, _hidden_states):
        return self.projected, None


class _IdentityOutputProjection(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None


class _FakeContextParallel:
    def __init__(self, *, d_inner: int, nheads: int, ngroups: int):
        self.d_inner_local_tpcp = d_inner
        self.nheads_local_tpcp = nheads
        self.ngroups_local_tpcp = ngroups
        self.cp_size = 1

    def pre_conv_ssm(self, tensor, packed_seq_params=None):
        del packed_seq_params
        return tensor

    def post_conv_ssm(self, tensor, packed_seq_params=None):
        del packed_seq_params
        return tensor

    def get_conv1d_weight(self):
        return self.conv_weight

    def get_conv1d_bias(self):
        return None

    def get_A_log(self):
        return self.a_log

    def get_dt_bias(self):
        return self.dt_bias


def _make_dispatch_mixer(*, cutedsl: bool, projected: torch.Tensor):
    """Build only the mixer state needed to exercise the chunk-kernel dispatch."""
    nheads, ngroups, d_state, headdim, num_householder = 4, 2, 3, 2, 3
    d_inner = nheads * headdim
    conv_dim = d_inner * num_householder + ngroups * d_state * (num_householder + 1)

    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    torch.nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(gdp_cutedsl_kernel=cutedsl, sequence_parallel=False)
    mixer.gdp_kernel_extra_kwargs = (
        {"num_chunk_states_to_recompute": 2} if cutedsl else {}
    )
    mixer.num_householder = num_householder
    mixer.d_state = d_state
    mixer.headdim = headdim
    mixer.rmsnorm = False
    mixer.activation = "silu"
    mixer.recompute_in_proj = False
    mixer.recompute_qkv = False
    mixer.in_proj = _IdentityProjection(projected.transpose(0, 1))
    mixer.out_proj = _IdentityOutputProjection()
    mixer.conv1d = SimpleNamespace(weight=torch.empty(conv_dim, 1, 1), bias=None)

    mixer.cp = _FakeContextParallel(d_inner=d_inner, nheads=nheads, ngroups=ngroups)
    mixer.cp.conv_weight = mixer.conv1d.weight
    mixer.cp.a_log = torch.zeros(nheads)
    mixer.cp.dt_bias = torch.zeros(nheads)
    return mixer


@pytest.mark.parametrize("cutedsl", [False, True])
def test_chunk_kernel_dispatch_layouts(monkeypatch, cutedsl):
    """The CuTeDSL path uses flat token rows while FLA retains its batched layout."""
    batch_size, seq_len = 2, 5
    nheads, ngroups, d_state, headdim, num_householder = 4, 2, 3, 2, 3
    d_inner = nheads * headdim
    projected_dim = (
        d_inner * (1 + num_householder)
        + ngroups * d_state * (num_householder + 1)
        + nheads * (num_householder + 1)
    )
    projected = torch.randn(batch_size, seq_len, projected_dim)
    mixer = _make_dispatch_mixer(cutedsl=cutedsl, projected=projected)

    # Isolate GDP layout/dispatch from causal-convolution numerics while retaining
    # the input strides so the CuTeDSL channel-last contract is covered.
    captured_conv = {}

    def fake_causal_conv1d(**kwargs):
        captured_conv.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(gdp_module, "causal_conv1d_fn", fake_causal_conv1d)
    monkeypatch.setattr(gdp_module, "l2_norm", lambda tensor: tensor)

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        if cutedsl:
            output = torch.arange(batch_size * seq_len * d_inner, dtype=torch.float32).reshape(
                batch_size * seq_len, d_inner
            )
        else:
            output = torch.arange(batch_size * seq_len * d_inner, dtype=torch.float32).reshape(
                batch_size, seq_len, nheads, headdim
            )
        return output, None

    mixer.gdp_kernel = fake_kernel
    hidden_states = torch.empty(seq_len, batch_size, 1)
    output, output_bias = mixer(hidden_states)

    assert output.shape == (seq_len, batch_size, d_inner)
    assert output_bias is None
    assert captured["use_qk_l2norm_in_kernel"] is cutedsl
    assert captured["num_householder"] == num_householder

    if cutedsl:
        assert captured["num_chunk_states_to_recompute"] == 2
        assert captured_conv["x"].stride(1) == 1
        assert not captured_conv["x"].is_contiguous()
        assert (
            captured_conv["x"].untyped_storage().data_ptr()
            == projected.untyped_storage().data_ptr()
        )
        assert captured["q"].shape == (batch_size * seq_len, nheads * d_state)
        assert captured["k"].shape == (
            batch_size * seq_len,
            num_householder * nheads * d_state,
        )
        assert captured["v"].shape == (
            batch_size * seq_len,
            num_householder * nheads * headdim,
        )
        assert captured["beta"].shape == (
            batch_size * seq_len,
            num_householder * nheads,
        )
        assert captured["g"].shape == (batch_size * seq_len, nheads)
        torch.testing.assert_close(
            captured["cu_seqlens"], torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
        )
    else:
        assert "num_chunk_states_to_recompute" not in captured
        assert captured_conv["x"].stride(2) == 1
        assert captured_conv["x"].is_contiguous()
        assert (
            captured_conv["x"].untyped_storage().data_ptr()
            != projected.untyped_storage().data_ptr()
        )
        assert captured["q"].shape == (batch_size, seq_len, nheads, d_state)
        assert captured["k"].shape == (
            batch_size,
            seq_len * num_householder,
            nheads,
            d_state,
        )
        assert captured["v"].shape == (
            batch_size,
            seq_len * num_householder,
            nheads,
            headdim,
        )
        assert captured["beta"].shape == (
            batch_size,
            seq_len * num_householder,
            nheads,
        )
        assert captured["g"].shape == (batch_size, seq_len, nheads)
        assert captured["cu_seqlens"] is None


def _make_guard_mixer():
    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    torch.nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(gdp_cutedsl_kernel=True, sequence_parallel=False)
    return mixer


def test_cutedsl_rejects_packed_sequences():
    mixer = _make_guard_mixer()
    with pytest.raises(NotImplementedError, match="unpacked sequences only"):
        mixer(torch.empty(1, 1, 1), packed_seq_params=object())


def test_cutedsl_rejects_dynamic_inference():
    mixer = _make_guard_mixer()
    context = SimpleNamespace(is_dynamic_batching=lambda: True)
    with pytest.raises(NotImplementedError, match="dynamic inference"):
        mixer(torch.empty(1, 1, 1), inference_context=context)


def test_cutedsl_rejects_static_decode():
    mixer = _make_guard_mixer()
    context = SimpleNamespace(
        is_dynamic_batching=lambda: False,
        is_static_batching=lambda: True,
        seqlen_offset=1,
    )
    with pytest.raises(NotImplementedError, match="static prefill but not decode"):
        mixer(torch.empty(1, 1, 1), inference_context=context)


def _make_real_config(
    *,
    cutedsl: bool,
    chunk_states_to_recompute: int = 2,
    recompute_modules: list[str] | None = None,
) -> TransformerConfig:
    # Match the production GDP head/state specialization while keeping the unit-test model small.
    return TransformerConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=1024,
        normalization="RMSNorm",
        bf16=True,
        mamba_num_heads=8,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_state_dim=128,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        gdp_cutedsl_kernel=cutedsl,
        gdp_num_chunk_states_to_recompute=chunk_states_to_recompute,
        recompute_granularity="selective" if recompute_modules else None,
        recompute_modules=recompute_modules,
    )


def _build_real_mixer(
    *,
    cutedsl: bool,
    chunk_states_to_recompute: int = 2,
    recompute_modules: list[str] | None = None,
):
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    config = _make_real_config(
        cutedsl=cutedsl,
        chunk_states_to_recompute=chunk_states_to_recompute,
        recompute_modules=recompute_modules,
    )
    submodules = GatedDeltaProductMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    )
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    mixer = GatedDeltaProductMixer(
        config=config,
        submodules=submodules,
        d_model=config.hidden_size,
        layer_number=1,
        pg_collection=pg_collection,
    )
    return mixer.cuda().bfloat16()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not HAVE_GDP_KERNELS, reason="requires FLA, gdp_attn, and Mamba dependencies")
class TestGDPCuTeDSLParity:
    @pytest.fixture(autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _build_pair(self):
        fla_mixer = _build_real_mixer(cutedsl=False)
        cutedsl_mixer = _build_real_mixer(cutedsl=True)
        cutedsl_mixer.load_state_dict(fla_mixer.state_dict())
        return fla_mixer, cutedsl_mixer

    def _build_recompute_pair(self):
        no_recompute = _build_real_mixer(cutedsl=True, chunk_states_to_recompute=0)
        recompute_two = _build_real_mixer(cutedsl=True, chunk_states_to_recompute=2)
        recompute_two.load_state_dict(no_recompute.state_dict())
        return no_recompute, recompute_two

    def _build_selective_recompute_pair(self, module: str):
        no_recompute = _build_real_mixer(cutedsl=False)
        recompute = _build_real_mixer(cutedsl=False, recompute_modules=[module])
        recompute.load_state_dict(no_recompute.state_dict())
        return no_recompute, recompute

    def test_unpacked_training_forward_backward_parity(self):
        fla_mixer, cutedsl_mixer = self._build_pair()
        fla_mixer.train()
        cutedsl_mixer.train()

        hidden = torch.randn(32, 2, 512, device="cuda", dtype=torch.bfloat16)
        hidden_fla = hidden.detach().clone().requires_grad_()
        hidden_cutedsl = hidden.detach().clone().requires_grad_()

        output_fla, _ = fla_mixer(hidden_fla)
        output_cutedsl, _ = cutedsl_mixer(hidden_cutedsl)
        torch.testing.assert_close(output_cutedsl, output_fla, rtol=5e-2, atol=5e-2)

        output_grad = torch.randn_like(output_fla)
        output_fla.backward(output_grad)
        output_cutedsl.backward(output_grad)
        torch.testing.assert_close(hidden_cutedsl.grad, hidden_fla.grad, rtol=5e-2, atol=5e-2)

        fla_parameters = dict(fla_mixer.named_parameters())
        cutedsl_parameters = dict(cutedsl_mixer.named_parameters())
        for name in ("in_proj.weight", "conv1d.weight", "dt_bias", "A_log", "out_proj.weight"):
            torch.testing.assert_close(
                cutedsl_parameters[name].grad,
                fla_parameters[name].grad,
                rtol=5e-2,
                atol=7e-2,
                msg=name,
            )

    def test_chunk_state_recompute_forward_backward_parity(self):
        no_recompute, recompute_two = self._build_recompute_pair()
        no_recompute.train()
        recompute_two.train()

        # Exercise multiple GDP chunks so the recomputed boundary states affect backward.
        hidden = torch.randn(256, 2, 512, device="cuda", dtype=torch.bfloat16)
        hidden_no_recompute = hidden.detach().clone().requires_grad_()
        hidden_recompute_two = hidden.detach().clone().requires_grad_()

        output_no_recompute, _ = no_recompute(hidden_no_recompute)
        output_recompute_two, _ = recompute_two(hidden_recompute_two)
        torch.testing.assert_close(
            output_recompute_two, output_no_recompute, rtol=1e-2, atol=1e-2
        )

        output_grad = torch.randn_like(output_no_recompute)
        output_no_recompute.backward(output_grad)
        output_recompute_two.backward(output_grad)
        torch.testing.assert_close(
            hidden_recompute_two.grad, hidden_no_recompute.grad, rtol=1e-2, atol=1e-2
        )

        no_recompute_parameters = dict(no_recompute.named_parameters())
        recompute_two_parameters = dict(recompute_two.named_parameters())
        for name in ("in_proj.weight", "conv1d.weight", "dt_bias", "A_log", "out_proj.weight"):
            torch.testing.assert_close(
                recompute_two_parameters[name].grad,
                no_recompute_parameters[name].grad,
                rtol=1e-2,
                atol=1e-2,
            )

    @pytest.mark.parametrize("module", ["gdp_in_proj", "gdp_qkv"])
    def test_selective_recompute_forward_backward_parity(self, module):
        no_recompute, recompute = self._build_selective_recompute_pair(module)
        no_recompute.train()
        recompute.train()

        hidden = torch.randn(32, 2, 512, device="cuda", dtype=torch.bfloat16)
        hidden_no_recompute = hidden.detach().clone().requires_grad_()
        hidden_recompute = hidden.detach().clone().requires_grad_()

        output_no_recompute, _ = no_recompute(hidden_no_recompute)
        output_recompute, _ = recompute(hidden_recompute)
        torch.testing.assert_close(output_recompute, output_no_recompute, rtol=1e-2, atol=1e-2)

        output_grad = torch.randn_like(output_no_recompute)
        output_no_recompute.backward(output_grad)
        output_recompute.backward(output_grad)
        torch.testing.assert_close(
            hidden_recompute.grad, hidden_no_recompute.grad, rtol=1e-2, atol=1e-2
        )

        no_recompute_parameters = dict(no_recompute.named_parameters())
        recompute_parameters = dict(recompute.named_parameters())
        for name in ("in_proj.weight", "conv1d.weight", "dt_bias", "A_log", "out_proj.weight"):
            torch.testing.assert_close(
                recompute_parameters[name].grad,
                no_recompute_parameters[name].grad,
                rtol=1e-2,
                atol=1e-2,
            )

    def test_static_prefill_output_and_state_parity(self):
        fla_mixer, cutedsl_mixer = self._build_pair()
        fla_mixer.eval()
        cutedsl_mixer.eval()
        hidden = torch.randn(32, 2, 512, device="cuda", dtype=torch.bfloat16)

        fla_context = StaticInferenceContext(max_batch_size=2, max_sequence_length=33)
        cutedsl_context = StaticInferenceContext(max_batch_size=2, max_sequence_length=33)
        # HybridBlock normally maintains this compatibility alias.
        fla_context.seqlen_offset = 0
        cutedsl_context.seqlen_offset = 0

        with torch.no_grad(), InferenceMode.active():
            output_fla, _ = fla_mixer(hidden, inference_context=fla_context)
            output_cutedsl, _ = cutedsl_mixer(hidden, inference_context=cutedsl_context)

        torch.testing.assert_close(output_cutedsl, output_fla, rtol=5e-2, atol=5e-2)
        fla_conv_state, fla_ssm_state = fla_context.key_value_memory_dict[1]
        cutedsl_conv_state, cutedsl_ssm_state = cutedsl_context.key_value_memory_dict[1]
        torch.testing.assert_close(cutedsl_conv_state, fla_conv_state)
        torch.testing.assert_close(cutedsl_ssm_state, fla_ssm_state, rtol=5e-2, atol=5e-2)
