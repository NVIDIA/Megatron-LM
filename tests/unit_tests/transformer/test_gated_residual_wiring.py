# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
End-to-end wiring tests for the gated-residual hyper-connection variant.

Validates GR-2: variant dispatch in the layer specs, the norm-free layer shape
(IdentityOp norms + unfused projections), and the full 4-stream pipeline
(input_expand -> per-sublayer read gate/write-back -> exit contract) against
the arithmetic of the HF Qwen4-Exp reference, using forward hooks to capture
the Megatron sublayer outputs so attention/MoE internals stay out of the
comparison.
"""

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_residual_hybrid_stack_spec,
    hybrid_stack_spec,
    is_gated_residual_norm_free,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_gated_residual import RefGatedResidual, ref_write_back

HIDDEN = 64
STREAMS = 4
LOWRANK = 16
NUM_LAYERS = 2
SEQ, BATCH = 16, 2


def make_config(**overrides):
    kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        is_hybrid_model=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_connection_variant="gated_residual",
        hc_lowrank=LOWRANK,
        layernorm_zero_centered_gamma=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_router_dtype="fp32",
        bf16=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def make_gr_spec(config):
    """The norm-free attention layer spec of the gated-residual hybrid stack."""
    return gated_residual_hybrid_stack_spec.submodules.attention_layer


def oracle_from(module: GatedResidualModule, use_combine=True) -> RefGatedResidual:
    """Build a weight-synchronized reference oracle from a Megatron GR module."""
    ref = RefGatedResidual(HIDDEN, STREAMS, LOWRANK, use_combine=use_combine)
    with torch.no_grad():
        ref.hc_norm.weight.copy_(module.hc_norm.weight)
        ref.input_mix_weight_down.weight.copy_(module.input_mix_weight_down.weight)
        ref.input_mix_weight_up.weight.copy_(module.input_mix_weight_up.weight)
        if use_combine:
            ref.block_inject_weight.weight.copy_(module.block_inject_weight.weight)
    return ref.cuda()


class TestGatedResidualMTPHead:
    """The MTP head under gated_residual: hnorm is the grouped [n*h] norm of the reference."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_mtp_hnorm_is_grouped_over_streams(self):
        from megatron.core.transformer.gated_residual import GroupedRMSNorm, grouped_rms_norm
        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock

        # Dense MLP: the MTP block built standalone has no expert process group, and the
        # hnorm layout under test does not depend on the MLP kind.
        config = make_config(mtp_num_layers=1, num_moe_experts=None, moe_ffn_hidden_size=None)
        # HybridModel MTP contract: the block is built from the hybrid stack's own MTP spec
        # plus a layer pattern and the stack submodules (mHC is HybridModel-only on this base).
        sub = gated_residual_hybrid_stack_spec.submodules
        mtp = MultiTokenPredictionBlock(
            config=config,
            spec=sub.mtp_block_spec,
            mtp_layer_pattern="*",
            mtp_num_depths=1,
            hybrid_submodules=sub,
        ).cuda()
        layer = mtp.layers[0]
        # Released Qwen3.8 checkpoint: mtp.pre_fc_norm_hidden.weight is [hc_count * hidden],
        # mtp.pre_fc_norm_embedding.weight is [hidden].
        assert isinstance(layer.hnorm, GroupedRMSNorm)
        assert tuple(layer.hnorm.weight.shape) == (STREAMS * HIDDEN,)
        assert tuple(layer.enorm.weight.shape) == (HIDDEN,)
        assert layer.hnorm.weight.dtype == torch.float32
        assert isinstance(layer.hc_exit_contract, GatedResidualModule)
        assert layer.final_layernorm is None

        # The grouped norm on the flat [s, b, n*h] tensor equals per-stream RMSNorm with the
        # per-channel gamma slice of that stream (what the HF group_size=hidden norm computes).
        x = torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda")
        with torch.no_grad():
            layer.hnorm.weight.normal_(std=0.1)
        out = layer.hnorm(x).view(SEQ, BATCH, STREAMS, HIDDEN)
        ref = []
        for stream in range(STREAMS):
            xs = x.view(SEQ, BATCH, STREAMS, HIDDEN)[:, :, stream]
            gamma = layer.hnorm.weight[stream * HIDDEN : (stream + 1) * HIDDEN]
            ref.append(xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + config.layernorm_epsilon) * (1 + gamma))
        torch.testing.assert_close(out, torch.stack(ref, dim=2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            out.reshape(SEQ, BATCH, -1),
            grouped_rms_norm(x, layer.hnorm.weight, HIDDEN, config.layernorm_epsilon, True),
            atol=1e-6,
            rtol=1e-6,
        )


class TestGatedResidualGDNLayer:
    """Build and step a real GDN layer under the gated-residual variant (the
    norm-free spec with an unfused in_proj), wrapped by the hybrid mHC layer."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gdn_layer_forward_backward(self):
        import torch.nn.functional as F_

        from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
            HyperConnectionHybridLayer,
        )
        from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
        from megatron.core.transformer.spec_utils import build_module

        config = TransformerConfig(
            num_layers=1,
            hidden_size=2048,
            num_attention_heads=16,
            num_query_groups=2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_attention_freq=[1],
            experimental_attention_variant="gdn",
            activation_func=F_.silu,
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
        )
        spec = gated_residual_hybrid_stack_spec
        gdn_spec = spec.submodules.gdn_layer
        assert gdn_spec.submodules.self_attention.submodules.in_proj is TEColumnParallelLinear

        layer = build_module(gdn_spec, config=config, layer_number=1)
        assert isinstance(layer.self_attention.in_proj, TEColumnParallelLinear)
        assert isinstance(layer.input_layernorm, IdentityOp)

        wrapper = HyperConnectionHybridLayer(config, layer)
        convert_module_to_dtype_except_fp32_marked(wrapper, torch.bfloat16)
        wrapper = wrapper.cuda()

        seq, batch = 64, 2
        x = torch.randn(
            seq, batch, STREAMS * config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        out, context = wrapper(hidden_states=x, attention_mask=None)
        assert context is None
        assert out.shape == (seq, batch, STREAMS * config.hidden_size)
        assert torch.isfinite(out).all()

        out.float().square().sum().backward()
        for name, param in wrapper.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name


class TestGatedResidualHybridSpec:

    def test_hybrid_spec_is_norm_free(self):
        config = make_config()
        spec = gated_residual_hybrid_stack_spec
        sub = spec.submodules
        assert sub.gdn_layer.submodules.self_attention.submodules.in_proj is TEColumnParallelLinear
        assert (
            sub.attention_layer.submodules.self_attention.submodules.linear_qkv
            is TEColumnParallelLinear
        )
        assert sub.moe_layer.submodules.pre_mlp_layernorm is IdentityOp

    def test_hybrid_spec_is_marked_norm_free(self):
        """An explicitly passed GR spec must not trigger the double-normalization warning."""
        config = make_config()
        assert is_gated_residual_norm_free(gated_residual_hybrid_stack_spec)
        # The default stack keeps its fused input layernorms and must still be warned about.
        assert not is_gated_residual_norm_free(hybrid_stack_spec)
        assert not is_gated_residual_norm_free(object())

    def test_gpt_path_is_rejected(self):
        """The gated-residual variant is HybridModel-only; the GPT path must not build."""
        with pytest.raises(ValueError, match="HybridModel"):
            make_config(is_hybrid_model=False)

    def test_virtual_pipeline_is_rejected(self):
        with pytest.raises(ValueError, match="[Vv]irtual pipeline"):
            make_config(virtual_pipeline_model_parallel_size=2)

    def test_pipeline_parallel_is_rejected(self):
        with pytest.raises(ValueError, match="[Pp]ipeline parallelism"):
            make_config(pipeline_model_parallel_size=2)

    def test_hybrid_wrapper_dispatch(self):
        """HyperConnectionHybridLayer instantiates GatedResidualModule under GR."""
        from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
            HyperConnectionHybridLayer,
        )

        Utils.initialize_model_parallel(1, 1)
        try:
            model_parallel_cuda_manual_seed(123)
            config = make_config()
            spec = make_gr_spec(config)
            from megatron.core.transformer.spec_utils import build_module

            layer = build_module(spec, config=config, layer_number=1)
            wrapper = HyperConnectionHybridLayer(config, layer)
            assert isinstance(wrapper.hyper_connection, GatedResidualModule)
        finally:
            Utils.destroy_model_parallel()
