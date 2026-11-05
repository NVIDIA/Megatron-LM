# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    TEFusedMLP,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def _make_submodules():
    return MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear)


def _make_config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=64,
        # Not a multiple of 32: TE <= 2.20 then uses its non-TMA GLU kernels, which also run on
        # GPUs with less shared memory per block (e.g. sm_120).
        ffn_hidden_size=112,
        num_attention_heads=4,
        # A dense layer of an MoE model: activation_func_clamp_value for SwiGLU requires MoE.
        num_moe_experts=4,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        bias_activation_fusion=False,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


@pytest.mark.skipif(TEFusedMLP is None, reason="TE operation-based MLP is unavailable")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTEFusedMLPActivation:
    """TEFusedMLP must compute the activation MLP computes for the same config, or refuse it."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("linear_scale", [None, 25.0])
    def test_rejects_situ_glu(self, linear_scale):
        config = _make_config(
            activation_func_tanh_clamp_scale=4.0,
            activation_func_tanh_clamp_scale_linear=linear_scale,
        )

        with pytest.raises(ValueError, match="activation_func_tanh_clamp_scale"):
            TEFusedMLP(config, _make_submodules(), ffn_hidden_size=config.ffn_hidden_size)

    @pytest.mark.skipif(
        not is_te_min_version("2.17.0.dev0"),
        reason="Clamped SwiGLU in the TE op fuser requires Transformer Engine >= 2.17.0.dev0",
    )
    def test_clamped_swiglu_matches_mlp(self):
        config = _make_config(activation_func_clamp_value=10.0)
        mlp = MLP(config, _make_submodules(), ffn_hidden_size=config.ffn_hidden_size).cuda()
        fused_mlp = TEFusedMLP(
            config, _make_submodules(), ffn_hidden_size=config.ffn_hidden_size
        ).cuda()
        with torch.no_grad():
            # FC1 outputs have a std of about 8, so the clamp at 10 is active on many elements.
            mlp.linear_fc1.weight.normal_(0.0, 1.0)
        fused_mlp.load_state_dict(mlp.state_dict())

        x = torch.randn(16, 2, config.hidden_size, device="cuda")
        x_ref = x.clone().requires_grad_()
        x_fused = x.clone().requires_grad_()
        out_ref, _ = mlp(x_ref)
        out_fused, _ = fused_mlp(x_fused)
        torch.testing.assert_close(out_fused, out_ref, rtol=1e-4, atol=1e-4)

        grad_output = torch.randn_like(out_ref)
        out_ref.backward(grad_output)
        out_fused.backward(grad_output)
        torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=1e-4, atol=1e-4)
        for param_ref, param_fused in zip(mlp.parameters(), fused_mlp.parameters()):
            torch.testing.assert_close(param_fused.grad, param_ref.grad, rtol=1e-4, atol=1e-4)
