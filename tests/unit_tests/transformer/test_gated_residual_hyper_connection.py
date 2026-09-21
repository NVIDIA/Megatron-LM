# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the Qwen4-Exp Gated Residual hyper connection.

The reference implementations below are straight ports of the HuggingFace
``Qwen4ExpTextGatedResidual`` / ``Qwen4ExpTextRMSNorm`` modules.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import (
    GatedResidualHyperConnection,
    GatedResidualOutputMixer,
    gated_residual_group_rmsnorm,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _reference_group_rmsnorm(x, weight, group_size, eps):
    xf = x.float().reshape(*x.shape[:-1], -1, group_size)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out.flatten(-2) * (1.0 + weight.float())).type_as(x)


def _reference_gated_residual(module, hyper_input):
    """HF Qwen4ExpTextGatedResidual.forward (use_combine=True)."""
    n, C = module.n, module.hidden_size
    hn = _reference_group_rmsnorm(hyper_input, module.hc_norm.weight, C, module.norm_eps)
    w = F.silu(F.linear(hn, module.input_mix_weight_down.weight) / n)
    w = torch.sigmoid(F.linear(w, module.input_mix_weight_up.weight))
    w = w.unflatten(-1, (n, C))
    mixed = (w * hn.unflatten(-1, (n, C))).mean(dim=-2)
    if module.block_inject_weight is None:
        return mixed, None
    inject = 2 * torch.sigmoid(F.linear(hn, module.block_inject_weight.weight) / n)
    return mixed, inject


def _make_config(hidden_size=32, n=4, rank=8, dtype=torch.float32):
    return TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_variant="gated_residual",
        mhc_num_residual_streams=n,
        mhc_gated_residual_rank=rank,
        params_dtype=dtype,
        layernorm_epsilon=1e-6,
    )


class TestGatedResidualHyperConnection:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_reference(self, dtype):
        config = _make_config(dtype=dtype)
        module = GatedResidualHyperConnection(config, layer_number=1).cuda()
        with torch.no_grad():
            module.hc_norm.weight.normal_(0, 0.1)
        s, b = 5, 2
        x = torch.randn(
            s, b, config.mhc_num_residual_streams * config.hidden_size, device="cuda", dtype=dtype
        )

        block_input, h_res, inject = module(x)
        ref_mixed, ref_inject = _reference_gated_residual(module, x)

        assert h_res is None
        assert block_input.shape == (s, b, config.hidden_size)
        assert inject.shape == (s, b, config.mhc_num_residual_streams)
        tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(block_input.float(), ref_mixed.float(), **tol)
        torch.testing.assert_close(inject.float(), ref_inject.float(), **tol)

    def test_residual_update(self):
        config = _make_config()
        module = GatedResidualHyperConnection(config, layer_number=1).cuda()
        n, C = config.mhc_num_residual_streams, config.hidden_size
        s, b = 3, 2
        residual = torch.randn(s, b, n * C, device="cuda")
        _, _, inject = module(residual)
        out = torch.randn(s, b, C, device="cuda")
        bias = torch.randn(C, device="cuda")

        updated = module.fused_h_res_h_post_bda(
            None, residual, inject, (out, bias), 0.0, True, False
        )

        expected = residual.view(s, b, n, C) + inject.unsqueeze(-1) * (out + bias).unsqueeze(2)
        torch.testing.assert_close(updated, expected.view(s, b, n * C), atol=1e-6, rtol=1e-6)

    def test_output_mixer_matches_reference_and_contracts(self):
        config = _make_config()
        mixer = GatedResidualOutputMixer(config, hidden_size=config.hidden_size, eps=1e-6).cuda()
        assert mixer.contracts_mhc_streams
        assert mixer.block_inject_weight is None
        x = torch.randn(4, 3, config.mhc_num_residual_streams * config.hidden_size, device="cuda")
        out = mixer(x)
        ref, ref_inject = _reference_gated_residual(mixer, x)
        assert ref_inject is None
        assert out.shape == (4, 3, config.hidden_size)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_group_rmsnorm_matches_reference(self):
        x = torch.randn(6, 2, 8 * 4, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.1
        out = gated_residual_group_rmsnorm(x, w, 4, 1e-6)
        ref = _reference_group_rmsnorm(x, w, 8, 1e-6)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_gradients_flow(self):
        config = _make_config()
        module = GatedResidualHyperConnection(config, layer_number=1).cuda()
        x = torch.randn(
            4,
            2,
            config.mhc_num_residual_streams * config.hidden_size,
            device="cuda",
            requires_grad=True,
        )
        block_input, _, inject = module(x)
        out = block_input * 2.0
        updated = module.fused_h_res_h_post_bda(None, x, inject, (out, None), 0.0, True, False)
        updated.square().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for name, p in module.named_parameters():
            assert p.grad is not None, name


class TestGatedResidualConfig:
    def test_requires_rank(self):
        with pytest.raises(ValueError):
            TransformerConfig(
                num_layers=2,
                hidden_size=32,
                num_attention_heads=4,
                enable_mhc_connections=True,
                mhc_variant="gated_residual",
            )

    def test_full_recompute_allowed(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            enable_mhc_connections=True,
            mhc_variant="gated_residual",
            mhc_gated_residual_rank=8,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        assert config.recompute_granularity == "full"
