# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""mHC input normalization variants selected by ``mhc_norm_eps*``.

``mhc_norm_eps_inside_sqrt=True`` makes the input normalization a standard RMSNorm,
``x * rsqrt(mean(x^2) + eps)``, which is the form GLM-5.3-Flash was trained with. The default
keeps the mHC paper's ``x / (rms(x) + eps)``. The two only differ materially for residual
streams whose per-token rms is near ``sqrt(eps)``, which is the regime this test exercises.
"""

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN, STREAMS, SEQ, BATCH = 64, 4, 5, 2


def _make_config(**overrides):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        add_bias_linear=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=0.01,
        **overrides,
    )
    return config


class TestHyperConnectionNormEps:
    """Input-normalization behavior of ``HyperConnectionModule``."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_eps_inside_sqrt_is_standard_rmsnorm(self):
        """The reciprocal matches ``rsqrt(mean(x^2) + eps)`` exactly."""
        eps = 1e-5
        config = _make_config(mhc_norm_eps=eps, mhc_norm_eps_inside_sqrt=True)
        with torch.device("cuda"):
            module = HyperConnectionModule(config, layer_number=1)

        # Per-token rms ~ 1e-2, i.e. below sqrt(eps): the epsilon placement is material here.
        x = 0.01 * torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda")
        with torch.no_grad():
            _, r = module._projection_and_get_norm(x)
            expected = torch.rsqrt(x.to(torch.float32).square().mean(dim=-1, keepdim=True) + eps)
        torch.testing.assert_close(r, expected, rtol=1e-5, atol=1e-6)

    def test_default_keeps_paper_normalization(self):
        """Without the flag the reciprocal stays ``1 / (rms(x) + eps)``."""
        eps = 1e-6
        config = _make_config(mhc_norm_eps=eps)
        assert config.mhc_norm_eps_inside_sqrt is False
        with torch.device("cuda"):
            module = HyperConnectionModule(config, layer_number=1)

        x = 0.01 * torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda")
        with torch.no_grad():
            _, r = module._projection_and_get_norm(x)
            x_fp32 = x.to(torch.float32)
            expected = 1.0 / (x_fp32.norm(dim=-1, keepdim=True) / (STREAMS * HIDDEN) ** 0.5 + eps)
        torch.testing.assert_close(r, expected, rtol=1e-5, atol=1e-6)

    def test_the_two_forms_disagree_at_small_scale(self):
        """Guards the reason the knob exists: at this scale the forms are not interchangeable."""
        eps = 1e-5
        x = 0.01 * torch.randn(SEQ, BATCH, STREAMS * HIDDEN, device="cuda", dtype=torch.float32)
        inside = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        outside = 1.0 / (x.norm(dim=-1, keepdim=True) / (STREAMS * HIDDEN) ** 0.5 + eps)
        assert (inside - outside).abs().max().item() > 1e-2 * outside.abs().max().item()

    def test_fused_mhc_rejects_eps_inside_sqrt(self):
        """The fused kernels implement the paper form only, so the combination is refused."""
        with pytest.raises(ValueError, match="mhc_norm_eps_inside_sqrt"):
            _make_config(mhc_norm_eps=1e-5, mhc_norm_eps_inside_sqrt=True, use_fused_mhc=True)
