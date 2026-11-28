# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from megatron.core.tensor_parallel.inference_layers import (
        InferenceLayerNormColumnParallelLinear,
    )


@pytest.mark.skipif(not HAVE_TE, reason="Requires Transformer Engine")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestInferenceLayerNormColumnParallelLinearInputDtype:
    """An fp32 residual stream feeds the fused norm + linear an fp32 input with bf16 parameters."""

    hidden_size = 256
    output_size = 512

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            normalization="RMSNorm",
            layernorm_epsilon=1e-5,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp32_residual_connection=True,
            add_bias_linear=False,
        )

    def _build(self, cls, config):
        return cls(
            self.hidden_size,
            self.output_size,
            config=config,
            init_method=torch.nn.init.normal_,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

    @torch.inference_mode()
    def test_fp32_input_matches_te_layer_norm_linear(self):
        config = self._config()
        reference = self._build(TELayerNormColumnParallelLinear, config).eval()
        layer = self._build(InferenceLayerNormColumnParallelLinear, config).eval()
        layer.load_state_dict(reference.state_dict())
        with torch.no_grad():
            # Non-trivial norm weights, so the norm is part of the comparison.
            ln = 1.0 + 0.1 * torch.randn_like(layer.layer_norm_weight)
            layer.layer_norm_weight.copy_(ln)
            reference.layer_norm_weight.copy_(ln)
        assert layer.weight.dtype == torch.bfloat16

        x = torch.randn(16, 2, self.hidden_size, device="cuda", dtype=torch.float32)
        expected = reference(x)[0]
        out = layer(x)[0]

        assert out.dtype == torch.bfloat16
        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, rtol=1.6e-2, atol=1e-2)
        # The cast is local: the caller's fp32 residual tensor is unchanged.
        assert x.dtype == torch.float32

    @torch.inference_mode()
    def test_fp32_input_equals_bf16_input(self):
        layer = self._build(InferenceLayerNormColumnParallelLinear, self._config()).eval()
        x = torch.randn(16, 2, self.hidden_size, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(layer(x)[0], layer(x.to(torch.bfloat16))[0], rtol=0, atol=0)
