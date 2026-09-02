# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the Transformer Engine fused layernorm and column-parallel linear wrapper."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE, TELayerNormColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
def test_fp32_residual_input_is_cast_without_mutating_residual():
    """Keep the residual FP32 while running the fused layer in BF16."""
    Utils.initialize_model_parallel(1, 1)

    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp32_residual_connection=True,
        )
        layer = TELayerNormColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=32,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        hidden_states = torch.randn(
            4, 2, config.hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
        )
        residual = hidden_states
        residual_before = residual.detach().clone()

        output, _ = layer(hidden_states)

        assert layer.layer_norm_weight.dtype == torch.bfloat16
        assert output.dtype == torch.bfloat16
        assert residual.dtype == torch.float32
        assert residual.data_ptr() == hidden_states.data_ptr()
        assert torch.equal(residual.detach(), residual_before)

        output.float().sum().backward()
        assert hidden_states.grad.dtype == torch.float32
    finally:
        Utils.destroy_model_parallel()
