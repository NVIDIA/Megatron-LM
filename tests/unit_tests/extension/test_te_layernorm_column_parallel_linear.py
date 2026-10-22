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


def _rms_norm(hidden_states, weight, eps):
    hidden_float = hidden_states.float()
    inv_rms = torch.rsqrt(hidden_float.square().mean(dim=-1, keepdim=True) + eps)
    return (hidden_float * inv_rms * weight.float()).to(hidden_states.dtype)


def _rms_norm_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
@pytest.mark.parametrize("bias, skip_bias_add", [(False, False), (True, True)])
def test_return_layernorm_output_appends_the_normalized_input(bias, skip_bias_add):
    """Opt-in third output is the fused norm's output; the default keeps two outputs."""
    Utils.initialize_model_parallel(1, 1)
    try:
        config = _rms_norm_config()
        torch.manual_seed(0)
        layers = []
        for return_layernorm_output in (False, True):
            layers.append(
                TELayerNormColumnParallelLinear(
                    input_size=config.hidden_size,
                    output_size=96,
                    config=config,
                    init_method=init_method_normal(config.init_method_std),
                    gather_output=False,
                    bias=bias,
                    skip_bias_add=skip_bias_add,
                    is_expert=False,
                    return_layernorm_output=return_layernorm_output,
                ).cuda()
            )
        default, returning = layers
        returning.load_state_dict(default.state_dict())
        with torch.no_grad():
            returning.layer_norm_weight.uniform_(0.5, 1.5)
            default.layer_norm_weight.copy_(returning.layer_norm_weight)
        assert TELayerNormColumnParallelLinear.supports_layernorm_output(config)

        hidden_states = torch.randn(
            8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        default_outputs = default(hidden_states)
        outputs = returning(hidden_states)

        assert len(default_outputs) == 2
        assert len(outputs) == 3
        output, output_bias, layernorm_output = outputs
        torch.testing.assert_close(output, default_outputs[0])
        if skip_bias_add:
            torch.testing.assert_close(output_bias, default_outputs[1])
        else:
            assert output_bias is None
        torch.testing.assert_close(
            layernorm_output,
            _rms_norm(
                hidden_states.detach(), returning.layer_norm_weight, config.layernorm_epsilon
            ),
        )

        # Backward runs whether or not the norm output is used, and the main path's gradients
        # are those of the default layer when it is not.
        output.float().sum().backward()
        returning_grad = hidden_states.grad.clone()
        hidden_states.grad = None
        default_outputs[0].float().sum().backward()
        torch.testing.assert_close(returning_grad, hidden_states.grad)
        torch.testing.assert_close(returning.weight.grad, default.weight.grad)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
def test_inference_layernorm_linear_returns_the_normalized_input_without_tp():
    """The inference-optimized fused layer appends its norm output at TP=1 only."""
    from types import SimpleNamespace

    from megatron.core.tensor_parallel.inference_layers import (
        InferenceLayerNormColumnParallelLinear,
    )

    assert InferenceLayerNormColumnParallelLinear.supports_layernorm_output(
        SimpleNamespace(tensor_model_parallel_size=1)
    )
    assert not InferenceLayerNormColumnParallelLinear.supports_layernorm_output(
        SimpleNamespace(tensor_model_parallel_size=2)
    )

    Utils.initialize_model_parallel(1, 1)
    try:
        config = _rms_norm_config(transformer_impl="inference_optimized", add_bias_linear=False)
        torch.manual_seed(0)
        layer = InferenceLayerNormColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=96,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            return_layernorm_output=True,
        ).cuda()
        with torch.no_grad():
            layer.layer_norm_weight.uniform_(0.5, 1.5)
        layer.eval()

        hidden_states = torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            output, output_bias, layernorm_output = layer(hidden_states)
            expected_norm = _rms_norm(
                hidden_states, layer.layer_norm_weight, config.layernorm_epsilon
            )
            expected_output = expected_norm @ layer.weight.t()

        assert output_bias is None
        torch.testing.assert_close(layernorm_output, expected_norm)
        torch.testing.assert_close(output, expected_output)
    finally:
        Utils.destroy_model_parallel()
