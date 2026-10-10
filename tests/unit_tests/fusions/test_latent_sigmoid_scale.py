# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent sigmoid sensitivity must preserve the other activation arguments."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("sigmoid_input_scale", [0.0, 0.5, 1.7, 1.0])
@pytest.mark.parametrize("mode", ["swiglu", "hard_clamp", "situ_gate", "situ_both"])
@pytest.mark.parametrize("variant", ["plain", "bias", "weighted"])
@pytest.mark.parametrize("shape", [(7, 16), (2, 7, 16)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
        ),
    ],
)
def test_sigmoid_scale_fused_forward_backward(sigmoid_input_scale, mode, variant, shape, device):
    """Compare custom backward (including bias/probabilities) with independent eager autograd."""
    torch.manual_seed(71)
    x = (torch.randn(shape, device=device) * 2).requires_grad_()
    bias = torch.randn(shape[-1], device=device).requires_grad_() if variant == "bias" else None
    weights = (
        torch.rand(x.numel() // shape[-1], 1, device=device).requires_grad_()
        if variant == "weighted"
        else None
    )
    gate, linear = (x if bias is None else x + bias).chunk(2, dim=-1)
    kwargs = {}
    if mode == "hard_clamp":
        kwargs["clamp_value"] = 1.25
        gate = gate.clamp(max=1.25)
        linear = linear.clamp(-1.25, 1.25)
    if mode.startswith("situ"):
        kwargs["gate_clamp_scale"] = 2.0
        gate_factor = 2.0 * torch.tanh(gate / 2.0)
        if mode == "situ_both":
            kwargs["linear_clamp_scale"] = 3.0
            linear = 3.0 * torch.tanh(linear / 3.0)
    else:
        gate_factor = gate
    reference = gate_factor * torch.sigmoid(sigmoid_input_scale * gate) * linear
    if weights is not None:
        reference = reference * weights.reshape(*shape[:-1], 1)
    grad = torch.randn_like(reference)
    inputs = [x] + ([bias] if bias is not None else []) + ([weights] if weights is not None else [])
    reference_grads = torch.autograd.grad(reference, inputs, grad)
    fused_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    fused_x = fused_inputs[0]
    kwargs["sigmoid_input_scale"] = sigmoid_input_scale
    if weights is not None:
        output = weighted_bias_swiglu_impl(fused_x, None, fused_inputs[1], **kwargs)
    else:
        output = bias_swiglu_impl(fused_x, fused_inputs[1] if bias is not None else None, **kwargs)
    fused_grads = torch.autograd.grad(output, fused_inputs, grad)
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    for actual, expected in zip(fused_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)


@pytest.mark.parametrize("sigmoid_input_scale", [0.0, -0.5, 0.5, 1.7])
@pytest.mark.parametrize(
    "down_gain,up_gain", [(False, False), (True, False), (False, True), (True, True)]
)
def test_sigmoid_scale_is_independent_of_projection_gains(sigmoid_input_scale, down_gain, up_gain):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_latent_size=8,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_latent_sigmoid_input_scale=sigmoid_input_scale,
        moe_latent_projection_scaling=down_gain,
        moe_latent_up_projection_scaling=up_gain,
    )
    assert config.moe_latent_sigmoid_input_scale == sigmoid_input_scale


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"moe_latent_size": None}, "positive moe_latent_size"),
        ({"moe_latent_size": 0}, "positive moe_latent_size"),
        ({"moe_latent_size": -2}, "positive moe_latent_size"),
        ({"num_moe_experts": None}, "num_moe_experts"),
        ({"gated_linear_unit": False}, "gated SiLU"),
        ({"use_transformer_engine_op_fuser": True}, "not supported by the TE op fuser"),
        ({"activation_func": F.gelu}, "gated SiLU"),
        ({"moe_latent_sigmoid_input_scale": float("inf")}, "must be finite"),
        ({"moe_latent_sigmoid_input_scale": float("nan")}, "must be finite"),
    ],
)
def test_sigmoid_scale_rejects_unsupported_config(kwargs, match):
    defaults = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_latent_size=8,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_latent_sigmoid_input_scale=1.7,
    )
    with pytest.raises(ValueError, match=match):
        TransformerConfig(**(defaults | kwargs))


def test_sigmoid_scale_default_preserves_dense_config():
    assert (
        TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=4
        ).moe_latent_sigmoid_input_scale
        == 1.0
    )
