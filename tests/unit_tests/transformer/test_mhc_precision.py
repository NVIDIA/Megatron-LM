# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    native_h_aggregate,
    native_h_aggregate_into,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.mark.parametrize("fp32_mixing", [False, True])
@pytest.mark.parametrize("input_scale", [1.0, 0.001])
def test_mhc_precision(fp32_mixing, input_scale):
    torch.manual_seed(42)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        layernorm_epsilon=1e-5,
        mhc_norm_eps_inside_sqrt=fp32_mixing,
        mhc_keep_mappings_in_fp32=fp32_mixing,
    )
    layer = HyperConnectionModule(config, layer_number=1).cuda()
    expected_eps = config.layernorm_epsilon if fp32_mixing else 1e-6
    assert layer.norm_eps == expected_eps
    with torch.no_grad():
        layer.mapping_proj.weight.normal_(std=0.2)
        layer.bias.normal_()
    residual = (torch.randn(16, 2, 128, device="cuda") * input_scale).to(torch.bfloat16)
    residual.requires_grad_()
    output = torch.randn(16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pre, post, comb = layer.compute_mappings(residual)
    expected_dtype = torch.float32 if fp32_mixing else torch.bfloat16
    assert pre.dtype == post.dtype == comb.dtype == expected_dtype

    x = residual.float()
    rms = (
        torch.rsqrt(x.square().mean(-1, keepdim=True) + expected_eps)
        if fp32_mixing
        else (x.norm(dim=-1, keepdim=True) / 128**0.5 + expected_eps).reciprocal()
    )
    scales = torch.cat(
        [layer.alpha_pre.expand(4), layer.alpha_post.expand(4), layer.alpha_res.expand(16)]
    )
    logits = (x @ layer.mapping_proj.weight.T) * rms * scales + layer.bias
    expected_pre = (logits[..., :4].sigmoid() + 1e-6).to(expected_dtype)
    expected_post = (2 * logits[..., 4:8].sigmoid()).to(expected_dtype)
    expected_comb = logits[..., 8:].reshape(16, 2, 4, 4).softmax(-1) + 1e-6
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(config.mhc_sinkhorn_iterations - 1):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + 1e-6)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    torch.testing.assert_close(pre, expected_pre)
    torch.testing.assert_close(post, expected_post)
    torch.testing.assert_close(comb, expected_comb.to(expected_dtype))

    aggregated = layer.aggregate(residual, pre)
    streams = residual.view(16, 2, 4, 32)
    expected_aggregate = (streams.float() * pre.float().unsqueeze(-1)).sum(2).to(residual.dtype)
    torch.testing.assert_close(aggregated, expected_aggregate, rtol=0.02, atol=0.01)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp32_mixing):
        actual = layer.fused_h_res_h_post_bda(
            comb, residual, post, (output, None), 0.0, True, False
        )
    expected_mix = torch.einsum("...ij,...ih->...jh", comb, streams.to(expected_dtype))
    expected_output = (
        expected_mix.float() + post.float().unsqueeze(-1) * output.float().unsqueeze(2)
    ).to(residual.dtype)
    torch.testing.assert_close(actual.view_as(streams), expected_output, rtol=0.02, atol=0.01)
    (actual.float().square().mean() + aggregated.float().square().mean()).backward()
    for tensor in (residual, output, layer.mapping_proj.weight, layer.bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("fp32_mappings", [False, True])
def test_mhc_caller_owned_aggregation_matches_regular_path(fp32_mappings):
    """The graph-input buffer must preserve FP32 mixing before the BF16 cast."""
    torch.manual_seed(43)
    x = torch.randn(8, 2, 4, 4096, device="cuda", dtype=torch.bfloat16)
    mapping_dtype = torch.float32 if fp32_mappings else torch.bfloat16
    weights = torch.randn(8, 2, 4, device="cuda", dtype=mapping_dtype)
    actual_input, reference_input = (x.clone().requires_grad_() for _ in range(2))
    actual_weights, reference_weights = (weights.clone().requires_grad_() for _ in range(2))
    out = torch.empty(8, 2, 4096, device="cuda", dtype=x.dtype)
    actual = native_h_aggregate_into(actual_input, actual_weights, out)
    expected = native_h_aggregate(reference_input, reference_weights)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    # The explicit-buffer custom op accumulates weight gradients in FP32.
    torch.testing.assert_close(
        actual_weights.grad, (grad.float().unsqueeze(2) * x.float()).sum(-1).to(mapping_dtype)
    )


@pytest.mark.parametrize("flag", ["mhc_norm_eps_inside_sqrt", "mhc_keep_mappings_in_fp32"])
def test_mhc_fused_rejects_unsupported_precision(flag):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        enable_mhc_connections=True,
        use_fused_mhc=True,
    )
    # Keep the module-level guard for configs modified after constructor validation.
    setattr(config, flag, True)
    with pytest.raises(ValueError, match="Fused mHC does not support"):
        HyperConnectionModule(config, layer_number=1)
