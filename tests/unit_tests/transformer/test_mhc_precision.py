# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""mHC precision contracts, including GLM against an independent official-model reference.

GLM reference: huggingface/transformers commit
93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py,
Glm5NextTextHyperConnection and Glm5NextTextHyperHead. The GLM path retains
HF's normalize-before-projection order, an intentional deviation from PR 7521.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.hyper_connection import (
    HyperConnectionModule,
    native_h_aggregate,
    native_h_aggregate_into,
)
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    """CI torchrun workers are independent replicas, each on its own local GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _assert_similarity(actual, expected, name, eps=1e-3):
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    actual, expected = actual.flatten().double(), expected.flatten().double()
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return
    cosine = F.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0)).item()
    similarity = (2 * (actual * expected).sum() / denominator).item()
    assert cosine > 1 - eps, f"{name}: cosine={cosine}"
    assert similarity > 1 - eps, f"{name}: tensor_similarity={similarity}"


def _official_mhc(hidden, parameters, num_streams, iterations, eps):
    """Torch-only translation preserving the pinned HF operation ordering."""
    flat = hidden.float()
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    logits = F.linear(normalized, parameters["mapping_proj.weight"])
    pre_logits, post_logits, residual_logits = logits.split(
        (num_streams, num_streams, num_streams**2), dim=-1
    )
    pre_bias, post_bias, residual_bias = parameters["bias"].split(
        (num_streams, num_streams, num_streams**2)
    )
    pre = (pre_logits * parameters["alpha_pre"] + pre_bias).sigmoid() + 1e-6
    post = 2 * (post_logits * parameters["alpha_post"] + post_bias).sigmoid()
    mixing = residual_logits * parameters["alpha_res"] + residual_bias
    mixing = mixing.unflatten(-1, (num_streams, num_streams)).softmax(-1) + 1e-6
    mixing = mixing / (mixing.sum(-2, keepdim=True) + 1e-6)
    for _ in range(iterations - 1):
        mixing = mixing / (mixing.sum(-1, keepdim=True) + 1e-6)
        mixing = mixing / (mixing.sum(-2, keepdim=True) + 1e-6)
    streams = hidden.unflatten(-1, (num_streams, -1))
    aggregated = (streams * pre.unsqueeze(-1)).sum(-2).to(hidden.dtype)
    return aggregated, post, mixing


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


@pytest.mark.parametrize("input_scale", [1.0, 1e-3])
@pytest.mark.parametrize("fp32_mappings", [False, True])
def test_glm_mhc_official_forward_and_all_gradients(input_scale, fp32_mappings):
    """Keep the production 4096-wide hidden state and four residual streams."""
    torch.manual_seed(42)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        num_residual_streams=4,
        enable_hyper_connections=True,
        layernorm_epsilon=1e-5,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=fp32_mappings,
        mhc_keep_pre_in_fp32=True,
        mhc_learned_output_contract=False,
    )
    module = HyperConnectionModule(config, layer_number=1).cuda()
    with torch.no_grad():
        module.mapping_proj.weight.normal_(std=0.05)
        module.bias.normal_(std=0.1)
    reference_parameters = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in module.named_parameters()
    }
    assert set(reference_parameters) == {
        "mapping_proj.weight",
        "bias",
        "alpha_pre",
        "alpha_post",
        "alpha_res",
    }
    hidden = (torch.randn(8, 2, 4 * 4096, device="cuda") * input_scale).bfloat16()
    actual_input = hidden.clone().requires_grad_()
    reference_input = hidden.clone().requires_grad_()
    branch = torch.randn(8, 2, 4096, device="cuda", dtype=torch.bfloat16)
    actual_branch = branch.clone().requires_grad_()
    reference_branch = branch.clone().requires_grad_()

    pre, post, mixing = module.compute_mappings(actual_input)
    mapping_dtype = torch.float32 if fp32_mappings else hidden.dtype
    assert pre.dtype == torch.float32
    assert post.dtype == mixing.dtype == mapping_dtype
    aggregated = module.aggregate(actual_input, pre)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = module.fused_h_res_h_post_bda(
            mixing, actual_input, post, (actual_branch, None), 0.0, True, False
        )
    expected_aggregate, expected_post, expected_mixing = _official_mhc(
        reference_input, reference_parameters, 4, config.mhc_sinkhorn_iterations, 1e-5
    )
    # The official decoder casts post and comb, but not pre, before mixing.
    # The all-FP32 profile is separately supported for actor/serving recipes.
    expected_post = expected_post.to(mapping_dtype)
    expected_mixing = expected_mixing.to(mapping_dtype)
    reference_streams = reference_input.unflatten(-1, (4, 4096)).to(mapping_dtype)
    expected = (
        (
            torch.einsum("...ij,...ih->...jh", expected_mixing, reference_streams)
            + expected_post.unsqueeze(-1) * reference_branch.unsqueeze(-2)
        )
        .flatten(-2)
        .to(hidden.dtype)
    )
    assert aggregated.dtype == actual.dtype == hidden.dtype
    for name, observed, reference in (
        ("aggregate", aggregated, expected_aggregate),
        ("post", post, expected_post),
        ("mixing", mixing, expected_mixing),
        ("output", actual, expected),
    ):
        _assert_similarity(observed, reference, name)

    aggregate_grad = torch.randn_like(aggregated)
    output_grad = torch.randn_like(actual)
    ((aggregated * aggregate_grad).float().sum() + (actual * output_grad).float().sum()).backward()
    (
        (expected_aggregate * aggregate_grad).float().sum() + (expected * output_grad).float().sum()
    ).backward()
    _assert_similarity(actual_input.grad, reference_input.grad, "input gradient")
    _assert_similarity(actual_branch.grad, reference_branch.grad, "branch gradient")
    for name, parameter in module.named_parameters():
        _assert_similarity(parameter.grad, reference_parameters[name].grad, name)


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
