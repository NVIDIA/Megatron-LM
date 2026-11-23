# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the explicit cast boundaries used by native Kimi layers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ops.norm.torch_norm import TorchRMSNorm
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _config(dtype, **kwargs):
    return TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=2,
        params_dtype=dtype,
        normalization="RMSNorm",
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cast_before_affine", [False, True])
def test_latent_norm_precision_and_replay(dtype, cast_before_affine):
    seeded()
    norm = TorchRMSNorm(
        config=_config(dtype), hidden_size=64, eps=1e-5, cast_before_affine=cast_before_affine
    ).cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(129, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    cotangent = torch.randn_like(x)
    values = x.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
    expected = (
        normalized.to(dtype) * norm.weight
        if cast_before_affine
        else (normalized * norm.weight.float()).to(dtype)
    )
    actual = norm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (x, norm.weight), cotangent),
        torch.autograd.grad(expected, (x, norm.weight), cotangent),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert_module_replays_bit_exact(
        norm, (x,), replays=3, contention=True, what="latent RMSNorm cast boundary"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sigmoid_fp32", [False, True])
def test_mla_sigmoid_precision_and_replay(dtype, sigmoid_fp32):
    seeded()
    module = SimpleNamespace(
        config=SimpleNamespace(
            gated_attention_proj_granularity="elementwise",
            mla_output_gate_sigmoid_fp32=sigmoid_fp32,
            attention_output_gate_fusion=False,
        )
    )
    x = torch.randn(129, 1, 128, device="cuda", dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    cotangent = torch.randn_like(x)

    def run(x, gate):
        return MLASelfAttention._apply_mla_output_gate(module, x, gate)

    expected = x * torch.sigmoid(gate.float() if sigmoid_fp32 else gate).to(dtype)
    actual = run(x, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (x, gate), cotangent),
        torch.autograd.grad(expected, (x, gate), cotangent),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert_replays_bit_exact(
        run, (x, gate), replays=3, contention=True, what="MLA output sigmoid dtype"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_normalized_attn_res_replay(dtype):
    seeded()
    config = _config(
        dtype,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        attn_res_impl="torch",
        attn_res_fp32_params=False,
        attn_res_query_as_matrix=True,
        attn_res_single_source_param_grad=False,
    )
    module = AttentionResidual(config).cuda()
    with torch.no_grad():
        module.pseudo_query.normal_(std=0.1)
        module.key_norm_weight.uniform_(0.5, 1.5)
    sources = tuple(
        torch.randn(129, 1, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(4)
    )
    assert module.pseudo_query.shape == (1, 128)
    assert module.pseudo_query.dtype == dtype
    assert_module_replays_bit_exact(
        module, (sources,), replays=3, contention=True, what="normalized-source attention residuals"
    )
    single = module((sources[0],))
    grads = torch.autograd.grad(single.sum(), (sources[0], *module.parameters()), allow_unused=True)
    assert grads[0] is not None
    assert all(grad is None for grad in grads[1:])


@pytest.mark.parametrize("query_init", ["zero", "linear"])
def test_attn_res_query_initialization(query_init):
    config = _config(
        torch.float32,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        attn_res_query_as_matrix=True,
        attn_res_query_init=query_init,
    )
    seeded()
    first = AttentionResidual(config).cuda()
    seeded()
    second = AttentionResidual(config).cuda()
    torch.testing.assert_close(first.pseudo_query, second.pseudo_query, rtol=0, atol=0)
    if query_init == "zero":
        assert torch.count_nonzero(first.pseudo_query) == 0
    else:
        assert torch.count_nonzero(first.pseudo_query) > 0
        assert first.pseudo_query.abs().max() <= 128**-0.5
