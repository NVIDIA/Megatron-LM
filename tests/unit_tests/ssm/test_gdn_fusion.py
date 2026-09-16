# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""First-order correctness coverage for the shape-specialized GDN fusions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm import gdn_fusion, gdn_gated_norm


def test_disabled_preparation_does_not_require_fla(monkeypatch):
    monkeypatch.setenv("MCORE_GDN_FUSION", "1")
    with patch.object(gdn_fusion, "_LINEAR_BWD", None):
        assert not gdn_fusion.enabled(SimpleNamespace(), torch.empty(0))


def test_disabled_preparation_does_not_inspect_inputs(monkeypatch):
    monkeypatch.setenv("MCORE_GDN_FUSION", "0")
    assert not gdn_fusion.enabled(SimpleNamespace(), torch.empty(0))


def _assert_gradients_close(actual, expected, atol=0.03):
    for got, ref in zip(actual, expected):
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert relative_l2.item() < 0.006
        torch.testing.assert_close(got, ref, atol=atol, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "length,scale,boundaries,has_bias,param_dtype",
    [
        (37, 1.0, None, True, torch.float32),
        (97, 1e-5, [0, 3, 33, 97], False, torch.bfloat16),
        (257, 1.0, [0, 1, 4, 15, 127, 257], True, torch.bfloat16),
        (32, 0.0, [0, 32], False, torch.bfloat16),
    ],
)
def test_preparation_forward_backward(length, scale, boundaries, has_bias, param_dtype):
    convolution = pytest.importorskip("fla.modules.convolution")
    normalization = pytest.importorskip("fla.modules.l2norm")
    if gdn_fusion._LINEAR_BWD is None:
        pytest.skip("The installed FLA version does not expose the linear backward kernel")
    torch.manual_seed(4321)
    projection = torch.randn((1, length, 5184), device="cuda", dtype=torch.bfloat16) * scale
    projection = projection[..., :5152].detach().requires_grad_()
    weight = (torch.randn((3072, 4), device="cuda", dtype=projection.dtype) * 0.1).requires_grad_()
    bias = (
        (torch.randn((3072,), device="cuda", dtype=projection.dtype) * 0.03).requires_grad_()
        if has_bias
        else None
    )
    alog = (torch.rand((16,), device="cuda", dtype=param_dtype) * 2).requires_grad_()
    dtbias = torch.ones((16,), device="cuda", dtype=param_dtype, requires_grad=True)
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int64) if boundaries else None
    qkv, gate, beta, alpha = projection.split((3072, 2048, 16, 16), -1)

    # GDN compiles this boundary, keeping exp(A_log) and its reduction in FP32.
    @torch.compile
    def prepare(conv, gate, beta, alpha, alog, dtbias):
        qk, value = conv.split((1024, 2048), -1)
        qk = normalization.l2norm(qk.reshape(1, length, 8, 128).contiguous())
        query, key = qk.split(4, -2)
        return (
            query.repeat_interleave(4, -2).contiguous(),
            key.repeat_interleave(4, -2).contiguous(),
            value.reshape(1, length, 16, 128).contiguous(),
            gate.reshape(1, length, 16, 128).contiguous(),
            -alog.exp() * F.softplus(alpha.float() + dtbias),
            beta.sigmoid(),
        )

    def unfused(conv_fn):
        conv, _ = conv_fn(qkv, weight, bias=bias, activation="silu", cu_seqlens=cu)
        return prepare(conv, gate, beta, alpha, alog, dtbias)

    expected = unfused(convolution.causal_conv1d)
    gradients = tuple(torch.randn_like(out) for out in expected)
    inputs = tuple(t for t in (projection, weight, bias, alog, dtbias) if t is not None)
    reference_gradients = torch.autograd.grad(expected, inputs, gradients)
    actual = gdn_fusion.fused_prepare(projection, weight, bias, alog, dtbias, cu)
    for index, (got, ref) in enumerate(zip(actual, expected)):
        if index in (3, 5):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        else:
            torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-2)
    _assert_gradients_close(torch.autograd.grad(actual, inputs, gradients), reference_gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("length,scale,zero_centered", [(37, 1.0, False), (97, 1e-5, True)])
def test_strided_output_norm_forward_backward(length, scale, zero_centered):
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(1234)
    norm = te.RMSNorm(
        128, eps=1e-6, params_dtype=torch.bfloat16, zero_centered_gamma=zero_centered, device="cuda"
    )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + (0 if zero_centered else 1))
    x = (
        torch.randn((1, length, 16, 128), device="cuda", dtype=torch.bfloat16) * scale
    ).requires_grad_()
    projection = torch.randn((1, length, 5152), device="cuda", dtype=x.dtype)
    gate = projection[..., 3072:5120].reshape(1, length, 16, 128).detach().requires_grad_()
    expected = (
        (norm(x.reshape(-1, 128)) * F.silu(gate.reshape(-1, 128).float()))
        .to(x.dtype)
        .reshape(x.shape)
    )
    actual = gdn_gated_norm.fused_gated_norm(x, gate, norm.weight, norm.eps, zero_centered)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    dy = torch.randn_like(actual)
    inputs = (x, gate, norm.weight)
    _assert_gradients_close(
        torch.autograd.grad(actual, inputs, dy),
        torch.autograd.grad(expected, inputs, dy),
        atol=0.05,
    )
