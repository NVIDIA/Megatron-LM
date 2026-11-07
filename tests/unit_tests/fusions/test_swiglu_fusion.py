# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situ_glu
from megatron.core.fusions import fused_bias_swiglu
from megatron.core.fusions.fused_bias_swiglu import (
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _clamped_swiglu_config(**kwargs):
    defaults = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
    )
    return TransformerConfig(**(defaults | kwargs))


def test_clamped_swiglu_config_accepts_positive_moe_clamp():
    assert _clamped_swiglu_config().activation_func_clamp_value == 10.0


@pytest.mark.parametrize("clamp_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_clamped_swiglu_config_requires_positive_clamp(clamp_value):
    with pytest.raises(ValueError, match="greater than zero"):
        _clamped_swiglu_config(activation_func_clamp_value=clamp_value)


def test_clamped_swiglu_config_rejects_linear_offset():
    with pytest.raises(ValueError, match="glu_linear_offset must be zero"):
        _clamped_swiglu_config(glu_linear_offset=1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_moe_experts": None}, "only supported with MoE"),
        ({"use_te_activation_func": True}, "use_te_activation_func must be False"),
    ],
)
def test_clamped_swiglu_config_rejects_unsupported_paths(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _clamped_swiglu_config(**kwargs)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("linear_clamp_scale", [None, 3.0], ids=("gate_only", "gate_and_linear"))
@pytest.mark.parametrize("with_bias", [False, True])
def test_situ_glu_fusion_matches_unfused(input_dtype, linear_clamp_scale, with_bias):
    """Fused SiTU-GLU must match the eager path taken when bias_activation_fusion is False."""
    torch.manual_seed(0)
    gate_clamp_scale = 5.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=2.0e-1)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    # Scaled up so a good share of the inputs land in the saturating region of the clamp.
    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = situ_glu(x + bias if with_bias else x, gate_clamp_scale, linear_clamp_scale)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_fused = bias_swiglu_impl(
        x_fused,
        bias_fused,
        gate_clamp_scale=gate_clamp_scale,
        linear_clamp_scale=linear_clamp_scale,
    )
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    if with_bias:
        assert bias_fused.grad.dtype == bias.grad.dtype
        assert torch.allclose(bias.grad, bias_fused.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_clamped_weighted_bias_swiglu(input_dtype):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    wgrad_tols = dict(rtol=1.0e-6, atol=2.0e-4)

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    # Reference: clamp and activate in FP32, then restore the input dtype.
    y_1, y_2 = torch.chunk(x.to(torch.float32), 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value))
        * y_2.clamp(min=-clamp_value, max=clamp_value)
        * weights
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    weights_fused = weights.detach().clone().requires_grad_(True)
    y_fused = weighted_bias_swiglu_impl(x_fused, None, weights_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    assert weights_fused.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_fused.grad, **wgrad_tols)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_clamped_bias_swiglu_impl(input_dtype, with_bias):
    clamp_value = 10.0

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = (torch.randn(16, 64, dtype=input_dtype, device="cuda") * 5.0).requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    x_fp32 = x.to(torch.float32)
    x_effective = x_fp32 + bias.to(torch.float32) if with_bias else x_fp32
    y_1, y_2 = torch.chunk(x_effective, 2, -1)
    y = (
        F.silu(y_1.clamp(min=None, max=clamp_value)) * y_2.clamp(min=-clamp_value, max=clamp_value)
    ).to(input_dtype)
    y.backward(bwd_input)

    x_fused = x.detach().clone().requires_grad_(True)
    bias_fused = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_fused = bias_swiglu_impl(x_fused, bias_fused, clamp_value=clamp_value)
    y_fused.backward(bwd_input.detach().clone())

    assert y_fused.dtype == y.dtype
    assert torch.allclose(y, y_fused, **tols)
    assert x_fused.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_fused.grad, **tols)
    if with_bias:
        assert bias_fused.grad.dtype == bias.grad.dtype
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0), bias_fused.grad.flatten().float().unsqueeze(0)
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bias_swiglu_impl_clamp_none_matches_unclamped(input_dtype, with_bias):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    else:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda").requires_grad_(True)
    bias = (
        torch.randn(64, dtype=input_dtype, device="cuda").requires_grad_(True)
        if with_bias
        else None
    )
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, bias)
    y.backward(bwd_input)

    x_explicit = x.detach().clone().requires_grad_(True)
    bias_explicit = bias.detach().clone().requires_grad_(True) if with_bias else None
    y_explicit = bias_swiglu_impl(x_explicit, bias_explicit, clamp_value=None)
    y_explicit.backward(bwd_input.detach().clone())

    assert torch.allclose(y, y_explicit, **tols)
    assert torch.allclose(x.grad, x_explicit.grad, **tols)
    if with_bias:
        bias_grad_cos = F.cosine_similarity(
            bias.grad.flatten().float().unsqueeze(0),
            bias_explicit.grad.flatten().float().unsqueeze(0),
        ).item()
        assert bias_grad_cos > 0.999, f"bias.grad cosine similarity = {bias_grad_cos:.6f}"


# The bias-free plain, clamped and weighted paths run the Triton kernels in fused_swiglu_triton.
# They must match the torch.compile functions they replace: within one bf16 ULP for activations
# and input grads (the kernels round once, Inductor may fuse differently), and to FP32 reduction
# order for the weighted d_w.
_TRITON_TOLS = {
    torch.bfloat16: dict(rtol=2.0**-7, atol=1.0e-5),
    torch.float32: dict(rtol=1.0e-5, atol=1.0e-6),
}
_TRITON_REQUIRED = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_bias_swiglu.HAVE_TRITON), reason="needs CUDA + Triton"
)


def _clamp_inputs(tokens, ffn, dtype, clamp_value):
    x = torch.randn(tokens, 2 * ffn, dtype=dtype, device="cuda") * 5.0
    if clamp_value is not None:
        # Exact boundary values exercise the clamp's gradient rule at +/-clamp_value.
        x[:, :4] = clamp_value
        x[:, ffn : ffn + 4] = -clamp_value
    return x


@_TRITON_REQUIRED
@pytest.mark.parametrize("ffn", [96, 768, 2048], ids=("tiny", "multi_row", "one_row"))
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("clamp_value", [None, 7.0], ids=("unclamped", "clamped"))
def test_triton_swiglu_matches_compiled(ffn, input_dtype, clamp_value):
    tols = _TRITON_TOLS[input_dtype]
    x = _clamp_inputs(67, ffn, input_dtype, clamp_value).requires_grad_(True)
    grad = torch.randn(67, ffn, dtype=input_dtype, device="cuda")

    y_ref = SwiGLUFunction.apply(x, False, False, clamp_value, None, None)
    (dx_ref,) = torch.autograd.grad(y_ref, x, grad)
    y = bias_swiglu_impl(x, None, clamp_value=clamp_value)
    (dx,) = torch.autograd.grad(y, x, grad)

    assert y.dtype == y_ref.dtype and dx.dtype == dx_ref.dtype
    torch.testing.assert_close(y, y_ref, **tols)
    torch.testing.assert_close(dx, dx_ref, **tols)


@_TRITON_REQUIRED
@pytest.mark.parametrize("ffn", [96, 768, 2048], ids=("tiny", "multi_row", "one_row"))
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("clamp_value", [None, 7.0], ids=("unclamped", "clamped"))
def test_triton_weighted_swiglu_matches_compiled(ffn, input_dtype, clamp_value):
    tols = _TRITON_TOLS[input_dtype]
    x = _clamp_inputs(67, ffn, input_dtype, clamp_value).requires_grad_(True)
    weights = torch.rand(67, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    grad = torch.randn(67, ffn, dtype=input_dtype, device="cuda")

    y_ref = WeightedSwiGLUFunction.apply(x, weights, False, clamp_value, None, None)
    dx_ref, dw_ref = torch.autograd.grad(y_ref, (x, weights), grad)
    y = weighted_bias_swiglu_impl(x, None, weights, clamp_value=clamp_value)
    dx, dw = torch.autograd.grad(y, (x, weights), grad)

    assert y.dtype == y_ref.dtype and dx.dtype == dx_ref.dtype and dw.dtype == dw_ref.dtype
    torch.testing.assert_close(y, y_ref, **tols)
    torch.testing.assert_close(dx, dx_ref, **tols)
    torch.testing.assert_close(dw, dw_ref, rtol=1.0e-4, atol=1.0e-4)


@_TRITON_REQUIRED
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 loads need sm89+",
)
@pytest.mark.parametrize("weighted", [False, True], ids=("plain", "weighted"))
def test_triton_swiglu_fp8_input_store_matches_compiled(weighted):
    x = torch.randn(67, 2 * 768, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weights = torch.rand(67, 1, dtype=torch.float32, device="cuda", requires_grad=True)
    grad = torch.randn(67, 768, dtype=torch.bfloat16, device="cuda")
    if weighted:
        y_ref = WeightedSwiGLUFunction.apply(x, weights, True, None, None, None)
        y = weighted_bias_swiglu_impl(x, None, weights, fp8_input_store=True)
        inputs = (x, weights)
    else:
        y_ref = SwiGLUFunction.apply(x, True, False, None, None, None)
        y = bias_swiglu_impl(x, None, fp8_input_store=True)
        inputs = (x,)
    grads_ref = torch.autograd.grad(y_ref, inputs, grad)
    grads = torch.autograd.grad(y, inputs, grad)

    torch.testing.assert_close(y, y_ref, **_TRITON_TOLS[torch.bfloat16])
    torch.testing.assert_close(grads[0], grads_ref[0], **_TRITON_TOLS[torch.bfloat16])
    if weighted:
        torch.testing.assert_close(grads[1], grads_ref[1], rtol=1.0e-4, atol=1.0e-4)


@_TRITON_REQUIRED
def test_triton_swiglu_3d_input_keeps_shape():
    x = torch.randn(5, 7, 2 * 768, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(35, 1, dtype=torch.float32, device="cuda")
    assert bias_swiglu_impl(x, None).shape == (5, 7, 768)
    assert weighted_bias_swiglu_impl(x, None, weights).shape == (5, 7, 768)


def test_triton_swiglu_routing():
    x = torch.empty(4, 8, device="cuda" if torch.cuda.is_available() else "cpu")
    expected = fused_bias_swiglu.HAVE_TRITON and x.is_cuda
    assert fused_bias_swiglu._use_triton_swiglu(x, None, None) == expected
    # SiTU-GLU and CPU tensors stay on the torch.compile path.
    assert not fused_bias_swiglu._use_triton_swiglu(x, 2.0, None)
    assert not fused_bias_swiglu._use_triton_swiglu(x, 2.0, 3.0)
    assert not fused_bias_swiglu._use_triton_swiglu(torch.empty(4, 8), None, None)
