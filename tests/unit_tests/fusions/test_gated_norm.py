# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward/backward parity and support checks for the GDN output fusion.

Numerical reference coverage is adapted from Layali Rashid's PR #7368.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_gated_norm as gated_norm


def _assert_gradients_close(actual, expected, atol=0.03):
    for got, ref in zip(actual, expected):
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert relative_l2.item() < 0.006
        torch.testing.assert_close(got, ref, atol=atol, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("length,scale,zero_centered", [(37, 1.0, False), (97, 1e-5, True)])
@pytest.mark.parametrize("contiguous_gate", [False, True])
def test_strided_output_norm_forward_backward(length, scale, zero_centered, contiguous_gate):
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
    if contiguous_gate:
        gate = gate.detach().contiguous().requires_grad_()
    expected = (
        (norm(x.reshape(-1, 128)) * F.silu(gate.reshape(-1, 128).float()))
        .to(x.dtype)
        .reshape(x.shape)
    )
    actual = gated_norm.fused_gated_norm(x, gate, norm.weight, norm.eps, zero_centered)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    dy = torch.randn_like(actual)
    inputs = (x, gate, norm.weight)
    gradients = torch.autograd.grad(actual, inputs, dy)
    _assert_gradients_close(gradients, torch.autograd.grad(expected, inputs, dy), atol=0.05)
    # Check repeated forward/backward results alongside the numerical reference.
    for _ in range(2):
        replay = gated_norm.fused_gated_norm(x, gate, norm.weight, norm.eps, zero_centered)
        replay_gradients = torch.autograd.grad(replay, inputs, dy)
        torch.testing.assert_close(replay, actual, atol=0, rtol=0)
        for got, ref in zip(replay_gradients, gradients):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.fixture
def norm_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    te = pytest.importorskip("transformer_engine.pytorch")
    norm = te.RMSNorm(128, eps=1e-6, params_dtype=torch.bfloat16, device="cuda")
    module = SimpleNamespace(
        config=SimpleNamespace(deterministic_mode=False),
        cp_size=1,
        activation="silu",
        out_norm=norm,
    )
    x = torch.randn((1, 3, 16, 128), device="cuda", dtype=torch.bfloat16)
    projection = torch.randn((1, 3, 5152), device="cuda", dtype=x.dtype)
    gate = projection[..., 3072:5120].reshape_as(x)
    return module, x, gate


@pytest.mark.parametrize("contiguous_gate", [False, True])
def test_validate_supported_layout(norm_inputs, contiguous_gate):
    module, x, gate = norm_inputs
    if contiguous_gate:
        gate = gate.contiguous()
    gated_norm.validate_gated_norm(module, x, gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 37, 16, 128), "sequence_projection"),
        ((2, 37, 16, 128), "sequence_projection"),
        ((3, 17, 8, 64), "sequence_projection"),
        ((2, 13, 4, 256), "sequence_projection"),
        ((3, 1, 3, 32), "sequence_projection"),
        ((2, 37, 8, 128), "batch_projection"),
        ((3, 17, 4, 64), "contiguous"),
        ((2, 13, 1, 256), "contiguous"),
        ((2, 17, 8, 128), "strided_elements"),
        ((2, 13, 4, 64), "strided_heads"),
        ((2, 17, 8, 128), "strided_x"),
        ((3, 13, 4, 64), "broadcast_gate"),
    ],
)
def test_generalized_layout_forward_backward(shape, layout, dtype, zero_centered):
    """Compare values and source-tensor gradients, including projection-backed batches."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(1234)
    batch, length, heads, dim = shape
    norm = te.RMSNorm(
        dim, eps=1e-6, params_dtype=dtype, zero_centered_gamma=zero_centered, device="cuda"
    )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + (0 if zero_centered else 1))
    scale = 1e-5 if zero_centered else 1.0
    x_shape = (length, batch, heads, dim * 2) if layout == "strided_x" else shape
    x_source = (torch.randn(x_shape, device="cuda", dtype=dtype) * scale).requires_grad_()
    x = x_source.permute(1, 0, 2, 3)[..., ::2] if layout == "strided_x" else x_source
    channels = heads * dim * 2 + 32
    if layout in ("sequence_projection", "strided_x"):
        source_shape = (length, batch, channels)
    elif layout == "batch_projection":
        source_shape = (batch, length, channels)
    elif layout == "strided_elements":
        source_shape = (batch, length, heads, dim * 2)
    elif layout == "strided_heads":
        source_shape = (batch, length, heads * 2, dim)
    elif layout == "broadcast_gate":
        source_shape = (1, length, heads, dim)
    else:
        source_shape = shape
    gate_source = torch.randn(source_shape, device="cuda", dtype=dtype).requires_grad_()
    if layout in ("sequence_projection", "strided_x"):
        gate = gate_source[..., 17 : 17 + heads * dim].view(length, batch, heads, dim)
        gate = gate.permute(1, 0, 2, 3)
    elif layout == "batch_projection":
        gate = gate_source[..., 17 : 17 + heads * dim].view(shape)
    elif layout == "strided_elements":
        gate = gate_source[..., ::2]
    elif layout == "strided_heads":
        gate = gate_source[:, :, ::2]
    elif layout == "broadcast_gate":
        gate = gate_source.expand(shape)
    else:
        gate = gate_source
    module = SimpleNamespace(
        config=SimpleNamespace(deterministic_mode=False),
        cp_size=4,
        activation="silu",
        out_norm=norm,
    )
    gated_norm.validate_gated_norm(module, x, gate)
    expected = (norm(x.reshape(-1, dim)).reshape(shape) * F.silu(gate.float())).to(dtype)
    actual = gated_norm.fused_gated_norm(x, gate, norm.weight, norm.eps, zero_centered)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    # Noncontiguous upstream gradients also need to be read in logical order.
    dy = torch.randn((*shape[:-1], dim * 2), device="cuda", dtype=dtype)[..., ::2]
    inputs = (x_source, gate_source, norm.weight)
    gradients = torch.autograd.grad(actual, inputs, dy)
    _assert_gradients_close(gradients, torch.autograd.grad(expected, inputs, dy), atol=0.05)
    replay = gated_norm.fused_gated_norm(x, gate, norm.weight, norm.eps, zero_centered)
    replay_gradients = torch.autograd.grad(replay, inputs, dy)
    torch.testing.assert_close(replay, actual, atol=0, rtol=0)
    for got, ref in zip(replay_gradients, gradients):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case,match",
    [
        ("deterministic", "deterministic_mode=False"),
        ("activation", "SiLU/Swish"),
        ("norm", "RMSNorm output"),
        ("cpu", "CUDA BF16"),
        ("dtype", "CUDA BF16"),
        ("head_dim", "power-of-two head dimension"),
        ("gate_rank", "matching output and gate shapes"),
        ("batch", "matching output and gate shapes"),
        ("heads", "matching output and gate shapes"),
        ("empty", "nonempty"),
        ("size", "matching output and gate shapes"),
        ("gate_dtype", "gate in the activation dtype or FP32"),
        ("gate_device", "same CUDA device"),
        ("weight_device", "RMSNorm weight"),
        ("weight_shape", "contiguous RMSNorm weight"),
        ("weight_stride", "contiguous RMSNorm weight"),
    ],
)
def test_validate_rejects_unsupported_inputs(norm_inputs, case, match):
    module, x, gate = norm_inputs
    if case == "deterministic":
        module.config.deterministic_mode = True
    elif case == "activation":
        module.activation = "gelu"
    elif case == "norm":
        module.out_norm = torch.nn.LayerNorm(128)
    elif case == "cpu":
        x = x.cpu()
    elif case == "dtype":
        x = x.float()
    elif case == "head_dim":
        x, gate = x[..., :63], gate[..., :63]
    elif case == "gate_rank":
        gate = gate.squeeze(0)
    elif case == "batch":
        gate = gate.expand(2, -1, -1, -1)
    elif case == "heads":
        gate = gate[:, :, :8, :]
    elif case == "empty":
        x, gate = x[:, :0], gate[:, :0]
    elif case == "size":
        gate = gate[:, :2]
    elif case == "gate_dtype":
        gate = gate.to(torch.int32)
    elif case == "gate_device":
        gate = gate.cpu()
    elif case == "weight_device":
        module.out_norm = module.out_norm.cpu()
    elif case == "weight_shape":
        module.out_norm.weight = torch.nn.Parameter(torch.ones(256, device=x.device))
    elif case == "weight_stride":
        module.out_norm.weight = torch.nn.Parameter(torch.ones(256, device=x.device)[::2])
    with pytest.raises(ValueError, match=match):
        gated_norm.validate_gated_norm(module, x, gate)
