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
    _assert_gradients_close(
        torch.autograd.grad(actual, inputs, dy),
        torch.autograd.grad(expected, inputs, dy),
        atol=0.05,
    )


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


@pytest.mark.parametrize(
    "case,match",
    [
        ("deterministic", "deterministic_mode=False"),
        ("cp", "context_parallel_size=1"),
        ("activation", "SiLU/Swish"),
        ("norm", "RMSNorm output"),
        ("cpu", "CUDA BF16"),
        ("dtype", "CUDA BF16"),
        ("x_stride", "contiguous core attention"),
        ("head_dim", "head dimension 128"),
        ("gate_rank", "gate shape"),
        ("batch", "gate shape"),
        ("heads", "gate shape"),
        ("empty", "nonempty"),
        ("size", "equal element counts"),
        ("gate_element_stride", "strides 1 and 128"),
        ("gate_head_stride", "strides 1 and 128"),
        ("gate_device", "same CUDA device"),
        ("weight_device", "RMSNorm weight"),
        ("weight_shape", "128-element"),
        ("weight_stride", "contiguous 128-element"),
    ],
)
def test_validate_rejects_unsupported_inputs(norm_inputs, case, match):
    module, x, gate = norm_inputs
    if case == "deterministic":
        module.config.deterministic_mode = True
    elif case == "cp":
        module.cp_size = 2
    elif case == "activation":
        module.activation = "gelu"
    elif case == "norm":
        module.out_norm = torch.nn.LayerNorm(128)
    elif case == "cpu":
        x = x.cpu()
    elif case == "dtype":
        x = x.float()
    elif case == "x_stride":
        x = torch.empty((1, 3, 16, 256), device=x.device, dtype=x.dtype)[..., ::2]
    elif case == "head_dim":
        x = x[..., :64].contiguous()
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
    elif case == "gate_element_stride":
        gate = torch.empty((1, 3, 16, 256), device=x.device, dtype=x.dtype)[..., ::2]
    elif case == "gate_head_stride":
        gate = torch.empty((1, 3, 32, 128), device=x.device, dtype=x.dtype)[:, :, ::2]
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
