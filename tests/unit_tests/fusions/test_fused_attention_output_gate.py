# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bitwise parity of the fused elementwise output gate with the eager cast-before-multiply gate."""

import pytest
import torch

from megatron.core.fusions.fused_attention_output_gate import (
    HAVE_TRITON,
    can_use_fused_elementwise_attention_output_gate,
    fused_elementwise_attention_output_gate_before_cast,
)
from megatron.core.transformer.attention_output_gate import (
    apply_attention_output_gate,
    apply_elementwise_attention_output_gate_before_cast,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="needs a GPU and Triton"
)
requires_triton = pytest.mark.skipif(not HAVE_TRITON, reason="needs Triton")

DTYPES = [torch.float32, torch.bfloat16, torch.float16]
DTYPE_IDS = ["fp32", "bf16", "fp16"]
# The second shape leaves a partial block at the end so the masked tail is exercised.
SHAPES = [(8, 2, 3072), (5, 3, 1001)]
# Gate logits around the FP32 subnormal band of the sigmoid (-88.72 .. -87.34), saturation in
# BF16 (sigmoid(8) rounds to one), large magnitudes, signed zeros and infinities.
EXTREME_GATES = [-1e4, -89.0, -88.5, -88.0, -87.5, -87.25, -20.0, -8.0, -0.0, 0.0]
EXTREME_GATES += [8.0, 20.0, 87.5, 88.75, 1e4, float("inf"), float("-inf")]
# Activations: ordinary values, zeros, the FP16 and FP32 extremes and subnormals.
EXTREME_VALUES = [1.0, -1.0, 3.0, 0.0, -0.0, 65504.0, -65504.0, 3.0e38, -3.0e38, 1e-39, 6e-8]


def _eager(core_attn_out, gate):
    return apply_attention_output_gate(core_attn_out, gate, "elementwise", cast_mode="before")


def _inputs(shape, dtype, gate_dtype, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(0)
    core_attn_out = (torch.randn(shape, device=device, generator=generator) * 3).to(dtype)
    gate = (torch.randn(shape, device=device, generator=generator) * 4).to(gate_dtype)
    grad_output = (torch.randn(shape, device=device, generator=generator) * 2).to(dtype)
    return core_attn_out, gate, grad_output


def _extreme_inputs(dtype, gate_dtype, device="cuda"):
    values = torch.tensor(EXTREME_VALUES, device=device, dtype=torch.float32)
    gates = torch.tensor(EXTREME_GATES, device=device, dtype=torch.float32)
    core_attn_out = values[:, None].expand(-1, len(EXTREME_GATES)).contiguous().to(dtype)
    gate = gates[None, :].expand(len(EXTREME_VALUES), -1).contiguous().to(gate_dtype)
    grad_output = (values[:, None] * 0.5 - gates[None, :] / 64).contiguous().to(dtype)
    return core_attn_out, gate, grad_output


def _grads(fn, core_attn_out, gate, grad_output):
    x = core_attn_out.detach().clone().requires_grad_(True)
    g = gate.detach().clone().requires_grad_(True)
    output = fn(x, g)
    output.backward(grad_output)
    return output, x.grad, g.grad


def _assert_same_bits(actual, expected):
    """Bitwise equality, with NaNs of any payload treated as equal."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    int_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }
    a = actual.contiguous().view(int_dtype[actual.dtype])
    e = expected.contiguous().view(int_dtype[expected.dtype])
    same = (a == e) | (actual.isnan() & expected.isnan())
    assert bool(same.all()), (actual[~same][:8], expected[~same][:8])


@requires_gpu
@pytest.mark.parametrize("shape", SHAPES, ids=["full-blocks", "masked-tail"])
@pytest.mark.parametrize("gate_dtype", DTYPES, ids=[f"gate-{i}" for i in DTYPE_IDS])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_fused_gate_matches_eager_bitwise(shape, dtype, gate_dtype):
    core_attn_out, gate, grad_output = _inputs(shape, dtype, gate_dtype)
    output, grad_x, grad_gate = _grads(
        fused_elementwise_attention_output_gate_before_cast, core_attn_out, gate, grad_output
    )
    expected, expected_grad_x, expected_grad_gate = _grads(_eager, core_attn_out, gate, grad_output)
    assert output.dtype == expected.dtype == dtype
    assert torch.equal(output, expected)
    assert grad_x.dtype == dtype and torch.equal(grad_x, expected_grad_x)
    assert grad_gate.dtype == gate_dtype and torch.equal(grad_gate, expected_grad_gate)


@requires_gpu
@pytest.mark.parametrize("gate_dtype", DTYPES, ids=[f"gate-{i}" for i in DTYPE_IDS])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_fused_gate_matches_eager_bitwise_at_extreme_values(dtype, gate_dtype):
    # Subnormal sigmoid outputs (gate logits near -88), BF16 saturation, infinities, signed
    # zeros and FP16/FP32 overflow of the products must round and flush exactly as eagerly.
    core_attn_out, gate, grad_output = _extreme_inputs(dtype, gate_dtype)
    output, grad_x, grad_gate = _grads(
        fused_elementwise_attention_output_gate_before_cast, core_attn_out, gate, grad_output
    )
    expected, expected_grad_x, expected_grad_gate = _grads(_eager, core_attn_out, gate, grad_output)
    _assert_same_bits(output, expected)
    _assert_same_bits(grad_x, expected_grad_x)
    _assert_same_bits(grad_gate, expected_grad_gate)


@requires_gpu
def test_fused_gate_saves_only_its_inputs():
    core_attn_out, gate, _ = _inputs(SHAPES[0], torch.bfloat16, torch.bfloat16)
    x = core_attn_out.requires_grad_(True)
    g = gate.requires_grad_(True)
    output = fused_elementwise_attention_output_gate_before_cast(x, g)
    assert type(output.grad_fn).__name__ == "_FusedElementwiseGateBeforeCastBackward"
    saved = output.grad_fn.saved_tensors
    assert len(saved) == 2
    assert saved[0].data_ptr() == x.data_ptr()
    assert saved[1].data_ptr() == g.data_ptr()


@requires_gpu
@pytest.mark.parametrize("which", ["core_attn_out", "gate"])
def test_fused_gate_partial_gradients(which):
    core_attn_out, gate, grad_output = _inputs(SHAPES[1], torch.bfloat16, torch.float32)
    x = core_attn_out.detach().clone().requires_grad_(which == "core_attn_out")
    g = gate.detach().clone().requires_grad_(which == "gate")
    fused_elementwise_attention_output_gate_before_cast(x, g).backward(grad_output)
    _, expected_grad_x, expected_grad_gate = _grads(_eager, core_attn_out, gate, grad_output)
    if which == "core_attn_out":
        assert g.grad is None and torch.equal(x.grad, expected_grad_x)
    else:
        assert x.grad is None and torch.equal(g.grad, expected_grad_gate)


@requires_gpu
def test_fused_gate_handles_empty_tensors():
    empty_x = torch.empty(0, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    empty_g = torch.empty(0, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = fused_elementwise_attention_output_gate_before_cast(empty_x, empty_g)
    assert output.shape == (0, 3072)
    output.backward(torch.empty_like(output))
    assert empty_x.grad.shape == (0, 3072) and empty_g.grad.shape == (0, 3072)


@requires_gpu
def test_helper_dispatches_to_the_fused_kernel_when_asked():
    core_attn_out, gate, _ = _inputs(SHAPES[0], torch.bfloat16, torch.bfloat16)
    x = core_attn_out.requires_grad_(True)
    g = gate.requires_grad_(True)
    fused = apply_elementwise_attention_output_gate_before_cast(x, g, fusion=True)
    unfused = apply_elementwise_attention_output_gate_before_cast(x, g)
    assert type(fused.grad_fn).__name__ == "_FusedElementwiseGateBeforeCastBackward"
    assert type(unfused.grad_fn).__name__ == "MulBackward0"
    assert torch.equal(fused, unfused)


@requires_gpu
def test_helper_keeps_views_on_the_unfused_path():
    # Non-contiguous inputs would force contiguous copies into the autograd graph; the helper
    # keeps them on the eager composition instead, with identical values and grads.
    base_x, base_g, base_grad = _inputs((4, 2, 4096), torch.bfloat16, torch.bfloat16)
    x = base_x[..., :3072].requires_grad_(True)
    g = base_g[..., :3072].requires_grad_(True)
    assert not x.is_contiguous()
    assert not can_use_fused_elementwise_attention_output_gate(x, g)
    output = apply_elementwise_attention_output_gate_before_cast(x, g, fusion=True)
    assert type(output.grad_fn).__name__ == "MulBackward0"
    output.backward(base_grad[..., :3072])
    expected, expected_grad_x, expected_grad_gate = _grads(
        _eager, base_x[..., :3072], base_g[..., :3072], base_grad[..., :3072]
    )
    assert torch.equal(output, expected)
    assert x.grad.shape == x.shape and torch.equal(x.grad, expected_grad_x)
    assert g.grad.shape == g.shape and torch.equal(g.grad, expected_grad_gate)


@requires_gpu
def test_helper_rejects_unsupported_cuda_dtypes():
    x = torch.zeros(2, 1, 16, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="attention_output_gate_fusion requires"):
        apply_elementwise_attention_output_gate_before_cast(x, x, fusion=True)


def test_can_use_rejects_unsupported_inputs():
    cpu_x = torch.zeros(2, 1, 16)
    cpu_g = torch.zeros(2, 1, 16)
    assert not can_use_fused_elementwise_attention_output_gate(cpu_x, cpu_g)
    # CPU tensors keep working through the unfused path when fusion is requested.
    assert torch.equal(
        apply_elementwise_attention_output_gate_before_cast(cpu_x, cpu_g, fusion=True),
        _eager(cpu_x, cpu_g),
    )
    if torch.cuda.is_available():
        x = torch.zeros(2, 1, 16, device="cuda", dtype=torch.bfloat16)
        assert not can_use_fused_elementwise_attention_output_gate(x, x[..., :8])
        assert not can_use_fused_elementwise_attention_output_gate(x, x.to(torch.float64))
        assert not can_use_fused_elementwise_attention_output_gate(x.transpose(0, 2), x)


def _mla_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=128,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        attention_output_gate=True,
        attention_output_gate_fusion=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


@requires_triton
def test_config_accepts_the_fusion_for_the_elementwise_mla_gate():
    config = _mla_config()
    assert config.attention_output_gate_fusion
    assert config.gated_attention_proj_granularity == "elementwise"


def test_config_rejects_the_fusion_without_an_elementwise_mla_gate():
    with pytest.raises(ValueError, match="requires attention_output_gate"):
        _mla_config(attention_output_gate=False)
    with pytest.raises(ValueError, match="elementwise"):
        _mla_config(gated_attention_proj_granularity="headwise")
    with pytest.raises(ValueError, match="multi_latent_attention"):
        TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            attention_output_gate=True,
            attention_output_gate_fusion=True,
        )
