# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent output-gate math and exact upstream rounding regressions.

The eager cast-before-multiply formula follows the public Ling MLA gate:
https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/9a98e35fe1c9ee255f78dd64771c7ae15a799481/modeling_bailing_moe_v3.py#L709-L715
The frozen regular Attention primitive is pinned to the main baseline:
https://github.com/NVIDIA/Megatron-LM/blob/724886e5c321bda42f7b75f972924b308164803d/megatron/core/transformer/attention.py#L1711-L1718
The new absorbed MLA headwise gate follows the frozen dev MLA primitive:
https://github.com/NVIDIA/Megatron-LM/blob/b4407209223d290bbf7d6f2167d4e2ae51b7a58d/megatron/core/transformer/multi_latent_attention.py
References use independent PyTorch operations and never call the production
helper. torch.compile matches upstream jit_fuser in the supported CI environment,
retaining compiler rounding when comparing FP32/BF16 outputs and VJPs.
Main supports the gate in absorbed DSA and existing regular Attention; ordinary
MLA, scalar Attention headwise gates, and KDA are outside this main counterpart.
Numerical parity cases require real CUDA; validation cases run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.attention import Attention
from megatron.core.transformer.attention_output_gate import (
    apply_attention_output_gate,
    validate_attention_output_gate_shapes,
)
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from tests.unit_tests.test_utilities import Utils

HY4_NUM_HEADS = 64
HY4_VALUE_HEAD_DIM = 256
DTYPES = [torch.float32, torch.bfloat16]


def _cuda_device():
    assert torch.cuda.is_available(), "CUDA regression tests require a real GPU."
    torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
    return torch.device("cuda")


@pytest.fixture
def cuda_device():
    return _cuda_device()


@pytest.fixture
def fresh_compile_cache():
    # Other attention tests share these wrappers and can exhaust Dynamo's
    # recompilation limit. Both sides of compiled parity must actually compile.
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def _inputs(dtype, device, granularity):
    shape = (2, 1, HY4_NUM_HEADS * HY4_VALUE_HEAD_DIM)
    values = torch.linspace(-2.0, 2.0, 2 * shape[-1], device=device).to(dtype).view(shape)
    gate_width = shape[-1] if granularity == "elementwise" else HY4_NUM_HEADS
    logits = torch.linspace(-8.0, 8.0, 2 * gate_width, device=device).to(dtype)
    return values, logits.view(2, 1, gate_width)


def _native_gate(values, logits, granularity, cast_mode):
    """Evaluate the independently specified FP32 sigmoid and dtype boundary."""
    scale = torch.sigmoid(logits.float())
    if cast_mode == "before":
        scale = scale.to(values.dtype)
    if granularity == "headwise":
        heads = logits.shape[-1]
        value_heads = values.unflatten(-1, (heads, values.shape[-1] // heads))
        output = (value_heads * scale.unsqueeze(-1)).flatten(-2)
    else:
        output = values * scale
    return output.to(values.dtype) if cast_mode == "after" else output


def _autograd_node_names(output):
    """Observe eager/compiled execution without replacing production code."""
    pending = [output.grad_fn]
    visited = set()
    node_names = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        node_names.add(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return node_names


def _assert_output_and_vjp(
    actual_fn,
    reference_fn,
    values,
    logits,
    actual_parameters=(),
    reference_parameters=(),
    expected_output_dtype=None,
    require_compiled=False,
):
    actual_values = values.detach().clone().requires_grad_(True)
    actual_logits = logits.detach().clone().requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    actual = actual_fn(actual_values, actual_logits)
    expected = reference_fn(reference_values, reference_logits)
    if require_compiled:
        for name, output in (("production", actual), ("reference", expected)):
            node_names = _autograd_node_names(output)
            assert "CompiledFunctionBackward" in node_names, (name, node_names)
    gradient = torch.linspace(-1.0, 1.0, actual.numel(), device=actual.device)
    gradient = gradient.to(actual.dtype).view_as(actual)
    actual_grads = torch.autograd.grad(
        actual, (actual_values, actual_logits, *actual_parameters), gradient
    )
    expected_grads = torch.autograd.grad(
        expected, (reference_values, reference_logits, *reference_parameters), gradient
    )
    assert actual.dtype == (
        values.dtype if expected_output_dtype is None else expected_output_dtype
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(actual_grads) == len(expected_grads)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    return actual_grads


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("granularity", ["elementwise", "headwise"])
@pytest.mark.parametrize("cast_mode", ["before", "after", "none"])
def test_attention_output_gate_matches_native(dtype, granularity, cast_mode, cuda_device):
    values, logits = _inputs(dtype, cuda_device, granularity)

    def actual_gate(x, g):
        output = apply_attention_output_gate(x, g, granularity, cast_mode=cast_mode)
        # The shared math helper is eager, preserving the explicit dtype
        # boundary for the absorbed MLA elementwise gate.
        assert "CompiledFunctionBackward" not in _autograd_node_names(output)
        if granularity == "elementwise":
            expected = _native_gate(x, g, granularity, cast_mode)
            assert type(output.grad_fn).__name__ == type(expected.grad_fn).__name__
        return output

    gradients = _assert_output_and_vjp(
        actual_gate,
        lambda x, g: _native_gate(x, g, granularity, cast_mode),
        values,
        logits,
        expected_output_dtype=torch.float32 if cast_mode == "none" else dtype,
    )
    # Sigmoid(8) rounds to BF16 one, but its saved FP32 value must retain a
    # nonzero VJP. This catches accidentally switching to activation sigmoid.
    assert logits.flatten()[-1] == 8
    assert gradients[1].flatten()[-1] != 0


@pytest.mark.parametrize("granularity", ["elementwise", "headwise"])
def test_cast_order_changes_bf16_forward(granularity, cuda_device):
    values, logits = _inputs(torch.bfloat16, cuda_device, granularity)
    values.fill_(1.5)
    logits.fill_(0.5)
    before = apply_attention_output_gate(values, logits, granularity, cast_mode="before")
    after = apply_attention_output_gate(values, logits, granularity, cast_mode="after")
    # BF16(sigmoid(0.5)) is 0.62109375. Its product with 1.5 is a BF16
    # midpoint rounded to 0.9296875; multiplying the FP32 sigmoid first
    # instead rounds to 0.93359375.
    torch.testing.assert_close(before, torch.full_like(values, 0.9296875), atol=0, rtol=0)
    torch.testing.assert_close(after, torch.full_like(values, 0.93359375), atol=0, rtol=0)
    assert not torch.equal(before, after)


@pytest.mark.parametrize("cast_mode", ["before", "after", "none"])
def test_packed_headwise_gate_matches_native(cast_mode, cuda_device):
    values, logits = _inputs(torch.bfloat16, cuda_device, "headwise")
    values = values.squeeze(1).transpose(0, 1).contiguous().transpose(0, 1)
    logits = logits.squeeze(1).transpose(0, 1).contiguous().transpose(0, 1)
    assert values.ndim == logits.ndim == 2
    assert not values.is_contiguous() and not logits.is_contiguous()
    _assert_output_and_vjp(
        lambda x, g: apply_attention_output_gate(x, g, "headwise", cast_mode=cast_mode),
        lambda x, g: _native_gate(x, g, "headwise", cast_mode),
        values,
        logits,
        expected_output_dtype=torch.float32 if cast_mode == "none" else values.dtype,
    )


@pytest.mark.parametrize(
    "value_shape,gate_shape,granularity",
    [
        ((2, 1, 64), (3, 1, 64), "elementwise"),
        ((2, 1, 64), (2, 1, 16), "elementwise"),
        ((2, 1, 64), (2, 64), "elementwise"),
        ((2, 1, 65), (2, 1, 16), "headwise"),
        ((2, 1, 64), (2, 1, 0), "headwise"),
        ((2, 1, 64), (2, 1, 16), "unsupported"),
        ((), (), "elementwise"),
        ((2, 1, 64), (), "headwise"),
    ],
)
def test_attention_output_gate_rejects_invalid_shapes(value_shape, gate_shape, granularity):
    values, logits = torch.empty(value_shape), torch.empty(gate_shape)
    with pytest.raises(ValueError):
        validate_attention_output_gate_shapes(values, logits, granularity)
    for cast_mode in ("before", "after", "none"):
        with pytest.raises(ValueError):
            apply_attention_output_gate(values, logits, granularity, cast_mode=cast_mode)


def test_attention_output_gate_rejects_invalid_cast_mode():
    values = torch.ones(2, 1, 64)
    for granularity, gate_width in (("elementwise", 64), ("headwise", 8)):
        logits = torch.zeros(2, 1, gate_width)
        for cast_mode in ("unsupported", None, True):
            with pytest.raises(ValueError, match="cast"):
                apply_attention_output_gate(values, logits, granularity, cast_mode=cast_mode)


@torch.compile
def _upstream_attention_gate(values, logits):
    """Frozen 724886e5 main Attention._apply_output_gate primitive."""
    value_dtype = values.dtype
    logits = logits.contiguous()
    logits = logits.view(*values.shape)
    values = values * torch.sigmoid(logits.float())
    return values.to(value_dtype)


@torch.compile
def _upstream_mla_headwise_gate(values, logits):
    """Frozen b4407209 dev MultiLatentAttention headwise primitive."""
    output_shape = values.shape
    values = values.view(*output_shape[:2], logits.size(-1), -1)
    scale = torch.sigmoid(logits.float()).to(values.dtype)
    values = values * scale.unsqueeze(-1)
    return values.reshape(output_shape)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("gate_layout", ["packed", "noncontiguous"])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_regular_attention_gate_preserves_upstream_rounding(dtype, gate_layout, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "elementwise")
    if gate_layout == "packed":
        logits = logits.view(2, 1, HY4_NUM_HEADS, HY4_VALUE_HEAD_DIM)
    else:
        logits = logits.transpose(0, -1).contiguous().transpose(0, -1)
        assert not logits.is_contiguous()
    _assert_output_and_vjp(
        lambda x, g: Attention._apply_output_gate(None, x, g),
        _upstream_attention_gate,
        values,
        logits,
        require_compiled=True,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_absorbed_mla_headwise_gate_preserves_upstream_rounding(dtype, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "headwise")

    def actual_gate(x, g):
        output = AbsorbedMLASelfAttention._apply_mla_headwise_output_gate(x, g)
        # Observe the actual autograd graph, without replacing the production
        # wrapper/compiler. Absorbed MLA compiles its private headwise wrapper
        # around the shared math helper in the supported CI environment.
        node_names = _autograd_node_names(output)
        assert "CompiledFunctionBackward" in node_names, node_names
        return output

    _assert_output_and_vjp(
        actual_gate, _upstream_mla_headwise_gate, values, logits, require_compiled=True
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_absorbed_mla_elementwise_gate_matches_native(dtype, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "elementwise")
    attention = SimpleNamespace(
        config=SimpleNamespace(gated_attention_proj_granularity="elementwise")
    )
    _assert_output_and_vjp(
        lambda x, g: AbsorbedMLASelfAttention._apply_mla_output_gate(attention, x, g),
        lambda x, g: _native_gate(x, g, "elementwise", "before"),
        values,
        logits,
    )
