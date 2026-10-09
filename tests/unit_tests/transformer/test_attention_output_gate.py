# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent output-gate math and exact upstream rounding regressions.

The eager cast-before-multiply formula follows the public Ling MLA gate:
https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/9a98e35fe1c9ee255f78dd64771c7ae15a799481/modeling_bailing_moe_v3.py#L709-L715
Cast-after-multiply, caller-owned final casting, and compiled primitives are pinned to MCore:
https://github.com/NVIDIA/Megatron-LM/tree/b4407209223d290bbf7d6f2167d4e2ae51b7a58d
The frozen primitives reproduce Attention.forward's scalar headwise gate,
Attention._apply_output_gate,
MultiLatentAttention._apply_mla_headwise_output_gate, and
KimiDeltaAttention._apply_gated_norm at that revision. They use independent
PyTorch operations and never call the production gate helper. torch.compile
matches the upstream jit_fuser in the supported CI environment. CUDA regression
cases preserve the original compiler rounding, rather than comparing compiled
BF16 arithmetic to an eager oracle. Scalar headwise gates intentionally share
MLA's layout flow; their eager regressions retain the original scalar oracle
for exact outputs, VJPs, and strides without requiring the old view-node order.
Numerical parity cases require real CUDA; shape-validation cases run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.attention_output_gate import (
    apply_attention_output_gate,
    validate_attention_output_gate_shapes,
)
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention
from tests.unit_tests.test_utilities import Utils

HY4_NUM_HEADS = 64
HY4_VALUE_HEAD_DIM = 256
LING_VALUE_HEAD_DIM = 128
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
        # boundary for elementwise MLA and scalar attention gates.
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


def _upstream_scalar_headwise_gate(values, logits):
    """Frozen b4407209 Attention.forward scalar headwise primitive (eager)."""
    gate_states = logits.view(*logits.shape[:2], -1, 1)
    values = values.view(*gate_states.shape[:3], -1)
    values = values * torch.sigmoid(gate_states.float()).to(values.dtype)
    return values.view(*gate_states.shape[:2], -1)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("batch_size", [2, 1], ids=["sbhd", "packed-restored"])
@pytest.mark.parametrize("noncontiguous_gate", [False, True], ids=["contiguous", "qkv-slice"])
def test_scalar_headwise_gate_preserves_upstream_output_and_vjp(
    dtype, batch_size, noncontiguous_gate, cuda_device
):
    shape = (2, batch_size, HY4_NUM_HEADS * HY4_VALUE_HEAD_DIM)
    values = torch.linspace(-2.0, 2.0, 2 * batch_size * shape[-1], device=cuda_device)
    values = values.to(dtype).view(shape)
    gate_width = HY4_NUM_HEADS + (16 if noncontiguous_gate else 0)
    logits = torch.linspace(-8.0, 8.0, 2 * batch_size * gate_width, device=cuda_device)
    logits = logits.to(dtype).view(2, batch_size, gate_width)[..., -HY4_NUM_HEADS:]
    assert logits.is_contiguous() != noncontiguous_gate

    def clone_input(tensor):
        # A plain clone would erase the gaps between QKV gate slices.
        return (
            torch.empty_strided(
                tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
            )
            .copy_(tensor)
            .requires_grad_(True)
        )

    actual_values, actual_logits = clone_input(values), clone_input(logits)
    reference_values, reference_logits = clone_input(values), clone_input(logits)
    assert actual_logits.stride() == reference_logits.stride() == logits.stride()
    actual = apply_attention_output_gate(
        actual_values, actual_logits, "headwise", cast_mode="before"
    )
    expected = _upstream_scalar_headwise_gate(reference_values, reference_logits)
    assert actual.shape == expected.shape == values.shape
    assert actual.dtype == expected.dtype == dtype
    assert actual.stride() == expected.stride()
    # Shared headwise layout changes the view-node order while retaining eager
    # execution and numerical compatibility with the original scalar gate.
    assert "CompiledFunctionBackward" not in _autograd_node_names(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.linspace(-1.0, 1.0, actual.numel(), device=cuda_device)
    gradient = gradient.to(dtype).view_as(actual)
    actual_grads = torch.autograd.grad(actual, (actual_values, actual_logits), gradient)
    expected_grads = torch.autograd.grad(expected, (reference_values, reference_logits), gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert actual_grads[1].flatten()[-1] != 0


@torch.compile
def _upstream_attention_gate(values, logits):
    """Frozen b4407209 Attention._apply_output_gate primitive."""
    value_dtype = values.dtype
    logits = logits.contiguous()
    logits = logits.view(*values.shape)
    values = values * torch.sigmoid(logits.float())
    return values.to(value_dtype)


@torch.compile
def _upstream_mla_headwise_gate(values, logits):
    """Frozen b4407209 MultiLatentAttention headwise primitive."""
    output_shape = values.shape
    values = values.view(*output_shape[:2], logits.size(-1), -1)
    scale = torch.sigmoid(logits.float()).to(values.dtype)
    values = values * scale.unsqueeze(-1)
    return values.reshape(output_shape)


@torch.compile
def _upstream_kda_gated_norm(values, logits, out_norm, value_head_dim):
    """Frozen b4407209 KimiDeltaAttention._apply_gated_norm primitive."""
    value_dtype = values.dtype
    values = values.reshape(-1, value_head_dim)
    values = out_norm(values)
    logits = logits.reshape(-1, value_head_dim)
    return (values * torch.sigmoid(logits.float())).to(value_dtype)


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
@pytest.mark.parametrize(
    "attention_class",
    [MultiLatentAttention, AbsorbedMLASelfAttention],
    ids=["ordinary", "absorbed"],
)
@pytest.mark.usefixtures("fresh_compile_cache")
def test_mla_headwise_gate_preserves_upstream_rounding(dtype, attention_class, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "headwise")

    def actual_gate(x, g):
        output = attention_class._apply_mla_headwise_output_gate(x, g)
        # Observe the actual autograd graph, without replacing the production
        # wrapper/compiler. Both MLA variants compile their private headwise
        # wrapper around the shared math helper in the supported CI environment.
        node_names = _autograd_node_names(output)
        assert "CompiledFunctionBackward" in node_names, node_names
        return output

    _assert_output_and_vjp(
        actual_gate, _upstream_mla_headwise_gate, values, logits, require_compiled=True
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_mla_elementwise_gate_matches_native(dtype, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "elementwise")
    attention = SimpleNamespace(
        config=SimpleNamespace(gated_attention_proj_granularity="elementwise")
    )
    _assert_output_and_vjp(
        lambda x, g: MultiLatentAttention._apply_mla_output_gate(attention, x, g),
        lambda x, g: _native_gate(x, g, "elementwise", "before"),
        values,
        logits,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("norm_dtype", DTYPES, ids=["norm-fp32", "norm-bf16"])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_kda_gated_norm_preserves_upstream_rounding(dtype, norm_dtype, cuda_device):
    values, logits = _inputs(dtype, cuda_device, "elementwise")
    actual_norm = torch.nn.RMSNorm(
        LING_VALUE_HEAD_DIM, eps=1e-6, dtype=norm_dtype, device=cuda_device
    )
    reference_norm = torch.nn.RMSNorm(
        LING_VALUE_HEAD_DIM, eps=1e-6, dtype=norm_dtype, device=cuda_device
    )
    with torch.no_grad():
        actual_norm.weight.copy_(torch.linspace(0.8, 1.2, LING_VALUE_HEAD_DIM, device=cuda_device))
    reference_norm.load_state_dict(actual_norm.state_dict())
    attention = SimpleNamespace(value_head_dim=LING_VALUE_HEAD_DIM, out_norm=actual_norm)
    gradients = _assert_output_and_vjp(
        lambda x, g: KimiDeltaAttention._apply_gated_norm(attention, x, g),
        lambda x, g: _upstream_kda_gated_norm(x, g, reference_norm, LING_VALUE_HEAD_DIM),
        values,
        logits,
        actual_parameters=(actual_norm.weight,),
        reference_parameters=(reference_norm.weight,),
        require_compiled=True,
    )
    assert torch.count_nonzero(gradients[2]) > 0


class _BF16OutputRMSNorm(torch.nn.RMSNorm):
    """Real RMSNorm with an explicit low-precision output boundary."""

    def forward(self, values):
        return super().forward(values).to(torch.bfloat16)


@pytest.mark.parametrize("norm_dtype", DTYPES, ids=["norm-fp32", "norm-bf16"])
@pytest.mark.usefixtures("fresh_compile_cache")
def test_kda_gated_norm_preserves_caller_cast_after_norm_downcast(norm_dtype, cuda_device):
    # The wrapper saves the FP32 input dtype before normalization. A norm that
    # returns BF16 must not cause the shared gate to round its FP32 product to
    # BF16 before the wrapper's original final cast back to FP32.
    values, logits = _inputs(torch.float32, cuda_device, "elementwise")
    logits.fill_(0.5)
    actual_norm = _BF16OutputRMSNorm(
        LING_VALUE_HEAD_DIM, eps=1e-6, dtype=norm_dtype, device=cuda_device
    )
    reference_norm = _BF16OutputRMSNorm(
        LING_VALUE_HEAD_DIM, eps=1e-6, dtype=norm_dtype, device=cuda_device
    )
    with torch.no_grad():
        actual_norm.weight.copy_(torch.linspace(0.8, 1.2, LING_VALUE_HEAD_DIM, device=cuda_device))
    reference_norm.load_state_dict(actual_norm.state_dict())
    attention = SimpleNamespace(value_head_dim=LING_VALUE_HEAD_DIM, out_norm=actual_norm)
    gradients = _assert_output_and_vjp(
        lambda x, g: KimiDeltaAttention._apply_gated_norm(attention, x, g),
        lambda x, g: _upstream_kda_gated_norm(x, g, reference_norm, LING_VALUE_HEAD_DIM),
        values,
        logits,
        actual_parameters=(actual_norm.weight,),
        reference_parameters=(reference_norm.weight,),
        require_compiled=True,
    )
    assert torch.count_nonzero(gradients[2]) > 0
    with torch.no_grad():
        norm_output = reference_norm(values.reshape(-1, LING_VALUE_HEAD_DIM))
        assert norm_output.dtype == torch.bfloat16
        unrounded_product = norm_output * torch.sigmoid(logits.reshape_as(norm_output).float())
        assert unrounded_product.dtype == torch.float32
        prematurely_rounded = unrounded_product.to(torch.bfloat16).to(values.dtype)
        assert not torch.equal(unrounded_product, prematurely_rounded)
