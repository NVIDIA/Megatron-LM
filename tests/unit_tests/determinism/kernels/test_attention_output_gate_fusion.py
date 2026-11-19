# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay coverage for the fused elementwise MLA output gate (Triton)."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions import fused_attention_output_gate
from megatron.core.fusions.fused_attention_output_gate import (
    HAVE_TRITON,
    fused_elementwise_attention_output_gate_before_cast,
)
from megatron.core.transformer.attention_output_gate import (
    apply_attention_output_gate,
    apply_elementwise_attention_output_gate_before_cast,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="needs a GPU and Triton"
)

# The second shape leaves a partial block so the masked tail of both kernels is replayed too.
SHAPES = [(8, 2, 3072), (5, 3, 1001)]
SHAPE_IDS = ["full-blocks", "partial-block"]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
DTYPE_IDS = ["bf16", "fp16", "fp32"]


def _gate_dtype(dtype):
    # The MLA path keeps the gate logits in FP32 next to BF16 activations; the other dtypes
    # cover gates in the activation dtype.
    return torch.float32 if dtype is torch.bfloat16 else dtype


def _inputs(shape, dtype):
    seeded()
    core_attn_out = (torch.randn(shape, device="cuda") * 3).to(dtype).requires_grad_()
    gate = (torch.randn(shape, device="cuda") * 4).to(_gate_dtype(dtype)).requires_grad_()
    grad_output = (torch.randn(shape, device="cuda") * 2).to(dtype)
    return {"core_attn_out": core_attn_out, "gate": gate}, {"out": grad_output}


def _eager(core_attn_out, gate):
    return apply_attention_output_gate(core_attn_out, gate, "elementwise", cast_mode="before")


@pytest.mark.parametrize("contention", [False, True], ids=["serial", "contention"])
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_fused_output_gate_kernels_replay(shape, dtype, contention):
    """Forward and backward kernels replay bit-exactly and match the eager gate bitwise."""
    inputs, grad_outputs = _inputs(shape, dtype)
    out, grads = assert_replays_bit_exact(
        fused_elementwise_attention_output_gate_before_cast,
        inputs,
        grad_outputs=grad_outputs,
        contention=contention,
        what="fused_elementwise_attention_output_gate_before_cast",
    )
    assert out["out"].dtype == dtype and tuple(out["out"].shape) == shape
    # The harness names input gradients after the flattened input dict.
    assert set(grads) == {"in.core_attn_out", "in.gate"}, sorted(grads)

    x = inputs["core_attn_out"].detach().clone().requires_grad_()
    g = inputs["gate"].detach().clone().requires_grad_()
    eager_out = _eager(x, g)
    eager_out.backward(grad_outputs["out"])
    assert bytes_equal(out["out"], eager_out.detach())
    assert bytes_equal(grads["in.core_attn_out"], x.grad)
    assert bytes_equal(grads["in.gate"], g.grad)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_module_entry_dispatches_to_the_kernels_and_replays(dtype):
    """The config-switch path (helper with fusion=True) takes the kernels on every replay."""
    inputs, grad_outputs = _inputs(SHAPES[1], dtype)

    def entry(core_attn_out, gate):
        return apply_elementwise_attention_output_gate_before_cast(core_attn_out, gate, fusion=True)

    with patch.object(
        fused_attention_output_gate,
        "fused_elementwise_attention_output_gate_before_cast",
        wraps=fused_attention_output_gate.fused_elementwise_attention_output_gate_before_cast,
    ) as fused:
        assert_replays_bit_exact(
            entry, inputs, grad_outputs=grad_outputs, what="apply_elementwise_attention_output_gate"
        )
    assert fused.call_count == 3
