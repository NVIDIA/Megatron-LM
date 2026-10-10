# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact output and VJP replay through the actual output-gate JIT entry points.

Regular Attention compiles the cast-after-multiply elementwise gate. Absorbed
MLA compiles the cast-before-multiply headwise gate, including the reduction
over the value channels in its gate gradient. The shared math is exercised
through these production wrappers, without replacing their compiler or kernels.
"""

import pytest
import torch

from megatron.core.transformer.attention import Attention
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

HY4_NUM_HEADS = 64
HY4_VALUE_HEAD_DIM = 256
TOKENS = 128


def _autograd_node_names(output):
    pending = [output.grad_fn]
    visited = set()
    names = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        names.add(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return names


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("entry_point", ["regular_elementwise", "absorbed_headwise"])
def test_compiled_attention_output_gate_replays_bit_exactly(dtype, entry_point):
    assert torch.cuda.is_available(), "Output-gate determinism tests require a real GPU."
    seeded()
    values = torch.randn(
        TOKENS,
        1,
        HY4_NUM_HEADS * HY4_VALUE_HEAD_DIM,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    if entry_point == "regular_elementwise":
        # Match the QKV-derived 4D gate consumed by Attention's original wrapper.
        logits = torch.randn(
            TOKENS,
            1,
            HY4_NUM_HEADS,
            HY4_VALUE_HEAD_DIM,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )

        def compiled_gate(x, g):
            return Attention._apply_output_gate(None, x, g)

    else:
        # Preserve a noncontiguous projected-gate layout across replays.
        logits = torch.randn(TOKENS, 1, HY4_NUM_HEADS + 16, device="cuda", dtype=dtype)[
            ..., -HY4_NUM_HEADS:
        ]
        logits = logits.detach().requires_grad_(True)
        assert not logits.is_contiguous()
        compiled_gate = AbsorbedMLASelfAttention._apply_mla_headwise_output_gate

    def run_gate(x, g):
        output = compiled_gate(x, g)
        nodes = _autograd_node_names(output)
        assert "CompiledFunctionBackward" in nodes, nodes
        return output

    upstream_gradient = torch.randn_like(values)
    outputs, gradients = assert_replays_bit_exact(
        run_gate,
        {"x": values, "g": logits},
        grad_outputs={"out": upstream_gradient},
        replays=3,
        contention=True,
        what=entry_point,
    )
    assert outputs["out"].shape == values.shape
    assert outputs["out"].dtype == dtype
    # Prevent replay from succeeding without checking either differentiable input.
    assert set(gradients) == {"in.x", "in.g"}
    for name, tensor in (("in.x", values), ("in.g", logits)):
        assert gradients[name].shape == tensor.shape
        assert gradients[name].dtype == tensor.dtype
        assert torch.count_nonzero(gradients[name]) > 0
