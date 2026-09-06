# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The clamped SwiGLU path must reach a fused kernel, and keep its arithmetic.

Only the unclamped variant of ``swiglu_with_probs`` reached a fused kernel. With
a clamp value set -- which DeepSeek-V4 always does -- it fell back to an eager
chunk / clamp / silu / multiply / cast written out in fp32, on
``[tokens * topk, moe_ffn * 2]``, the largest activation in the model. Core's
implementations already take ``clamp_value``, so the fallback is gone.

These tests pin the arithmetic against that removed expression, since "fused"
and "clamped at the right operand" are not visible in any output shape.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.experts import swiglu_with_probs

pytestmark = [pytest.mark.mlite]

TOKENS, FFN = 8, 16


def _reference(y: torch.Tensor, probs: torch.Tensor | None, limit: float) -> torch.Tensor:
    """The eager expression this replaced, kept verbatim as the contract."""
    gate, up = y.chunk(2, dim=-1)
    up = torch.clamp(up.float(), min=-limit, max=limit)
    gate = torch.clamp(gate.float(), max=limit)
    out = torch.nn.functional.silu(gate) * up
    if probs is not None:
        out = out * probs
    return out.to(dtype=y.dtype)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("with_probs", [False, True])
def test_clamped_swiglu_matches_eager_reference(with_probs: bool) -> None:
    """Same values as the eager fallback, on the device that has the kernels."""
    torch.manual_seed(0)
    # Values well outside the clamp so the clamping actually participates; a test
    # that never saturates would pass with the clamp dropped entirely.
    y = torch.randn(TOKENS, FFN * 2, device="cuda", dtype=torch.bfloat16) * 8
    probs = (
        torch.rand(TOKENS, 1, device="cuda", dtype=torch.bfloat16) if with_probs else None
    )
    limit = 3.0
    assert y.float().abs().max() > limit, "fixture does not exercise the clamp"

    actual = swiglu_with_probs(y, probs, limit)
    expected = _reference(y, probs, limit)
    assert actual.shape == (TOKENS, FFN)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpus(1)
def test_clamped_and_unclamped_differ_on_saturating_input() -> None:
    """Guard the guard: the clamp must change the result on this fixture.

    Without this, a regression that passes ``clamp_value=None`` through would
    still satisfy the test above whenever the inputs happen not to saturate.
    """
    torch.manual_seed(0)
    y = torch.randn(TOKENS, FFN * 2, device="cuda", dtype=torch.bfloat16) * 8
    clamped = swiglu_with_probs(y, None, 3.0)
    unclamped = swiglu_with_probs(y, None, 0.0)
    assert not torch.allclose(clamped, unclamped, rtol=1e-2, atol=1e-2)


@pytest.mark.gpus(1)
def test_clamped_swiglu_is_differentiable() -> None:
    """Gradient must flow, and match the eager expression's."""
    torch.manual_seed(0)
    y = (torch.randn(TOKENS, FFN * 2, device="cuda", dtype=torch.float32) * 8).requires_grad_()
    ref_in = y.detach().clone().requires_grad_()

    swiglu_with_probs(y, None, 3.0).sum().backward()
    _reference(ref_in, None, 3.0).sum().backward()

    assert y.grad is not None and torch.isfinite(y.grad).all()
    torch.testing.assert_close(y.grad, ref_in.grad, rtol=1e-4, atol=1e-4)
