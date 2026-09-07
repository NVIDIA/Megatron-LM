# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gradients that go straight into ``main_grad`` must equal the ones that did not.

Three paths in this branch stopped materialising a weight gradient and folding
it in with ``main_grad.add_(grad)``, and started accumulating inside the
backward instead: the vocabulary projection, the embedding table, and the
attention projections. Together they were 17.6 + 8.6 + 8.6 ms per step of pure
addition, and removing them is what took this configuration below Megatron
Core's step time.

The risk they carry is not visible in a shape or a loss. A gradient that is
accumulated once instead of twice, scattered to the wrong rows, or rounded to
bf16 on the way in still produces finite numbers of the right size, and the
step still runs. Two runs of the same tree with the same seed differ by 0.006
in the loss, so the loss cannot be the judge here; these compare the
accumulator itself, over more than one microbatch, against the arithmetic that
was removed.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.parallel.linear import (
    AccumulatingLinear,
    _EmbeddingAccumulatingIntoMainGrad,
)

pytestmark = [pytest.mark.mlite]

# ``main_grad`` only ever exists on CUDA -- the distributed optimizer allocates
# it there -- and Core's accumulating linear has no CPU kernel, so these run on
# the device the paths they cover actually run on.
DEVICE = "cuda"

TOKENS, IN, OUT, VOCAB = 16, 8, 12, 32
MICROBATCHES = 3


def _reference_accumulation(module, inputs) -> torch.Tensor:
    """What DDP used to do: dense grad per microbatch, added into fp32."""
    accumulated = torch.zeros_like(module.weight, dtype=torch.float32)
    for x in inputs:
        module(x).sum().backward()
        accumulated += module.weight.grad.data.float()
        module.weight.grad = None
    return accumulated


@pytest.mark.gpus(1)
def test_linear_accumulates_the_same_gradient_it_used_to_add() -> None:
    """``AccumulatingLinear`` must match the add it replaced, over 3 microbatches."""
    torch.manual_seed(0)
    inputs = [torch.randn(TOKENS, IN, device=DEVICE) for _ in range(MICROBATCHES)]

    reference = AccumulatingLinear(IN, OUT, bias=False).to(DEVICE)
    expected = _reference_accumulation(reference, inputs)
    assert not hasattr(reference.weight, "main_grad"), "reference must take the stock path"

    fused = AccumulatingLinear(IN, OUT, bias=False).to(DEVICE)
    with torch.no_grad():
        fused.weight.copy_(reference.weight)
    fused.weight.main_grad = torch.zeros(OUT, IN, dtype=torch.float32, device=DEVICE)
    fused.weight.grad_added_to_main_grad = False
    for x in inputs:
        fused(x).sum().backward()

    assert fused.weight.main_grad.abs().max() > 0, "fused path produced no gradient"
    torch.testing.assert_close(fused.weight.main_grad, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpus(1)
def test_linear_without_main_grad_keeps_the_stock_path() -> None:
    """Guard the gate: no accumulator means ``param.grad``, not a silent no-op.

    An optimizer that allocates no ``main_grad`` must still see a gradient. If
    the gate inverted, training would proceed with weights that never update --
    finite, correctly shaped, and wrong.
    """
    torch.manual_seed(0)
    module = AccumulatingLinear(IN, OUT, bias=False).to(DEVICE)
    module(torch.randn(TOKENS, IN, device=DEVICE)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.abs().max() > 0


@pytest.mark.gpus(1)
def test_embedding_scatters_to_the_same_rows_it_used_to_add() -> None:
    """The scatter must land the same values in the same rows as the dense path."""
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, IN, device=DEVICE, requires_grad=True)
    ids = [torch.randint(0, VOCAB, (TOKENS,), device=DEVICE) for _ in range(MICROBATCHES)]

    expected = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    for i in ids:
        out = weight[i]
        out.sum().backward()
        expected += weight.grad.data.float()
        weight.grad = None

    fused = weight.detach().clone().requires_grad_()
    fused.main_grad = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    fused.grad_added_to_main_grad = False
    for i in ids:
        _EmbeddingAccumulatingIntoMainGrad.apply(fused, i).sum().backward()

    assert fused.main_grad.abs().max() > 0, "scatter produced no gradient"
    torch.testing.assert_close(fused.main_grad, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.gpus(1)
def test_embedding_scatter_is_row_selective() -> None:
    """Guard the guard: rows never looked up must stay exactly zero.

    A scatter that ignored the indices -- or used the wrong axis -- would still
    produce a finite accumulator of the right shape, and the test above would
    pass on a fixture where every row happened to be touched.
    """
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, IN, device=DEVICE, requires_grad=True)
    weight.main_grad = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    weight.grad_added_to_main_grad = False
    ids = torch.zeros(4, dtype=torch.long, device=DEVICE)  # only row 0

    _EmbeddingAccumulatingIntoMainGrad.apply(weight, ids).sum().backward()

    assert weight.main_grad[0].abs().max() > 0
    assert torch.equal(weight.main_grad[1:], torch.zeros(VOCAB - 1, IN, device=DEVICE))


@pytest.mark.gpus(1)
def test_embedding_reports_a_gradient_for_ddp() -> None:
    """DDP asserts a gradient exists whenever overlap_grad_reduce is on.

    Returning ``None`` here is not a missing optimisation but a crash, which is
    why Core hands back a placeholder from its own fused path. This pins that
    the placeholder is returned and the flag is set, so the hook sees the
    parameter and then skips the add.
    """
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, IN, device=DEVICE, requires_grad=True)
    weight.main_grad = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    weight.grad_added_to_main_grad = False

    _EmbeddingAccumulatingIntoMainGrad.apply(weight, torch.randint(0, VOCAB, (TOKENS,), device=DEVICE)).sum().backward()

    assert weight.grad_added_to_main_grad is True
    assert weight.grad is not None, "DDP would assert on a missing gradient"
