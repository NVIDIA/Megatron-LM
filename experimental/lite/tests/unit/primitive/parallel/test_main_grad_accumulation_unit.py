# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gradients that go straight into ``main_grad`` must equal the ones that did not."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.optimizers.megatron_wrap import _enable_wgrad_accumulation_fusion
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
    """Guard the gate: no accumulator means ``param.grad``, not a silent no-op."""
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
    """Guard the guard: rows never looked up must stay exactly zero."""
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
    """DDP asserts a gradient exists whenever overlap_grad_reduce is on."""
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, IN, device=DEVICE, requires_grad=True)
    weight.main_grad = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    weight.grad_added_to_main_grad = False

    _EmbeddingAccumulatingIntoMainGrad.apply(weight, torch.randint(0, VOCAB, (TOKENS,), device=DEVICE)).sum().backward()

    assert weight.grad_added_to_main_grad is True
    assert weight.grad is not None, "DDP would assert on a missing gradient"




class _FakeTELinear(torch.nn.Module):
    """Stands in for a TE linear: carries the flag and a weight, nothing else."""

    def __init__(self, with_main_grad: bool) -> None:
        super().__init__()
        self.fuse_wgrad_accumulation = False
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        if with_main_grad:
            self.weight.main_grad = torch.zeros(2, 2, dtype=torch.float32)


class _Plain(torch.nn.Module):
    """A module with a weight but no TE flag -- must not grow one."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.weight.main_grad = torch.zeros(2, 2, dtype=torch.float32)


def test_only_switches_modules_whose_weights_have_main_grad() -> None:
    chunk = torch.nn.Module()
    chunk.ready = _FakeTELinear(with_main_grad=True)
    chunk.not_ready = _FakeTELinear(with_main_grad=False)
    chunk.plain = _Plain()

    assert _enable_wgrad_accumulation_fusion([chunk]) == 1
    assert chunk.ready.fuse_wgrad_accumulation is True
    # No main_grad to accumulate into: TE would dereference a buffer that the
    # optimizer never allocated, so this one has to stay off.
    assert chunk.not_ready.fuse_wgrad_accumulation is False
    assert not hasattr(chunk.plain, "fuse_wgrad_accumulation")


def test_is_idempotent_and_reports_nothing_switched_on_a_second_pass() -> None:
    """Re-wrapping must not double-count; the caller reads the return value."""
    chunk = torch.nn.Module()
    chunk.ready = _FakeTELinear(with_main_grad=True)
    assert _enable_wgrad_accumulation_fusion([chunk]) == 1
    assert _enable_wgrad_accumulation_fusion([chunk]) == 0


@pytest.mark.gpus(1)
def test_fused_accumulation_matches_the_add_it_replaces() -> None:
    """Same accumulated gradient as ``main_grad.add_(grad)``, over two microbatches."""
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    shape = (64, 64)
    inputs = [torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    reference = te.Linear(*shape, bias=False, params_dtype=torch.bfloat16, device="cuda")
    fused = te.Linear(*shape, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        fused.weight.copy_(reference.weight)

    accumulated = torch.zeros(shape, device="cuda", dtype=torch.float32)
    for x in inputs:
        reference(x).sum().backward()
        accumulated.add_(reference.weight.grad.data)
        reference.weight.grad = None

    fused.weight.main_grad = torch.zeros(shape, device="cuda", dtype=torch.float32)
    fused.weight.grad_added_to_main_grad = False
    chunk = torch.nn.Module()
    chunk.fc = fused
    assert _enable_wgrad_accumulation_fusion([chunk]) == 1
    for x in inputs:
        fused(x).sum().backward()

    assert fused.weight.main_grad.abs().max() > 0, "fusion produced no gradient at all"
    torch.testing.assert_close(fused.weight.main_grad, accumulated, rtol=2e-2, atol=2e-2)
