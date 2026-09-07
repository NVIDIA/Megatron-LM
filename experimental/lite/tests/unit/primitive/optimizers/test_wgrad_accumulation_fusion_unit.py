# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight gradients must land in ``main_grad``, and land there unchanged.

Megatron-Core's DDP folds each weight gradient into the fp32 accumulator one
parameter at a time (``param.main_grad.add_(param.grad.data)``). Transformer
Engine can do it inside the wgrad GEMM instead, which is what Core itself does
everywhere; lite never switched it on. ``_enable_wgrad_accumulation_fusion``
flips it after the DDP wrap, when ``main_grad`` first exists.

Two things need pinning. The selection has to be self-gating -- a module whose
weights have no ``main_grad`` must be left alone, or TE will look for a buffer
that was never allocated. And the accumulated values have to match the add that
was removed, which is not visible in any shape: a fusion that silently dropped
a contribution, or accumulated in bf16, would still produce finite gradients of
the right size.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.optimizers.megatron_wrap import _enable_wgrad_accumulation_fusion

pytestmark = [pytest.mark.mlite]


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
    """Same accumulated gradient as ``main_grad.add_(grad)``, over two microbatches.

    One microbatch would not exercise accumulation at all, so this runs two and
    compares against the explicit add on an identical unfused module. The
    tolerance is set at bf16 GEMM resolution; the failure this guards against
    (a dropped or bf16-rounded accumulation) is far larger.
    """
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
