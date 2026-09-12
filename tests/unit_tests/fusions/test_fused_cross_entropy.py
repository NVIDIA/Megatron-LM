# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy


class _FakeTPGroup:
    """Single-rank stand-in for a tensor-parallel group so no distributed init is needed."""

    def rank(self):
        return 0

    def size(self):
        return 1


@pytest.fixture
def single_rank_all_reduce(monkeypatch):
    """With one rank, all-reduce is the identity; skip torch.distributed entirely."""

    def fake_all_reduce(tensor, op=None, group=None):
        return tensor

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_cross_entropy_grad_matches_logits_dtype(single_rank_all_reduce, logits_dtype):
    """The backward must emit the gradient in the dtype the logits arrived in.

    A gradient of a different dtype makes autograd insert a full-size cast of the [s, b, v]
    gradient before the output layer's backward. This pins the dtype for fp32, bf16 and fp16
    logits so a regression to a hardcoded dtype fails for the two it would get wrong.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    logits = torch.randn(4, 2, 8, dtype=logits_dtype, device=device, requires_grad=True)
    target = torch.randint(0, 8, (4, 2), device=device)

    loss = fused_vocab_parallel_cross_entropy(logits, target, _FakeTPGroup())
    loss.sum().backward()

    assert logits.grad is not None
    assert logits.grad.dtype == logits_dtype
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_fused_cross_entropy_matches_torch_reference(single_rank_all_reduce, logits_dtype):
    """Loss and gradient agree with torch.nn.functional.cross_entropy on a single rank."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    logits = torch.randn(4, 2, 8, dtype=logits_dtype, device=device, requires_grad=True)
    target = torch.randint(0, 8, (4, 2), device=device)
    # Snapshot before the fused call: for fp32 logits the fused path's `.float()` is a no-op,
    # and its in-place subtract-max / exp would otherwise corrupt the reference input.
    ref_logits = logits.detach().clone().float().requires_grad_(True)

    loss = fused_vocab_parallel_cross_entropy(logits, target, _FakeTPGroup())
    loss.sum().backward()

    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.reshape(-1, 8), target.reshape(-1), reduction="none"
    ).reshape(4, 2)
    ref_loss.sum().backward()

    tols = (
        dict(rtol=1e-5, atol=1e-5) if logits_dtype == torch.float32 else dict(rtol=2e-2, atol=1e-2)
    )
    torch.testing.assert_close(loss.float(), ref_loss, **tols)
    torch.testing.assert_close(logits.grad.float(), ref_logits.grad, **tols)
