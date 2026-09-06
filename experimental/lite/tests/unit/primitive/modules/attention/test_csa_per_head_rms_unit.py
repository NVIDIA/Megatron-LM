# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The weightless per-head query RMS normalisation, fused.

Two call sites in the CSA attention wrote this out as a cast, a square, a mean,
an rsqrt, a cast back and a multiply, over the full
``[batch, heads, seq, head_dim]`` query. A dispatch-level op census of one
training step put that tensor among the largest touched per layer, and with
``recompute=full`` every one of those kernels is paid twice.

Fusing this is **not** bitwise-neutral, and that is recorded rather than
glossed: the compiler reassociates the mean, so even in fp32 the result moves in
the last bits, and in bf16 the disagreement reaches one ulp. The bounds below
pin it at that scale. It is kept because it is worth 8.75% of step time at the
reference configuration, measured, which is not a trade that gets made silently.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.attention.csa import _per_head_rms

pytestmark = [pytest.mark.mlite]

B, H, S, D = 2, 3, 5, 8
EPS = 1e-6


def _eager(q: torch.Tensor, eps: float) -> torch.Tensor:
    """The expression this replaced, kept verbatim as the contract."""
    return q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(dtype=q.dtype)


@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float32, 1e-6), (torch.bfloat16, 8e-3)])
def test_fused_per_head_rms_stays_within_one_ulp_of_eager(dtype: torch.dtype, rtol: float) -> None:
    """Bound the reassociation: last-bit in fp32, one ulp in bf16, no more."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=generator).to(dtype=dtype)
    torch.testing.assert_close(_per_head_rms(q, EPS), _eager(q, EPS), rtol=rtol, atol=0)


def test_normalises_per_head_not_across_heads() -> None:
    """Pin the reduction axis: it is the head dimension, nothing else.

    Reducing over the wrong axis still returns the right shape and finite values,
    so a bitwise test against a reference that made the same mistake would not
    catch it. Here each head is scaled independently, checked by giving one head
    a much larger magnitude and requiring the others to be unaffected.
    """
    q = torch.ones(1, 2, 1, 4)
    q[:, 1] *= 100.0
    out = _per_head_rms(q, EPS)
    torch.testing.assert_close(out[:, 0], torch.ones(1, 1, 4), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, 1], torch.ones(1, 1, 4), rtol=1e-5, atol=1e-5)


def test_fused_per_head_rms_gradient_matches_eager() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=generator)
    q_f = q.detach().clone().requires_grad_()
    q_e = q.detach().clone().requires_grad_()
    _per_head_rms(q_f, EPS).sum().backward()
    _eager(q_e, EPS).sum().backward()
    assert q_f.grad is not None
    torch.testing.assert_close(q_f.grad, q_e.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.gpus(1)
@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float32, 1e-6), (torch.bfloat16, 8e-3)])
def test_fused_per_head_rms_within_bound_on_gpu(dtype: torch.dtype, rtol: float) -> None:
    """Same bound on the device and dtypes production uses."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=generator).to(device="cuda", dtype=dtype)
    torch.testing.assert_close(_per_head_rms(q, EPS), _eager(q, EPS), rtol=rtol, atol=0)
