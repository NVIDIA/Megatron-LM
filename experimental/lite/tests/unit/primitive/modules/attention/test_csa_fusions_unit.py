# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fusions swapped into CSA and the MoE experts, each bounded against what it replaced."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.attention.csa import _per_head_rms, apply_partial_rope
from megatron.lite.primitive.modules.experts import swiglu_with_probs

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
    """Pin the reduction axis: it is the head dimension, nothing else."""
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

TOTAL, HEADS, HEAD_DIM, GROUPS = 17, 8, 6, 2
ROPE_DIM = 4


def test_thd_grouping_is_a_free_view_of_the_bshd_reshape() -> None:
    """``(total, np, hn) -> (1, total, g, per*hn)`` must be the removed reshape."""
    per = HEADS // GROUPS
    t = torch.randn(TOTAL, HEADS, HEAD_DIM)
    view = t.view(1, TOTAL, GROUPS, per * HEAD_DIM)
    reshape = (
        t.permute(1, 0, 2)
        .unsqueeze(0)
        .contiguous()
        .transpose(1, 2)
        .reshape(1, TOTAL, GROUPS, per * HEAD_DIM)
    )
    assert torch.equal(view, reshape)


def test_grouping_is_head_major_not_head_dim_major() -> None:
    """Guard the guard: a group axis taken over the wrong stride must differ."""
    per = HEADS // GROUPS
    t = torch.randn(TOTAL, HEADS, HEAD_DIM)
    correct = t.view(1, TOTAL, GROUPS, per * HEAD_DIM)
    transposed = t.transpose(1, 2).reshape(1, TOTAL, GROUPS, per * HEAD_DIM)
    assert not torch.equal(correct, transposed)


def _cos_sin(dtype: torch.dtype, device: str):
    """Full-length ``cat(freqs, freqs)`` tables, as the CSA builders return."""
    pos = torch.arange(TOTAL, device=device, dtype=torch.float32)
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM)
    )
    freqs = pos.unsqueeze(-1) * inv.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.gpus(1)
def test_fused_inverse_rope_matches_apply_partial_rope() -> None:
    """Core's fused inverse must agree with ``apply_partial_rope(cos, -sin)``."""
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_out_of_place

    torch.manual_seed(0)
    dtype = torch.bfloat16
    cu = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
    cos, sin = _cos_sin(dtype, "cuda")
    nope = HEAD_DIM - ROPE_DIM
    context = torch.randn(TOTAL, HEADS, HEAD_DIM, device="cuda", dtype=dtype)

    fused = fused_mla_rope_out_of_place(
        context, cos, sin, nope, ROPE_DIM, cu, 0, 1, inverse=True, remove_interleaving=True
    )
    eager = apply_partial_rope(
        context.permute(1, 0, 2).unsqueeze(0), cos.unsqueeze(0), -sin.unsqueeze(0), ROPE_DIM
    )
    eager = eager.squeeze(0).permute(1, 0, 2)
    scale = eager.float().abs().max()
    assert (fused.float() - eager.float()).abs().max() < 2e-2 * scale


@pytest.mark.gpus(1)
def test_forward_rotation_is_not_mistaken_for_the_inverse() -> None:
    """Negative control: ``inverse=False`` must not satisfy the bound above."""
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_out_of_place

    torch.manual_seed(0)
    dtype = torch.bfloat16
    cu = torch.tensor([0, TOTAL], device="cuda", dtype=torch.int32)
    cos, sin = _cos_sin(dtype, "cuda")
    nope = HEAD_DIM - ROPE_DIM
    context = torch.randn(TOTAL, HEADS, HEAD_DIM, device="cuda", dtype=dtype)

    wrong = fused_mla_rope_out_of_place(
        context, cos, sin, nope, ROPE_DIM, cu, 0, 1, inverse=False, remove_interleaving=True
    )
    eager = apply_partial_rope(
        context.permute(1, 0, 2).unsqueeze(0), cos.unsqueeze(0), -sin.unsqueeze(0), ROPE_DIM
    )
    eager = eager.squeeze(0).permute(1, 0, 2)
    scale = eager.float().abs().max()
    assert (wrong.float() - eager.float()).abs().max() > 2e-2 * scale

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
    """Guard the guard: the clamp must change the result on this fixture."""
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


class _Params:
    """Stands in for PackedSeqParams: an object the layers all share."""


def test_rope_tables_are_built_once_per_packed_batch() -> None:
    """A second layer asking for the same tables must get the first one's."""
    from megatron.lite.primitive.modules.attention.csa import rope_tables_for_packed_batch

    params = _Params()
    cu = torch.tensor([0, 4], dtype=torch.int32)
    kw = dict(config=None, use_yarn=False, device=torch.device("cpu"), dtype=torch.float32)
    first = rope_tables_for_packed_batch(params, cu, 0, 4, 4, 10000.0, **kw)
    second = rope_tables_for_packed_batch(params, cu, 0, 4, 4, 10000.0, **kw)
    assert first[0] is second[0] and first[1] is second[1]


def test_rope_tables_differ_across_batches_and_offsets() -> None:
    """Guard the guard: the tables must not outlive what defines them.

    A cache that ignored the rank offset, or that lived longer than the batch,
    would return the first batch's positions for the second -- finite, correctly
    shaped, and wrong. Both axes are checked because either alone would pass.
    """
    from megatron.lite.primitive.modules.attention.csa import rope_tables_for_packed_batch

    cu = torch.tensor([0, 4], dtype=torch.int32)
    kw = dict(config=None, use_yarn=False, device=torch.device("cpu"), dtype=torch.float32)
    a = rope_tables_for_packed_batch(_Params(), cu, 0, 4, 4, 10000.0, **kw)
    b = rope_tables_for_packed_batch(_Params(), cu, 0, 4, 4, 10000.0, **kw)
    assert a[0] is not b[0], "a new batch must not reuse the previous batch's tables"

    same = _Params()
    near = rope_tables_for_packed_batch(same, cu, 0, 4, 4, 10000.0, **kw)
    far = rope_tables_for_packed_batch(same, cu, 4, 4, 4, 10000.0, **kw)
    assert not torch.allclose(near[0], far[0]), "a different rank offset must rebuild"
