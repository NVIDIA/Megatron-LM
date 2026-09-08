# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contracts for the fusions and gradient paths this branch swaps in."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.attention.csa import (
    _ROPE_TABLES,
    _per_head_rms,
    apply_partial_rope,
    build_compressed_rope_cos_sin,
    rope_table,
)
from megatron.lite.primitive.modules.attention.hca import HyperConnection, split_sinkhorn
from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.optimizers.megatron_wrap import _enable_wgrad_accumulation_fusion
from megatron.lite.primitive.parallel.linear import (
    AccumulatingLinear,
    _EmbeddingAccumulatingIntoMainGrad,
)

pytestmark = [pytest.mark.mlite]


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every case; the fixtures below rely on fixed draws."""
    torch.manual_seed(0)


B, H, S, D = 2, 3, 5, 8
EPS = 1e-6


def _eager(q: torch.Tensor, eps: float) -> torch.Tensor:
    """The expression this replaced, kept verbatim as the contract."""
    return q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(dtype=q.dtype)


def test_normalises_per_head_not_across_heads() -> None:
    """Pin the reduction axis: it is the head dimension, nothing else."""
    q = torch.ones(1, 2, 1, 4)
    q[:, 1] *= 100.0
    out = _per_head_rms(q, EPS)
    torch.testing.assert_close(out[:, 0], torch.ones(1, 1, 4), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, 1], torch.ones(1, 1, 4), rtol=1e-5, atol=1e-5)


@pytest.mark.gpus(1)
@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float32, 1e-6), (torch.bfloat16, 8e-3)])
def test_fused_per_head_rms_within_bound_on_gpu(dtype: torch.dtype, rtol: float) -> None:
    """Same bound on the device and dtypes production uses."""
    q = torch.randn(B, H, S, D).to(device="cuda", dtype=dtype)
    torch.testing.assert_close(_per_head_rms(q, EPS), _eager(q, EPS), rtol=rtol, atol=0)

TOTAL, HEADS, HEAD_DIM, GROUPS = 17, 8, 6, 2
ROPE_DIM = 4


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
    y = torch.randn(TOKENS, FFN * 2, device="cuda", dtype=torch.bfloat16) * 8
    clamped = swiglu_with_probs(y, None, 3.0)
    unclamped = swiglu_with_probs(y, None, 0.0)
    assert not torch.allclose(clamped, unclamped, rtol=1e-2, atol=1e-2)


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
    module = AccumulatingLinear(IN, OUT, bias=False).to(DEVICE)
    module(torch.randn(TOKENS, IN, device=DEVICE)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.abs().max() > 0


@pytest.mark.gpus(1)
def test_embedding_scatters_to_the_same_rows_it_used_to_add() -> None:
    """The scatter must land the same values in the same rows as the dense path."""
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
    weight = torch.randn(VOCAB, IN, device=DEVICE, requires_grad=True)
    weight.main_grad = torch.zeros(VOCAB, IN, dtype=torch.float32, device=DEVICE)
    weight.grad_added_to_main_grad = False
    ids = torch.zeros(4, dtype=torch.long, device=DEVICE)  # only row 0

    _EmbeddingAccumulatingIntoMainGrad.apply(weight, ids).sum().backward()

    assert weight.main_grad[0].abs().max() > 0
    assert torch.equal(weight.main_grad[1:], torch.zeros(VOCAB - 1, IN, device=DEVICE))


S, B, N, C = 3, 2, 4, 16


def _make_inputs(dtype: torch.dtype, device: str):
    """Build (x, residual, post, comb) with a real Sinkhorn-normalised ``comb``."""
    mixes = torch.randn(S, B, (2 + N) * N).to(device=device, dtype=dtype)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    base = torch.zeros((2 + N) * N, device=device, dtype=torch.float32)
    _, post, comb = split_sinkhorn(mixes, scale, base, N, 3, 1e-6)
    x = torch.randn(S, B, C).to(device=device, dtype=dtype)
    residual = torch.randn(S, B, N, C).to(device=device, dtype=dtype)
    return x, residual, post, comb


def _reference_post(x, residual, post, comb):
    """The pre-fusion expression, kept verbatim as the numerical contract."""
    dtype = x.dtype
    placed = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2)
    mixed = torch.matmul(comb.to(dtype), residual.to(dtype))
    return placed + mixed


@pytest.mark.gpus(1)
def test_orientation_is_not_symmetric() -> None:
    """Guard the guard: passing ``comb`` un-transposed must disagree."""
    _, residual, _, comb = _make_inputs(torch.float32, "cuda")
    assert not torch.allclose(comb, comb.transpose(-1, -2), rtol=1e-3, atol=1e-3)
    correct = torch.matmul(comb, residual)
    swapped = torch.matmul(comb.transpose(-1, -2), residual)
    assert not torch.allclose(correct, swapped, rtol=1e-3, atol=1e-3)


def _reference_forward(module: HyperConnection, x: torch.Tensor) -> torch.Tensor:
    """The pre-fusion aggregation, kept verbatim as the numerical contract."""
    if x.dim() == 3:
        x = x.unsqueeze(2).expand(*x.shape[:2], module.hc_mult, x.size(-1))
    shape, dtype = x.shape, x.dtype
    xf = x.flatten(2)
    rms_inv = 1.0 / (xf.norm(dim=-1, keepdim=True) / math.sqrt(xf.shape[-1]) + module.eps)
    mixes = F.linear(xf, module.fn.to(device=x.device, dtype=dtype)) * rms_inv
    pre, _, _ = split_sinkhorn(
        mixes, module.scale, module.base, module.hc_mult, module.sinkhorn_iters, module.eps
    )
    return torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(dtype)


@pytest.mark.gpus(1)
def test_post_matches_pre_fusion_reference_bf16_gpu() -> None:
    """The fused path itself, in the dtype and on the device production uses."""
    x, residual, post, comb = _make_inputs(torch.bfloat16, "cuda")
    expected = _reference_post(x, residual, post, comb)
    actual = HyperConnection.post(x, residual, post, comb)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpus(1)
def test_forward_aggregation_matches_pre_fusion_reference_gpu() -> None:
    """Same, for the aggregation half: this is the only cover for the fused kernel."""
    module = HyperConnection(hidden_size=C, hc_mult=N, sinkhorn_iters=3, eps=1e-6).cuda()
    x = torch.randn(S, B, N, C, device="cuda")
    expected = _reference_forward(module, x)
    actual, _, _ = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


_ROPE_KW = dict(config=None, use_yarn=False, device=torch.device("cpu"), dtype=torch.float32)


def test_rope_table_is_shared_across_calls_and_grows_on_demand() -> None:
    """One table per parameter set, rebuilt only when a longer span is asked for."""
    _ROPE_TABLES.clear()
    first = rope_table(8, 4, 10000.0, **_ROPE_KW)
    assert rope_table(8, 4, 10000.0, **_ROPE_KW)[0] is first[0], "same span must reuse"
    assert rope_table(4, 4, 10000.0, **_ROPE_KW)[0] is first[0], "shorter span must reuse"
    grown = rope_table(64, 4, 10000.0, **_ROPE_KW)
    assert grown[0] is not first[0], "a longer span must rebuild"
    assert len(_ROPE_TABLES) == 1, "growing must replace the entry, not add one"


def test_rope_table_rows_match_building_from_those_positions() -> None:
    """Gathering rows from the shared table equals building for those positions."""
    positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    direct_cos, direct_sin = build_compressed_rope_cos_sin(positions, 4, 10000.0, **_ROPE_KW)
    cos, sin = rope_table(16, 4, 10000.0, **_ROPE_KW)
    rows = positions.view(-1)
    torch.testing.assert_close(cos[0].index_select(0, rows), direct_cos[0])
    torch.testing.assert_close(sin[0].index_select(0, rows), direct_sin[0])
    shifted = cos[0].index_select(0, rows + 4)
    assert not torch.allclose(shifted, direct_cos[0]), "wrong rows must not match"

