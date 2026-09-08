# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contracts for the fusions and gradient paths swapped in on this branch."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.attention.csa import (
    _per_head_rms,
    apply_partial_rope,
    rope_tables_for_packed_batch,
)
from megatron.lite.primitive.modules.attention.hca import HyperConnection, split_sinkhorn
from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.optimizers.megatron_wrap import _enable_wgrad_accumulation_fusion
from megatron.lite.primitive.parallel.linear import (
    AccumulatingLinear,
    _EmbeddingAccumulatingIntoMainGrad,
)

pytestmark = [pytest.mark.mlite]
DEVICE = "cuda"


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

    params = _Params()
    cu = torch.tensor([0, 4], dtype=torch.int32)
    kw = dict(config=None, use_yarn=False, device=torch.device("cpu"), dtype=torch.float32)
    first = rope_tables_for_packed_batch(params, cu, 0, 4, 4, 10000.0, **kw)
    second = rope_tables_for_packed_batch(params, cu, 0, 4, 4, 10000.0, **kw)
    assert first[0] is second[0] and first[1] is second[1]


def test_rope_tables_differ_across_batches_and_offsets() -> None:
    """Guard the guard: the tables must not outlive what defines them."""

    cu = torch.tensor([0, 4], dtype=torch.int32)
    kw = dict(config=None, use_yarn=False, device=torch.device("cpu"), dtype=torch.float32)
    a = rope_tables_for_packed_batch(_Params(), cu, 0, 4, 4, 10000.0, **kw)
    b = rope_tables_for_packed_batch(_Params(), cu, 0, 4, 4, 10000.0, **kw)
    assert a[0] is not b[0], "a new batch must not reuse the previous batch's tables"

    same = _Params()
    near = rope_tables_for_packed_batch(same, cu, 0, 4, 4, 10000.0, **kw)
    far = rope_tables_for_packed_batch(same, cu, 4, 4, 4, 10000.0, **kw)
    assert not torch.allclose(near[0], far[0]), "a different rank offset must rebuild"


# ``main_grad`` only ever exists on CUDA -- the distributed optimizer allocates
# it there -- and Core's accumulating linear has no CPU kernel, so these run on
# the device the paths they cover actually run on.
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


@pytest.mark.gpus(1)
def test_fused_accumulation_matches_the_add_it_replaces() -> None:
    """Same accumulated gradient as ``main_grad.add_(grad)``, over two microbatches."""

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


S, B, N, C = 3, 2, 4, 16


def _make_inputs(dtype: torch.dtype, device: str):
    """Build (x, residual, post, comb) with a real Sinkhorn-normalised ``comb``."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    mixes = torch.randn(S, B, (2 + N) * N, generator=generator).to(device=device, dtype=dtype)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    base = torch.zeros((2 + N) * N, device=device, dtype=torch.float32)
    _, post, comb = split_sinkhorn(mixes, scale, base, N, 3, 1e-6)
    x = torch.randn(S, B, C, generator=generator).to(device=device, dtype=dtype)
    residual = torch.randn(S, B, N, C, generator=generator).to(device=device, dtype=dtype)
    return x, residual, post, comb


def _reference_post(x, residual, post, comb):
    """The pre-fusion expression, kept verbatim as the numerical contract."""
    dtype = x.dtype
    placed = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2)
    mixed = torch.matmul(comb.to(dtype), residual.to(dtype))
    return placed + mixed


def test_orientation_is_not_symmetric() -> None:
    """Guard the guard: passing ``comb`` un-transposed must disagree."""
    _, residual, _, comb = _make_inputs(torch.float32, "cpu")
    assert not torch.allclose(comb, comb.transpose(-1, -2), rtol=1e-3, atol=1e-3)
    correct = torch.matmul(comb, residual)
    swapped = torch.matmul(comb.transpose(-1, -2), residual)
    assert not torch.allclose(correct, swapped, rtol=1e-3, atol=1e-3)


def test_post_is_differentiable_through_both_terms() -> None:
    """Both the mixing and the placement term must carry gradient."""
    x, residual, post, comb = _make_inputs(torch.float32, "cpu")
    x = x.detach().requires_grad_(True)
    residual = residual.detach().requires_grad_(True)
    HyperConnection.post(x, residual, post, comb).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert x.grad.abs().sum() > 0
    assert residual.grad.abs().sum() > 0


def _legacy_sinkhorn(comb_logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """The inline loop that Core's Sinkhorn replaced, kept to bound the change."""
    comb = torch.exp(comb_logits - comb_logits.max(dim=-1, keepdim=True).values)
    for _ in range(iters):
        comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=eps)
        comb = comb / comb.sum(dim=-2, keepdim=True).clamp(min=eps)
    return comb


def test_sinkhorn_output_is_doubly_stochastic() -> None:
    """Whatever the regularisation, the projection must still do its job."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    mixes = torch.randn(S, B, (2 + N) * N, generator=generator)
    _, _, comb = split_sinkhorn(mixes, torch.ones(3), torch.zeros((2 + N) * N), N, 20, 1e-6)
    ones = torch.ones(S, B, N)
    torch.testing.assert_close(comb.sum(dim=-1), ones, rtol=0, atol=1e-3)
    torch.testing.assert_close(comb.sum(dim=-2), ones, rtol=0, atol=1e-3)


def test_sinkhorn_regularisation_change_is_below_bf16_resolution() -> None:
    """Bound the numerical cost of moving to Core's Sinkhorn."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    mixes = torch.randn(S, B, (2 + N) * N, generator=generator)
    scale = torch.ones(3)
    base = torch.zeros((2 + N) * N)
    _, _, comb = split_sinkhorn(mixes, scale, base, N, 20, 1e-6)

    comb_mix = mixes.split([N, N, N * N], dim=-1)[2]
    legacy = _legacy_sinkhorn(comb_mix.view(S, B, N, N), 20, 1e-6)
    assert (comb - legacy).abs().max().item() < 1e-3


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
    torch.manual_seed(0)
    module = HyperConnection(hidden_size=C, hc_mult=N, sinkhorn_iters=3, eps=1e-6).cuda()
    x = torch.randn(S, B, N, C, device="cuda")
    expected = _reference_forward(module, x)
    actual, _, _ = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


class _FakeCSA(torch.nn.Module):
    pass


def _model_with(attention: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.attn = attention
    return model


def test_dense_multi_row_batch_is_refused_at_the_boundary() -> None:
    """``[B, S]`` with ``B > 1`` must raise here, naming the constraint."""

    model = _model_with(CompressedSparseAttention.__new__(CompressedSparseAttention))
    torch.nn.Module.__init__(model.attn)

    batch = type("B", (), {"input_ids": torch.zeros(4, 8, dtype=torch.long)})()
    with pytest.raises(NotImplementedError) as excinfo:
        protocol._prepare_model_forward_kwargs(model, batch)

    message = str(excinfo.value)
    assert "B=4" in message, "the message must name the batch that was rejected"
    assert "pack" in message.lower(), "the message must name the way out"


@pytest.mark.parametrize("shape", [(8,), (1, 8)])
def test_packed_shapes_still_take_the_packed_route(shape: tuple[int, ...]) -> None:
    """Guard the guard: the refusal must not swallow the shapes that do work."""

    called = {}

    def _fake_packed(model, batch):
        called["yes"] = True
        return {}

    original = protocol._prepare_packed_batch_kwargs
    protocol._prepare_packed_batch_kwargs = _fake_packed
    try:
        batch = type("B", (), {"input_ids": torch.zeros(*shape, dtype=torch.long)})()
        protocol._prepare_model_forward_kwargs(_model_with(_FakeCSA()), batch)
    finally:
        protocol._prepare_packed_batch_kwargs = original

    assert called.get("yes"), f"shape {shape} should have taken the packed route"


def test_models_without_csa_keep_the_dense_route() -> None:
    """The refusal is CSA's constraint, not the batch builder's."""

    reached = {}

    def _fake_dense(model, kwargs):
        reached["yes"] = True
        return kwargs

    original_dense = protocol._prepare_contiguous_cp_kwargs
    original_base = protocol._base_model_forward_kwargs
    protocol._prepare_contiguous_cp_kwargs = _fake_dense
    protocol._base_model_forward_kwargs = lambda batch: {}
    try:
        batch = type("B", (), {"input_ids": torch.zeros(4, 8, dtype=torch.long)})()
        protocol._prepare_model_forward_kwargs(_model_with(_FakeCSA()), batch)
    finally:
        protocol._prepare_contiguous_cp_kwargs = original_dense
        protocol._base_model_forward_kwargs = original_base

    assert reached.get("yes"), "a non-CSA model must still reach the dense builder"
