# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Engram gate (MCORE_ENGRAM_FUSED_GATE=1) against FP64 autograd and the eager gate.

The eager gate in ``Engram.forward`` is the reference. The fused kernels compute in FP32 like
it and use the same element-wise routines, but sum over the hidden size (and over streams and
tokens in backward) in a different order, so the checks are:

* the closed-form gradients the kernels implement equal FP64 autograd of the reference;
* given dot, the gate and d gate / d dot are bitwise equal to PyTorch's FP32 autograd,
  including dot = +-0, |dot| on and around the clamp at 1e-6, saturated gates, +-inf and NaN;
* on well-conditioned data (|dot| bounded away from 0 by construction), values and all
  gradients agree with FP64 autograd to FP32 accuracy (FP32 tensors) or to one BF16 ulp
  (BF16 tensors), and with the eager gate to one BF16 ulp; on arbitrary data both paths
  inherit the cancellation of the reference's FP32 math, so the module tests compare normwise;
* near the clamp, where dot itself is ill-conditioned in any FP32 implementation, backward
  is checked against the closed form evaluated at the kernel's own forward values;
* masked tokens are inert, runs are bitwise reproducible, and CUDA graphs replay exactly;
* the MXFP8 lookup's padded projection output goes in whole (no slice copy in backward),
  broadcast masks behave as in the eager gate, and without Triton the eager gate runs.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.deepseek_v41 import engram_gate
from megatron.core.models.deepseek_v41.engram import Engram
from megatron.core.models.deepseek_v41.engram_gate import fused_engram_gate
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not engram_gate.HAVE_TRITON, reason="requires CUDA and Triton"
)

_EPS = 1e-6
# torch.clamp_min(x, 1e-6) on FP32 compares with 1e-6 rounded to FP32.
_THRESHOLD = torch.tensor(1e-6, dtype=torch.float32).item()
# Whether the installed PyTorch's clamp_min backward passes the gradient at the minimum.
_GRAD_AT_THRESHOLD = engram_gate._clamp_min_grad_at_min()


def _device():
    index = int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _eager_gate(hidden_states, kv, q_weight, k_weight, token_mask, n, d, eps):
    """``Engram.forward``'s eager gate after ``kv.transpose(0, 1)``, verbatim."""
    key, value = kv.split([n * d, d], dim=-1)
    key = key.float().unflatten(-1, (n, d))
    h = hidden_states.float().unflatten(-1, (n, d))
    rstd = (h.square().mean(-1) + eps).rsqrt() * (key.square().mean(-1) + eps).rsqrt()
    gamma = (q_weight.float() * k_weight.float()).view(n, d)
    dot = (h * gamma * key).sum(-1)
    dot = dot * rstd * d**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.transpose(0, 1).unsqueeze(-1), 0)
    return (
        (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).flatten(-2).to(hidden_states.dtype)
    )


def _forward64(hidden_states, kv, q_weight, k_weight, token_mask, n, d, eps):
    """The eager gate in FP64; returns the output and (rh, rk, dot, gate)."""
    key, value = kv.double().split([n * d, d], dim=-1)
    key = key.unflatten(-1, (n, d))
    h = hidden_states.double().unflatten(-1, (n, d))
    rh = (h.square().mean(-1) + eps).rsqrt()
    rk = (key.square().mean(-1) + eps).rsqrt()
    gamma = (q_weight.double() * k_weight.double()).view(n, d)
    dot = (h * gamma * key).sum(-1) * (rh * rk) * d**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(_THRESHOLD).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.transpose(0, 1).unsqueeze(-1), 0)
    out = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
    return out, (rh, rk, dot, gate)


def _closed_form64(hidden_states, kv, q_weight, k_weight, token_mask, grad, n, d, eps, state=None):
    """Gradients of the gate in FP64 from the closed form the kernels implement.

    ``state`` = (rh, rk, dot, gate) as [sequence, batch, n] replaces the FP64 forward values,
    which checks the backward at a given (e.g. the kernel's own) forward state.
    """
    h = hidden_states.double().unflatten(-1, (n, d))
    key, v = kv.double().split([n * d, d], dim=-1)
    k = key.unflatten(-1, (n, d))
    qw, kw = q_weight.double().view(n, d), k_weight.double().view(n, d)
    gamma = qw * kw
    if state is None:
        _, state = _forward64(hidden_states, kv, q_weight, k_weight, token_mask, n, d, eps)
    rh, rk, dot, gate = (t.double() for t in state)
    g = grad.double().unflatten(-1, (n, d))
    # sigmoid, copysign, sqrt, clamp_min (gradient where |dot| > 1e-6, and at equality on
    # PyTorch versions whose clamp_min backward uses >=) and abs, chained.
    passes = dot.abs() >= _THRESHOLD if _GRAD_AT_THRESHOLD else dot.abs() > _THRESHOLD
    grad_u = (g * v.unsqueeze(-2)).sum(-1) * (1 - gate) * gate
    grad_dot = torch.where(passes, grad_u / (2 * dot.abs().clamp_min(_THRESHOLD).sqrt()), 0.0)
    if token_mask is not None:
        grad_dot = grad_dot.masked_fill(~token_mask.transpose(0, 1).unsqueeze(-1), 0)
    cross = grad_dot * d**-0.5 * rh * rk
    grad_h = g + cross[..., None] * gamma * k - (grad_dot * dot * rh**2 / d)[..., None] * h
    grad_k = cross[..., None] * gamma * h - (grad_dot * dot * rk**2 / d)[..., None] * k
    grad_v = (gate[..., None] * g).sum(-2)
    grad_gamma = (cross[..., None] * h * k).sum((0, 1))
    return (
        grad_h.flatten(-2),
        torch.cat([grad_k.flatten(-2), grad_v], -1),
        (grad_gamma * kw).flatten(),
        (grad_gamma * qw).flatten(),
    )


def _inputs(length, batch, n, d, dtype, *, seed=0, masked=False, near_threshold=False):
    """Hidden states, a [batch, sequence] kv buffer (seen transposed), weights and a mask.

    Keys are correlated with the hidden states (random sign and strength per token and
    stream) and gamma is about 1 / sqrt(d), so |dot| stays well away from 0 and the clamp:
    there, FP32 evaluation of dot is well conditioned. ``near_threshold`` then rewrites the
    first tokens: an all-zero hidden state (dot = 0), a key orthogonal to h * gamma, and
    dots of +-2e-6, 1.2e-6 and -5e-7, all of them ill conditioned on purpose.
    """
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(length, batch, n, d, generator=gen, dtype=torch.float64)
    strength = torch.rand(batch, length, n, 1, generator=gen, dtype=torch.float64) + 0.5
    sign = torch.randint(0, 2, (batch, length, n, 1), generator=gen).double() * 2 - 1
    keys = sign * strength * hidden.transpose(0, 1) + torch.randn(
        batch, length, n, d, generator=gen, dtype=torch.float64
    )
    value = torch.randn(batch, length, d, generator=gen, dtype=torch.float64)
    q = d**-0.25 * (1 + 0.3 * torch.randn(n * d, generator=gen, dtype=torch.float64))
    k = d**-0.25 * (1 + 0.3 * torch.randn(n * d, generator=gen, dtype=torch.float64))
    if near_threshold:
        gamma = (q * k).view(n, d)
        hidden[0, 0] = 0.0
        w = hidden[1, 0, 0] * gamma[0]
        keys[0, 1, 0] -= (keys[0, 1, 0] @ w) / (w @ w) * w
        for i, target in enumerate((2e-6, -2e-6, 1.2e-6, -5e-7)[:n]):
            w = hidden[2, 0, i] * gamma[i]
            r = keys[0, 2, i] - (keys[0, 2, i] @ w) / (w @ w) * w
            rh = ((hidden[2, 0, i] ** 2).mean() + _EPS).rsqrt()
            rk = ((r**2).mean() + _EPS).rsqrt()
            keys[0, 2, i] = r + target / (rh * rk * d**-0.5) / (w @ w) * w
    kv = torch.cat([keys.flatten(-2), value], -1)
    mask = None
    if masked:
        mask = torch.rand(batch, length, generator=gen) > 0.25
        mask[0, :3] = True
    device = _device()
    cast = lambda t: t.to(dtype=dtype, device=device)
    return (
        cast(hidden.flatten(-2)),
        cast(kv),
        cast(q),
        cast(k),
        None if mask is None else mask.to(device),
    )


def _run(fn, inputs, grad, n, d):
    """Output and gradients w.r.t. hidden, the [batch, sequence] kv buffer, q and k weights.

    Also returns the fused gate's saved FP32 [tokens, 4, streams] state (None for eager).
    """
    hidden, kv, q, k, mask = inputs
    leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, kv, q, k)]
    out = fn(leaves[0], leaves[1].transpose(0, 1), leaves[2], leaves[3], mask, n, d, _EPS)
    saved = getattr(out.grad_fn, "saved_tensors", None)
    state = saved[-1].clone() if saved else None
    grads = torch.autograd.grad(out, leaves, grad)
    return [out.detach()] + list(grads), state


def _reference(inputs, grad, n, d):
    hidden, kv, q, k, mask = inputs
    leaves = [t.detach().double().requires_grad_(True) for t in (hidden, kv, q, k)]
    out, state = _forward64(
        leaves[0], leaves[1].transpose(0, 1), leaves[2], leaves[3], mask, n, d, _EPS
    )
    grads = torch.autograd.grad(out, leaves, grad.double())
    return [out.detach()] + list(grads), [t.detach() for t in state]


def _normwise(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).abs().max() / expected.abs().max()).item()


def _assert_bf16_ulps(actual, expected, what, ulps=1.0, floor=2.0**-20):
    """|actual - expected| <= ulps BF16 ulps (of the larger magnitude) + floor * max|expected|."""
    a, e = actual.double(), expected.double()
    magnitude = torch.maximum(a.abs(), e.abs()).clamp_min(2.0**-126)
    bound = ulps * torch.exp2(torch.floor(torch.log2(magnitude)) - 7) + floor * e.abs().max()
    excess = (a - e).abs() - bound
    assert (excess <= 0).all(), (
        f"{what}: {int((excess > 0).sum())}/{a.numel()} elements exceed {ulps} BF16 ulp "
        f"(worst excess {excess.max().item():.3e})"
    )


_NAMES = ("out", "grad_hidden", "grad_kv", "grad_q_weight", "grad_k_weight")
# (sequence, batch, streams, hidden size, mask): hidden sizes 64, 96, 1000 and the proxy's 5120,
# B = 1 and B > 1 (kv non-contiguous after the transpose), 2, 3 (padded) and 4 streams.
_SHAPES = [
    (64, 1, 4, 5120, True),
    (16, 3, 4, 96, True),
    (24, 2, 3, 1000, False),
    (32, 1, 4, 64, False),
    (8, 2, 2, 64, True),
]


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "S{}-B{}-n{}-d{}-mask{}".format(*s))
def test_closed_form_gradients_match_fp64_autograd(shape):
    """The kernels' gradient formulas, re-derived, equal autograd of the reference in FP64."""
    length, batch, n, d, masked = shape
    length = min(length, 8)
    inputs = _inputs(length, batch, n, d, torch.float64, masked=masked, near_threshold=True)
    grad = torch.randn(length, batch, n * d, dtype=torch.float64, device=inputs[0].device)
    expected, (_, _, dot, _) = _reference(inputs, grad, n, d)
    # dot = 0 (zero hidden state) and |dot| on both sides of the clamp are covered.
    assert (dot[0, 0] == 0).all() and (dot[2, 0].abs() < 3e-6).all()
    hidden, kv, q, k, mask = inputs
    closed = _closed_form64(hidden, kv.transpose(0, 1), q, k, mask, grad, n, d, _EPS)
    closed = [closed[0], closed[1].transpose(0, 1), closed[2], closed[3]]
    for name, actual, reference in zip(_NAMES[1:], closed, expected[1:]):
        assert _normwise(actual, reference) < 1e-13, name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "S{}-B{}-n{}-d{}-mask{}".format(*s))
def test_fused_gate_matches_fp64_reference_and_eager(shape, dtype):
    """Forward and every gradient against FP64 autograd and against the eager gate."""
    length, batch, n, d, masked = shape
    inputs = _inputs(length, batch, n, d, dtype, seed=1, masked=masked)
    grad = torch.randn(length, batch, n * d, device=inputs[0].device).to(dtype)
    fused, _ = _run(fused_engram_gate, inputs, grad, n, d)
    eager, _ = _run(_eager_gate, inputs, grad, n, d)
    reference, (_, _, dot, _) = _reference(inputs, grad, n, d)
    assert dot.abs().min() > 0.05  # well conditioned by construction
    for name, f, e, r in zip(_NAMES, fused, eager, reference):
        assert f.dtype == e.dtype and f.shape == e.shape, name
        if dtype == torch.float32:
            # FP32 rounding of the sums over d (and over tokens for the weights) leaves normwise
            # errors of a few FP32 epsilons (<= 4.1e-7 measured); the eager gate's are alike.
            assert _normwise(f, r) < 2e-6, (name, _normwise(f, r), _normwise(e, r))
            assert _normwise(f, e) < 2e-6, name
        else:
            # Both paths round an FP32 value to BF16: at most one BF16 ulp apart from the
            # correctly rounded FP64 result and from each other.
            _assert_bf16_ulps(f, r, f"{name} vs FP64")
            _assert_bf16_ulps(f, e, f"{name} vs eager")
            assert _normwise(f, r) <= 1.25 * _normwise(e, r) + 2.0**-20, name
    # The output and every gradient keep the dtypes autograd gives the eager gate.
    assert fused[0].dtype == dtype


def _same_bits(actual, expected):
    """Elementwise: equal FP32 bit patterns (so +0 != -0), or both NaN."""
    same = actual.view(torch.int32) == expected.view(torch.int32)
    return same | (torch.isnan(actual) & torch.isnan(expected))


def test_gate_and_its_derivative_match_pytorch_bitwise():
    """Given dot, the gate and d gate / d dot equal PyTorch's FP32 autograd bit for bit.

    Covers |dot| on and around the clamp at 1e-6, +-0, saturated gates (gate == 1 or below
    1e-30, from |dot| ~ 277 on, where d gate / d dot is a zero whose sign must match), the
    windows where the gate or the derivative is subnormal, +-inf (autograd gives NaN there)
    and NaN, and random bit patterns of every finite magnitude.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def gate_kernel(
        dot_ptr,
        grad_ptr,
        gate_ptr,
        grad_dot_ptr,
        n,
        AT_THRESHOLD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < n
        dot = tl.load(dot_ptr + offset, valid, other=1.0)
        gate = engram_gate._gate_from_dot(dot, 1e-6)
        grad_dot = engram_gate._dot_grad_from_gate_grad(
            tl.load(grad_ptr + offset, valid, other=0.0), gate, dot, 1e-6, AT_THRESHOLD
        )
        tl.store(gate_ptr + offset, gate, valid)
        tl.store(grad_dot_ptr + offset, grad_dot, valid)

    device = _device()
    threshold = torch.tensor([1e-6], dtype=torch.float32)
    near = [
        threshold,
        torch.nextafter(threshold, torch.zeros(1)),
        torch.nextafter(threshold, torch.ones(1)),
    ]
    zeros = [0.0, -0.0]
    large = [276.75, -276.75, 300.0, -300.0, 7400.0, -7600.0, -8000.0, 3.4e38, -3.4e38]
    special = (
        torch.cat(near + [-t for t in near]).tolist()
        + zeros
        + [1e-30, -1e-30, 1e-7, -5e-7, 2e-6, -2e-6, 0.25, -1.5, 30.0, -90.0, 200.0]
        + large
        + [float("inf"), float("-inf"), float("nan")]
    )
    gen = torch.Generator().manual_seed(3)
    bits = torch.randint(0, 2**31 - 1, (1 << 20,), generator=gen, dtype=torch.int64)
    finite = bits.to(torch.int32).view(torch.float32)
    finite = finite[torch.isfinite(finite)]
    finite = torch.where(torch.rand(finite.numel(), generator=gen) < 0.5, -finite, finite)
    # Gate 1 (with 1 - gate subnormal on the way) and gate subnormal, then 0.
    window = torch.linspace(7300.0, 8100.0, 200_001)
    dot = torch.cat(
        [
            torch.tensor(special, dtype=torch.float32),
            torch.randn(4000, generator=gen) * torch.logspace(-8, 1, 4000),
            finite,
            window,
            -window,
        ]
    ).to(device)
    grad = torch.randn(dot.numel(), generator=gen).to(device)
    gate, grad_dot = torch.empty_like(dot), torch.empty_like(dot)
    block = 1024
    gate_kernel[(triton.cdiv(dot.numel(), block),)](
        dot, grad, gate, grad_dot, dot.numel(), AT_THRESHOLD=_GRAD_AT_THRESHOLD, BLOCK=block
    )

    leaf = dot.clone().requires_grad_(True)
    reference = torch.sigmoid(torch.copysign(leaf.abs().clamp_min(1e-6).sqrt(), leaf))
    (reference_grad,) = torch.autograd.grad(reference, leaf, grad)
    reference = reference.detach()
    for name, actual, expected in (("gate", gate, reference), ("grad", grad_dot, reference_grad)):
        same = _same_bits(actual, expected)
        assert same.all(), (
            f"{name}: {int((~same).sum())} of {dot.numel()} differ, e.g. dot="
            f"{dot[~same][:4].tolist()} fused={actual[~same][:4].tolist()} "
            f"torch={expected[~same][:4].tolist()}"
        )
    # The clamp blocks the gradient just below 1e-6 and passes it just above; at 1e-6 itself it
    # follows the installed PyTorch (strict > in recent versions); +-0 gets none.
    for sign in (0, 3):
        assert grad_dot[sign + 1] == 0 and grad_dot[sign + 2] != 0
        assert (grad_dot[sign] != 0) == _GRAD_AT_THRESHOLD
    assert (grad_dot[6:8] == 0).all() and gate[6] > 0.5 > gate[7]
    # Saturated gates (exactly 1 or 0) get +0, never -0: autograd adds copysign's zero gradient
    # for its sign input last. An infinite dot gets NaN (copysign's backward divides inf by inf).
    saturated = ((gate == 1) | (gate == 0)) & torch.isfinite(dot)
    assert saturated.sum() > 1000
    assert (grad_dot[saturated] == 0).all() and not torch.signbit(grad_dot[saturated]).any()
    infinite = slice(len(special) - 3, len(special) - 1)  # +inf, -inf (NaN is last)
    assert torch.isnan(grad_dot[infinite]).all()


@pytest.mark.parametrize("masked", [False, True])
def test_near_threshold_tokens_follow_the_kernel_forward_state(masked):
    """dot = 0 and |dot| around 1e-6: forward state and backward at that state are exact.

    There, dot cancels catastrophically, so any FP32 evaluation differs from FP64 in the first
    digits. The gate must still be exactly the function of the kernel's own dot, and the
    backward must be the closed form at the kernel's own (rh, rk, dot, gate).
    """
    length, batch, n, d = 8, 2, 4, 1000
    inputs = _inputs(length, batch, n, d, torch.float32, seed=2, masked=masked, near_threshold=True)
    hidden, kv, q, k, mask = inputs
    grad = torch.randn(length, batch, n * d, device=hidden.device)
    fused, state = _run(fused_engram_gate, inputs, grad, n, d)
    rh, rk, dot, gate = state.view(length, batch, 4, -1)[..., :n].unbind(2)
    _, (rh64, rk64, dot64, _) = _reference(inputs, grad, n, d)
    # rh and rk are well conditioned; dot is accurate relative to c0 * r * sum |h * gamma * k|.
    assert _normwise(rh, rh64) < 1e-6 and _normwise(rk, rk64) < 1e-6
    hh = hidden.double().unflatten(-1, (n, d))
    kk = kv.transpose(0, 1)[..., : n * d].double().unflatten(-1, (n, d))
    magnitude = (hh * (q.double() * k.double()).view(n, d) * kk).abs().sum(-1)
    assert ((dot - dot64).abs() <= 1e-5 * magnitude * rh64 * rk64 * d**-0.5 + 1e-30).all()
    assert (dot[0, 0] == 0).all()  # all-zero hidden state
    # The saved gate is PyTorch's FP32 gate of the saved dot (0 for masked tokens).
    expected_gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if mask is not None:
        expected_gate = expected_gate.masked_fill(~mask.transpose(0, 1).unsqueeze(-1), 0)
    assert torch.equal(gate, expected_gate)
    closed = _closed_form64(
        hidden, kv.transpose(0, 1), q, k, mask, grad, n, d, _EPS, state=(rh, rk, dot, gate)
    )
    closed = [closed[0], closed[1].transpose(0, 1), closed[2], closed[3]]
    for name, actual, expected in zip(_NAMES[1:], fused[1:], closed):
        assert _normwise(actual, expected) < 2e-6, (name, _normwise(actual, expected))
    # Out is h + gate * v at the kernel's gate.
    value = kv[..., n * d :].transpose(0, 1).double()
    expected_out = (
        hidden.double().unflatten(-1, (n, d)) + gate.double()[..., None] * value[:, :, None]
    ).flatten(-2)
    assert _normwise(fused[0], expected_out) < 1e-6


def test_masked_tokens_are_inert():
    """Masked tokens pass h and the output gradient through and contribute nothing else."""
    length, batch, n, d = 16, 3, 4, 96
    inputs = _inputs(length, batch, n, d, torch.bfloat16, seed=4, masked=True)
    hidden, kv, q, k, mask = inputs
    grad = torch.randn(length, batch, n * d, device=hidden.device).bfloat16()
    first, _ = _run(fused_engram_gate, inputs, grad, n, d)
    masked = ~mask.transpose(0, 1)  # [sequence, batch]
    assert masked.any() and (~masked).any()
    out, grad_hidden, grad_kv, grad_q, grad_k = first
    assert torch.equal(out[masked], hidden[masked])
    assert torch.equal(grad_hidden[masked], grad[masked])
    assert torch.count_nonzero(grad_kv.transpose(0, 1)[masked]) == 0
    # Changing everything about masked tokens changes nothing else, bit for bit.
    noise = torch.randn_like(hidden) * 10
    other_hidden = torch.where(masked[..., None], noise, hidden)
    other_kv = torch.where(mask[..., None], kv, torch.randn_like(kv) * 10)
    second, _ = _run(fused_engram_gate, (other_hidden, other_kv, q, k, mask), grad, n, d)
    keep = ~masked
    for name, a, b in zip(_NAMES[:3], first[:3], second[:3]):
        if name == "grad_kv":
            a, b = a.transpose(0, 1), b.transpose(0, 1)
        assert torch.equal(a[keep], b[keep]), name
    assert torch.equal(first[3], second[3]) and torch.equal(first[4], second[4])


def _poison_allocations(monkeypatch):
    """Make torch.empty / torch.empty_like return NaN-filled floating-point tensors."""

    def poisoned(factory):
        def make(*args, **kwargs):
            tensor = factory(*args, **kwargs)
            return tensor.fill_(float("nan")) if tensor.is_floating_point() else tensor

        return make

    monkeypatch.setattr(torch, "empty", poisoned(torch.empty))
    monkeypatch.setattr(torch, "empty_like", poisoned(torch.empty_like))


@pytest.mark.parametrize("n", [3, 4])
def test_results_do_not_depend_on_uninitialized_memory(monkeypatch, n):
    """NaN-filled torch.empty / torch.empty_like buffers change nothing, bit for bit.

    Every output element is written, and the padding lanes of 3 streams (padded to 4) are
    never read unmasked.
    """
    length, batch, d = 16, 2, 96
    inputs = _inputs(length, batch, n, d, torch.bfloat16, seed=12, masked=True)
    grad = torch.randn(length, batch, n * d, device=inputs[0].device).bfloat16()
    clean, _ = _run(fused_engram_gate, inputs, grad, n, d)
    _poison_allocations(monkeypatch)
    dirty, _ = _run(fused_engram_gate, inputs, grad, n, d)
    monkeypatch.undo()
    for name, a, b in zip(_NAMES, clean, dirty):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name


@pytest.mark.parametrize(
    "length,batch,rows", [(35, 1, 64), (64, 1, 64), (12, 3, 64)], ids=["T35-rows64", "T64", "B3"]
)
def test_padded_token_rows_go_in_whole(monkeypatch, length, batch, rows):
    """The MXFP8 lookup's projection output, [rows >= tokens, width] with the tokens
    batch-major in its first rows, goes into the gate whole.

    Output and gradients equal those of its first rows seen as [sequence, batch] (what the
    eager gate slices) bit for bit; the kv gradient keeps the padded shape, with pad rows the
    gate itself zeroes; and autograd records no slice, view or transpose between kv and the
    gate, so backward makes no zero-filled copy of the [rows, width] gradient.
    """
    n, d = 4, 96
    hidden, kv, q, k, mask = _inputs(length, batch, n, d, torch.bfloat16, seed=14, masked=True)
    tokens, width = length * batch, (n + 1) * d
    padded = torch.full((rows, width), float("nan"), dtype=kv.dtype, device=kv.device)
    padded[:tokens] = kv.reshape(tokens, width)  # NaN pad rows: the gate must never read them
    grad = torch.randn(length, batch, n * d, device=hidden.device).bfloat16()

    def run(through_view):
        leaves = [t.detach().clone().requires_grad_(True) for t in (hidden, padded, q, k)]
        kv_in = leaves[1]
        if through_view:
            kv_in = kv_in[:tokens].view(batch, length, width).transpose(0, 1)
        out = fused_engram_gate(leaves[0], kv_in, leaves[2], leaves[3], mask, n, d, _EPS)
        kv_node = out.grad_fn.next_functions[1][0]
        assert through_view or (
            type(kv_node).__name__ == "AccumulateGrad" and kv_node.variable is leaves[1]
        ), type(kv_node).__name__
        return [out.detach()] + list(torch.autograd.grad(out, leaves, grad))

    viewed = run(True)
    whole = run(False)
    assert whole[2].shape == (rows, width)
    assert torch.count_nonzero(whole[2][tokens:]) == 0
    for name, a, b in zip(_NAMES, whole, viewed):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name
    _poison_allocations(monkeypatch)
    dirty = run(False)
    monkeypatch.undo()
    for name, a, b in zip(_NAMES, whole, dirty):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name
    with pytest.raises(ValueError):
        fused_engram_gate(hidden, padded[: tokens - 1], q, k, mask, n, d, _EPS)


@pytest.mark.parametrize("shape", ["1xS", "Bx1"])
def test_token_mask_broadcasts_like_the_eager_gate(shape):
    """A [1, sequence] or [batch, 1] mask broadcasts over [batch, sequence], as in eager."""
    length, batch, n, d = 16, 3, 4, 96
    hidden, kv, q, k, mask = _inputs(length, batch, n, d, torch.bfloat16, seed=15, masked=True)
    small = mask[:1] if shape == "1xS" else mask[:, :1]
    small[0, 0] = False  # mask at least one row of the broadcast dimension
    dense = small.expand(batch, length).contiguous()
    grad = torch.randn(length, batch, n * d, device=hidden.device).bfloat16()
    broadcast, _ = _run(fused_engram_gate, (hidden, kv, q, k, small), grad, n, d)
    expanded, _ = _run(fused_engram_gate, (hidden, kv, q, k, dense), grad, n, d)
    eager, _ = _run(_eager_gate, (hidden, kv, q, k, small), grad, n, d)
    for name, a, b, e in zip(_NAMES, broadcast, expanded, eager):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name
        assert a.shape == e.shape, name
        _assert_bf16_ulps(a, e, f"{name} vs eager")
    for bad in (mask[:2], mask[0], mask.long()):  # not broadcastable to [3, S]; 1-D; not bool
        with pytest.raises(ValueError):
            fused_engram_gate(hidden, kv.transpose(0, 1), q, k, bad, n, d, _EPS)


def test_forward_without_autograd():
    """no_grad and inference_mode forwards (evaluation) equal the training forward bitwise."""
    length, batch, n, d = 16, 2, 4, 96
    hidden, kv, q, k, mask = _inputs(length, batch, n, d, torch.bfloat16, seed=13, masked=True)
    args = (hidden.requires_grad_(True), kv.transpose(0, 1), q, k, mask, n, d, _EPS)
    training = fused_engram_gate(*args).detach()
    with torch.no_grad():
        no_grad = fused_engram_gate(*args)
    engram_gate._clamp_min_grad_at_min.cache_clear()  # the probe must also work in here
    with torch.inference_mode():
        inference = fused_engram_gate(*args)
    assert engram_gate._clamp_min_grad_at_min() == _GRAD_AT_THRESHOLD
    for out in (no_grad, inference):
        assert torch.equal(out.view(torch.int16), training.view(torch.int16))


def test_run_to_run_bitwise_determinism():
    """Identical inputs give bitwise identical outputs and gradients (no atomics)."""
    length, batch, n, d = 512, 1, 4, 5120
    inputs = _inputs(length, batch, n, d, torch.bfloat16, seed=5, masked=True)
    grad = torch.randn(length, batch, n * d, device=inputs[0].device).bfloat16()
    first, _ = _run(fused_engram_gate, inputs, grad, n, d)
    for _ in range(2):
        again, _ = _run(fused_engram_gate, inputs, grad, n, d)
        for name, a, b in zip(_NAMES, first, again):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name


@pytest.mark.parametrize(
    "masked,padded",
    [(False, False), (True, False), (True, True)],
    ids=["plain", "masked", "masked-padded-rows"],
)
def test_cuda_graph_capture_and_replay(masked, padded):
    """Forward and backward capture in one CUDA graph and replay new data exactly.

    ``padded`` passes kv as the MXFP8 lookup's [rows, width] projection output, with NaN pad
    rows that are never read and come back zero in the captured gradient.
    """
    length, batch, n, d = 64, 2, 4, 1000
    tokens, width = length * batch, (n + 1) * d
    inputs = _inputs(length, batch, n, d, torch.bfloat16, seed=6, masked=masked)
    hidden, kv, q, k, mask = inputs
    if padded:
        pad = torch.full((24, width), float("nan"), dtype=kv.dtype, device=kv.device)
        kv = torch.cat([kv.reshape(tokens, width), pad])
    static = [t.detach().clone().requires_grad_(True) for t in (hidden, kv, q, k)]
    static_mask = None if mask is None else mask.clone()
    static_grad = torch.randn(length, batch, n * d, device=hidden.device).bfloat16()

    def step():
        kv_in = static[1] if padded else static[1].transpose(0, 1)
        out = fused_engram_gate(static[0], kv_in, static[2], static[3], static_mask, n, d, _EPS)
        return [out] + list(torch.autograd.grad(out, static, static_grad))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # compile the kernels outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed in (7, 8):
        new = _inputs(length, batch, n, d, torch.bfloat16, seed=seed, masked=masked)
        new_grad = torch.randn(length, batch, n * d, device=hidden.device).bfloat16()
        with torch.no_grad():
            for target, source in zip(static, new[:4]):
                if target.shape != source.shape:  # padded kv: replace the token rows only
                    target = target[:tokens].view(source.shape)
                target.copy_(source)
            static_grad.copy_(new_grad)
            if static_mask is not None:
                static_mask.copy_(new[4])
        graph.replay()
        expected, _ = _run(fused_engram_gate, new, new_grad, n, d)
        captured_kv_grad = captured[2]
        if padded:
            assert torch.count_nonzero(captured_kv_grad[tokens:]) == 0
            captured_kv_grad = captured_kv_grad[:tokens].view(batch, length, width)
        actual = captured[:2] + [captured_kv_grad] + captured[3:]
        for name, a, b in zip(_NAMES, actual, expected):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name


def _bf16_engram(n, d):
    """A small non-FP8 Engram (nn.Linear projection, local table) on the current GPU."""
    config = SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=True,
        init_method=lambda t: torch.nn.init.normal_(t, std=0.05),
        mhc_num_residual_streams=n,
        hidden_size=d,
        layernorm_epsilon=_EPS,
    )
    layout = EngramLayout(
        max_ngram_size=2,
        layer_ids=(0,),
        num_embeddings=(198,),
        primes=(((97, 101),),),
        n_heads=2,
        head_dim=32,
    )
    model = Engram(config, layout, 0, SimpleNamespace(ep=None, tp=None, expt_dp=None))
    model = model.to(_device())
    with torch.no_grad():
        gen = torch.Generator().manual_seed(9)
        model.q_weight.copy_(1 + 0.3 * torch.randn(n * d, generator=gen))
        model.k_weight.copy_(1 + 0.3 * torch.randn(n * d, generator=gen))
    return model


_VIEW_NODES = ("SliceBackward0", "ViewBackward0", "TransposeBackward0", "UnsafeViewBackward0")


def _kv_gradient_path(out):
    """Autograd node names from the fused gate's kv input down to the first non-view node."""
    node = out.grad_fn.next_functions[1][0]
    names = [type(node).__name__]
    while names[-1] in _VIEW_NODES:
        node = node.next_functions[0][0]
        names.append(type(node).__name__)
    return names


def _module_step(model, hidden, hashes, mask, grad, fused, monkeypatch, context=None):
    """Output, input and parameter gradients, and the fused gate's kv gradient path (or None)."""
    from contextlib import nullcontext

    monkeypatch.setattr(engram_gate, "FUSED_ENGRAM_GATE", fused)
    model.zero_grad(set_to_none=True)
    leaf = hidden.detach().clone().requires_grad_(True)
    with context if context is not None else nullcontext():
        out = model(leaf, hashes, mask)
    gate_node = type(out.grad_fn).__name__ == "FusedEngramGateBackward"
    path = _kv_gradient_path(out) if gate_node else None
    out.backward(grad)
    grads = {"hidden": leaf.grad}
    grads.update({name: p.grad.clone() for name, p in model.named_parameters()})
    return out.detach(), grads, path


@pytest.mark.parametrize("masked", [False, True])
def test_engram_forward_dispatches_to_the_fused_gate(monkeypatch, masked):
    """Engram.forward with the switch on matches its eager gate; off, the eager code runs."""
    n, d, length, batch = 4, 96, 24, 2
    model = _bf16_engram(n, d)
    device = _device()
    gen = torch.Generator().manual_seed(10)
    hidden = torch.randn(length, batch, n * d, generator=gen).bfloat16().to(device)
    hashes = torch.randint(0, 198, (batch, length, 1, 2), generator=gen).to(device)
    mask = (torch.rand(batch, length, generator=gen) > 0.25).to(device) if masked else None
    grad = torch.randn(length, batch, n * d, generator=gen).bfloat16().to(device)
    eager_out, eager_grads, path = _module_step(
        model, hidden, hashes, mask, grad, False, monkeypatch
    )
    assert path is None  # the eager gate ran
    # Off, Engram.forward is exactly the reference used above.
    with torch.no_grad():
        kv = model.wkv(model.lookup(hashes)).transpose(0, 1)
        reference = _eager_gate(hidden, kv, model.q_weight, model.k_weight, mask, n, d, _EPS)
    assert torch.equal(eager_out, reference)
    fused_out, fused_grads, path = _module_step(
        model, hidden, hashes, mask, grad, True, monkeypatch
    )
    assert path is not None  # the fused gate ran
    # Here keys are not correlated with the hidden states, so some tokens have a small |dot|,
    # whose FP32 evaluation (eager or fused) is ill conditioned; accuracy is covered above with
    # controlled conditioning, so the module path is compared normwise: within one BF16 ulp of
    # the largest element, for the output and every input and parameter gradient.
    assert _normwise(fused_out, eager_out) < 2.0**-7
    assert (
        set(fused_grads)
        == set(eager_grads)
        == {"hidden", "q_weight", "k_weight", "wkv.weight", "embed.tables.0.weight"}
    )
    for name in eager_grads:
        assert _normwise(fused_grads[name], eager_grads[name]) < 2.0**-7, name


def test_engram_forward_without_triton_keeps_the_eager_gate(monkeypatch):
    """With the switch on but Triton unavailable, Engram.forward runs the eager gate."""
    n, d, length, batch = 4, 96, 24, 2
    model = _bf16_engram(n, d)
    device = _device()
    gen = torch.Generator().manual_seed(16)
    hidden = torch.randn(length, batch, n * d, generator=gen).bfloat16().to(device)
    hashes = torch.randint(0, 198, (batch, length, 1, 2), generator=gen).to(device)
    mask = (torch.rand(batch, length, generator=gen) > 0.25).to(device)
    grad = torch.randn(length, batch, n * d, generator=gen).bfloat16().to(device)
    eager_out, eager_grads, _ = _module_step(model, hidden, hashes, mask, grad, False, monkeypatch)
    monkeypatch.setattr(engram_gate, "HAVE_TRITON", False)
    out, grads, path = _module_step(model, hidden, hashes, mask, grad, True, monkeypatch)
    assert path is None
    assert torch.equal(out, eager_out)
    for name in eager_grads:
        assert torch.equal(grads[name], eager_grads[name]), name


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="TE MXFP8 needs Blackwell",
)
@pytest.mark.parametrize(
    "fp8_param,length",
    [(False, 64), (True, 35), (True, 64)],
    ids=["bf16-table", "mxfp8-table-T35", "mxfp8-table-T64"],
)
def test_mxfp8_engram_fused_gate(monkeypatch, fp8_param, length):
    """The TE projection's (output, bias) tuple and the MXFP8 lookup's padded rows.

    With an MXFP8 table the projection output has the lookup's rows, padded to a multiple of
    32 (35 tokens: 29 pad rows; 64 tokens: none). It must reach the gate whole: the kv
    gradient goes straight to the projection's backward, with no slice (a zero-filled copy of
    the whole gradient), view or transpose node in between.
    """
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.models.test_v41_engram import _cuda_mxfp8_engram
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        model = _cuda_mxfp8_engram(
            ProcessGroupCollection.use_mpu_process_groups(), fp8_param=fp8_param
        )
        assert model.fp8_projection and model.fp8_lookup == fp8_param
        # BF16 table: TE's FP8 GEMM needs a token count that is a multiple of 8.
        gen = torch.Generator().manual_seed(11)
        hidden = torch.randn(length, 1, 128, generator=gen).bfloat16().cuda()
        hashes = torch.randint(0, 198, (1, length, 1, 2), generator=gen).cuda()
        grad = torch.randn(length, 1, 128, generator=gen).bfloat16().cuda()
        args = (model, hidden, hashes, None, grad)
        context = lambda: get_fp8_context(model.config)
        eager_out, eager_grads, _ = _module_step(*args, False, monkeypatch, context())
        fused_out, fused_grads, path = _module_step(*args, True, monkeypatch, context())
        # MXFP8 table: the padded [rows, width] projection output itself; BF16 table: its
        # [batch, sequence] output seen transposed, a free view in backward.
        assert path[-1] == "_LinearBackward" and "SliceBackward0" not in path, path
        assert path == ["_LinearBackward"] or not fp8_param, path
        assert _normwise(fused_out, eager_out) < 2.0**-7
        for name in ("hidden", "q_weight", "k_weight"):
            assert _normwise(fused_grads[name], eager_grads[name]) < 2.0**-7, name
        # grad_kv is MXFP8-quantized for the projection's backward, so where the BF16 grad_kv
        # of the two paths differs, a flipped rounding can move an element by one E4M3 step
        # (up to 2^-3 relative); the weight and table gradients sum many such elements.
        assert set(fused_grads) == set(eager_grads)
        for name in eager_grads:
            assert _normwise(fused_grads[name], eager_grads[name]) < 2.0**-4, name
    finally:
        Utils.destroy_model_parallel()
