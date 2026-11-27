# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from megatron.core.optimizer import cute_dsl_multi_tensor

CHUNK = 65536


def _sm100_with_cute_dsl():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


requires_kernels = pytest.mark.skipif(
    not _sm100_with_cute_dsl(), reason="needs an SM100+ GPU and nvidia-cutlass-dsl"
)


class _Recorder:
    def __init__(self, ret=None):
        self.calls = 0
        self.ret = ret

    def __call__(self, *args):
        self.calls += 1
        return self.ret


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.setattr(cute_dsl_multi_tensor, "_kernels", None)


def test_count_zeros_falls_through_when_disabled(disabled):
    assert cute_dsl_multi_tensor.count_zeros(3, [torch.zeros(3)]) is None


def test_wrappers_fall_through_when_disabled(disabled):
    te_adam, te_l2norm = _Recorder(), _Recorder(ret=("norm", "per_tensor"))
    adam = cute_dsl_multi_tensor.adam_impl(te_adam)
    l2norm = cute_dsl_multi_tensor.l2norm_impl(te_l2norm)
    lists = [[torch.zeros(4)] for _ in range(4)]
    adam(CHUNK, None, lists, 1e-3, 0.9, 0.95, 1e-8, 1, 1, 1, 0.0)
    assert te_adam.calls == 1
    assert l2norm(CHUNK, None, lists[:1], False) == ("norm", "per_tensor")
    assert te_l2norm.calls == 1


def _sizes(total_chunks, seed=0):
    """Many small tensors, some spanning several chunks."""
    rng = np.random.default_rng(seed)
    sizes, chunks = [], 0
    while chunks < total_chunks:
        kind = rng.random()
        if kind < 0.4:
            n = int(rng.integers(1, 20000))
        elif kind < 0.8:
            n = int(rng.integers(20000, 3 * CHUNK))
        else:
            n = int(rng.integers(3 * CHUNK, 30 * CHUNK))
        sizes.append(n)
        chunks += (n + CHUNK - 1) // CHUNK
    return sizes


@pytest.fixture
def kernels():
    assert cute_dsl_multi_tensor.enable()
    from megatron.core.optimizer import cute_dsl_multi_tensor_kernels

    return cute_dsl_multi_tensor_kernels


@requires_kernels
def test_adam_matches_te(kernels):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import multi_tensor_applier

    torch.manual_seed(0)
    sizes = _sizes(1000)
    g = [torch.randn(n, device="cuda") * 1e-3 for n in sizes]
    p = [torch.randn(n, device="cuda") * 0.02 for n in sizes]
    m = [torch.randn(n, device="cuda") * 1e-4 for n in sizes]
    v = [torch.rand(n, device="cuda") * 1e-7 for n in sizes]
    p_ref, m_ref, v_ref = [[t.clone() for t in lst] for lst in (p, m, v)]
    noop = torch.zeros(1, dtype=torch.int32, device="cuda")
    args = (8e-4, 0.9, 0.95, 1e-8)
    adam = cute_dsl_multi_tensor.adam_impl(tex.multi_tensor_adam)
    for step in range(1, 4):
        # Fresh grad objects every step, the same moment objects: the cached table must
        # re-read the grads.
        gs = [t * step for t in g]
        multi_tensor_applier(adam, noop, [gs, p, m, v], *args, step, 1, 1, 0.1)
        multi_tensor_applier(
            tex.multi_tensor_adam, noop, [gs, p_ref, m_ref, v_ref], *args, step, 1, 1, 0.1
        )
    # New moment objects (a reloaded optimizer state) must miss the cached table.
    m, v = [t.clone() for t in m], [t.clone() for t in v]
    multi_tensor_applier(adam, noop, [g, p, m, v], *args, 4, 1, 1, 0.1)
    multi_tensor_applier(tex.multi_tensor_adam, noop, [g, p_ref, m_ref, v_ref], *args, 4, 1, 1, 0.1)
    # Swapped storage under the same moment objects must be detected.
    m[0].data = m[0].data.clone()
    multi_tensor_applier(adam, noop, [g, p, m, v], *args, 5, 1, 1, 0.1)
    multi_tensor_applier(tex.multi_tensor_adam, noop, [g, p_ref, m_ref, v_ref], *args, 5, 1, 1, 0.1)
    # Same fp32 terms as TE; only rounding of the fused operations may differ, so compare each
    # tensor relative to its largest magnitude.
    for ours, ref in ((p, p_ref), (m, m_ref), (v, v_ref)):
        for a, b in zip(ours, ref):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-5 * b.abs().max().item())


@requires_kernels
def test_l2norm_matches_te(kernels):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import multi_tensor_applier

    torch.manual_seed(0)
    g = [torch.randn(n, device="cuda") for n in _sizes(1000)]
    p = [t * 2 for t in g]
    misaligned = [t[1:] for t in g[:50] if t.numel() > 1]
    noop = torch.zeros(1, dtype=torch.int32, device="cuda")
    l2norm = cute_dsl_multi_tensor.l2norm_impl(tex.multi_tensor_l2norm)
    # g and p share one numel-keyed table; back-to-back calls must each see their own
    # addresses.
    outs = [multi_tensor_applier(l2norm, noop, [lst], False)[0] for lst in (g, p, misaligned)]
    for lst, ours in zip((g, p, misaligned), outs):
        ref, _ = multi_tensor_applier(tex.multi_tensor_l2norm, noop, [lst], False)
        torch.testing.assert_close(ours, ref, rtol=1e-5, atol=0)
    assert noop.item() == 0
    # Repeated launches are deterministic.
    first = multi_tensor_applier(l2norm, noop, [g], False)[0].item()
    for _ in range(3):
        assert multi_tensor_applier(l2norm, noop, [g], False)[0].item() == first


@requires_kernels
def test_l2norm_non_finite_sets_noop_and_bf16_falls_back(kernels):
    noop = torch.zeros(1, dtype=torch.int32, device="cuda")
    bad = [torch.randn(1000, device="cuda") for _ in range(5)]
    bad[3][7] = float("inf")
    kernels.multi_tensor_l2norm(bad, noop)
    assert noop.item() == 1
    assert kernels.multi_tensor_l2norm([bad[0].bfloat16()], noop) is None


@requires_kernels
def test_count_zeros_matches_count_nonzero(kernels):
    torch.manual_seed(0)
    buf = torch.randn(20_000_000, device="cuda").bfloat16()
    buf[torch.rand(buf.numel(), device="cuda") < 0.33] = 0
    buf[5], buf[7], buf[11] = -0.0, float("nan"), float("inf")
    cuts = [0, 64 * 1000, 64 * 70_001, 64 * 200_000, buf.numel()]
    regions = [buf[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    f32 = [torch.randn(n, device="cuda") for n in (1, 777, CHUNK * 3 + 5)]
    for t in f32:
        t[torch.rand(t.numel(), device="cuda") < 0.5] = 0
    f16 = [torch.randn(100_003, device="cuda").half()]
    f16[0][::3] = 0
    cases = (regions, regions + f32, regions + f32 + f16, [buf[1:1_000_001], buf[3:4_000_003]])
    for tensors in cases:
        base = sum(t.numel() for t in tensors) + 7
        ours = cute_dsl_multi_tensor.count_zeros(base, tensors)
        assert ours.dtype is torch.int64 and ours.dim() == 0
        assert ours.item() == base - sum(torch.count_nonzero(t).item() for t in tensors)
    # Same numels, other storage: the cached table must re-read the addresses.
    other = buf.clone()
    other[: buf.numel() // 2] = 0
    regions2 = [other[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    base = buf.numel()
    first = cute_dsl_multi_tensor.count_zeros(base, regions)
    second = cute_dsl_multi_tensor.count_zeros(base, regions2)
    assert first.item() == base - torch.count_nonzero(buf).item()
    assert second.item() == base - torch.count_nonzero(other).item()
    # Unsupported inputs fall back; empty tensors count nothing.
    assert cute_dsl_multi_tensor.count_zeros(0, [buf[:10].to(torch.int32)]) is None
    assert cute_dsl_multi_tensor.count_zeros(0, [buf[:20:2]]) is None
    assert cute_dsl_multi_tensor.count_zeros(9, [buf[:0]]).item() == 9
