# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU tests for the fused CuTe DSL KDA backward behind the internal KDA adapter."""

import importlib.util

import pytest
import torch


def _fused_kda_bwd_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec("cutlass") is None:
        return False
    return True


pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not _fused_kda_bwd_available(), reason="needs SM100, FLA and the CuTe DSL runtime"
    ),
]

_HEADS = 4
_DIM = 128


def _adapter():
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import kda as adapter

    return adapter


def _force_path(monkeypatch, path):
    """Pin the backward to the tape pipeline or the fused recurrence."""
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_kda_bwd_cute import (
        launcher,
    )

    monkeypatch.setattr(launcher, "prefer_tape", lambda *args: path == "tape")


def _inputs(batch, seq_len, gate_range, seed=0, heads=_HEADS):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    shape = (batch, seq_len, heads, _DIM)
    q = torch.nn.functional.normalize(randn(*shape), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(randn(*shape), dim=-1).bfloat16()
    v = randn(*shape).bfloat16()
    raw_g = randn(*shape).bfloat16()
    beta = torch.rand(batch, seq_len, heads, generator=generator, device="cuda")
    low, high = gate_range
    A_log = (low + (high - low) * torch.rand(heads, generator=generator, device="cuda")).float()
    dt_bias = 0.1 * randn(heads * _DIM)
    do = randn(*shape).bfloat16()
    return (q, k, v, raw_g, beta, A_log, dt_bias), do


def _grads(chunk_kda, leaves, do, cu_seqlens, safe_gate, lower_bound):
    leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    q, k, v, raw_g, beta, A_log, dt_bias = leaves
    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return (out, *torch.autograd.grad(out, leaves, do))


@pytest.mark.parametrize(
    "lengths, safe_gate, lower_bound, A_log_range",
    [
        ([130, 64, 257], False, None, (-1.0, 1.0)),
        ([130, 64, 257], True, -5.0, (-1.0, 1.0)),
        # Strong gates push many diagonal blocks past DIAG_SPAN_LIMIT (CUDA-core fallback).
        ([130, 64, 257], False, None, (1.5, 2.5)),
        (None, False, None, (-1.0, 1.0)),
    ],
    ids=["unbounded", "bounded", "strong_unbounded", "dense_batch2"],
)
@pytest.mark.parametrize("path", ["tape", "fused"])
def test_fused_bwd_matches_fla(monkeypatch, path, lengths, safe_gate, lower_bound, A_log_range):
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.delenv("MCORE_KDA_FUSED_BWD", raising=False)
    _force_path(monkeypatch, path)
    if lengths is None:
        leaves, do = _inputs(2, 192, A_log_range)
        cu_seqlens = None
    else:
        leaves, do = _inputs(1, sum(lengths), A_log_range)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )

    # The fused backward evaluates the intra-chunk terms exactly for any gate, so the
    # reference is FLA's exact path (safe_gate=False) on the same gate activation:
    # FLA's safe_gate M=16 approximation alone moves ddt_bias by about 5e-2.
    reference = _grads(fla_chunk_kda, leaves, do, cu_seqlens, False, lower_bound)
    before = adapter.get_fused_kda_bwd_launch_count()
    candidate = _grads(adapter.chunk_kda, leaves, do, cu_seqlens, safe_gate, lower_bound)
    assert adapter.get_fused_kda_bwd_launch_count() == before + 1

    names = ("out", "dq", "dk", "dv", "draw_g", "dbeta", "dA_log", "ddt_bias")
    for name, got, want in zip(names, candidate, reference):
        assert torch.isfinite(got).all(), name
        relative_l2 = ((got.float() - want.float()).norm() / want.float().norm()).item()
        assert relative_l2 < 1e-2, f"{name}: relative L2 {relative_l2:.3e}"


@pytest.mark.parametrize("path", ["tape", "fused"])
def test_fused_bwd_byte_offsets_past_2gib(monkeypatch, path):
    """Rows of the FP32 gate cumsum past token 65536 at 64 heads lie more than
    2 GiB into the tensor, so their byte offsets need 64 bits."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.delenv("MCORE_KDA_FUSED_BWD", raising=False)
    _force_path(monkeypatch, path)
    seq_len = 66000
    leaves, do = _inputs(1, seq_len, (-1.0, 1.0), heads=64)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    reference = _grads(fla_chunk_kda, leaves, do, cu_seqlens, False, None)
    candidate = _grads(adapter.chunk_kda, leaves, do, cu_seqlens, False, None)
    names = ("out", "dq", "dk", "dv", "draw_g", "dbeta", "dA_log", "ddt_bias")
    for name, got, want in zip(names, candidate, reference):
        # The last chunks hold the rows past the 2 GiB boundary.
        if got.dim() == 4:
            got, want = got[:, 65536:], want[:, 65536:]
        assert torch.isfinite(got).all(), name
        relative_l2 = ((got.float() - want.float()).norm() / want.float().norm()).item()
        assert relative_l2 < 1e-2, f"{name}: relative L2 {relative_l2:.3e}"


def test_fused_bwd_can_be_disabled(monkeypatch):
    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.setenv("MCORE_KDA_FUSED_BWD", "0")
    leaves, do = _inputs(1, 128, (-1.0, 1.0))
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
    before = adapter.get_fused_kda_bwd_launch_count()
    _grads(adapter.chunk_kda, leaves, do, cu_seqlens, False, None)
    assert adapter.get_fused_kda_bwd_launch_count() == before


@pytest.mark.parametrize("path", ["tape", "fused"])
def test_fused_bwd_compiles_once_across_packings(monkeypatch, path):
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels.fused_kda_bwd_cute import (
        launcher,
    )

    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.delenv("MCORE_KDA_FUSED_BWD", raising=False)
    _force_path(monkeypatch, path)
    # Token count, chunk count, tail lengths and sequence count all change, and
    # with a tape so does the number of CTAs per (sequence, head).
    packings = ([64], [130, 64, 257], [1, 63, 65, 128, 200], [512, 0, 77])
    sizes = []
    for seed, lengths in enumerate(packings):
        leaves, do = _inputs(1, sum(lengths), (-1.0, 1.0), seed=seed)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )
        reference = _grads(fla_chunk_kda, leaves, do, cu_seqlens, False, None)
        candidate = _grads(adapter.chunk_kda, leaves, do, cu_seqlens, False, None)
        for got, want in zip(candidate, reference):
            relative_l2 = ((got.float() - want.float()).norm() / want.float().norm()).item()
            assert relative_l2 < 1e-2, f"{lengths}: relative L2 {relative_l2:.3e}"
        sizes.append(launcher.compile_cache_size())
    assert sizes[1:] == sizes[:1] * (len(sizes) - 1), sizes


def test_fused_bwd_saved_state_matches_recompute(monkeypatch):
    from functools import partial

    adapter = _adapter()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    monkeypatch.delenv("MCORE_KDA_FUSED_BWD", raising=False)
    lengths = [130, 64, 257]
    leaves, do = _inputs(1, sum(lengths), (-1.0, 1.0))
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    saved = _grads(
        partial(adapter.chunk_kda, recompute_h=False), leaves, do, cu_seqlens, True, -5.0
    )
    recomputed = _grads(
        partial(adapter.chunk_kda, recompute_h=True), leaves, do, cu_seqlens, True, -5.0
    )
    # Same math either way; only the order of the backward's floating-point
    # reductions may differ between runs.
    for got, want in zip(saved, recomputed):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)
