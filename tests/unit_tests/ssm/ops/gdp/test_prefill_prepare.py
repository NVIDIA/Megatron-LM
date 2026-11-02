# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Equivalence tests for the fused GDP prefill preparation kernel.

The reference below is the eager sequence the kernel replaces in
``GatedDeltaProductMixer.ssm_prefill``: ``_prepare_qkv``'s post-conv split,
Householder interleave and GQA expansion (non-CuTeDSL layout, L2 norm left to the
chunk kernel), followed by ``_compute_gating``.
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core.ssm.ops.gdp.prepare import gdp_prefill_prepare


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def prefill_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N):
    """Eager reference: the ops between the varlen conv and the chunked recurrence."""
    value, key, query = torch.split(x, [H * P * M, G * N * M, G * N], dim=-1)
    value = rearrange(value, "b l (m h p) -> b (l m) h p", m=M, p=P).contiguous()
    key = rearrange(key, "b l (m g n) -> b (l m) g n", m=M, n=N).contiguous()
    query = rearrange(query, "b l (g n) -> b l g n", n=N).contiguous()
    if H // G > 1:
        query = query.repeat_interleave(H // G, dim=-2)
        key = key.repeat_interleave(H // G, dim=-2)

    b, a = torch.split(ba, [H * M, H], dim=-1)
    b, a = b.contiguous(), a.contiguous()
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = rearrange(beta, "b l (m h) -> b (l m) h", m=M).contiguous()
    return query, key, value, beta, g


def _inputs(n, T, M, H, G, P, N, dtype, dt_bias_dtype, seed=1234):
    torch.manual_seed(seed)
    conv_dim = H * P * M + (M + 1) * G * N
    # `ba` reaches the kernel as a slice of the input projection, so exercise the
    # strided-row path rather than handing it a dense tensor. `x` is the varlen
    # conv's dense output in production; a strided slice is exercised as well.
    packed = torch.randn(n, T, conv_dim + (M + 1) * H, device="cuda", dtype=dtype)
    x, ba = torch.split(packed, [conv_dim, (M + 1) * H], dim=-1)
    A_log = torch.randn(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H, device="cuda", dtype=dt_bias_dtype)
    return x, ba, A_log, dt_bias


@pytest.mark.internal
class TestGDPPrefillPrepare:
    """Match the fused kernel against the eager sequence it replaces."""

    @pytest.mark.parametrize("num_householder", [1, 2, 3])
    @pytest.mark.parametrize("num_heads,num_groups", [(8, 8), (8, 2), (20, 20), (4, 1)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("n,seq_len", [(1, 1), (1, 7), (1, 257), (2, 33)])
    def test_matches_reference(self, num_householder, num_heads, num_groups, dtype, n, seq_len):
        _requires_cuda()
        M, H, G, P, N = num_householder, num_heads, num_groups, 64, 32
        x, ba, A_log, dt_bias = _inputs(n, seq_len, M, H, G, P, N, dtype, dtype)

        out = gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N)
        ref = prefill_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N)

        query, key, value, beta, g = out
        T = seq_len
        assert query.shape == (n, T, H, N)
        assert key.shape == (n, T * M, H, N)
        assert value.shape == (n, T * M, H, P)
        assert beta.shape == (n, T * M, H)
        assert g.shape == (n, T, H)
        assert g.dtype == torch.float32
        assert value.dtype == dtype and beta.dtype == dtype
        # Every output must match bit for bit, not just closely: the GDP inference
        # functional tests compare against golden values produced by the eager path.
        for got, want, name in zip(out, ref, ("query", "key", "value", "beta", "g")):
            assert got.is_contiguous(), f"{name} must be contiguous like the eager path"
            assert torch.equal(got, want), f"{name} is not bitwise equal to the eager path"

    @pytest.mark.parametrize("dt_bias_dtype", [torch.float32, torch.bfloat16])
    def test_full_step_non_power_of_two_heads(self, dt_bias_dtype):
        """A 4096-token prefill step with a non-power-of-two head count, so the kernel's
        head block is masked."""
        _requires_cuda()
        M, H, G, P, N = 3, 12, 12, 64, 128
        x, ba, A_log, dt_bias = _inputs(1, 4096, M, H, G, P, N, torch.bfloat16, dt_bias_dtype)
        out = gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N)
        ref = prefill_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N)
        for got, want, name in zip(out, ref, ("query", "key", "value", "beta", "g")):
            assert torch.equal(got, want), f"{name} is not bitwise equal to the eager path"

    def test_dense_input(self):
        """The varlen conv hands the kernel a dense `[1, T, width]` tensor."""
        _requires_cuda()
        M, H, G, P, N = 2, 8, 4, 32, 64
        x, ba, A_log, dt_bias = _inputs(1, 100, M, H, G, P, N, torch.bfloat16, torch.float32)
        x = x.contiguous()
        out = gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N)
        ref = prefill_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N)
        for got, want, name in zip(out, ref, ("query", "key", "value", "beta", "g")):
            assert torch.equal(got, want), f"{name} is not bitwise equal to the eager path"

    def test_empty_step(self):
        """A zero-token step allocates empty outputs and launches nothing."""
        _requires_cuda()
        M, H, G, P, N = 2, 4, 2, 16, 16
        x, ba, A_log, dt_bias = _inputs(1, 0, M, H, G, P, N, torch.bfloat16, torch.float32)
        query, key, value, beta, g = gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N)
        assert query.shape == (1, 0, H, N) and key.shape == (1, 0, H, N)
        assert value.shape == (1, 0, H, P) and beta.shape == (1, 0, H) and g.shape == (1, 0, H)

    @pytest.mark.parametrize("a_value", [100.0, 20.0, 0.0, -20.0, -80.0])
    def test_softplus_tails(self, a_value):
        """softplus saturates to the identity on the way up and keeps its significant
        digits on the way down, matching F.softplus bitwise at both tails."""
        _requires_cuda()
        M, H, G, P, N = 1, 2, 2, 16, 16
        x, ba, A_log, dt_bias = _inputs(1, 3, M, H, G, P, N, torch.float32, torch.float32)
        ba = ba.clone()
        ba[..., M * H :] = a_value
        dt_bias = torch.zeros_like(dt_bias)
        out = gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N)
        ref = prefill_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N)
        assert torch.equal(out[4], ref[4])
