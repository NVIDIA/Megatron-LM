# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Equivalence tests for the fused GDP decode preparation kernel.

The reference below is the eager sequence the kernel replaces, copied from
``GatedDeltaProductMixer.ssm_decode``: ``_prepare_qkv``'s post-conv split and
GQA expansion, ``_compute_gating``, and the Householder interleave.
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core.ssm.ops.gdp.decode_prepare import gdp_decode_prepare


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def decode_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N):
    """Eager reference: the ops between the conv update and the recurrent kernel."""
    value, key, query = torch.split(x, [H * P * M, G * N * M, G * N], dim=-1)
    value = rearrange(value, "b l (m h p) -> b (l m) h p", m=M, p=P).contiguous()
    key = rearrange(key, "b l (m g n) -> b (l m) g n", m=M, n=N).contiguous()
    query = rearrange(query, "b l (g n) -> b l g n", n=N).contiguous()
    if H // G > 1:
        query = query.repeat_interleave(H // G, dim=-2)
        key = key.repeat_interleave(H // G, dim=-2)

    b, a = torch.split(ba, [H * M, H], dim=-1)
    beta = b.contiguous().sigmoid()
    g = -A_log.float().exp() * F.softplus(a.contiguous().float() + dt_bias)
    beta = rearrange(beta, "b l (m h) -> b (l m) h", m=M).contiguous()

    g_new = g.new_zeros(g.shape[0], g.shape[1], M, g.shape[2])
    g_new[:, :, 0] = g
    g = rearrange(g_new, "n t m h -> n (t m) h")
    query_new = query.new_zeros(query.shape[0], query.shape[1], M, query.shape[2], query.shape[3])
    query_new[:, :, -1] = query
    query = rearrange(query_new, "n t m h d -> n (t m) h d")
    return query, key, value, beta, g


@pytest.mark.internal
class TestGDPDecodePrepare:
    """Match the fused kernel against the eager sequence it replaces."""

    @pytest.mark.parametrize("num_householder", [1, 2, 3])
    @pytest.mark.parametrize("num_heads,num_groups", [(8, 8), (8, 2), (4, 1)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_matches_reference(self, num_householder, num_heads, num_groups, dtype):
        _requires_cuda()
        torch.manual_seed(1234)
        M, H, G, P, N = num_householder, num_heads, num_groups, 64, 32
        n = 5
        device = "cuda"

        conv_dim = H * P * M + (M + 1) * G * N
        # `x` and `ba` reach the kernel as slices of one projection output, so
        # exercise the strided-row path rather than handing it dense tensors.
        packed = torch.randn(n, 1, conv_dim + (M + 1) * H, device=device, dtype=dtype)
        x, ba = torch.split(packed, [conv_dim, (M + 1) * H], dim=-1)
        A_log = torch.randn(H, device=device, dtype=torch.float32)
        dt_bias = torch.randn(H, device=device, dtype=dtype)

        out = gdp_decode_prepare(
            x,
            ba,
            A_log=A_log,
            dt_bias=dt_bias,
            num_householder=M,
            num_heads=H,
            num_groups=G,
            head_dim=P,
            state_dim=N,
        )
        ref = decode_prepare_ref(x, ba, A_log, dt_bias, M, H, G, P, N)

        query, key, value, beta, g = out
        assert query.shape == (n, M, H, N)
        assert key.shape == (n, M, H, N)
        assert value.shape == (n, M, H, P)
        assert beta.shape == (n, M, H)
        assert g.shape == (n, M, H)
        assert g.dtype == torch.float32
        assert value.dtype == dtype and beta.dtype == dtype

        # Every output must match bit for bit, not just closely: the GDP inference
        # functional tests compare against golden values produced by the eager
        # path, so a ulp in `beta` or `g` compounds through the recurrence. That
        # holds only while the kernel's math goes through libdevice rather than
        # Triton's approximate fp32 `exp` and `/` -- if a Triton or torch upgrade
        # changes either side's lowering, this is where it should surface.
        names = ("query", "key", "value", "beta", "g")
        for got, want, name in zip(out, ref, names):
            assert torch.equal(got, want), f"{name} is not bitwise equal to the eager path"

    def test_decay_and_query_placement(self):
        """The decay sits on the first Householder copy, the query on the last."""
        _requires_cuda()
        torch.manual_seed(0)
        M, H, G, P, N = 3, 4, 2, 16, 16
        n = 2
        x = torch.randn(n, 1, H * P * M + (M + 1) * G * N, device="cuda", dtype=torch.bfloat16)
        ba = torch.randn(n, 1, (M + 1) * H, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        query, _, _, _, g = gdp_decode_prepare(
            x,
            ba,
            A_log=A_log,
            dt_bias=dt_bias,
            num_householder=M,
            num_heads=H,
            num_groups=G,
            head_dim=P,
            state_dim=N,
        )
        assert (query[:, :-1] == 0).all()
        assert (query[:, -1] != 0).any()
        assert (g[:, 1:] == 0).all()
        assert (g[:, 0] != 0).any()

    @pytest.mark.parametrize("a_value", [100.0, 20.0, 0.0, -20.0, -80.0])
    def test_softplus_tails(self, a_value):
        """softplus saturates to the identity on the way up and keeps its
        significant digits on the way down, where `log(1 + exp(x))` would flush
        to zero and only `log1p` is accurate."""
        _requires_cuda()
        M, H, G, P, N = 1, 2, 2, 16, 16
        x = torch.zeros(1, 1, H * P * M + (M + 1) * G * N, device="cuda", dtype=torch.float32)
        ba = torch.zeros(1, 1, (M + 1) * H, device="cuda", dtype=torch.float32)
        ba[..., M * H :] = a_value
        A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)

        _, _, _, _, g = gdp_decode_prepare(
            x,
            ba,
            A_log=A_log,
            dt_bias=dt_bias,
            num_householder=M,
            num_heads=H,
            num_groups=G,
            head_dim=P,
            state_dim=N,
        )
        expected = -F.softplus(torch.full((1, H), a_value, device="cuda"))
        torch.testing.assert_close(g[:, 0], expected, rtol=1e-6, atol=0.0)
