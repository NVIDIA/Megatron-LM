# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the Triton varlen causal conv1d kernel.

Tests correctness of `causal_conv1d_varlen_fn` against a reference implementation
that loops over requests calling `causal_conv1d_fn` with `initial_states`.
"""

import pytest
import torch

from megatron.core.ssm.ops.causal_conv1d_varlen import causal_conv1d_varlen_fn

try:
    from causal_conv1d import causal_conv1d_fn

    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False


def _reference_conv1d_varlen(x, weight, bias, cu_seqlens, initial_states, activation="silu"):
    """Reference: per-request loop calling causal_conv1d_fn with initial_states."""
    num_requests = cu_seqlens.shape[0] - 1
    conv_dim = x.shape[1]
    d_conv = weight.shape[1]
    parts = []
    for r in range(num_requests):
        start = cu_seqlens[r].item()
        end = cu_seqlens[r + 1].item()
        if end <= start:
            continue
        seq_len_r = end - start
        if initial_states is not None:
            init_r = initial_states[r : r + 1]  # (1, conv_dim, d_conv-1)
            # causal_conv1d_fn with initial_states requires channels-last layout
            # for both x and initial_states: create as (1, L, C) then transpose
            x_r = x[start:end].unsqueeze(0).transpose(1, 2)  # channels-last (1, C, L)
            init_r = init_r.permute(0, 2, 1).contiguous().transpose(1, 2)  # channels-last
        else:
            init_r = None
            x_r = x[start:end].T.unsqueeze(0).contiguous()  # (1, conv_dim, seq_len)
        out_r = causal_conv1d_fn(
            x=x_r, weight=weight, bias=bias, activation=activation, initial_states=init_r
        )
        parts.append(out_r.squeeze(0).T.contiguous())  # (seq_len, conv_dim)
    return torch.cat(parts, dim=0) if parts else torch.empty(0, conv_dim, device=x.device)


@pytest.mark.skipif(not HAS_CAUSAL_CONV1D, reason="causal_conv1d not installed")
class TestCausalConv1dVarlen:
    """Test causal_conv1d_varlen_fn against per-request causal_conv1d_fn reference."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_single_request(self, dtype):
        """Single request should match causal_conv1d_fn exactly."""
        torch.manual_seed(42)
        conv_dim, d_conv, seq_len = 64, 4, 32
        device = "cuda"

        x = torch.randn(seq_len, conv_dim, dtype=dtype, device=device)
        weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
        bias = torch.randn(conv_dim, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        initial_states = torch.randn(1, conv_dim, d_conv - 1, dtype=dtype, device=device)

        out = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, initial_states)
        ref = _reference_conv1d_varlen(x, weight, bias, cu_seqlens, initial_states)

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_multiple_requests_varying_lengths(self, dtype):
        """Multiple requests with different sequence lengths."""
        torch.manual_seed(123)
        conv_dim, d_conv = 128, 4
        seq_lens = [10, 25, 3, 50, 8]
        device = "cuda"

        total_tokens = sum(seq_lens)
        x = torch.randn(total_tokens, conv_dim, dtype=dtype, device=device)
        weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
        bias = torch.randn(conv_dim, dtype=dtype, device=device)

        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

        num_requests = len(seq_lens)
        initial_states = torch.randn(num_requests, conv_dim, d_conv - 1, dtype=dtype, device=device)

        out = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, initial_states)
        ref = _reference_conv1d_varlen(x, weight, bias, cu_seqlens, initial_states)

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)

    def test_seqlen_shorter_than_d_conv(self):
        """Sequence shorter than d_conv should use initial_states for all taps."""
        torch.manual_seed(7)
        conv_dim, d_conv = 32, 4
        seq_lens = [2, 1, 3]  # All shorter than d_conv
        device = "cuda"
        dtype = torch.float32

        total_tokens = sum(seq_lens)
        x = torch.randn(total_tokens, conv_dim, dtype=dtype, device=device)
        weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
        bias = torch.randn(conv_dim, dtype=dtype, device=device)

        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

        num_requests = len(seq_lens)
        initial_states = torch.randn(num_requests, conv_dim, d_conv - 1, dtype=dtype, device=device)

        out = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, initial_states)
        ref = _reference_conv1d_varlen(x, weight, bias, cu_seqlens, initial_states)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_zero_initial_states(self):
        """Zero initial_states should produce same result as None initial_states."""
        torch.manual_seed(99)
        conv_dim, d_conv = 64, 4
        seq_lens = [16, 24]
        device = "cuda"
        dtype = torch.float32

        total_tokens = sum(seq_lens)
        x = torch.randn(total_tokens, conv_dim, dtype=dtype, device=device)
        weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
        bias = torch.randn(conv_dim, dtype=dtype, device=device)

        cu_seqlens_list = [0]
        for sl in seq_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + sl)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

        num_requests = len(seq_lens)
        zero_states = torch.zeros(num_requests, conv_dim, d_conv - 1, dtype=dtype, device=device)

        out_zero = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, zero_states)
        out_none = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, None)

        torch.testing.assert_close(out_zero, out_none, atol=1e-5, rtol=1e-5)

    def test_nonzero_vs_zero_initial_states_differ(self):
        """Non-zero initial_states should produce different results from zero."""
        torch.manual_seed(55)
        conv_dim, d_conv = 64, 4
        seq_len = 16
        device = "cuda"
        dtype = torch.float32

        x = torch.randn(seq_len, conv_dim, dtype=dtype, device=device)
        weight = torch.randn(conv_dim, d_conv, dtype=dtype, device=device)
        bias = torch.randn(conv_dim, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        nonzero_states = torch.randn(1, conv_dim, d_conv - 1, dtype=dtype, device=device)

        out_nonzero = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, nonzero_states)
        out_none = causal_conv1d_varlen_fn(x, weight, bias, cu_seqlens, None)

        # First few tokens should differ (those that depend on initial state)
        assert not torch.allclose(
            out_nonzero[: d_conv - 1], out_none[: d_conv - 1], atol=1e-5
        ), "Non-zero initial states should produce different outputs for early tokens"
