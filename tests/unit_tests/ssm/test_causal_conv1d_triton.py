# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.ops.causal_conv1d_triton import (
    causal_conv1d_update,
    gather_conv_state,
    roll_conv_varlen_states,
    scatter_conv_state,
)


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------- Reference Implementations ---------------------- #


def roll_conv_varlen_states_ref(conv_varlen_states, cu_seqlens):
    """Reference: roll each [D, W] slice by (seqlen % W) positions."""
    B, D, W = conv_varlen_states.shape
    out = torch.empty_like(conv_varlen_states)
    for b in range(B):
        seqlen = (cu_seqlens[b + 1] - cu_seqlens[b]).item()
        shift = seqlen % W
        for d in range(D):
            for w in range(W):
                src = (w - shift + W) % W
                out[b, d, w] = conv_varlen_states[b, d, src]
    return out


def gather_conv_state_ref(conv_state, batch_indices, cache_seqlens, d_conv):
    """Reference: read last (d_conv-1) elements from circular buffer."""
    B = batch_indices.shape[0]
    D = conv_state.shape[1]
    state_len = conv_state.shape[2]
    out = torch.zeros((B, D, d_conv - 1), device=conv_state.device, dtype=conv_state.dtype)
    for b in range(B):
        req_idx = batch_indices[b].item()
        if req_idx < 0:
            continue
        seq_len = cache_seqlens[b].item()
        for d in range(D):
            for w in range(d_conv - 1):
                val = seq_len - d_conv + 1 + w
                if val < 0:
                    continue
                idx = (val + state_len) % state_len
                out[b, d, w] = conv_state[req_idx, d, idx]
    return out


def scatter_conv_state_ref(conv_state, xBC, batch_indices, cache_seqlens):
    """Reference: write newest chunk into circular buffer."""
    state_len = conv_state.shape[2]
    chunk_len = xBC.shape[-1]
    copy_len = min(chunk_len, state_len)
    xBC_tail = xBC[..., -copy_len:]
    B, D, _ = xBC_tail.shape
    for b in range(B):
        req_idx = batch_indices[b].item()
        if req_idx < 0:
            continue
        seq_len = cache_seqlens[b].item()
        for d in range(D):
            for w in range(copy_len):
                idx = (seq_len + chunk_len - copy_len + w) % state_len
                conv_state[req_idx, d, idx] = xBC_tail[b, d, w]


def causal_conv1d_update_ref(x, conv_state, weight, bias, silu_activation):
    """Reference: linear (non-circular) causal conv1d update."""
    batch, seq_len, dim = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    out = torch.empty_like(x)
    for b in range(batch):
        for s in range(seq_len):
            # Shift state left by 1
            conv_state[b, :, :-1] = conv_state[b, :, 1:].clone()
            conv_state[b, :, -1] = x[b, s, :]
            # Convolution over the last `width` elements
            window = conv_state[b, :, state_len - width : state_len].float()
            w = weight.float()
            val = (window * w).sum(dim=1)
            if bias is not None:
                val = val + bias.float()
            if silu_activation:
                val = val * torch.sigmoid(val)
            out[b, s, :] = val.to(x.dtype)
    return out


# ---------------------- Tests ---------------------- #


@pytest.mark.internal
class TestRollConvVarlenStates:

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("B,D,W", [(1, 4, 4), (3, 8, 4), (2, 16, 3)])
    def test_matches_reference(self, B, D, W):
        torch.manual_seed(42)
        conv_states = torch.randn(B, D, W, device="cuda", dtype=torch.float32)
        seqlens = torch.randint(1, 20, (B,), device="cuda", dtype=torch.int32)
        cu_seqlens = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

        result = roll_conv_varlen_states(conv_states, cu_seqlens)
        expected = roll_conv_varlen_states_ref(conv_states, cu_seqlens)

        torch.testing.assert_close(result, expected)

    def test_zero_shift(self):
        """When all seqlens are multiples of W, no rolling should occur."""
        B, D, W = 2, 4, 4
        conv_states = torch.randn(B, D, W, device="cuda", dtype=torch.float32)
        cu_seqlens = torch.tensor([0, W, 2 * W], device="cuda", dtype=torch.int32)

        result = roll_conv_varlen_states(conv_states, cu_seqlens)
        torch.testing.assert_close(result, conv_states)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_dtypes(self, dtype):
        B, D, W = 2, 8, 4
        conv_states = torch.randn(B, D, W, device="cuda", dtype=dtype)
        cu_seqlens = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)

        result = roll_conv_varlen_states(conv_states, cu_seqlens)
        expected = roll_conv_varlen_states_ref(conv_states, cu_seqlens)

        torch.testing.assert_close(result, expected)


@pytest.mark.internal
class TestGatherConvState:

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("d_conv", [2, 3, 4])
    def test_matches_reference(self, d_conv):
        torch.manual_seed(42)
        B, D, state_len = 3, 8, 16
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.arange(B, device="cuda", dtype=torch.int32)
        cache_seqlens = torch.randint(
            d_conv, state_len + 10, (B,), device="cuda", dtype=torch.int32
        )

        result = gather_conv_state(conv_state, batch_indices, cache_seqlens, d_conv)
        expected = gather_conv_state_ref(conv_state, batch_indices, cache_seqlens, d_conv)

        torch.testing.assert_close(result, expected)

    def test_negative_batch_index_zeros_output(self):
        B, D, state_len, d_conv = 2, 4, 8, 4
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.tensor([-1, 0], device="cuda", dtype=torch.int32)
        cache_seqlens = torch.tensor([5, 5], device="cuda", dtype=torch.int32)

        result = gather_conv_state(conv_state, batch_indices, cache_seqlens, d_conv)

        # First batch should be all zeros due to negative index
        torch.testing.assert_close(
            result[0], torch.zeros(D, d_conv - 1, device="cuda", dtype=torch.float32)
        )

    def test_small_seqlen(self):
        """When seq_len < d_conv - 1, early positions should be zero-padded."""
        B, D, state_len, d_conv = 1, 4, 8, 4
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
        cache_seqlens = torch.tensor([1], device="cuda", dtype=torch.int32)

        result = gather_conv_state(conv_state, batch_indices, cache_seqlens, d_conv)
        expected = gather_conv_state_ref(conv_state, batch_indices, cache_seqlens, d_conv)

        torch.testing.assert_close(result, expected)


@pytest.mark.internal
class TestScatterConvState:

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("chunk_len", [4, 8, 20])
    def test_matches_reference(self, chunk_len):
        torch.manual_seed(42)
        B, D, state_len = 3, 8, 16
        conv_state_triton = torch.zeros(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_ref = conv_state_triton.clone()
        xBC = torch.randn(B, D, chunk_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.arange(B, device="cuda", dtype=torch.int32)
        cache_seqlens = torch.randint(0, 20, (B,), device="cuda", dtype=torch.int32)

        scatter_conv_state(conv_state_triton, xBC, batch_indices, cache_seqlens)
        scatter_conv_state_ref(conv_state_ref, xBC, batch_indices, cache_seqlens)

        torch.testing.assert_close(conv_state_triton, conv_state_ref)

    def test_negative_batch_index_noop(self):
        B, D, state_len, chunk_len = 2, 4, 8, 4
        conv_state = torch.zeros(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_orig = conv_state.clone()
        xBC = torch.randn(2, D, chunk_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.tensor([-1, -1], device="cuda", dtype=torch.int32)
        cache_seqlens = torch.tensor([0, 0], device="cuda", dtype=torch.int32)

        scatter_conv_state(conv_state, xBC, batch_indices, cache_seqlens)

        torch.testing.assert_close(conv_state, conv_state_orig)

    def test_chunk_larger_than_state(self):
        """When chunk_len > state_len, only last state_len tokens should be written."""
        B, D, state_len = 1, 4, 4
        chunk_len = 10
        conv_state = torch.zeros(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_ref = conv_state.clone()
        xBC = torch.randn(B, D, chunk_len, device="cuda", dtype=torch.float32)
        batch_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
        cache_seqlens = torch.tensor([0], device="cuda", dtype=torch.int32)

        scatter_conv_state(conv_state, xBC, batch_indices, cache_seqlens)
        scatter_conv_state_ref(conv_state_ref, xBC, batch_indices, cache_seqlens)

        torch.testing.assert_close(conv_state, conv_state_ref)


@pytest.mark.internal
class TestCausalConv1dUpdate:

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_linear_no_bias(self, width):
        torch.manual_seed(42)
        B, seq_len, D, state_len = 2, 3, 64, 8
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state_triton = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_ref = conv_state_triton.clone()
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)

        result = causal_conv1d_update(
            x,
            conv_state_triton,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=None,
        )
        expected = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias=None, silu_activation=False
        )

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(conv_state_triton, conv_state_ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_linear_with_bias(self, width):
        torch.manual_seed(42)
        B, seq_len, D, state_len = 2, 3, 64, 8
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state_triton = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_ref = conv_state_triton.clone()
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        bias = torch.randn(D, device="cuda", dtype=torch.float32)

        result = causal_conv1d_update(
            x,
            conv_state_triton,
            weight,
            bias=bias,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=None,
        )
        expected = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias=bias, silu_activation=False
        )

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_linear_with_silu(self, width):
        torch.manual_seed(42)
        B, seq_len, D, state_len = 2, 1, 64, 8
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state_triton = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        conv_state_ref = conv_state_triton.clone()
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        bias = torch.randn(D, device="cuda", dtype=torch.float32)

        result = causal_conv1d_update(
            x,
            conv_state_triton,
            weight,
            bias=bias,
            silu_activation="silu",
            cache_seqlens=None,
            conv_state_indices=None,
        )
        expected = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias=bias, silu_activation=True
        )

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_2d_input(self):
        """Test that 2D input (B, D) is handled correctly and returns 2D output."""
        torch.manual_seed(42)
        B, D, state_len, width = 2, 64, 8, 4
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)

        result = causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=None,
        )

        assert result.dim() == 2
        assert result.shape == (B, D)

    def test_conv_state_indices(self):
        """Test that conv_state_indices correctly maps batch to state entries."""
        torch.manual_seed(42)
        B, D, state_len, width = 2, 64, 8, 4
        num_states = 4
        x = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(num_states, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        # Map batch 0 -> state 2, batch 1 -> state 0
        state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)

        # Run with indices
        conv_state_copy = conv_state.clone()
        result = causal_conv1d_update(
            x,
            conv_state_copy,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=state_indices,
        )

        # Run without indices by manually reordering
        conv_state_reordered = conv_state[state_indices.long()].clone()
        expected = causal_conv1d_update(
            x,
            conv_state_reordered,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=None,
        )

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_negative_state_index_zeros_output(self):
        """Padding batch entries (index < 0) should produce zero output."""
        torch.manual_seed(42)
        B, D, state_len, width = 2, 64, 8, 4
        x = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        state_indices = torch.tensor([-1, 0], device="cuda", dtype=torch.int32)

        result = causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=state_indices,
        )

        # Batch 0 (padded) should be all zeros
        torch.testing.assert_close(result[0], torch.zeros(1, D, device="cuda", dtype=torch.float32))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision(self, dtype):
        torch.manual_seed(42)
        B, seq_len, D, state_len, width = 2, 1, 64, 8, 4
        x = torch.randn(B, seq_len, D, device="cuda", dtype=dtype)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=dtype)
        weight = torch.randn(D, width, device="cuda", dtype=dtype)

        result = causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=None,
            silu_activation=False,
            cache_seqlens=None,
            conv_state_indices=None,
        )

        assert result.dtype == dtype
        assert result.shape == (B, seq_len, D)
        assert torch.isfinite(result).all()
