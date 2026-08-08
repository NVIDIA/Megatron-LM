# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.ops.causal_conv1d_triton import causal_conv1d_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------- Reference Implementations ---------------------- #


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
            x, conv_state_triton, weight, bias=None, silu_activation=False, conv_state_indices=None
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
            x, conv_state_triton, weight, bias=bias, silu_activation=False, conv_state_indices=None
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
            x, conv_state_triton, weight, bias=bias, silu_activation="silu", conv_state_indices=None
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
            x, conv_state, weight, bias=None, silu_activation=False, conv_state_indices=None
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
            x, conv_state, weight, bias=None, silu_activation=False, conv_state_indices=None
        )

        assert result.dtype == dtype
        assert result.shape == (B, seq_len, D)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_intermediate_state(self, width):
        """Test that intermediate conv states are correctly stored at each sequence step."""
        torch.manual_seed(42)
        B, seq_len, D, state_len = 2, 4, 64, 8
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)

        # Allocate intermediate state buffer: (B, seq_len, D, state_len)
        int_states = torch.zeros(B, seq_len, D, state_len, device="cuda", dtype=torch.float32)

        # Run with intermediate state recording
        conv_state_copy = conv_state.clone()
        result = causal_conv1d_update(
            x,
            conv_state_copy,
            weight,
            bias=None,
            silu_activation=False,
            conv_state_indices=None,
            intermediate_conv_states=int_states,
        )

        # Verify by running step-by-step and checking each intermediate
        conv_state_ref = conv_state.clone()
        for s in range(seq_len):
            conv_state_ref[:, :, :-1] = conv_state_ref[:, :, 1:].clone()
            conv_state_ref[:, :, -1] = x[:, s, :]
            torch.testing.assert_close(int_states[:, s, :, :], conv_state_ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_state_len_eq_width_fast_path(self, width):
        """Cover the ``state_len == WIDTH`` fast path (the common Mamba
        configuration where d_conv == width).

        The other tests use ``state_len = 8`` so they always fall through to
        the explicit shift loop. Here ``state_len = width`` exercises the
        register-resident shift and the matching ``HAS_INT_STATE`` branch.
        """
        torch.manual_seed(42)
        B, seq_len, D = 2, 4, 64
        state_len = width
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state_initial = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        int_states = torch.zeros(B, seq_len, D, state_len, device="cuda", dtype=torch.float32)

        conv_state_triton = conv_state_initial.clone()
        conv_state_ref = conv_state_initial.clone()

        result = causal_conv1d_update(
            x,
            conv_state_triton,
            weight,
            bias=None,
            silu_activation=False,
            conv_state_indices=None,
            intermediate_conv_states=int_states,
        )
        expected = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias=None, silu_activation=False
        )

        # Output and final state match the reference.
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(conv_state_triton, conv_state_ref, atol=1e-5, rtol=1e-5)

        # Per-step intermediate states match a manual replay.
        replay_state = conv_state_initial.clone()
        for s in range(seq_len):
            replay_state[:, :, :-1] = replay_state[:, :, 1:].clone()
            replay_state[:, :, -1] = x[:, s, :]
            torch.testing.assert_close(int_states[:, s, :, :], replay_state, atol=1e-5, rtol=1e-5)

    def test_intermediate_state_with_indices(self):
        """Test intermediate states work correctly with conv_state_indices mapping."""
        torch.manual_seed(42)
        B, seq_len, D, state_len, width = 2, 3, 64, 8, 4
        num_states = 4
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(num_states, D, state_len, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)

        # Intermediate states are indexed by state_batch_coord (i.e., req index, not batch index)
        int_states = torch.zeros(
            num_states, seq_len, D, state_len, device="cuda", dtype=torch.float32
        )

        conv_state_copy = conv_state.clone()
        result = causal_conv1d_update(
            x,
            conv_state_copy,
            weight,
            bias=None,
            silu_activation=False,
            conv_state_indices=state_indices,
            intermediate_conv_states=int_states,
        )

        # The final intermediate state at last seq step should match the final conv_state
        for b_idx in range(B):
            req_idx = state_indices[b_idx].item()
            torch.testing.assert_close(
                int_states[req_idx, seq_len - 1, :, :],
                conv_state_copy[req_idx, :, :],
                atol=1e-5,
                rtol=1e-5,
            )

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_immediate_free_token_state_update(self, width):
        """Immediate free-token update (state_len == width, the Mamba-2 case).

        The free token (s == 0) window is written straight into the live
        conv_state, drafted tokens (s >= 1) are checkpointed at intermediate
        index s - 1, and outputs are identical to the standard path. Verified
        against the standard path's full-window intermediate buffer.
        """
        torch.manual_seed(7)
        B, seq_len, D = 2, 4, 64  # 1 free token + 3 drafted tokens
        state_len = width
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        bias = torch.randn(D, device="cuda", dtype=torch.float32)
        conv_state_initial = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)

        # Standard path: full (seq_len) intermediate buffer; conv_state rolls to the end.
        cs_ref = conv_state_initial.clone()
        int_ref = torch.zeros(B, seq_len, D, state_len, device="cuda", dtype=torch.float32)
        out_ref = causal_conv1d_update(
            x,
            cs_ref,
            weight,
            bias=bias,
            silu_activation="silu",
            conv_state_indices=None,
            intermediate_conv_states=int_ref,
        )

        # Immediate path: intermediate buffer has only seq_len - 1 (drafted) slots.
        cs_imm = conv_state_initial.clone()
        int_imm = torch.zeros(B, seq_len - 1, D, state_len, device="cuda", dtype=torch.float32)
        out_imm = causal_conv1d_update(
            x,
            cs_imm,
            weight,
            bias=bias,
            silu_activation="silu",
            conv_state_indices=None,
            intermediate_conv_states=int_imm,
            immediate_state_update=True,
        )

        # Outputs are identical to the standard path.
        torch.testing.assert_close(out_imm, out_ref, atol=1e-4, rtol=1e-4)
        # Live conv_state holds the free-token (s == 0) window.
        torch.testing.assert_close(cs_imm, int_ref[:, 0], atol=1e-5, rtol=1e-5)
        # Drafted-token windows are shifted by one: int_imm[:, j] == int_ref[:, j + 1].
        for j in range(seq_len - 1):
            torch.testing.assert_close(int_imm[:, j], int_ref[:, j + 1], atol=1e-5, rtol=1e-5)

    def test_immediate_state_update_requires_state_len_eq_width(self):
        """The flag is only valid for state_len == width (Mamba-2)."""
        B, seq_len, D, width, state_len = 2, 3, 64, 4, 8  # state_len != width
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        int_states = torch.zeros(B, seq_len - 1, D, state_len, device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias=None,
                silu_activation=False,
                conv_state_indices=None,
                intermediate_conv_states=int_states,
                immediate_state_update=True,
            )

    def test_immediate_state_update_requires_intermediate(self):
        """The flag requires an intermediate buffer to be provided."""
        B, seq_len, D, width = 2, 3, 64, 4
        state_len = width
        x = torch.randn(B, seq_len, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, width, device="cuda", dtype=torch.float32)
        conv_state = torch.randn(B, D, state_len, device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias=None,
                silu_activation=False,
                conv_state_indices=None,
                intermediate_conv_states=None,
                immediate_state_update=True,
            )
