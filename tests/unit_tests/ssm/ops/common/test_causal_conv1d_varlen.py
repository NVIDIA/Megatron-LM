# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the varlen causal conv1d ops.

`causal_conv1d_varlen_fn` (the Triton kernel) is checked against a reference that
loops over requests calling `causal_conv1d_fn` with `initial_states`.

`causal_conv1d_varlen_carry_states` (the conv-state update) is checked against a
reference that concatenates the previous state with the slice. Under chunked
prefill a request's conv state has to survive being handed a slice at a time:
deriving it from the slice alone is fine while every slice is at least `d_conv`
tokens long, and wrong the moment one is not -- which chunked prefill makes
reachable, since a prompt's final chunk can be as short as two tokens.
"""

import pytest
import torch

from megatron.core.ssm.ops.common.causal_conv1d_varlen import (
    causal_conv1d_varlen_carry_states,
    causal_conv1d_varlen_fn,
)

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


def _carry_reference(x, cu_seqlens, previous_states):
    """Right-align `[previous_tokens..., slice_tokens...]` into a d_conv window."""
    num_requests, conv_dim, d_conv = previous_states.shape
    out = torch.empty_like(previous_states)
    for i in range(num_requests):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        # The previous state already holds the d_conv tokens before the slice.
        history = torch.cat([previous_states[i], x[start:end].transpose(0, 1)], dim=1)
        out[i] = history[:, -d_conv:]
    return out


def _run_carry(lengths, d_conv=4, conv_dim=3, seed=0):
    torch.manual_seed(seed)
    cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    x = torch.randn(int(cu_seqlens[-1]), conv_dim)
    previous_states = torch.randn(len(lengths), conv_dim, d_conv)
    got = causal_conv1d_varlen_carry_states(x, cu_seqlens, previous_states)
    return got, _carry_reference(x, cu_seqlens, previous_states)


class TestCausalConv1dVarlenCarryStates:
    """Conv-state update that carries prior history across a chunked-prefill slice."""

    @pytest.mark.internal
    @pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 9])
    def test_matches_reference_for_every_slice_length(self, length):
        """Slices shorter, equal to, and longer than d_conv all round-trip."""
        got, expected = _run_carry([length])
        torch.testing.assert_close(got, expected)

    @pytest.mark.internal
    def test_long_slice_ignores_previous_state(self):
        """A slice of at least d_conv tokens fully determines the new state.

        This is the case the non-carrying `causal_conv1d_varlen_states` also gets
        right, so it pins the equivalence: nothing of the old state leaks through.
        """
        torch.manual_seed(1)
        cu_seqlens = torch.tensor([0, 7], dtype=torch.int32)
        x = torch.randn(7, 3)
        first = causal_conv1d_varlen_carry_states(x, cu_seqlens, torch.randn(1, 3, 4))
        second = causal_conv1d_varlen_carry_states(x, cu_seqlens, torch.randn(1, 3, 4))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[0], x[-4:].transpose(0, 1))

    @pytest.mark.internal
    def test_short_slice_carries_history(self):
        """A 2-token slice keeps the two taps that predate it.

        Deriving the state from the slice alone would zero-fill those two columns,
        which is exactly what corrupts the first decode step after a short final
        prefill chunk.
        """
        torch.manual_seed(2)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
        x = torch.randn(2, 3)
        previous = torch.randn(1, 3, 4)
        got = causal_conv1d_varlen_carry_states(x, cu_seqlens, previous)

        torch.testing.assert_close(got[0, :, :2], previous[0, :, 2:])
        torch.testing.assert_close(got[0, :, 2:], x.transpose(0, 1))
        assert not torch.allclose(got[0, :, :2], torch.zeros_like(got[0, :, :2]))

    @pytest.mark.internal
    def test_mixed_batch_including_padding_requests(self):
        """Zero-length padding requests keep their state; real ones update."""
        lengths = [0, 2, 5, 0, 130]
        got, expected = _run_carry(lengths)
        torch.testing.assert_close(got, expected)

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "lengths", [[2], [4], [0, 2, 5, 0, 130]], ids=["short", "exact", "mixed"]
    )
    def test_precomputed_plan_matches_the_inline_path(self, lengths):
        """The hoisted gather plan must agree with deriving it in place.

        Dynamic batching builds the plan once per step in `MambaMetadata` and
        every SSM layer reuses it, so the two code paths have to stay identical.
        """
        d_conv, conv_dim = 4, 3
        torch.manual_seed(7)
        cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
        x = torch.randn(int(cu_seqlens[-1]), conv_dim)
        previous_states = torch.randn(len(lengths), conv_dim, d_conv)

        inline = causal_conv1d_varlen_carry_states(x, cu_seqlens, previous_states)

        # Same arithmetic MambaMetadata._update_conv_carry_plan performs.
        starts = cu_seqlens[:-1].to(torch.int64)
        offsets = (
            (cu_seqlens[1:].to(torch.int64) - starts)[:, None]
            - d_conv
            + torch.arange(d_conv, dtype=torch.int64)[None, :]
        )
        planned = causal_conv1d_varlen_carry_states(
            x,
            None,
            previous_states,
            token_indices=(starts[:, None] + offsets).clamp(min=0),
            prev_columns=(offsets + d_conv).clamp(0, d_conv - 1),
            from_slice=offsets >= 0,
        )

        torch.testing.assert_close(inline, planned)

    @pytest.mark.internal
    def test_partial_plan_is_rejected(self):
        """Passing some but not all of the plan is a caller bug, not a fallback."""
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
        with pytest.raises(AssertionError, match="in full or not at all"):
            causal_conv1d_varlen_carry_states(
                torch.randn(4, 3),
                cu_seqlens,
                torch.randn(1, 3, 4),
                token_indices=torch.zeros(1, 4, dtype=torch.int64),
            )

    @pytest.mark.internal
    @pytest.mark.parametrize("field", ["prev_columns", "from_slice"])
    def test_plan_shape_mismatch_is_rejected(self, field):
        """Every plan tensor is checked, not just `token_indices`.

        A wrongly-shaped `prev_columns` or `from_slice` would otherwise fail deep
        inside `gather`/`where`, or broadcast silently.
        """
        plan = {
            "token_indices": torch.zeros(1, 4, dtype=torch.int64),
            "prev_columns": torch.zeros(1, 4, dtype=torch.int64),
            "from_slice": torch.zeros(1, 4, dtype=torch.bool),
        }
        plan[field] = plan[field][:, :2]
        with pytest.raises(AssertionError, match=f"carry plan {field} describes"):
            causal_conv1d_varlen_carry_states(torch.randn(4, 3), None, torch.randn(1, 3, 4), **plan)

    @pytest.mark.internal
    def test_out_of_range_discarded_column_does_not_read_a_foreign_token(self):
        """Columns `from_slice` discards may name a token past the end of `x`.

        Those lanes are rewritten to index 0 rather than clamped to the last
        token, so a plan/`x` mismatch on a *live* column cannot quietly fold
        another request's final token into this request's state.
        """
        d_conv, conv_dim = 4, 3
        torch.manual_seed(11)
        x = torch.randn(4, conv_dim)
        previous_states = torch.randn(1, conv_dim, d_conv)

        # Every column is discarded, but each names a token well past `x`.
        got = causal_conv1d_varlen_carry_states(
            x,
            None,
            previous_states,
            token_indices=torch.full((1, d_conv), 999, dtype=torch.int64),
            prev_columns=torch.arange(d_conv, dtype=torch.int64)[None, :],
            from_slice=torch.zeros(1, d_conv, dtype=torch.bool),
        )
        torch.testing.assert_close(got, previous_states)

    @pytest.mark.internal
    def test_zero_length_request_is_a_no_op(self):
        torch.manual_seed(3)
        cu_seqlens = torch.tensor([0, 0], dtype=torch.int32)
        previous = torch.randn(1, 3, 4)
        got = causal_conv1d_varlen_carry_states(torch.randn(0, 3), cu_seqlens, previous)
        torch.testing.assert_close(got, previous)
