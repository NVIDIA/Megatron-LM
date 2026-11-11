# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update


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


@pytest.mark.internal
class TestInt32Offsets:
    """Slot and batch offsets at the int32 limit, at Nemotron-H 8B's conv (TP 1: 10,240 channels,
    d_conv 4, so 40,960 state elements per slot).

    The inference context gives the kernel one layer's view of ``[layers, slots, ...]`` states,
    so a wrapped offset lands silently in another layer. The ``large`` cases (opt in with
    ``-m large``; about 10 GiB of device memory) put an offset at 2**31 elements, or just below
    it; the small cases run the same checks far below the limit.
    """

    dim, width = 10240, 4

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize(
        "slots, steps",
        [
            pytest.param(4, 1, id="small"),
            pytest.param(4, 8, id="small-snapshots"),
            pytest.param(52429, 1, marks=pytest.mark.large, id="state-below"),
            pytest.param(52430, 1, marks=pytest.mark.large, id="state-past"),
            pytest.param(6554, 8, marks=pytest.mark.large, id="snapshots-below"),
            pytest.param(6555, 8, marks=pytest.mark.large, id="snapshots-past"),
        ],
    )
    def test_last_slot_of_a_layer_matches_its_own_run(self, slots, steps):
        """A request decodes into the last slot of layer 1, with int32 slot indices. Its state
        starts past 2**31 elements at 52,430 slots (52,429 is the control); with 7 draft tokens,
        its snapshots (327,680 elements per slot) start past it at 6,555 slots (6,554 is the
        control)."""
        torch.manual_seed(0)
        dim, width = self.dim, self.width
        x = torch.randn(1, steps, dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        initial_state = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
        # Two layers of states and snapshots, filled with a sentinel.
        states = torch.full((2, slots, dim, width), 7.0, device="cuda", dtype=torch.bfloat16)
        states[1, -1] = initial_state
        snapshots = None
        if steps > 1:
            snapshots = torch.full(
                (2, slots, steps, dim, width), 7.0, device="cuda", dtype=torch.bfloat16
            )

        out = causal_conv1d_update(
            x,
            states[1],
            weight,
            bias,
            "silu",
            torch.tensor([slots - 1], dtype=torch.int32, device="cuda"),
            intermediate_conv_states=None if snapshots is None else snapshots[1],
        )

        alone_state = initial_state[None].clone()
        alone_snapshots = None if snapshots is None else torch.zeros_like(snapshots[1, :1])
        alone = causal_conv1d_update(
            x,
            alone_state,
            weight,
            bias,
            "silu",
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            intermediate_conv_states=alone_snapshots,
        )
        torch.testing.assert_close(out, alone, rtol=0, atol=0)
        torch.testing.assert_close(states[1, -1], alone_state[0], rtol=0, atol=0)
        assert bool((states[0] == 7.0).all()), "the update wrote into layer 0's states"
        if snapshots is not None:
            torch.testing.assert_close(snapshots[1, -1], alone_snapshots[0], rtol=0, atol=0)
            assert bool((snapshots[0] == 7.0).all()), "the update wrote into layer 0's snapshots"

    @pytest.mark.parametrize(
        "requests",
        [
            pytest.param(64, id="small"),
            pytest.param(115705, marks=pytest.mark.large, id="below"),
            pytest.param(115706, marks=pytest.mark.large, id="past"),
        ],
    )
    def test_batch_offset_of_a_projection_view(self, requests):
        """MambaMixer passes x as a column slice of the in_proj output, so its request stride is
        the whole in_proj row (18,560 elements): the last request's x starts past 2**31 elements
        at 115,706 one-token requests (115,705 is the control). Every request but the last is
        padding, so only the last one holds a slot."""
        torch.manual_seed(0)
        dim, width = self.dim, self.width
        d_inner, nheads = 8192, 128
        projection = torch.full(
            (requests, 1, d_inner + dim + nheads), 7.0, device="cuda", dtype=torch.bfloat16
        )
        projection[-1].normal_()
        weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        initial_state = torch.randn(1, dim, width, device="cuda", dtype=torch.bfloat16)
        indices = torch.full((requests,), -1, dtype=torch.int64, device="cuda")
        indices[-1] = 0

        state = initial_state.clone()
        out = causal_conv1d_update(
            projection[..., d_inner : d_inner + dim], state, weight, bias, "silu", indices
        )

        alone_state = initial_state.clone()
        alone = causal_conv1d_update(
            projection[-1:, :, d_inner : d_inner + dim].clone(),
            alone_state,
            weight,
            bias,
            "silu",
            torch.zeros(1, dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(out[-1:], alone, rtol=0, atol=0)
        torch.testing.assert_close(state, alone_state, rtol=0, atol=0)
