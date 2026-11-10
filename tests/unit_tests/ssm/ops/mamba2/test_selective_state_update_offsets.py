# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# `selective_state_update` (megatron/core/ssm/ops/mamba2/mamba_ssm.py) at the int32 offset limit.

import pytest
import torch

from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.internal
class TestInt32Offsets:
    """Slot and batch offsets at the int32 limit, at Nemotron-H 8B's heads (TP 1: 128 heads of 64,
    d_state 128, 8 groups, so 2**20 state elements per slot).

    The inference context gives the kernel one layer's view of ``[layers, slots, ...]`` states,
    so a wrapped offset lands silently in another layer. The ``large`` cases (opt in with
    ``-m large``; about 20 GiB of device memory) put an offset at 2**31 elements, or just below
    it; the small cases run the same checks far below the limit.
    """

    nheads, headdim, dstate, ngroups = 128, 64, 128, 8

    def setup_method(self, method):
        _requires_cuda()

    def _params(self):
        nheads, headdim, dstate = self.nheads, self.headdim, self.dstate
        A = (-torch.exp(torch.randn(nheads, device="cuda"))).view(-1, 1, 1)
        D = torch.randn(nheads, device="cuda")[:, None].expand(nheads, headdim)
        dt_bias = torch.randn(nheads, device="cuda")[:, None].expand(nheads, headdim)
        return A.expand(nheads, headdim, dstate), D, dt_bias

    @pytest.mark.parametrize(
        "slots, steps",
        [
            pytest.param(4, 1, id="small"),
            pytest.param(4, 8, id="small-snapshots"),
            pytest.param(2048, 1, marks=pytest.mark.large, id="state-below"),
            pytest.param(2049, 1, marks=pytest.mark.large, id="state-past"),
            pytest.param(256, 8, marks=pytest.mark.large, id="snapshots-below"),
            pytest.param(257, 8, marks=pytest.mark.large, id="snapshots-past"),
        ],
    )
    def test_last_slot_of_a_layer_matches_its_own_run(self, slots, steps):
        """A request decodes into the last slot of layer 1, with int32 slot indices. Its state
        starts at 2**31 elements at 2,049 slots (2,048 is the control); with 7 draft tokens, its
        snapshots start there at 257 slots (256 is the control)."""
        torch.manual_seed(0)
        nheads, headdim, dstate, ngroups = self.nheads, self.headdim, self.dstate, self.ngroups
        x = torch.randn(1, steps, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        dt = torch.randn(1, steps, nheads, device="cuda", dtype=torch.bfloat16)
        dt = dt[..., None].expand(1, steps, nheads, headdim)
        B = torch.randn(1, steps, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
        C = torch.randn(1, steps, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
        A, D, dt_bias = self._params()
        initial_state = torch.randn(nheads, headdim, dstate, device="cuda")
        # Two layers of states and snapshots, filled with a sentinel.
        states = torch.full((2, slots, nheads, headdim, dstate), 7.0, device="cuda")
        states[1, -1] = initial_state
        snapshots = None
        if steps > 1:
            snapshots = torch.full((2, slots, steps, nheads, headdim, dstate), 7.0, device="cuda")

        out = selective_state_update(
            states[1],
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=torch.tensor([slots - 1], dtype=torch.int32, device="cuda"),
            intermediate_ssm_states=None if snapshots is None else snapshots[1],
        )

        alone_state = initial_state[None].clone()
        alone_snapshots = None if snapshots is None else torch.zeros_like(snapshots[1, :1])
        alone = selective_state_update(
            alone_state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            intermediate_ssm_states=alone_snapshots,
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
            pytest.param(14464, marks=pytest.mark.large, id="below"),
            pytest.param(14465, marks=pytest.mark.large, id="past"),
        ],
    )
    def test_batch_offsets_of_projection_views(self, requests):
        """MambaMixer passes dt as a view of the in_proj output, so its request stride is the
        whole in_proj row (18,560 elements) times the 8 tokens of a step with 7 draft tokens:
        the last request's dt starts at 2**31 elements at 14,465 requests (14,464 is the
        control). Every request but the last is padding, so only the last one holds a slot."""
        torch.manual_seed(0)
        nheads, headdim, dstate, ngroups = self.nheads, self.headdim, self.dstate, self.ngroups
        steps = 8
        d_inner = nheads * headdim
        conv_dim = d_inner + 2 * ngroups * dstate
        projection = torch.full(
            (requests, steps, d_inner + conv_dim + nheads), 0.5, device="cuda", dtype=torch.bfloat16
        )
        projection[-1].normal_()
        conv_out = torch.full((requests, steps, conv_dim), 0.5, device="cuda", dtype=torch.bfloat16)
        conv_out[-1].normal_()
        A, D, dt_bias = self._params()
        initial_state = torch.randn(1, nheads, headdim, dstate, device="cuda")

        def views(projection, conv_out):
            # As MambaMixer splits zxBCdt and the conv output (mamba_mixer.py).
            dt = projection[..., d_inner + conv_dim :, None].expand(-1, -1, nheads, headdim)
            x = conv_out[..., :d_inner].unflatten(-1, (nheads, headdim))
            B = conv_out[..., d_inner : d_inner + ngroups * dstate].unflatten(-1, (ngroups, dstate))
            C = conv_out[..., d_inner + ngroups * dstate :].unflatten(-1, (ngroups, dstate))
            return x, dt, B, C

        state = initial_state.clone()
        indices = torch.full((requests,), -1, dtype=torch.int64, device="cuda")
        indices[-1] = 0
        x, dt, B, C = views(projection, conv_out)
        out = selective_state_update(
            state, x, dt, A, B, C, D, dt_bias=dt_bias, dt_softplus=True, state_batch_indices=indices
        )

        alone_state = initial_state.clone()
        x, dt, B, C = views(projection[-1:].clone(), conv_out[-1:].clone())
        alone = selective_state_update(
            alone_state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(out[-1:], alone, rtol=0, atol=0)
        torch.testing.assert_close(state, alone_state, rtol=0, atol=0)
