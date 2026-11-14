# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Unit tests for `selective_state_update` (megatron/core/ssm/ops/mamba2/mamba_ssm.py).

import pytest
import torch

from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.internal
class TestSelectiveStateUpdateInputPurity:
    """Regression tests: selective_state_update must never mutate its inputs.

    When called WITHOUT `intermediate_ssm_states`, the launcher used to
    substitute `x` as a dummy pointer with all-zero strides. The
    `HAS_INT_STATE` heuristic (`int_state_ptr is not None`) then saw a
    non-None pointer and enabled the intermediate-state dump, which collapsed
    its whole store grid onto `x[0, 0, 0, 0]` — corrupting `x` after the
    call and racing other programs' loads of `x` (nondeterministic
    head-0/dim-0 output errors). This hit every hybrid non-spec decode step.
    The fix passes `None` instead of a dummy.
    """

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("seq_len", [1, 3])
    def test_inputs_not_mutated_without_intermediates(self, seq_len):
        """seq_len == 1 is the production non-spec decode shape."""
        torch.manual_seed(0)
        device = "cuda"
        B, nheads, headdim, dstate, ngroups = 4, 8, 64, 64, 2
        x = torch.randn(B, seq_len, nheads, headdim, device=device)
        dt = (
            torch.randn(B, seq_len, nheads, device=device)
            .unsqueeze(-1)
            .expand(B, seq_len, nheads, headdim)
        )
        A = (
            (-torch.rand(nheads, device=device) - 1.0)
            .view(nheads, 1, 1)
            .expand(nheads, headdim, dstate)
        )
        Bm = torch.randn(B, seq_len, ngroups, dstate, device=device)
        Cm = torch.randn(B, seq_len, ngroups, dstate, device=device)
        D = torch.randn(nheads, device=device).unsqueeze(-1).expand(nheads, headdim)
        dt_bias = (torch.rand(nheads, device=device) - 4.0).unsqueeze(-1).expand(nheads, headdim)
        state = torch.randn(B, nheads, headdim, dstate, device=device)
        inputs = {"x": x, "dt": dt, "A": A, "B": Bm, "C": Cm, "D": D, "dt_bias": dt_bias}
        before = {k: v.clone() for k, v in inputs.items()}

        selective_state_update(
            state,
            x,
            dt,
            A,
            Bm,
            Cm,
            D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=torch.arange(B, dtype=torch.int64, device=device),
        )
        torch.cuda.synchronize()

        for name, tensor in inputs.items():
            assert torch.equal(tensor, before[name]), (
                f"selective_state_update mutated input '{name}' when called "
                f"without intermediate_ssm_states (dummy-pointer regression)"
            )


@pytest.mark.internal
class TestSelectiveStateUpdateSnapshotContract:
    """Multi-token snapshots must be what a serial decode of that prefix holds.

    Speculative decoding restores `intermediate_ssm_states[slot, i]` when only part
    of a draft is accepted, so that entry has to equal the state cache after
    decoding tokens `0..i` one launch at a time: same arithmetic, same rounding.
    The kernel keeps the state in an FP32 accumulator, but a serial decode stores
    it to the cache dtype and the next launch reads the *rounded* value back. A
    multi-token launch that rounds only the values it stores therefore drifts from
    serial decode once per token in BF16, while the first token -- the one a
    next-output check tends to look at -- still matches. FP32 caches are exact, so
    an accuracy test in FP32 cannot see it at all.
    """

    num_slots = 6
    nheads = 4
    headdim = 32
    dstate = 32
    ngroups = 2
    seq_len = 4

    def setup_method(self, method):
        _requires_cuda()

    def _make_inputs(self, device, dtype):
        """One `seq_len`-token decode step plus an out-of-order slot map."""
        torch.manual_seed(1234)
        batch = 3
        x = torch.randn(
            batch, self.seq_len, self.nheads, self.headdim, device=device, dtype=dtype
        )
        # Per-element `dt` input rather than a broadcast bias, i.e. the general
        # non-tied-H-dim layout.
        dt = torch.randn(
            batch, self.seq_len, self.nheads, self.headdim, device=device, dtype=dtype
        )
        A = -torch.rand(self.nheads, self.headdim, self.dstate, device=device) - 1.0
        B = torch.randn(
            batch, self.seq_len, self.ngroups, self.dstate, device=device, dtype=dtype
        )
        C = torch.randn(
            batch, self.seq_len, self.ngroups, self.dstate, device=device, dtype=dtype
        )
        D = torch.randn(self.nheads, self.headdim, device=device, dtype=dtype)
        dt_bias = torch.rand(self.nheads, self.headdim, device=device, dtype=dtype) - 4.0
        # Slots deliberately out of order, so a snapshot written at the batch
        # position instead of the cache slot fails here.
        state_indices = torch.tensor([4, 0, 3], dtype=torch.int64, device=device)
        initial_state = torch.randn(
            self.num_slots, self.nheads, self.headdim, self.dstate, device=device, dtype=dtype
        )
        return x, dt, A, B, C, D, dt_bias, state_indices, initial_state

    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    def test_snapshots_match_token_by_token_decode(self, state_dtype):
        """Snapshot `i` equals the cache after decoding tokens `0..i` serially."""
        device = "cuda"
        x, dt, A, B, C, D, dt_bias, indices, initial_state = self._make_inputs(
            device, state_dtype
        )

        state = initial_state.clone()
        intermediate = torch.full(
            (self.num_slots, self.seq_len, self.nheads, self.headdim, self.dstate),
            float("nan"),
            device=device,
            dtype=state_dtype,
        )
        common = dict(
            z=None, dt_bias=dt_bias, dt_softplus=True, state_batch_indices=indices
        )
        out = selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            intermediate_ssm_states=intermediate,
            **common,
        )

        replay_state = initial_state.clone()
        for token in range(self.seq_len):
            replay_out = selective_state_update(
                replay_state,
                x[:, token],
                dt[:, token],
                A,
                B[:, token],
                C[:, token],
                D,
                **common,
            )
            assert torch.equal(intermediate[indices, token], replay_state[indices]), (
                f"snapshot for token {token} does not match a serial decode that stopped "
                f"there (state cache dtype {state_dtype})"
            )
            if state_dtype == torch.bfloat16:
                # Worth pinning down once the state matches: the output is a reduction
                # of the state, and for a 1-token vs an S-token launch the compiler is
                # free to schedule that reduction differently, which shows up in FP32
                # at ~1e-6 with or without this contract holding. In BF16 the store
                # rounds those bits away, so here the outputs must match exactly.
                assert torch.equal(out[:, token], replay_out), (
                    f"output for token {token} differs from serial decode "
                    f"(state cache dtype {state_dtype})"
                )

        assert torch.equal(state, replay_state), (
            f"post-step state differs from serial decode (state cache dtype {state_dtype})"
        )
