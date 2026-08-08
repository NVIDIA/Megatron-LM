# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the immediate-free-token-update path of selective_state_update."""

import pytest
import torch

from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_factorized_rollback,
)
from megatron.core.ssm.ops.mamba_ssm import selective_state_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.internal
class TestSelectiveStateUpdateImmediate:

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("num_spec", [1, 2, 4])
    def test_immediate_free_token_equivalence(self, num_spec):
        """The immediate path must produce identical outputs and accepted states
        to the standard path, while writing the free-token (s == 0) state to the
        live buffer and the drafted-token states to intermediate[s - 1]."""
        torch.manual_seed(0)
        device, dtype = "cuda", torch.float32
        B, nheads, headdim, dstate, ngroups = 3, 4, 16, 32, 1
        S = num_spec + 1  # verification-window length (free token + drafts)

        x = torch.randn(B, S, nheads, headdim, device=device, dtype=dtype)
        dt = torch.randn(B, S, nheads, headdim, device=device, dtype=dtype)
        A = -torch.rand(nheads, headdim, dstate, device=device, dtype=dtype) - 0.5
        Bm = torch.randn(B, S, ngroups, dstate, device=device, dtype=dtype)
        Cm = torch.randn(B, S, ngroups, dstate, device=device, dtype=dtype)
        D = torch.randn(nheads, headdim, device=device, dtype=dtype)
        dt_bias = torch.randn(nheads, headdim, device=device, dtype=dtype)
        state_init = torch.randn(B, nheads, headdim, dstate, device=device, dtype=dtype)
        batch_indices = torch.arange(B, device=device, dtype=torch.int64)

        # Standard path: full (S) intermediate buffer; final state persisted to `state`.
        state_ref = state_init.clone()
        int_ref = torch.zeros(B, S, nheads, headdim, dstate, device=device, dtype=dtype)
        out_ref = selective_state_update(
            state_ref,
            x,
            dt,
            A,
            Bm,
            Cm,
            D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=batch_indices,
            intermediate_ssm_states=int_ref,
        )

        # Immediate path: intermediate buffer has only num_spec (drafted) slots.
        state_imm = state_init.clone()
        int_imm = torch.zeros(B, S - 1, nheads, headdim, dstate, device=device, dtype=dtype)
        out_imm = selective_state_update(
            state_imm,
            x,
            dt,
            A,
            Bm,
            Cm,
            D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=batch_indices,
            intermediate_ssm_states=int_imm,
            immediate_state_update=True,
        )

        # Outputs identical at every position.
        torch.testing.assert_close(out_imm, out_ref, atol=1e-4, rtol=1e-4)
        # Live state holds the free-token (s == 0) state.
        torch.testing.assert_close(state_imm, int_ref[:, 0], atol=1e-4, rtol=1e-4)
        # Drafted-token states are shifted by one: int_imm[:, j] == int_ref[:, j + 1].
        for j in range(S - 1):
            torch.testing.assert_close(int_imm[:, j], int_ref[:, j + 1], atol=1e-4, rtol=1e-4)

    def test_immediate_requires_intermediate(self):
        """immediate_state_update=True without an intermediate buffer must raise."""
        device = "cuda"
        B, nheads, headdim, dstate, ngroups = 2, 4, 16, 32, 1
        S = 3
        x = torch.randn(B, S, nheads, headdim, device=device)
        dt = torch.randn(B, S, nheads, headdim, device=device)
        A = -torch.rand(nheads, headdim, dstate, device=device) - 0.5
        Bm = torch.randn(B, S, ngroups, dstate, device=device)
        Cm = torch.randn(B, S, ngroups, dstate, device=device)
        state = torch.randn(B, nheads, headdim, dstate, device=device)
        with pytest.raises(AssertionError):
            selective_state_update(
                state, x, dt, A, Bm, Cm, immediate_state_update=True
            )


@pytest.mark.internal
class TestFactorizedStoreRoundTrip:
    """End-to-end check of the fused factor store (in selective_state_update) plus the
    fused rank-1 rewind: storing factors and reconstructing the all-accepted state must
    match the full sequential recurrence computed by the normal kernel."""

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("num_spec", [1, 2, 4])
    @pytest.mark.parametrize("factor_dtype", [torch.float32, torch.bfloat16])
    def test_store_then_rewind_matches_full_recurrence(self, num_spec, factor_dtype):
        torch.manual_seed(0)
        device = "cuda"
        N = 6  # decode requests == Mamba state slots (1:1 mapping)
        nheads, headdim, dstate, ngroups = 8, 16, 32, 2
        npg = nheads // ngroups
        S = num_spec + 1  # window length (free token + drafts)

        # Per-head dt / dt_bias and per-head scalar A, broadcast with stride-0 expands so
        # the kernel runs the TIE_HDIM (per-head scalar) path used by the factor store.
        raw_dt = torch.randn(N, S, nheads, device=device) * 0.5
        dt = raw_dt.unsqueeze(-1).expand(N, S, nheads, headdim)
        dt_bias = torch.randn(nheads, device=device).unsqueeze(-1).expand(nheads, headdim)
        a_log = torch.randn(nheads, device=device)
        A = (-torch.exp(a_log)).view(nheads, 1, 1).expand(nheads, headdim, dstate)
        x = torch.randn(N, S, nheads, headdim, device=device)
        Bm = torch.randn(N, S, ngroups, dstate, device=device)
        Cm = torch.randn(N, S, ngroups, dstate, device=device)
        state0 = torch.randn(N, nheads, headdim, dstate, device=device)
        batch_indices = torch.arange(N, device=device, dtype=torch.int64)

        # Ground truth: the normal kernel runs the full-window recurrence in place.
        state_full = state0.clone()
        selective_state_update(
            state_full, x, dt, A, Bm, Cm, D=None, z=None,
            dt_bias=dt_bias, dt_softplus=True, state_batch_indices=batch_indices,
        )

        # Factorized store: the kernel writes (dx, B, alpha) and leaves the state untouched.
        factor_dx = torch.zeros(1, N, S, nheads, headdim, device=device, dtype=factor_dtype)
        factor_B = torch.zeros(1, N, S, ngroups, dstate, device=device, dtype=factor_dtype)
        factor_alpha = torch.zeros(1, N, S, nheads, device=device, dtype=torch.float32)
        state_skip = state0.clone()
        selective_state_update(
            state_skip, x, dt, A, Bm, Cm, D=None, z=None,
            dt_bias=dt_bias, dt_softplus=True, state_batch_indices=batch_indices,
            skip_state_write=True,
            factor_dx=factor_dx[0], factor_B=factor_B[0], factor_alpha=factor_alpha[0],
        )
        torch.testing.assert_close(state_skip, state0)  # live state left untouched

        # Rewind with every drafted token accepted -> the whole window is applied.
        state_rewind = state0.clone().unsqueeze(0)  # (L=1, N, nheads, headdim, dstate)
        prefill = torch.zeros(N, dtype=torch.int32, device=device)
        accepted = torch.full((N,), num_spec, dtype=torch.int64, device=device)
        mamba_state_factorized_rollback(
            factor_dx, factor_B, factor_alpha, state_rewind,
            prefill, batch_indices, accepted, num_layers=1, nheads_per_group=npg,
        )

        tol = 2e-2 if factor_dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(state_rewind[0], state_full, atol=tol, rtol=tol)
