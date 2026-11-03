# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Unit tests for `selective_state_update` (megatron/core/ssm/ops/mamba2/mamba_ssm.py).

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _decode_recurrence(state, x, dt, A, B, C, D, dt_bias):
    """The Mamba-2 decode recurrence with softplus, token by token, in the dtype of its inputs.

    state (batch, nheads, dim, dstate); x, dt (batch, seqlen, nheads, dim); A (nheads, dim, dstate);
    B, C (batch, seqlen, ngroups, dstate); D, dt_bias (nheads, dim). Returns the outputs
    (batch, seqlen, nheads, dim) and the state after each token (batch, seqlen, nheads, dim, dstate).
    """
    heads_per_group = x.shape[2] // B.shape[2]
    B = B.repeat_interleave(heads_per_group, dim=2)[..., None, :]
    C = C.repeat_interleave(heads_per_group, dim=2)[..., None, :]
    outs, states = [], []
    for t in range(x.shape[1]):
        dt_t = F.softplus(dt[:, t] + dt_bias)
        state = torch.exp(A * dt_t[..., None]) * state + (dt_t * x[:, t])[..., None] * B[:, t]
        outs.append((state * C[:, t]).sum(-1) + D * x[:, t])
        states.append(state)
    return torch.stack(outs, dim=1), torch.stack(states, dim=1)


def _assert_as_accurate_as_torch(actual, reference, trusted, what, factor=2.0):
    """`actual`'s max-abs and rms errors against the fp64 `reference` must be at most `factor` times
    those of `trusted` (the same math in torch, in `actual`'s dtype), plus one ulp of the largest
    reference magnitude at one element."""
    _, exponent = math.frexp(reference.abs().max().item())
    ulp = math.ldexp(torch.finfo(actual.dtype).eps, exponent - 1)
    errors = [(t.double() - reference).abs() for t in (actual, trusted)]
    for metric, norm, floor in (
        ("max-abs", lambda e: e.max().item(), ulp),
        ("rms", lambda e: e.square().mean().sqrt().item(), ulp / math.sqrt(reference.numel())),
    ):
        kernel_error, torch_error = (norm(e) for e in errors)
        assert kernel_error <= factor * torch_error + floor, (
            f"{what}: {metric} error against fp64 is {kernel_error:.3e}, more than {factor}x "
            f"torch's {torch_error:.3e} (+ {floor:.1e})"
        )


@pytest.mark.internal
class TestSelectiveStateUpdateAccuracy:
    """With an fp32 state, a decode step must be about as accurate as torch fp32 doing the same
    math: at most twice its error against an fp64 reference.

    The kernel used to compute softplus as log(exp(x) + 1), which rounds exp(x) + 1 to fp32 before
    the log. At the dt + dt_bias Mamba decodes with (dt_bias is initialized to softplus^-1 of
    [1e-3, 1e-1]) that is a relative error of up to ~5e-5 in the step size, which exp(A * dt) with
    |A| up to 16 carries into the fp32 state on every step: up to ~10x torch's error.
    """

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize("seq_len", [1, 4], ids=["decode", "speculative"])
    @pytest.mark.parametrize("tie_hdim", [True, False], ids=["broadcast_dt_A", "dense_dt_A"])
    def test_fp32_state_matches_fp64(self, tie_hdim, seq_len):
        """Production shapes and dtypes: 8 heads, 1 group, head dim 64, dstate 128, bf16
        activations, fp32 A, D, dt_bias and state. With `tie_hdim`, dt, A and dt_bias are
        broadcast over the head dim as MambaMixer passes them (the kernel's TIE_HDIM branch);
        otherwise they are dense (its general branch). seq_len 4 is a speculative step, which
        also writes the state after each token."""
        torch.manual_seed(0)
        device = "cuda"
        batch, nheads, headdim, dstate, ngroups = 4, 8, 64, 128, 1
        x = torch.randn(batch, seq_len, nheads, headdim, device=device, dtype=torch.bfloat16)
        B = torch.randn(batch, seq_len, ngroups, dstate, device=device, dtype=torch.bfloat16)
        C = torch.randn(batch, seq_len, ngroups, dstate, device=device, dtype=torch.bfloat16)
        dt = torch.randn(batch, seq_len, nheads, device=device, dtype=torch.bfloat16)
        # MambaMixer's initialization: A in -[1, 16], softplus(dt_bias) log-uniform in [1e-3, 1e-1].
        A = -torch.empty(nheads, device=device).uniform_(1.0, 16.0)
        dt_init = torch.empty(nheads, device=device).uniform_(math.log(1e-3), math.log(1e-1)).exp()
        dt_bias = dt_init + torch.log(-torch.expm1(-dt_init))
        D = torch.randn(nheads, device=device)

        dt = dt[..., None].expand(batch, seq_len, nheads, headdim)
        A = A[:, None, None].expand(nheads, headdim, dstate)
        dt_bias = dt_bias[:, None].expand(nheads, headdim)
        D = D[:, None].expand(nheads, headdim)
        if not tie_hdim:
            dt, A, dt_bias = dt.contiguous(), A.contiguous(), dt_bias.contiguous()
        state0 = torch.randn(batch, nheads, headdim, dstate, device=device)
        state = state0.clone()
        per_token_states = (
            torch.empty(batch, seq_len, nheads, headdim, dstate, device=device)
            if seq_len > 1
            else None
        )

        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=torch.arange(batch, dtype=torch.int64, device=device),
            intermediate_ssm_states=per_token_states,
        )

        inputs = (state0, x, dt, A, B, C, D, dt_bias)
        _, reference = _decode_recurrence(*(t.double() for t in inputs))
        _, trusted = _decode_recurrence(*(t.float() for t in inputs))
        _assert_as_accurate_as_torch(state, reference[:, -1], trusted[:, -1], "final state")
        if per_token_states is not None:
            _assert_as_accurate_as_torch(per_token_states, reference, trusted, "per-token states")


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
