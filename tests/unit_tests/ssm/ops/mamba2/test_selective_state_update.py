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
