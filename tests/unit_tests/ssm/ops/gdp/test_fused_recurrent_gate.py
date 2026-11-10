# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The decay the fused recurrent kernel forms itself: GDN's decode gate.

`GatedDeltaNet.ssm_decode` hands `fused_recurrent_gated_delta_rule_update` the raw gate
input `a` together with `A_log` and `dt_bias`, and the kernel forms the log decay
`g = -exp(A_log) * softplus(a + dt_bias)` in fp32. `GatedDeltaNet.ssm_prefill` forms the
same `g` in torch.

The kernel used to compute softplus as `log(1 + exp(x))` with approximate `ex2`/`lg2`
(FLA's inline PTX). Rounding `1 + exp(x)` before the log leaves an absolute error of at
least 2**-24, about 1e-4 relative where softplus is small, and `exp(A_log)` (up to 16)
carries it into the decay of the state on every step.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tests.unit_tests.ssm.ops.mamba2.test_selective_state_update import _assert_as_accurate_as_torch

# GDN's TransformerConfig defaults (the Qwen3-Next shape): 16 key heads, 32 value heads, head
# dims 128. Rows 3 and 6 are CUDA-graph padding (-1); slots 1, 4 and 9 belong to no row.
_H, _HV, _K, _V = 16, 32, 128, 128
_SLOTS = (5, 0, 7, -1, 2, 8, -1, 3)
_NUM_SLOTS = 10


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _decode(q, k, v, g, beta, state, slots, **gate):
    """One GDN decode step as `GatedDeltaNet.ssm_decode` launches it, in place on `state`."""
    o, _ = fused_recurrent_gated_delta_rule_update(
        q,
        k,
        v,
        g=g,
        beta=beta,
        state=state,
        state_indices=torch.tensor(slots, dtype=torch.int32, device=state.device),
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        **gate,
    )
    return o


def _activations(n, H, HV, K, V, device="cuda"):
    """bf16 q, k [n, 1, H, K], v [n, 1, HV, V] and beta logits [n, 1, HV]."""
    q = torch.randn(n, 1, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(n, 1, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(n, 1, HV, V, device=device, dtype=torch.bfloat16)
    beta = torch.randn(n, 1, HV, device=device, dtype=torch.bfloat16)
    return q, k, v, beta


def _gdn_decode_state(k, v, a, b, A_log, dt_bias, state):
    """The state after one GDN decode step, in the dtype of the inputs: k [n, H, K];
    v [n, HV, V]; a, b [n, HV]; A_log, dt_bias [HV]; state [n, HV, K, V]."""
    k = k.repeat_interleave(v.shape[1] // k.shape[1], dim=1)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    state = state * g.exp()[..., None, None]
    u = torch.sigmoid(b)[..., None] * (v - torch.einsum("nhkv,nhk->nhv", state, k))
    return state + k[..., None] * u[..., None, :]


def _assert_bitwise_equal(actual, expected, what):
    if torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        return
    bits = torch.int16 if actual.element_size() == 2 else torch.int32
    differ = actual.view(bits) != expected.view(bits)
    worst = (actual.float() - expected.float()).abs().max().item()
    raise AssertionError(
        f"{what}: {differ.sum().item()} of {differ.numel()} elements differ from decode with "
        f"the eager gate (max |diff| {worst:.3e})"
    )


@pytest.mark.internal
class TestGDNDecodeGate:
    """The in-kernel gate is torch's gate: bit for bit, and as accurate."""

    def setup_method(self, method):
        _requires_cuda()

    @pytest.mark.parametrize(
        "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"]
    )
    @pytest.mark.parametrize(
        "params_dtype", [torch.float32, torch.bfloat16], ids=["fp32_params", "bf16_params"]
    )
    def test_matches_the_eager_gate_bit_for_bit(self, state_dtype, params_dtype):
        """Decode with the in-kernel gate equals decode with `g` formed in torch as
        `GatedDeltaNet.ssm_prefill` forms it, bit for bit: a token decays the state the same
        way whichever phase processes it. `a + dt_bias` spans small and large softplus values
        and both sides of softplus's threshold 20; GDN's gate parameters follow
        `params_dtype`."""
        torch.manual_seed(0)
        n = len(_SLOTS)
        q, k, v, b = _activations(n, _H, _HV, _K, _V)
        A_log = torch.empty(_HV, device="cuda").uniform_(1.0, 16.0).log().to(params_dtype)
        dt_bias = torch.randn(_HV, device="cuda").to(params_dtype)
        a = torch.randn(n, 1, _HV, device="cuda") * 8
        a[0, 0, :4] = torch.tensor([-30.0, 19.5, 20.5, 40.0], device="cuda")
        a = a.bfloat16()
        state0 = torch.randn(_NUM_SLOTS, _HV, _K, _V, device="cuda").to(state_dtype)

        state = state0.clone()
        o = _decode(q, k, v, a, b, state, _SLOTS, A_log=A_log, dt_bias=dt_bias)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        state_eager = state0.clone()
        o_eager = _decode(q, k, v, g, b, state_eager, _SLOTS)

        _assert_bitwise_equal(state, state_eager, "state")
        _assert_bitwise_equal(o, o_eager, "output")

    def test_decay_within_4_eps_of_fp64(self):
        """Read the decay out of single steps, bounded against fp64 alone. With zero keys a
        step adds nothing, so a unit state becomes exp(g) itself, rounded once.

        At exp(A_log) = 16, the steepest decay rate GDN initializes, and softplus(a + dt_bias)
        between about 5e-4 and 4e-3, exp(g) must be within 4 eps: libdevice's exp and log1p
        are within 2 and 1 ulp, the kernel's `exp` (ex2.approx) within 2 ulp, and |g| < 0.07
        keeps g's own error below an eps of exp(g). `log(1 + exp(x))` was off by about 1e-4
        relative in softplus here, several eps in exp(g).
        """
        torch.manual_seed(0)
        n, H = 64, 8
        q, _, v, b = _activations(n, H, H, 16, 16)
        k = torch.zeros(n, 1, H, 16, device="cuda", dtype=torch.bfloat16)
        A_log = torch.full((H,), math.log(16.0), device="cuda")
        dt_bias = torch.randn(H, device="cuda")
        a = torch.empty(n, 1, H, device="cuda").uniform_(-7.5, -5.5) - dt_bias
        arg = a + dt_bias  # the fp32 sum the kernel takes softplus of
        state = torch.ones(n, H, 16, 16, device="cuda")

        _decode(q, k, v, a, b, state, list(range(n)), A_log=A_log, dt_bias=dt_bias)

        decay = torch.exp(-A_log.double().exp() * F.softplus(arg.double()))[:, 0]
        expected = decay[..., None, None].expand_as(state)
        error = (state.double() - expected).abs() / expected / torch.finfo(torch.float32).eps
        worst = error.flatten().argmax()
        i, h = (worst // error[0].numel()).item(), (worst // (16 * 16) % H).item()
        assert error.max().item() <= 4, (
            f"exp(g) is {error.max().item():.1f} eps from fp64 (bound 4) at a + dt_bias = "
            f"{arg[i, 0, h].item()!r}: kernel {state[i, h, 0, 0].item()!r}, "
            f"fp64 {decay[i, h].item()!r}"
        )

    def test_fp32_state_matches_fp64(self):
        """One decode step at GDN's default shape with an fp32 state
        (`--mamba-inference-ssm-states-dtype fp32`) must be within 3x torch fp32's error
        against fp64, plus one ulp.

        Slow decays: dt_bias puts softplus(a + dt_bias) between about 1e-3 and 1e-1, as Mamba
        and GDP initialize it, where `log(1 + exp(x))` made the state's error about 10x
        torch's. Factor 3 rather than 2: the kernel decays the carried state with
        `ex2.approx` (up to 2 ulp) where torch's exp is accurate, which measured up to 2.1x
        torch's error in the fp32 states of correct decode steps on H100.
        """
        torch.manual_seed(0)
        n = len(_SLOTS)
        q, k, v, b = _activations(n, _H, _HV, _K, _V)
        a = torch.randn(n, 1, _HV, device="cuda", dtype=torch.bfloat16)
        A_log = torch.empty(_HV, device="cuda").uniform_(1.0, 16.0).log()
        dt = torch.empty(_HV, device="cuda").uniform_(math.log(1e-3), math.log(1e-1)).exp()
        dt_bias = dt + torch.log(-torch.expm1(-dt))
        state0 = torch.randn(_NUM_SLOTS, _HV, _K, _V, device="cuda")
        state = state0.clone()

        _decode(q, k, v, a, b, state, _SLOTS, A_log=A_log, dt_bias=dt_bias)

        live = [i for i, slot in enumerate(_SLOTS) if slot >= 0]
        slots = [_SLOTS[i] for i in live]
        inputs = (k[live, 0], v[live, 0], a[live, 0], b[live, 0], A_log, dt_bias)
        reference = _gdn_decode_state(*(t.double() for t in inputs), state0[slots].double())
        trusted = _gdn_decode_state(*(t.float() for t in inputs), state0[slots])
        _assert_as_accurate_as_torch(state[slots], reference, trusted, "final state", factor=3.0)
