# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The FLA context-parallel kernels the GDP chunkwise CP backend launches directly.

``megatron/core/ssm/context_parallel/gdp.py`` launches FLA's CP summary kernels and its CP merge
kernel itself, outside FLA's wrappers, for both BLH (several sequences per launch) and varlen
boundary fragments. Each launch must replay bit-exactly and meet the contract chunkwise CP builds
on, checked against a computation that does not go through those launches:

- forward summary ``[h_local | M]``: a fragment started from a state ``h_in`` ends at
  ``M @ h_in + h_local``. FLA's chunk scan (``chunk_gated_delta_rule_fwd_h``) gives ``h_local``
  from a zero state, and ``M @ h_in`` from ``h_in`` with zero values;
- backward summary ``[dh_local | dM]``: a gradient ``dh_out`` at the fragment's end reaches its
  start as ``dM @ dh_out + dh_local``. FLA's backward chunk scan
  (``chunk_gated_delta_rule_bwd_dhu``) gives ``dh_local`` with no gradient at the end, and
  ``dM @ dh_out`` from ``dh_out`` alone;
- merge: the summaries compose from a zero state in rank order (forward) or in reverse rank order
  (backward), recomputed here in FP64.

The two parts of each summary are checked separately, so that neither can hide behind the other.
"""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

try:
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# GDP chunkwise CP also runs on Blackwell, so GB200 CI runs these too.
pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed"),
]

BATCH, LENGTH, KEY_HEADS, VALUE_HEADS, VALUE_DIM = 2, 200, 2, 4, 64
VARLEN_CU_SEQLENS = (0, 77, BATCH * LENGTH)


@pytest.fixture(scope="module")
def gdp():
    """gdp.py's FLA backend. With FLA installed, an import failure (an FLA that renamed or dropped
    the CP kernels gdp.py launches) is an error, not a skip."""
    from megatron.core.ssm.context_parallel import gdp

    return gdp


def _fragments(key_dim, varlen):
    """BF16 activations as the backend receives them, and gdp.py's FP32 gate: per-chunk
    cumulative sums in log2 space."""
    seeded()
    shape = (BATCH, LENGTH)
    k = torch.nn.functional.normalize(
        torch.randn(*shape, KEY_HEADS, key_dim, device="cuda"), dim=-1
    )
    beta = torch.rand(*shape, VALUE_HEADS, 1, device="cuda")
    tensors = dict(
        q=torch.randn(*shape, KEY_HEADS, key_dim, device="cuda"),
        k=k,
        # beta * k keeps every per-token transition a contraction, as the WY form does.
        w=beta * k.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2),
        u=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
        do=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
        dv=torch.randn(*shape, VALUE_HEADS, VALUE_DIM, device="cuda"),
    )
    tensors = {name: t.bfloat16() for name, t in tensors.items()}
    tensors["g"] = -0.05 * torch.rand(*shape, VALUE_HEADS, device="cuda")
    cu_seqlens = None
    if varlen:
        tensors = {name: t.flatten(0, 1).unsqueeze(0) for name, t in tensors.items()}
        cu_seqlens = torch.tensor(VARLEN_CU_SEQLENS, dtype=torch.int32, device="cuda")
    tensors["g"] = chunk_local_cumsum(
        tensors["g"],
        chunk_size=64,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    return tensors, cu_seqlens


def _apply(summary, state):
    """One affine step ``summary[..., V:] @ state + summary[..., :V]`` of a [K, V+K] summary."""
    return summary[..., VALUE_DIM:] @ state + summary[..., :VALUE_DIM]


def _assert_close(actual, expected, tolerance, what):
    """Relative error over the whole state. A mis-mapped argument or grid is off by O(1)."""
    error = ((actual.double() - expected.double()).norm() / expected.double().norm()).item()
    assert error < tolerance, f"{what}: relative error {error:.3g} (tolerance {tolerance})"


# The chunk scans round their running state to BF16 at every chunk, while the summaries keep the
# state transform in FP32.
SUMMARY_TOLERANCE = 2e-2


@pytest.mark.parametrize("varlen", [False, True], ids=["blh", "varlen"])
@pytest.mark.parametrize("key_dim", [64, 96, 128])
def test_fla_gdp_cp_forward_summary(gdp, key_dim, varlen):
    t, cu_seqlens = _fragments(key_dim, varlen)
    what = f"FLA GDP CP forward summary [K={key_dim}, {'varlen' if varlen else 'blh'}]"
    out, _ = assert_replays_bit_exact(
        lambda **x: gdp._compute_fragment_summary(**x, cu_seqlens=cu_seqlens),
        dict(k=t["k"], w=t["w"], u=t["u"], g=t["g"]),
        backward=False,
        contention=True,
        what=what,
    )
    h_in = torch.randn(BATCH, VALUE_HEADS, key_dim, VALUE_DIM, device="cuda")

    def final_state(u, initial_state):
        _, _, state = chunk_gated_delta_rule_fwd_h(
            k=t["k"],
            w=t["w"],
            u=u,
            g=t["g"],
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None if cu_seqlens is None else cu_seqlens.long(),
        )
        # The forward summary covers the trailing boundary fragment: the pack's last sequence.
        return state[-1] if varlen else state

    h_local = final_state(t["u"], torch.zeros_like(h_in))
    transformed = final_state(torch.zeros_like(t["u"]), h_in)
    if varlen:
        h_in = h_in[-1]
    summary = out["out"]
    _assert_close(summary[..., :VALUE_DIM], h_local, SUMMARY_TOLERANCE, f"{what} h_local")
    _assert_close(summary[..., VALUE_DIM:] @ h_in, transformed, SUMMARY_TOLERANCE, f"{what} M")


@pytest.mark.parametrize("varlen", [False, True], ids=["blh", "varlen"])
@pytest.mark.parametrize("key_dim", [64, 96, 128])
def test_fla_gdp_cp_backward_summary(gdp, key_dim, varlen):
    t, cu_seqlens = _fragments(key_dim, varlen)
    what = f"FLA GDP CP backward summary [K={key_dim}, {'varlen' if varlen else 'blh'}]"
    out, _ = assert_replays_bit_exact(
        lambda **x: gdp._compute_backward_fragment_summary(
            **x, scale=key_dim**-0.5, cu_seqlens=cu_seqlens
        ),
        dict(q=t["q"], k=t["k"], w=t["w"], g=t["g"], do=t["do"], dv=t["dv"]),
        backward=False,
        contention=True,
        what=what,
    )
    dh_out = torch.randn(BATCH, VALUE_HEADS, key_dim, VALUE_DIM, device="cuda")

    def initial_state_grad(do, dv, final_state_grad):
        _, grad, _ = chunk_gated_delta_rule_bwd_dhu(
            q=t["q"],
            k=t["k"],
            w=t["w"],
            do=do,
            dv=dv,
            g=t["g"],
            h0=torch.zeros_like(dh_out),
            dht=final_state_grad,
            scale=key_dim**-0.5,
            cu_seqlens=None if cu_seqlens is None else cu_seqlens.long(),
        )
        # The backward summary covers the leading boundary fragment: the pack's first sequence.
        return grad[0] if varlen else grad

    dh_local = initial_state_grad(t["do"], t["dv"], torch.zeros_like(dh_out))
    transformed = initial_state_grad(torch.zeros_like(t["do"]), torch.zeros_like(t["dv"]), dh_out)
    if varlen:
        dh_out = dh_out[0]
    summary = out["out"]
    _assert_close(summary[..., :VALUE_DIM], dh_local, SUMMARY_TOLERANCE, f"{what} dh_local")
    _assert_close(summary[..., VALUE_DIM:] @ dh_out, transformed, SUMMARY_TOLERANCE, f"{what} dM")


@pytest.mark.parametrize("batched", [False, True], ids=["varlen", "blh"])
@pytest.mark.parametrize("forward", [True, False], ids=["fwd", "bwd"])
@pytest.mark.parametrize("key_dim", [64, 96, 128])
def test_fla_gdp_cp_merge(gdp, key_dim, forward, batched):
    seeded()
    state = (BATCH,) if batched else ()
    summaries = 0.2 * torch.randn(
        3, *state, VALUE_HEADS, key_dim, VALUE_DIM + key_dim, device="cuda"
    )
    output = torch.empty(*state, VALUE_HEADS, key_dim, VALUE_DIM, device="cuda")
    what = f"FLA GDP CP merge [K={key_dim}, {'fwd' if forward else 'bwd'}]"
    out, _ = assert_replays_bit_exact(
        lambda packed_summaries: gdp._merge_affine_summaries(
            packed_summaries, output, forward
        ).clone(),
        dict(packed_summaries=summaries),
        backward=False,
        contention=True,
        what=what,
    )
    # Forward composes the preceding ranks' summaries 0, 1, 2; backward the following ranks'
    # summaries 2, 1, 0.
    expected = torch.zeros(output.shape, dtype=torch.float64, device="cuda")
    for summary in summaries.double() if forward else summaries.double().flip(0):
        expected = _apply(summary, expected)
    _assert_close(out["out"], expected, 1e-3, what)
