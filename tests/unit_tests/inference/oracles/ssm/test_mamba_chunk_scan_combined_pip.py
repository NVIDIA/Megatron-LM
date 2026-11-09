# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for pip `mamba_ssm.mamba_chunk_scan_combined(..., return_final_states=True)` at its inference
call site, the static-batching Mamba prefill (`MambaMixer._static_prefill`, megatron/core/ssm/mamba_mixer.py),
which returns each request's final SSM state and copies it into the static cache.

Before this module the call was the trusted implementation of the varlen scan oracle
(test_chunk_scan_varlen.py) but was never tested itself; its final-state handoff was reached only by the
static-decode prefill cases (test_mamba_static_decode.py), which check the conv cache and fail on main for
an unrelated decode bug (#8568) at b > 1.

Contract (written from the Mamba-2 recurrence, mamba_ssm's documented API and the call site, before running
the kernel):

* Kernel: x (b, L, nheads, headdim), dt (b, L, nheads), B, C (b, L, ngroups, dstate) in the model dtype;
  A (nheads,) fp32; D (nheads,) or (nheads, headdim) fp32; z (b, L, nheads, headdim) or None; dt_bias
  (nheads,) fp32; dt_softplus; no initial states; `state_dtype` where the installed mamba_ssm takes it
  (the call site passes `mamba_training_ssm_states_dtype`, by default the model dtype; without that
  parameter, the cases that name a state dtype run as the call site then does, without it). Each batch row is
  an independent request from a zero state:

      dt' = softplus(dt + dt_bias);  h_t = exp(dt' A) h_{t-1} + dt' x_t (outer) B_t,g(h)
      y_t = (C_t . h_t + D x_t) * z_t sigmoid(z_t)   (no gate without z);   final = h_L

  y in x's dtype. `state_dtype` sets the dtype of the chunk states the scan reads; the final state is
  the fp32 carry, returned in fp32 whatever `state_dtype` says (H100, pdx 7986854), and the call site
  copies it into the cache's dtype, so its dtype is not part of the contract.
* Call site: a static prefill of b prompts (`forward` with a StaticInferenceContext at offset 0) leaves,
  in the static caches, conv_state[i] = the prompt's last d_conv pre-conv xBC columns, zero-padded on the
  left (exact) and ssm_state[i] = request i's final state after the conv and scan of its prompt, in the
  cache dtype; out_proj's input is the scan output (rounded to the activation dtype) through the gated
  group RMSNorm (z goes to the scan only without rmsnorm); D is reshaped to (nheads, headdim) with
  D_has_hdim.
* Tolerance class: floating point. Reference: the token-by-token recurrence in fp64 (no chunking; for the
  call site, the conv too, from the captured in_proj output). Trusted: Megatron's own varlen prefill
  kernels — `mamba_chunk_scan_combined_varlen` (and `causal_conv1d_varlen_fn` at the call site) on the
  same requests with production chunk metadata — an independent implementation of the same chunked
  algorithm and roundings (itself checked against fp64 by test_chunk_scan_varlen.py). Exact: the conv
  cache, read-only inputs.

Inputs: b 1 and 3; prompt lengths 1, 3, 4, 5 (around d_conv), 127, 128, 129, 259 (around the chunk);
chunk 128 (the mixer's) and 256; (nheads, ngroups) (8, 1), (8, 8), (1, 1); headdim 64 and 48; dstate
128 and 64; D per head, per channel and none; z on and off; bf16 and fp32; state_dtype default, bf16 and
fp32. Production: the Nemotron-H 8B layer (128 heads of 64, 8 groups, dstate 128, chunk 128).

Main vs open PRs: no open PR changes this call. #8568 changes only the static decode.
"""

import pytest
import torch

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn
from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su
from tests.unit_tests.inference.oracles.ssm.test_causal_conv1d_varlen import varlen_conv_reference
from tests.unit_tests.inference.oracles.ssm.test_chunk_scan_varlen import _scan_reference
from tests.unit_tests.test_utilities import Utils

DEVICE = "cuda"


def _pip_scan():
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        pytest.fail("mamba_ssm (the kernel under test) is not installed", pytrace=False)
    return mamba_chunk_scan_combined


def _takes_state_dtype():
    from megatron.core.ssm.mamba_mixer import MAMBA_HAS_STATE_DTYPE

    return MAMBA_HAS_STATE_DTYPE


def flat(t):
    """(b, L, ...) -> (b * L, ...): the batch rows as a varlen pack."""
    return t.reshape(-1, *t.shape[2:])


def trusted_scan(x, dt, A, B, C, D, z, dt_bias, chunk_size, state_dtype):
    """Megatron's varlen scan on the batch rows as a pack of equal-length requests: (y, final)."""
    batch, length = x.shape[:2]
    meta = su.production_metadata([length] * batch, 0, 0, chunk_size, DEVICE)
    out = torch.zeros_like(flat(x))
    final = mamba_chunk_scan_combined_varlen(
        flat(x),
        flat(dt),
        A,
        flat(B),
        flat(C),
        chunk_size,
        cu_chunk_seqlens=meta["cu_chunk_seqlens"],
        last_chunk_indices=meta["last_chunk_indices"],
        seq_idx=meta["seq_idx"],
        out=out,
        D=D,
        z=None if z is None else flat(z),
        dt_bias=dt_bias,
        dt_softplus=True,
        state_dtype=state_dtype,
    )
    return out.view_as(x), final


def reference_scan(x, dt, A, B, C, D, z, dt_bias):
    """fp64 recurrence per batch row from zero: (y (b, L, H, P), final (b, H, P, N))."""
    batch, length, heads, headdim = x.shape
    dstate = B.shape[-1]
    h0 = x.new_zeros(batch, heads, headdim, dstate)
    starts = [i * length for i in range(batch)]
    y, final, _ = _scan_reference(
        flat(x),
        flat(dt),
        A,
        flat(B),
        flat(C),
        D,
        None if z is None else flat(z),
        dt_bias,
        h0,
        starts,
        [length] * batch,
        [],
    )
    return y.view_as(x), final


def run_pip_scan(**case):
    """`_run_pip_scan` with both scans pinned to batch-invariant mode's tiles (one compile per shape);
    `timed=True` leaves pip's autotuner timing its candidates, as outside batch-invariant mode."""
    if case.pop("timed", False):
        with su.pinned_scan_kernels(include_pip=False):
            return _run_pip_scan(**case)
    with su.pinned_scan_kernels():
        return _run_pip_scan(**case)


def _run_pip_scan(
    *, batch, length, chunk_size, heads_groups, headdim, dstate, d_kind, with_z, dtype, state_dtype
):
    scan = _pip_scan()
    g = ou.make_generator(device=DEVICE)
    heads, groups = heads_groups

    def rand(*shape):
        return torch.randn(*shape, device=DEVICE, generator=g).to(dtype)

    x, dt = rand(batch, length, heads, headdim), rand(batch, length, heads)
    B, C = rand(batch, length, groups, dstate), rand(batch, length, groups, dstate)
    z = rand(batch, length, heads, headdim) if with_z else None
    A = -torch.empty(heads, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    dt0 = torch.empty(heads, device=DEVICE).uniform_(1e-3, 1e-1, generator=g)
    dt_bias = dt0 + torch.log(-torch.expm1(-dt0))
    D = {
        "per_head": torch.randn(heads, device=DEVICE, generator=g),
        "per_channel": torch.randn(heads, headdim, device=DEVICE, generator=g),
        None: None,
    }[d_kind]
    # As the call site: `state_dtype` only where the installed mamba_ssm takes it (else the default).
    kwargs = {}
    if state_dtype is not None and _takes_state_dtype():
        kwargs["state_dtype"] = state_dtype
    inputs = dict(x=x, dt=dt, B=B, C=C, z=z, D=D, dt_bias=dt_bias, A=A)
    kept = {k: v.clone() for k, v in inputs.items() if v is not None}
    y, final = scan(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        return_final_states=True,
        **kwargs,
    )
    for name, value in kept.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    assert y.shape == x.shape and y.dtype == dtype
    assert final.shape == (batch, heads, headdim, dstate)
    args = (x, dt, A, B, C, D, z, dt_bias)
    ref_y, ref_final = ou.fp64_reference(reference_scan, *args)
    # y: Megatron's scan with the same state dtype, which sets the chunk states the scan reads.
    t_y, _ = trusted_scan(*args, chunk_size, kwargs.get("state_dtype"))
    # The final state: the fp32 carry, stored in the dtype mamba_ssm returns (fp32 at H100, pdx
    # 7986854, whatever state_dtype says; the call site casts it into the cache).
    _, t_final = trusted_scan(*args, chunk_size, final.dtype)
    ou.assert_close_to_reference(y, ref_y, t_y, what="y")
    ou.assert_close_to_reference(final, ref_final, t_final, what="final states")


def _cases():
    factors = {
        "batch": (3, 1),
        "length": (129, 1, 3, 4, 5, 127, 128, 259),
        "chunk_size": (su.PRODUCTION_CHUNK, 256),
        "heads_groups": ((8, 1), (8, 8), (1, 1)),
        "headdim": (64, 48),
        "dstate": (128, 64),
        "d_kind": ("per_head", "per_channel", None),
        "with_z": (False, True),
        "dtype": (torch.bfloat16, torch.float32),
        "state_dtype": (None, torch.bfloat16, torch.float32),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                f"b{case['batch']}",
                f"L{case['length']}",
                f"c{case['chunk_size']}",
                "h{}g{}".format(*case["heads_groups"]),
                f"p{case['headdim']}",
                f"n{case['dstate']}",
                f"D_{case['d_kind']}",
                "z" if case["with_z"] else "noz",
                str(case["dtype"]).replace("torch.", ""),
                "state_" + str(case["state_dtype"]).replace("torch.", ""),
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_pip_scan_with_final_states_matches_fp64(case):
    """The kernel as the static prefill calls it: y and the final states of every batch row."""
    run_pip_scan(**case)


@ou.requires_sm(ou.HOPPER)
def test_production_layer():
    """Production: the Nemotron-H 8B layer at TP 1 (128 heads of 64, 8 groups, dstate 128, chunk 128),
    a static batch of 4 prompts of 257 tokens, bf16, per-head D, rmsnorm (no z), default state dtype.
    """
    run_pip_scan(
        timed=True,
        batch=4,
        length=257,
        chunk_size=128,
        heads_groups=(128, 8),
        headdim=64,
        dstate=128,
        d_kind="per_head",
        with_z=False,
        dtype=torch.bfloat16,
        state_dtype=None,
    )


# ---------------------------------------------------------------------------------------------
# The call site: MambaMixer static prefill
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _gated_norm(y, z, norm_w, groups, eps, round_to=None):
    """The mixer's gated group RMSNorm (rmsnorm=True, norm after the gate): y and z (..., d_inner)."""
    if round_to is not None:
        y = y.to(round_to).to(z.dtype)
    y = y * (z * torch.sigmoid(z))
    shape = y.shape
    y = y.reshape(*shape[:-1], groups, -1)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y.reshape(shape) * norm_w


def _split(zxBCdt, heads, groups, dstate, d_inner):
    seq = zxBCdt.transpose(0, 1).contiguous()  # (b, L, proj)
    return seq.split([d_inner, d_inner + 2 * groups * dstate, heads], -1)


def _mixer_reference(
    zxBCdt, conv_w, conv_b, dt_bias, A, D, norm_w, *, heads, groups, headdim, dstate, eps
):
    """fp64: each request's conv (zero history), scan and gated norm over its prompt.

    zxBCdt (L, b, proj); D (H,) or (H, P). Returns (out_proj input (b, L, d_inner), final states).
    """
    length, batch, _ = zxBCdt.shape
    d_inner = heads * headdim
    z, xBC, dt = _split(zxBCdt, heads, groups, dstate, d_inner)
    conv = varlen_conv_reference(
        flat(xBC), conv_w, conv_b, None, [i * length for i in range(batch)], [length] * batch
    ).view(batch, length, -1)
    x, B, C = conv.split([d_inner, groups * dstate, groups * dstate], -1)
    y, final = reference_scan(
        x.reshape(batch, length, heads, headdim),
        dt,
        A,
        B.reshape(batch, length, groups, dstate),
        C.reshape(batch, length, groups, dstate),
        D,
        None,
        dt_bias,
    )
    return _gated_norm(y.reshape(batch, length, d_inner), z, norm_w, groups, eps), final


def _mixer_trusted(
    zxBCdt, conv_w, conv_b, dt_bias, A, D, norm_w, *, heads, groups, headdim, dstate, eps, chunk
):
    """Megatron's dynamic-prefill kernels on the same prompts, in the model's dtypes, then the gated
    norm in fp32 on the bf16 scan output (the module's interface rounding)."""
    length, batch, _ = zxBCdt.shape
    d_inner = heads * headdim
    z, xBC, dt = _split(zxBCdt, heads, groups, dstate, d_inner)
    cu = torch.arange(0, (batch + 1) * length, length, device=DEVICE, dtype=torch.int32)
    conv = causal_conv1d_varlen_fn(
        flat(xBC).contiguous(), conv_w.contiguous(), conv_b, cu, activation="silu"
    ).view(batch, length, -1)
    x, B, C = conv.split([d_inner, groups * dstate, groups * dstate], -1)
    y, final = trusted_scan(
        x.reshape(batch, length, heads, headdim).contiguous(),
        dt.contiguous(),
        A,
        B.reshape(batch, length, groups, dstate).contiguous(),
        C.reshape(batch, length, groups, dstate).contiguous(),
        D,
        None,
        dt_bias,
        chunk,
        torch.float32,
    )
    y = _gated_norm(
        y.reshape(batch, length, d_inner).float(), z.float(), norm_w.float(), groups, eps
    )
    return y.to(zxBCdt.dtype), final


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("prompt", [pytest.param(n, id=f"L{n}") for n in (3, 4, 5, 127, 128, 129)])
@pytest.mark.parametrize(
    "head_config,D_has_hdim", [("h8g8", False), ("h8g1", True), ("h1g1", False)]
)
def test_static_prefill_hands_each_request_its_final_state(
    model_parallel, head_config, D_has_hdim, prompt
):
    """A static prefill of b = 3 prompts through `MambaMixer.forward`: the conv cache holds each prompt's
    last d_conv inputs (exact); the SSM cache each request's own final state and out_proj's input each
    token's gated-normed scan output (both bounded against the fp64 conv + scan + norm of the captured
    projection, per request and per token)."""
    from tests.unit_tests.inference.oracles.ssm.test_mamba_static_decode import (
        D_CONV,
        HIDDEN,
        Capture,
        build_mixer,
    )

    _pip_scan()
    mixer = build_mixer(head_config, D_has_hdim=D_has_hdim)
    batch, g = 3, ou.make_generator(device=DEVICE)
    context = StaticInferenceContext(max_batch_size=batch, max_sequence_length=512)
    context.sequence_len_offset = context.seqlen_offset = 0
    hidden = torch.randn(prompt, batch, HIDDEN, device=DEVICE, generator=g).bfloat16()
    capture = Capture(mixer)
    try:
        with torch.no_grad(), InferenceMode.active():
            mixer(hidden, inference_context=context)
    finally:
        capture.remove()
    conv_state, ssm_state = context.key_value_memory_dict[mixer.layer_number]
    zxBCdt, y = capture.zxBCdt[-1], capture.y[-1]  # (L, b, proj), (L, b, d_inner)
    heads, groups = mixer.nheads_local_tp, mixer.ngroups_local_tp
    xBC = zxBCdt[:, :, mixer.d_inner_local_tp : -heads]
    window = xBC.permute(1, 2, 0)[..., -D_CONV:]
    expected = torch.zeros_like(conv_state)
    expected[..., D_CONV - window.shape[-1] :] = window.to(conv_state.dtype)
    ou.assert_exact(conv_state, expected, what="conv cache after the prefill")

    D = mixer.D.detach().float()
    if D_has_hdim:
        D = D.view(heads, mixer.headdim)
    kw = dict(
        heads=heads, groups=groups, headdim=mixer.headdim, dstate=mixer.d_state, eps=mixer.norm.eps
    )
    params = (
        zxBCdt,
        mixer.conv1d_weight.detach()[:, 0, :],
        mixer.conv1d_bias.detach(),
        mixer.dt_bias.detach().float(),
        -torch.exp(mixer.A_log.detach().float()),
        D,
        mixer.norm.weight.detach(),
    )
    ref_y, ref_final = ou.fp64_reference(_mixer_reference, *params, **kw)
    with su.pinned_scan_kernels(include_pip=False):  # the trusted scan's tile does not matter
        t_y, t_final = _mixer_trusted(*params, **kw, chunk=mixer.chunk_size)
    ou.assert_close_to_reference(
        ssm_state.flatten(1),
        ref_final.flatten(1),
        t_final.to(ssm_state.dtype).flatten(1),
        dim=-1,
        what="SSM cache after the prefill",
    )
    ou.assert_close_to_reference(
        y.transpose(0, 1).flatten(0, 1),
        ref_y.flatten(0, 1),
        t_y.flatten(0, 1),
        dim=-1,
        what="out_proj input (gated-normed scan output)",
    )


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: b 2, a 5-token prompt, 2 heads of 48, dstate 64, per-channel D
    and z."""
    run_pip_scan(
        batch=2,
        length=5,
        chunk_size=128,
        heads_groups=(2, 1),
        headdim=48,
        dstate=64,
        d_kind="per_channel",
        with_z=True,
        dtype=torch.bfloat16,
        state_dtype=None,
    )


def test_case_list_covers_every_factor():
    """Every batch, length class, layout, D kind, gate, dtype and state dtype occurs (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in (
        "batch",
        "length",
        "chunk_size",
        "heads_groups",
        "d_kind",
        "with_z",
        "state_dtype",
    ):
        assert len({str(c[factor]) for c in cases}) > 1
