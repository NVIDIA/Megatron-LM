# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for pip `causal_conv1d.causal_conv1d_update` at its Megatron call site, the batch-invariant Mamba
decode conv (`MambaMixer.ssm_decode` with `batch_invariant_mode`, megatron/core/ssm/mamba_mixer.py).

Before this module the call site had only a differential end-to-end check (a cached prefix against an
uncached one, both through this kernel); its output was never compared with a reference, and what the
kernel does with the `-1` padding indices of a CUDA-graph decode batch was unverified (audit record:
"relies on unverified pip semantics for -1 indices").

Contract (written from the call site and causal-conv1d's documented API, before running the kernel):

* The call site asserts one token per request, no snapshot buffer and int32 indices, and passes
  x = xBC.to(conv_state.dtype).squeeze(1) (B, D) — a split view of the projection for a bf16 state —,
  conv_state (slots, D, W) (a layer's slice of the context's state tensor), weight (D, W) and bias (D,) in
  the state's dtype (for an fp32 state, #8646's cached FP32 copies), activation "silu", and
  conv_state_indices = the decode batch indices: a slot per row, -1 for the padding rows of a CUDA-graph
  batch.
* For a row with slot s >= 0: conv_state[s] <- [conv_state[s][:, 1:], x] (shift, append; exact), and
  out[b] = silu(bias + sum_j weight[:, j] * conv_state[s][:, j]) in x's dtype, fp32 accumulation.
* A padding row must not write any slot, nor any memory outside the state tensor: production's
  conv_state is `mamba_conv_states[layer]`, so a write through slot -1 would land in the previous layer's
  last slot (or before the allocation for layer 0). Its output is unspecified: `_decode_step_prep_kernel`
  zeroes the scan output of a padding entry. Slots no row maps to are untouched; inputs are read-only.
* Batch invariance (the reason for this call site: "Match the causal-conv1d arithmetic used by the
  training forward"): decoding a sequence token by token from a zero state gives, bit for bit, the
  training conv's (pip `causal_conv1d_fn`) output at each position, at the call site's output dtype.
* Tolerance class: floating point for `out` (reference: fp64; trusted: the same math in torch fp32 on the
  device); exact for the state (copies), untouched slots, the guard bands around the state tensor (a whole
  slot on each side, so a write through slot -1 is caught without leaving the allocation), and the
  bitwise parity above.

Variants: the pip kernel's width is the call site's d_conv (4); the state dtype bf16 (x a strided view)
and fp32 (x a contiguous cast); D 64, 65, 10240 (Nemotron-H 8B at TP 1); batches with and without padding
rows, all padding; shuffled slots and unmapped slots. The call itself: `test_mixer_call_site` runs
`MambaMixer.ssm_decode` in batch-invariant mode with a spy on the kernel, so the arguments the mixer
builds (the split, the casts, #8646's cached parameters, the indices) are checked against the mixer's
own parameters.

Main vs open PRs: no open PR changes this call site. #8646 (on main) makes it pass cached FP32 weight and
bias for an FP32 state, bitwise the per-step casts.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"
D_CONV = 4  # MambaMixer default
PRODUCTION_DIM = 4096 * 2 + 2 * 8 * 128  # Nemotron-H 8B at TP 1


def _pip():
    """pip causal-conv1d (in the CI image), or a failure on a GPU without it."""
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        pytest.fail("causal_conv1d (the kernel under test) is not installed", pytrace=False)
    return causal_conv1d_update, causal_conv1d_fn


def reference_step(x, state, weight, bias):
    """One decode step per row: (out (B, D), new states (B, D, W)) in the inputs' dtype."""
    new = torch.cat([state[..., 1:], x[..., None]], dim=-1)
    return F.silu((new * weight).sum(-1) + bias), new


def call_site_inputs(batch, dim, state_dtype, *, padding, extra_slots, g):
    """The arguments as `MambaMixer.ssm_decode` builds them in batch-invariant mode."""
    nslots = batch + extra_slots
    slots = torch.randperm(nslots, generator=torch.Generator().manual_seed(5))[:batch]
    for p in padding:
        slots[p] = -1
    indices = slots.to(device=DEVICE, dtype=torch.int32)
    # zxBCdt (B, 1, z + xBC + dt) in bf16; xBC is a split view of it.
    proj = torch.randn(batch, 1, 2 * dim + 9, device=DEVICE, generator=g).bfloat16()
    xBC = proj[..., dim + 4 : 2 * dim + 4]
    x = xBC.to(state_dtype).squeeze(1)
    conv1d_weight = torch.randn(dim, 1, D_CONV, device=DEVICE, generator=g).bfloat16() * 0.5
    conv1d_bias = torch.randn(dim, device=DEVICE, generator=g).bfloat16()
    weight = conv1d_weight[:, 0, :].to(state_dtype)
    bias = conv1d_bias.to(state_dtype)
    state0 = torch.randn(nslots, dim, D_CONV, device=DEVICE, generator=g).to(state_dtype)
    return dict(x=x, weight=weight, bias=bias, state0=state0, indices=indices, slots=slots)


def run_step(batch, dim, state_dtype, *, padding=(1,), extra_slots=3):
    """One call-site launch on a guarded state tensor, checked against the contract."""
    update, _ = _pip()
    g = ou.make_generator(device=DEVICE)
    d = call_site_inputs(batch, dim, state_dtype, padding=padding, extra_slots=extra_slots, g=g)
    slot_bytes = dim * D_CONV * state_dtype.itemsize
    cache = ou.guarded_like(d["state0"], guard_bytes=2 * slot_bytes)
    kept = {k: d[k].clone() for k in ("x", "weight", "bias", "indices")}
    out = update(
        d["x"], cache.tensor, d["weight"], d["bias"], "silu", conv_state_indices=d["indices"]
    )
    ou.check_guards(conv_state=cache)
    for name, value in kept.items():
        ou.assert_exact(d[name], value, what=f"{name} is read-only")
    assert out.shape == d["x"].shape and out.dtype == state_dtype
    live = [b for b in range(batch) if int(d["slots"][b]) >= 0]
    mapped = [int(d["slots"][b]) for b in live]
    unmapped = [s for s in range(d["state0"].shape[0]) if s not in mapped]
    ou.assert_exact(cache.tensor[unmapped], d["state0"][unmapped], what="slots no row maps to")
    if not live:
        return
    x, state = d["x"][live], d["state0"][mapped]
    ref_out, ref_state = ou.fp64_reference(reference_step, x, state, d["weight"], d["bias"])
    t_out, _ = reference_step(x.float(), state.float(), d["weight"].float(), d["bias"].float())
    ou.assert_exact(cache.tensor[mapped], ref_state.to(state_dtype), what="new conv states")
    ou.assert_close_to_reference(out[live], ref_out, t_out.to(state_dtype), what="conv output")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("dim", [64, 65, 2 * 1024 + 2 * 128 + 1])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_call_site_step_matches_fp64(state_dtype, dim):
    """A decode step at the call site's arguments: shuffled slots, a padding row, unmapped slots."""
    run_step(9, dim, state_dtype)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_padding_rows_write_no_slot(state_dtype):
    """Padding rows only (an idle CUDA-graph batch) and a mostly padded batch: no slot and no byte
    around the state tensor may change (a write through slot -1 lands in the front guard band)."""
    run_step(4, 65, state_dtype, padding=(0, 1, 2, 3))
    run_step(8, 65, state_dtype, padding=(0, 2, 3, 5, 7))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_production_decode_batch(state_dtype):
    """Production: the Nemotron-H 8B conv (D 10240, W 4) at a decode batch of 64 with padding rows."""
    run_step(64, PRODUCTION_DIM, state_dtype, padding=(3, 62, 63), extra_slots=64)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_decode_matches_the_training_conv_bitwise(state_dtype):
    """Batch invariance: 37 tokens decoded one at a time from a zero state give, bit for bit at the
    activation dtype (bf16; an fp32 state's output is cast back as the call site does), the training
    conv `causal_conv1d_fn` over the whole sequence, for 3 sequences in shuffled slots."""
    update, conv_fn = _pip()
    g = ou.make_generator(device=DEVICE)
    batch, dim, length = 3, 129, 37
    seq = torch.randn(batch, length, dim, device=DEVICE, generator=g).bfloat16()
    conv1d_weight = torch.randn(dim, 1, D_CONV, device=DEVICE, generator=g).bfloat16() * 0.5
    conv1d_bias = torch.randn(dim, device=DEVICE, generator=g).bfloat16()
    training = conv_fn(
        seq.transpose(1, 2), conv1d_weight[:, 0, :], conv1d_bias, activation="silu"
    ).transpose(1, 2)
    indices = torch.tensor([4, 0, 2], dtype=torch.int32, device=DEVICE)
    state = torch.zeros(5, dim, D_CONV, device=DEVICE, dtype=state_dtype)
    weight, bias = conv1d_weight[:, 0, :].to(state_dtype), conv1d_bias.to(state_dtype)
    decoded = []
    for t in range(length):
        out = update(
            seq[:, t].to(state_dtype), state, weight, bias, "silu", conv_state_indices=indices
        )
        decoded.append(out.to(torch.bfloat16))
    ou.assert_exact(torch.stack(decoded, 1), training, what="decode vs training conv (bitwise)")


@pytest.fixture
def model_parallel():
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_mixer_call_site(model_parallel, monkeypatch, state_dtype):
    """Through `MambaMixer.ssm_decode` in batch-invariant mode (the flag set on a built mixer): the
    kernel's output, recorded by a spy around it, against the fp64 conv of the mixer's own parameters
    and the step's projection; the conv states exact; padding rows' slots untouched. An fp32 state
    takes #8646's cached FP32 weight and bias."""
    import megatron.core.ssm.mamba_mixer as mixer_module
    from tests.unit_tests.inference.oracles.ssm.test_mamba_static_decode import build_mixer

    update, _ = _pip()
    calls = []

    def spy(x, conv_state, weight, bias, activation, conv_state_indices=None):
        out = update(x, conv_state, weight, bias, activation, conv_state_indices=conv_state_indices)
        calls.append(out.clone())
        return out

    monkeypatch.setattr(mixer_module, "causal_conv1d_update_cuda", spy)
    mixer = build_mixer("h8g1")
    monkeypatch.setattr(mixer.config, "batch_invariant_mode", True)
    g = ou.make_generator(device=DEVICE)
    batch, slots = 5, 8
    indices = torch.tensor([6, -1, 0, 3, -1], dtype=torch.int32, device=DEVICE)
    conv_dim = mixer.conv1d_weight.shape[0]
    conv0 = torch.randn(slots, conv_dim, D_CONV, device=DEVICE, generator=g).to(state_dtype)
    conv_state = conv0.clone()
    ssm_state = torch.zeros(
        slots, mixer.nheads_local_tp, mixer.headdim, mixer.d_state, device=DEVICE
    )
    # zxBCdt = [z (d_inner), xBC (d_inner + 2 G N), dt (nheads)], as in_proj produces it.
    proj = 2 * mixer.d_inner_local_tp + 2 * mixer.ngroups_local_tp * mixer.d_state
    proj += mixer.nheads_local_tp
    zxBCdt = torch.randn(batch, 1, proj, device=DEVICE, generator=g).bfloat16()
    with torch.no_grad():
        mixer.ssm_decode(zxBCdt, conv_state, ssm_state, batch_indices=indices)
    assert len(calls) == 1, "the batch-invariant branch calls the pip kernel once"
    live = [b for b in range(batch) if int(indices[b]) >= 0]
    mapped = [int(indices[b]) for b in live]
    xBC = zxBCdt[:, 0, mixer.d_inner_local_tp : -mixer.nheads_local_tp]
    weight = mixer.conv1d_weight.detach()[:, 0, :]
    bias = mixer.conv1d_bias.detach()
    args = (xBC[live].to(state_dtype), conv0[mapped], weight, bias)
    ref_out, ref_state = ou.fp64_reference(reference_step, *args)
    t_out, _ = reference_step(*[a.float() for a in args])
    unmapped = [s for s in range(slots) if s not in mapped]
    ou.assert_exact(conv_state[unmapped], conv0[unmapped], what="slots no row maps to")
    ou.assert_exact(conv_state[mapped], ref_state.to(state_dtype), what="new conv states")
    ou.assert_close_to_reference(
        calls[0][live], ref_out, t_out.to(state_dtype), what="pip conv output at the call site"
    )


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: D 65, a padding row, unmapped slots, bf16 state."""
    run_step(5, 65, torch.bfloat16, padding=(2,))
