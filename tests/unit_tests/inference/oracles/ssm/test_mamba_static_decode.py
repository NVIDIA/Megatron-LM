# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the static-batching Mamba decode dispatch `MambaMixer._static_decode -> ssm_decode`
(megatron/core/ssm/mamba_mixer.py:574-591, :1108-1304), which feeds `causal_conv1d_update` and
`selective_state_update` (the decode kernel, see test_selective_state_update.py).

Contract (written from the Mamba-2 layer math and the static-batching interface, before reading the
dispatch code):

* A decode step of a static batch of b requests: `MambaMixer.forward(hidden (1, b, d_model),
  StaticInferenceContext with seqlen_offset > 0)`. The mixer keeps per-request caches conv_state
  (b, conv_dim, d_conv) and ssm_state (b, nheads, headdim, dstate) in the context; request i (batch column
  i) owns slot i of both. With zxBCdt = in_proj(hidden) split into z (d_inner), xBC (d_inner + 2 G N) and
  dt (nheads), for each request i independently:

      conv_state[i] <- [conv_state[i][:, 1:], xBC_i]                    (shift left, append; exact copies)
      xBC'_i  = silu(sum_w conv_state[i] * conv_weight + conv_bias)      split into x (H, P), B, C (G, N)
      dt'     = softplus(dt_i + dt_bias);  A = -exp(A_log)
      h_i     = exp(dt' A) * ssm_state[i] + dt' x (outer) B_{g(h)};      ssm_state[i] <- h_i
      y_i     = C_{g(h)} . h_i + D * x            (D per head, or per channel with D_has_hdim)
      y_i     = RMSNorm_groups(y_i * silu(z_i)) * norm_weight   (rmsnorm=True, ngroups norm groups)
              = y_i * silu(z_i)                                  (rmsnorm=False, gated inside the scan)
      out_i   = out_proj(y_i)

  Requests never interact: request i's outputs and slots depend only on its own inputs and slots.
* Interface roundings (the module's dtype contract): the conv output is rounded to the activation dtype,
  the scan output to the activation dtype before the norm, the SSM state to the cache dtype
  (in_proj.weight's dtype) once per step.
* Tolerance class: floating point, bounded by a trusted implementation: the same mixer math in torch fp32
  on the same device with the interface roundings above (`mixer_decode_reference` with `round_to`). The
  reference is that math in fp64 without roundings. The new conv state is a copy: exact. Checked outputs:
  the out_proj input y (captured with a forward pre-hook), the new conv state and the new SSM state;
  in_proj and out_proj are TE GEMMs outside this dispatch and are not re-derived.

Dispatch paths (from the code, read after the contract): `forward` -> `_static_decode` when the static
context's seqlen_offset > 0 (:550-553), else `_mamba_chunk` -> `_static_prefill` (pip causal-conv1d and
mamba_chunk_scan_combined) which seeds the caches; `ssm_decode`'s Triton conv and decode-kernel branch
(the `batch_invariant_mode` branch is dynamic-only, asserted at :546); rmsnorm on/off (z gating in the norm
or in the kernel); D_has_hdim on/off. Inputs: b in {1, 2, 3, 8}; heads x groups (8, 8), (8, 1) and one head
(the TP = H case); prompt lengths 3, 4, 5 (around d_conv = 4, the F.pad handoff at :634-636) and 127, 128,
129 (around the prefill scan's chunk_size = 128); 1 and 3 consecutive decode steps; a NaN in one request.
bf16 model (production): with an fp32 model the step also carries finding O7-N1 (the decode kernel's fp32
softplus, see test_selective_state_update.py), which the kernel-level oracle covers. The static caches
take in_proj.weight's dtype (`_get_states_from_cache`), so a bf16 model keeps a bf16 SSM state and O7-N1
cannot show here.

Production sizes (`PRODUCTION_LAYERS`, the Mamba layers of the Nemotron-H configs under
examples/post_training/modelopt/conf/nvidia/): Nemotron-H 8B (hidden 4096, 128 heads of 64, 8 groups,
dstate 128), Nemotron Nano 9B v2 (hidden 4480, 128 heads of 80: a head dim that is not a multiple of the
Blackwell tile's BLOCK_SIZE_M = 32) and Nemotron-H 56B (hidden 8192, 256 heads of 64, dstate 256: the
Hopper (4, 8) tile for dstate > 128), at TP 1, two chained decode steps each. One request (passes on
main) and a full static batch (64, or 32 for the 56B layer; needs #8568). The decode kernel's tile
depends on the architecture (BLOCK_SIZE_M, num_warps = (4, 4)/(4, 8) on Hopper, (32, 8) on Blackwell
for dstate > 64, mamba_ssm.py:380-395), so the production and memory cases also run on GB200.

Memory: `test_static_decode_caches_in_guard_bands` hands the mixer static caches that sit in guard bands
(the context's cache slots, reused because the batch size matches) and checks them after two steps; it is
the file's `sanitizer` case (b = 1, so it passes on main).

Main vs open PRs:
* #8568 (d6, fix/d6-static-mamba-batch) makes `_static_decode` pass a batch-first (b, 1, proj) input.
  On main the sequence-first (1, b, proj) projection reaches `ssm_decode`, which reads it as one request
  with b tokens: every case with b > 1 fails on main (requests 1..b-1 are decoded on slot 0's state and
  their own slots never advance) and passes with #8568. The b = 1 cases pass on main.
* #8569 (s10, fix/s10-mamba-d-has-hdim-decode) gives the decode kernel D as (nheads, headdim). On main a
  D_has_hdim mixer passes `repeat(D, "h -> h p")` of the d_inner-long D, so the kernel reads D[head] for
  every channel: the `D_has_hdim` cases (b = 1, so independent of #8568) fail on main and pass with #8569.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

HIDDEN = 256
D_CONV = 4  # MambaMixer default (mamba_mixer.py:180)
PREFILL_CHUNK = 128  # MambaMixer chunk_size default (mamba_mixer.py:195)
HEAD_CONFIGS = {"h8g8": (None, 8), "h8g1": (None, 1), "h1g1": (1, 1)}  # (mamba_num_heads, groups)

PRODUCTION_LAYERS = {
    # name: (hidden, mamba_num_heads, mamba_head_dim, mamba_num_groups, mamba_state_dim, full batch),
    # from examples/post_training/modelopt/conf/nvidia/{Nemotron-H-8B-Base-8K, NVIDIA-Nemotron-Nano-9B-v2,
    # Nemotron-H-56B-Base-8K}.sh (the 56B file sets only the state dim; the rest are the defaults,
    # transformer_config.py mamba_* and MambaMixer expand=2). The full batch is a large static batch
    # (the 56B layer's SSM state is 4 MiB elements per request, so it gets half).
    "nemotron_h_8b": (4096, 128, 64, 8, 128, 64),
    "nemotron_nano_9b_v2": (4480, 128, 80, 8, 128, 64),
    "nemotron_h_56b": (8192, 256, 64, 8, 256, 32),
}


# ---------------------------------------------------------------------------------------------
# fp64 reference of one mixer decode step
# ---------------------------------------------------------------------------------------------


def mixer_decode_reference(
    zxBCdt,
    conv0,
    ssm0,
    conv_w,
    conv_b,
    dt_bias,
    A_log,
    D,
    norm_w,
    *,
    heads,
    groups,
    headdim,
    dstate,
    rmsnorm,
    eps,
    round_to=None,
    state_dtype=None,
):
    """One Mamba-2 decode step per request, written from the layer math.

    zxBCdt (b, proj); conv0 (b, conv_dim, d_conv); ssm0 (b, H, P, N); conv_w (conv_dim, d_conv); D (H,) or
    (H * P,). `round_to`: the activation dtype at the module's interfaces (None: no rounding, the fp64
    reference); `state_dtype`: the SSM cache dtype the state is stored in. Returns (y (b, d_inner), new
    conv state, new SSM state).
    """

    def rnd(t, dtype):
        return t if dtype is None else t.to(dtype).to(t.dtype)

    b, d_inner = zxBCdt.shape[0], heads * headdim
    z, xBC, dt = zxBCdt.split([d_inner, d_inner + 2 * groups * dstate, heads], -1)
    conv = torch.cat([conv0[..., 1:], xBC[..., None]], -1)
    xBC = rnd(F.silu((conv * conv_w).sum(-1) + conv_b), round_to)
    x, B, C = xBC.split([d_inner, groups * dstate, groups * dstate], -1)
    x = x.view(b, heads, headdim)
    group_of_head = torch.arange(heads, device=x.device) // (heads // groups)
    B = B.view(b, groups, dstate)[:, group_of_head]
    C = C.view(b, groups, dstate)[:, group_of_head]
    delta = F.softplus(dt + dt_bias)
    decay = torch.exp(delta * -torch.exp(A_log))
    h = decay[..., None, None] * ssm0 + (delta[..., None] * x)[..., None] * B[:, :, None, :]
    y = (h * C[:, :, None, :]).sum(-1) + D.view(heads, -1) * x
    y = y.reshape(b, d_inner)
    gate = z * torch.sigmoid(z)
    if rmsnorm:
        y = rnd(y, round_to) * gate
        y = y.view(b, groups, -1)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        y = y.reshape(b, d_inner) * norm_w
    else:
        y = y * gate
    return y, conv, rnd(h, state_dtype)


# ---------------------------------------------------------------------------------------------
# Mixer under test
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def build_mixer(head_config, *, rmsnorm=True, D_has_hdim=False, dtype=torch.bfloat16):
    """A MambaMixer (TE spec, TP = 1) with random parameters that make every term visible.

    `head_config` names a small layout (`HEAD_CONFIGS`, hidden `HIDDEN`, default head dim and dstate)
    or a production layer (`PRODUCTION_LAYERS`, initialized on the GPU: its in_proj has up to 3e8
    parameters).
    """
    if head_config in PRODUCTION_LAYERS:
        hidden, num_heads, head_dim, groups, dstate, _ = PRODUCTION_LAYERS[head_config]
        shape = dict(mamba_head_dim=head_dim, mamba_state_dim=dstate, use_cpu_initialization=False)
    else:
        hidden, (num_heads, groups) = HIDDEN, HEAD_CONFIGS[head_config]
        shape = dict(use_cpu_initialization=True)
    config = TransformerConfig(
        hidden_size=hidden,
        num_layers=1,
        num_attention_heads=1,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        mamba_num_heads=num_heads,
        mamba_num_groups=groups,
        **shape,
    )
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        hidden,
        D_has_hdim=D_has_hdim,
        rmsnorm=rmsnorm,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
    ).cuda()
    g = ou.make_generator(device="cuda")
    with torch.no_grad():
        # D = 1 everywhere (the init) would hide a per-head/per-channel mix-up; so would a unit norm gain.
        mixer.D.copy_(torch.randn(mixer.D.shape, device="cuda", generator=g))
        dt0 = torch.empty(mixer.dt_bias.shape, device="cuda").uniform_(1e-3, 1e-1, generator=g)
        mixer.dt_bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))
        mixer.A_log.copy_(torch.empty_like(mixer.A_log).uniform_(0.0, 2.77, generator=g))
        if rmsnorm:
            weight = mixer.norm.weight
            weight.copy_(1 + 0.5 * torch.randn(weight.shape, device="cuda", generator=g))
    mixer.eval()
    mixer.refresh_cache()
    return mixer


class Capture:
    """Records in_proj's output (zxBCdt) and out_proj's input (y) on every forward."""

    def __init__(self, mixer):
        self.zxBCdt, self.y = [], []
        self.handles = [
            mixer.in_proj.register_forward_hook(
                lambda mod, args, out: self.zxBCdt.append(out[0].detach().clone())
            ),
            mixer.out_proj.register_forward_pre_hook(
                lambda mod, args: self.y.append(args[0].detach().clone())
            ),
        ]

    def remove(self):
        for handle in self.handles:
            handle.remove()


def mixer_params(mixer):
    """The parameters the reference needs, in the layout the math uses."""
    return dict(
        conv_w=mixer.conv1d_weight.detach()[:, 0, :],
        conv_b=mixer.conv1d_bias.detach(),
        dt_bias=mixer.dt_bias.detach(),
        A_log=mixer.A_log.detach(),
        D=mixer.D.detach(),
        norm_w=mixer.norm.weight.detach() if mixer.rmsnorm else None,
    )


def check_step(mixer, zxBCdt, y, conv_before, ssm_before, conv_after, ssm_after, what):
    """One decode step's outputs against the fp64 reference, bounded by the fp32 trusted implementation."""
    params = mixer_params(mixer)
    kw = dict(
        heads=mixer.nheads_local_tp,
        groups=mixer.ngroups_local_tp,
        headdim=mixer.headdim,
        dstate=mixer.d_state,
        rmsnorm=mixer.rmsnorm,
        eps=mixer.norm.eps if mixer.rmsnorm else 0.0,
    )
    args = (zxBCdt, conv_before, ssm_before, *params.values())
    ref_y, ref_conv, ref_ssm = ou.fp64_reference(mixer_decode_reference, *args, **kw)
    f32 = [None if t is None else t.float() for t in args]
    t_y, _, t_ssm = mixer_decode_reference(
        *f32, **kw, round_to=zxBCdt.dtype, state_dtype=ssm_after.dtype
    )
    # The conv state is the previous window shifted left with the new xBC appended: a copy.
    xBC = zxBCdt[:, mixer.d_inner_local_tp : -mixer.nheads_local_tp].to(conv_after.dtype)
    expected_conv = torch.cat([conv_before[..., 1:], xBC[..., None]], -1)
    ou.assert_exact(conv_after, expected_conv, what=f"{what}: conv state")
    assert ref_conv.shape == conv_after.shape
    ou.assert_close_to_reference(
        ssm_after, ref_ssm, t_ssm.to(ssm_after.dtype), what=f"{what}: SSM state"
    )
    ou.assert_close_to_reference(
        y, ref_y, t_y.to(y.dtype), dim=-1, what=f"{what}: y (out_proj input)"
    )


def static_context(batch, offset):
    context = StaticInferenceContext(max_batch_size=batch, max_sequence_length=512)
    context.sequence_len_offset = offset
    context.seqlen_offset = offset  # set by HybridStack.forward in a model (hybrid_block.py:622)
    return context


def decode_steps(mixer, context, hiddens):
    """Run decode steps through `MambaMixer.forward`; returns per-step (zxBCdt, y, states before/after)."""
    batch = hiddens[0].shape[1]
    conv_state, ssm_state = mixer._get_states_from_cache(context, batch)
    capture = Capture(mixer)
    steps = []
    try:
        with torch.no_grad(), InferenceMode.active():
            for hidden in hiddens:
                before = conv_state.clone(), ssm_state.clone()
                out, _ = mixer(hidden, inference_context=context)
                assert out.shape == hidden.shape
                steps.append(
                    dict(
                        zxBCdt=capture.zxBCdt[-1][0],
                        y=capture.y[-1][0],
                        conv_before=before[0],
                        ssm_before=before[1],
                        conv_after=conv_state.clone(),
                        ssm_after=ssm_state.clone(),
                    )
                )
                context.sequence_len_offset += 1
                context.seqlen_offset = context.sequence_len_offset
    finally:
        capture.remove()
    return steps


def random_states(mixer, context, batch, g):
    """Allocate the static caches and give each request its own random states, as a prefill would."""
    conv_state, ssm_state = mixer._get_states_from_cache(context, batch)
    conv_state.copy_(torch.randn(conv_state.shape, device="cuda", generator=g))
    ssm_state.copy_(torch.randn(ssm_state.shape, device="cuda", generator=g))
    return conv_state, ssm_state


def guarded_states(mixer, context, batch, g):
    """Like `random_states`, but the static caches are views into guard-banded buffers.

    The context's cache slot is filled before the first forward with the shapes and dtypes
    `_get_states_from_cache` allocates (`mamba_state_shapes_per_request`), and the mixer's cached
    batch size set to match, so the mixer reuses them in place (seqlen_offset > 0: not zeroed).
    """
    conv_shape, ssm_shape = mixer.mamba_state_shapes_per_request()
    conv = ou.guarded_empty((batch, *conv_shape), mixer.conv1d_weight.dtype)
    ssm = ou.guarded_empty((batch, *ssm_shape), mixer.in_proj.weight.dtype)
    conv.tensor.copy_(torch.randn(conv.shape, device="cuda", generator=g))
    ssm.tensor.copy_(torch.randn(ssm.shape, device="cuda", generator=g))
    context.key_value_memory_dict[mixer.layer_number] = (conv.tensor, ssm.tensor)
    mixer.cached_batch_size = batch
    conv_state, ssm_state = mixer._get_states_from_cache(context, batch)
    assert conv_state is conv.tensor and ssm_state is ssm.tensor, "the mixer reallocated its caches"
    return conv, ssm


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def _dispatch_cases():
    params = []
    # Batch, head layout and gate placement, pairwise (D per head: independent of #8569).
    for case in ou.pairwise_cases(
        {"batch": (1, 2, 3, 8), "head_config": tuple(HEAD_CONFIGS), "rmsnorm": (True, False)}
    ):
        case["D_has_hdim"] = False
        params.append(case)
    # D per channel at b = 1 (independent of #8568).
    for head_config in HEAD_CONFIGS:
        for rmsnorm in (True, False):
            params.append(dict(batch=1, head_config=head_config, rmsnorm=rmsnorm, D_has_hdim=True))
    return [
        pytest.param(
            c,
            id="b{batch}-{head_config}-{norm}-{d}".format(
                norm="rmsnorm" if c["rmsnorm"] else "zgate",
                d="D_hdim" if c["D_has_hdim"] else "D_head",
                **c,
            ),
        )
        for c in params
    ]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _dispatch_cases())
def test_static_decode_step_matches_fp64(model_parallel, case):
    """One static decode step from random per-request states: each request on its own slots."""
    mixer = build_mixer(case["head_config"], rmsnorm=case["rmsnorm"], D_has_hdim=case["D_has_hdim"])
    batch = case["batch"]
    g = ou.make_generator(device="cuda")
    context = static_context(batch, offset=8)
    random_states(mixer, context, batch, g)
    hidden = torch.randn(1, batch, HIDDEN, device="cuda", generator=g).bfloat16()
    (step,) = decode_steps(mixer, context, [hidden])
    check_step(mixer, **step, what="step")


@ou.requires_sm(ou.HOPPER)
def test_three_decode_steps_chain_per_request(model_parallel):
    """Three consecutive steps, b = 3: each step against the reference from the states the previous step
    left, and the whole chain against the fp64 chain from the starting states."""
    mixer = build_mixer("h8g1")
    batch, g = 3, ou.make_generator(device="cuda")
    context = static_context(batch, offset=8)
    random_states(mixer, context, batch, g)
    hiddens = [
        torch.randn(1, batch, HIDDEN, device="cuda", generator=g).bfloat16() for _ in range(3)
    ]
    steps = decode_steps(mixer, context, hiddens)
    for k, step in enumerate(steps):
        check_step(mixer, **step, what=f"step {k}")
    # The chain from the starting states: fp64 throughout vs fp32 with the module's roundings.
    params = mixer_params(mixer)
    kw = dict(
        heads=mixer.nheads_local_tp,
        groups=mixer.ngroups_local_tp,
        headdim=mixer.headdim,
        dstate=mixer.d_state,
        rmsnorm=True,
        eps=mixer.norm.eps,
    )
    conv_r = conv_t = steps[0]["conv_before"]
    ssm_r = steps[0]["ssm_before"].double()
    ssm_t = steps[0]["ssm_before"].float()
    for k, step in enumerate(steps):
        ref = ou.fp64_reference(
            mixer_decode_reference, step["zxBCdt"], conv_r, ssm_r, *params.values(), **kw
        )
        f32 = [None if t is None else t.float() for t in (step["zxBCdt"], conv_t, ssm_t)]
        f32 += [None if t is None else t.float() for t in params.values()]
        trusted = mixer_decode_reference(
            *f32, **kw, round_to=torch.bfloat16, state_dtype=torch.bfloat16
        )
        ou.assert_close_to_reference(
            step["y"], ref[0], trusted[0].bfloat16(), dim=-1, what=f"chained y, step {k}"
        )
        conv_r, ssm_r = ref[1], ref[2]
        conv_t, ssm_t = trusted[1].bfloat16(), trusted[2]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(n, id=f"prompt{n}[{why}]")
        for n, why in (
            (3, "d_conv-1"),
            (4, "d_conv"),
            (5, "d_conv+1"),
            (127, "chunk-1"),
            (128, "chunk"),
            (129, "chunk+1"),
        )
    ],
)
def test_prefill_then_decode_hands_over_each_request(model_parallel, prompt):
    """A static prefill (pip kernels) of b = 3 prompts, then two decode steps. The conv cache after the
    prefill holds each request's last d_conv pre-conv inputs, zero-padded on the left (exact); each decode
    step continues each request from its own slots."""
    mixer = build_mixer("h8g8")
    batch, g = 3, ou.make_generator(device="cuda")
    context = static_context(batch, offset=0)
    prompt_hidden = torch.randn(prompt, batch, HIDDEN, device="cuda", generator=g).bfloat16()
    capture = Capture(mixer)
    try:
        with torch.no_grad(), InferenceMode.active():
            mixer(prompt_hidden, inference_context=context)
    finally:
        capture.remove()
    conv_state, _ = context.key_value_memory_dict[mixer.layer_number]
    xBC = capture.zxBCdt[-1][:, :, mixer.d_inner_local_tp : -mixer.nheads_local_tp]
    window = xBC.permute(1, 2, 0)[..., -D_CONV:]  # (b, conv_dim, min(prompt, d_conv))
    expected = torch.zeros_like(conv_state)
    expected[..., D_CONV - window.shape[-1] :] = window.to(conv_state.dtype)
    ou.assert_exact(conv_state, expected, what="conv cache after the prefill")

    context.sequence_len_offset = context.seqlen_offset = prompt
    hiddens = [
        torch.randn(1, batch, HIDDEN, device="cuda", generator=g).bfloat16() for _ in range(2)
    ]
    for k, step in enumerate(decode_steps(mixer, context, hiddens)):
        check_step(mixer, **step, what=f"decode step {k} after a {prompt}-token prefill")


@ou.requires_sm(ou.HOPPER)
def test_nan_in_one_request_stays_in_that_request(model_parallel):
    """A NaN hidden state for request 1 must not reach requests 0 and 2 (their outputs and slots)."""
    mixer = build_mixer("h8g1")
    batch, g = 3, ou.make_generator(device="cuda")
    context = static_context(batch, offset=8)
    random_states(mixer, context, batch, g)
    hidden = torch.randn(1, batch, HIDDEN, device="cuda", generator=g).bfloat16()
    hidden[0, 1] = float("nan")
    (step,) = decode_steps(mixer, context, [hidden])
    assert torch.isnan(step["y"][1]).all()
    for name in ("y", "conv_after", "ssm_after"):
        assert torch.isfinite(step[name][[0, 2]]).all(), f"NaN leaked into another request's {name}"
    keep = [0, 2]
    check_step(
        mixer, **{k: v[keep] for k, v in step.items()}, what="requests 0 and 2 beside a NaN request"
    )


def _production_decode(layer, batch):
    """Two chained static decode steps of a production Mamba layer from random per-request states,
    each step against the fp64 reference from the states the previous step left."""
    mixer = build_mixer(layer)
    g = ou.make_generator(device="cuda")
    context = static_context(batch, offset=8)
    random_states(mixer, context, batch, g)
    hiddens = [
        torch.randn(1, batch, mixer.d_model, device="cuda", generator=g).bfloat16()
        for _ in range(2)
    ]
    for k, step in enumerate(decode_steps(mixer, context, hiddens)):
        check_step(mixer, **step, what=f"{layer} b={batch} step {k}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("layer", list(PRODUCTION_LAYERS))
def test_production_layer_one_request(model_parallel, layer):
    """Production Mamba layers (`PRODUCTION_LAYERS`), one request: passes on main."""
    _production_decode(layer, 1)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("layer", list(PRODUCTION_LAYERS))
def test_production_layer_full_batch(model_parallel, layer):
    """Production Mamba layers at a full static batch (64 requests; 32 for the 56B layer). Fails on
    main and passes with #8568 (on main the b requests are decoded as one request of b tokens)."""
    _production_decode(layer, PRODUCTION_LAYERS[layer][-1])


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.sanitizer
def test_static_decode_caches_in_guard_bands(model_parallel):
    """Small case for compute-sanitizer (b = 1, per-head D: passes on main): two decode steps of the
    h8g1 layout (8 heads share one B/C group; dstate 128, so the Blackwell (32, 8) tile on GB200) on
    static caches that sit in guard bands. The conv update and the decode kernel write the caches in
    place: nothing outside them may change, and each step must match the reference."""
    mixer = build_mixer("h8g1")
    g = ou.make_generator(device="cuda")
    context = static_context(1, offset=8)
    conv, ssm = guarded_states(mixer, context, 1, g)
    hiddens = [torch.randn(1, 1, HIDDEN, device="cuda", generator=g).bfloat16() for _ in range(2)]
    steps = decode_steps(mixer, context, hiddens)
    ou.check_guards(conv_state=conv, ssm_state=ssm)
    for k, step in enumerate(steps):
        check_step(mixer, **step, what=f"guarded caches, step {k}")


def test_case_list_isolates_each_pr():
    """Every D_has_hdim case runs at b = 1 and every b > 1 case with a per-head D (CPU check)."""
    cases = [p.values[0] for p in _dispatch_cases()]
    assert all(c["batch"] == 1 for c in cases if c["D_has_hdim"])
    assert {c["batch"] for c in cases} == {1, 2, 3, 8}
    assert {c["head_config"] for c in cases} == set(HEAD_CONFIGS)
