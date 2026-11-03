# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``fused_moe.cutlass_fused_moe`` at Megatron's call sites in
``InferenceGroupedMLP._flashinfer_forward`` (``megatron/core/transformer/moe/experts.py``): BF16
routed MoE with squared ReLU or ``ClampedRelu2``, and MXFP8 x MXFP8 with ``ClampedRelu2`` through
``flashinfer_cutlass_mxfp8_moe`` (``megatron/core/inference/moe/flashinfer_mxfp8.py``).

Contract (from the call sites, MCore's MoE math and FlashInfer's documented API):

* Per token ``t``: ``out[t] = sum_k p[t, k] * W2[e] @ act(W1[e] @ x[t])`` over routes to the
  local experts ``[ep_rank * E_local, (ep_rank + 1) * E_local)``; -1 and non-local routes add
  nothing. ``act`` is ``relu(z)**2`` or, with ``activation_func_tanh_clamp_scale = L``,
  ``(L * tanh(relu(z) / L))**2`` (FlashInfer ``ClampedRelu2``, SM10x only). FP32 routing
  weights; ``W1 [E_local, I, H]``, ``W2 [E_local, H, I]`` BF16 (or canonical MXFP8, whose
  scales Megatron views as CUTLASS's packed int32 without moving bytes).
* BF16 precision (FlashInfer ``cutlass_fused_moe_kernels.cuh``): GEMMs accumulate in FP32.
  ``Relu2`` is not fused: GEMM1 writes its output in BF16 and a separate kernel applies the
  activation to it, in FP32, and stores BF16; so the FC1 output is rounded to BF16 before the
  activation, as in MCore's own BF16 path (TE GroupedLinear and the torch backend output BF16
  before the activation). ``ClampedRelu2`` is fused into the GEMM1 epilogue (FlashInfer requires
  a fusion tactic for it) and runs on the FP32 accumulator. Either way the FC2 input is BF16.
  The routes are summed into the BF16 output (the default fused finalize uses BF16 atomics;
  with ``use_fused_finalize=False`` a separate finalize kernel sums them). Megatron takes
  element 0 of FlashInfer's returned list: the ``[T, H]`` BF16 output.
* MXFP8: the input is quantized to MXFP8 with swizzled scales, the FC2 input is requantized to
  MXFP8 inside the kernel; output BF16 ``[T, H]``.
* MXFP8 ``ClampedRelu2`` (``flashinfer_cutlass_mxfp8_moe``; MCore's ``clamped_squared_relu``,
  ``fusions/fused_weighted_squared_relu.py``): FC1 accumulates the exact products of the decoded
  MXFP8 operands in FP32; the FC2 input is ``mxfp8(act(z))``, the activation of the FC1 output
  requantized in blocks of 32 along the intermediate dimension with the RCEIL scale (the
  smallest ``2**e`` with ``amax <= 448 * 2**e``: NVIDIA's MXFP8 recipe, as Megatron's own
  quantizer, TE and FlashInfer's ``mxfp8_quantize``) and round-to-nearest-even E4M3 codes.
  ``act`` is evaluated in FP32; an admissible kernel may use an approximate ``tanh`` (FlashInfer:
  ``tanh.approx.f32``, relative error about 2**-11) and may round the activation to BF16 first,
  as MCore's BF16 path does (``clamped_squared_relu`` returns the input dtype). The two together
  move an activation by less than 2**-8.4 of itself, so they only matter within that distance of
  an E4M3 rounding boundary or of a scale boundary. All-zero blocks (every FC1 output of the
  block <= 0) give zero FC2 inputs; a zero token gives a zero output row.
* Inputs are read only. Open PR #8574 (s3) passes ``use_fused_finalize=False`` under
  deterministic or batch-invariant configs; both finalize paths are covered here and the
  results are the same contract.

Oracle: ``spec.moe_reference`` in fp64. BF16: random ``x`` and ``W2``; ClampedRelu2 random
``W1``, Relu2 a rank-1 ``W1`` (``Bf16Problem``: the BF16-stored FC1 output then has one rounding
decision per token, kept away from BF16 ties); the reference rounds to BF16 where the contract
stores BF16 (the FC1 output for Relu2, the FC2 input) and the trusted implementation is torch in
fp32 (TF32 off) with the same BF16 storage and BF16 route accumulation;
tolerance class floating point, per row, factor 2 (one BF16 rounding of the FC2 input more or
less moves the output by less than half a BF16 ulp, so the bound does not depend on which side
of a rounding boundary each implementation lands). MXFP8 ``ClampedRelu2``, two regimes:

* Saturated (``test_mxfp8_clamped_relu2_matches_fp64_moe``): a rank-1 ``W1`` and tokens whose FC1
  outputs are all beyond 40 L in magnitude, so the activation is exactly ``L**2`` or 0 whatever
  the kernel's tanh, and the FC2 input is MXFP8-exact.
* Unsaturated (``ClampedMxfp8Problem``): dense MX-exact tokens, a rank-1 ``W1`` and a regime of
  ``|z| / L`` from 2**-13 up to 40 (around 0, the knee at L, the point near 2.4 L where the
  activation reaches ``L**2`` at FP8 precision, and FP32 saturation near 9 L). The FC2 input is
  not exact there, so the reference requantizes the fp64 activation with the spec MXFP8 quantizer;
  the tokens are redrawn until moving every activation by up to ``ACT_MARGIN = 2**-7`` (twice the
  admissible evaluation error above, plus the FP32 noise of FC1) changes no FC2-input code or
  scale, so every admissible kernel has exactly the reference's FC2 input. The trusted
  implementation (fp32 torch activation, the same spec requantization, fp32 FC2, BF16 route
  sum) is checked to produce that same FC2 input; then the bound is per row, factor 2, as above.
  The sweep also keeps each block amax where the RCEIL and the OCP floor scales agree;
  ``test_mxfp8_fc2_input_scale_is_rceil_near_power_of_two_activations`` pins RCEIL where they
  differ (a power-of-two ``L**2`` approached from below).

Variants: activation {Relu2, ClampedRelu2 with L = 0.5 (mostly saturated) and 8 (mostly
linear)}, finalize {fused, unfused}, every GEMM2 tactic FlashInfer reports valid (Relu2), tokens
1, 64/65, 128/129 and 1024, H and I 128, 256/384 and production (2688, 1856), E_local 1 and 8,
top-k 1, 2, 6, EP (size, rank) (1, 0), (2, 1), (4, 2); an empty expert and all -1 rows. MXFP8
ClampedRelu2 unsaturated: L 0.5, 3.5, 6, 8, 16 (1/L inexact for 3.5 and 6; ``L**2`` off the E4M3
grid for 3.5) x regime, zero tokens and all-zero FC2-input blocks, and every GEMM1 tactic (the
fused activation + MXFP8 epilogue: FlashInfer generates CTA 128 x {64, 128, 256}, cluster 1x1 and
2x1, static and dynamic) and every GEMM2 tactic FlashInfer reports valid, both finalizes.
Hardware: BF16 Relu2 on H100 (FlashInfer's SM90 CUTLASS module) and Blackwell; ClampedRelu2
and MXFP8 on Blackwell only. On H100 the CUTLASS MoE module must already be built
(flashinfer-jit-cache, or a prebuild: its JIT build takes about 40 minutes).
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

# The BF16 Relu2 activation as the contract evaluates it (FC1 output stored in BF16).
BF16_RELU2 = spec.relu2_of_bf16


def _cutlass_module_ready():
    """Fail or skip on Hopper when FlashInfer's CUTLASS fused-MoE module is not built (its
    JIT build takes ~40 min there); on Blackwell it builds on first use (~8 min)."""
    spec.require_flashinfer()
    from flashinfer.jit import fused_moe as fused_moe_jit

    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    gen = getattr(fused_moe_jit, f"gen_cutlass_fused_moe_sm{sm}_module", None)
    if major == 9 and (gen is None or not gen().is_compiled):
        reason = (
            "FlashInfer's SM90 CUTLASS fused-MoE module is not built: install flashinfer-jit-cache"
            " or run flashinfer.fused_moe.core.get_cutlass_fused_moe_module('90') once"
        )
        if ou.session_target_sm() is not None:
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)


def _experts(w1, w2, *, ep_size, ep_rank, activation, clamp=None, mxfp8=False):
    """The attributes ``InferenceGroupedMLP._flashinfer_forward`` reads."""
    from flashinfer.fused_moe import ActivationType

    act = getattr(ActivationType, activation)
    return SimpleNamespace(
        _uses_mxfp8_weights=mxfp8,
        _nvfp4_weights=None,
        _fc1_weight=w1,
        _fc2_weight=w2,
        _flashinfer_activation_type=act,
        _activation_clamp_scale=clamp,
        _flashinfer_clamp_limit=(
            None if clamp is None else torch.full((1,), clamp, dtype=torch.float32, device="cuda")
        ),
        _nvls_dispatcher=False,
        _flashinfer_mxfp8_token_capacity=None,
        num_local_experts=w1.shape[0] if not mxfp8 else w1.data.shape[0],
        ep_group=SimpleNamespace(size=lambda: ep_size, rank=lambda: ep_rank),
        config=SimpleNamespace(deterministic_mode=False, batch_invariant_mode=False),
    )


def _forward(experts, x, ids, probs):
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    before = [t.clone() for t in (x, ids, probs)]
    out, bias = InferenceGroupedMLP._flashinfer_forward(experts, x, ids, probs)
    assert bias is None
    for name, t, b in zip(("hidden", "routing", "probs"), (x, ids, probs), before):
        ou.assert_exact(t, b, what=f"{name} is read-only")
    return out


def _unfused_finalize(monkeypatch):
    """Run the call sites' ``cutlass_fused_moe`` (the BF16 one in experts.py, the MXFP8 one in
    flashinfer_mxfp8.py) with the non-fused finalize, whatever the call site passes."""
    from flashinfer import fused_moe

    from megatron.core.inference.moe import flashinfer_mxfp8

    original = fused_moe.cutlass_fused_moe
    assert "use_fused_finalize" in inspect.signature(original).parameters

    def unfused(*args, **kwargs):
        return original(*args, **{**kwargs, "use_fused_finalize": False})

    monkeypatch.setattr(fused_moe, "cutlass_fused_moe", unfused)
    monkeypatch.setattr(flashinfer_mxfp8, "cutlass_fused_moe", unfused, raising=False)


class Bf16Problem:
    """BF16 routed-MoE inputs on CUDA: random, or (``rank1``) with a rank-1 FC1.

    Unfused Relu2 stores the FC1 output in BF16, so an FC1 output within FP32 noise of a BF16
    rounding boundary may round either way, and one such element with a large activation moves
    its whole output row by up to an ulp (seen on H100). ``rank1`` makes ``W1[e] = u v^T`` with
    ``u = +-2**-n``: every FC1 output of a token is ``u_i * c`` with one ``c = v . x`` (up to a
    power of two per expert), so one rounding decision per token remains, and tokens whose ``c``
    is that close to a BF16 tie are redrawn. ``x`` and ``W2`` stay random and dense.
    """

    def __init__(
        self, *, tokens, hidden, inter, local, ep_size, ep_rank, top_k, pattern, seed=0, rank1=False
    ):
        g = ou.make_generator(ou.DEFAULT_SEED + seed)
        self.local, self.offset = local, ep_rank * local
        experts = local * ep_size
        if rank1:
            u, v = spec.rank1_factors(
                local, inter, hidden, row_block=32, generator=g, v_scale_exponents=(-1, 1)
            )
            w1 = (u[:, :, None] * v[:, None, :] * 2.0**-3).bfloat16()
        else:
            w1 = (torch.randn(local, inter, hidden, generator=g) / hidden**0.5).bfloat16()
        self.w1 = w1.cuda()
        self.w2 = (torch.randn(local, hidden, inter, generator=g) / inter**0.5).bfloat16().cuda()
        x = torch.randn(tokens, hidden, generator=g).bfloat16()
        if rank1:
            v0 = (v[0] / v[0, 0]).bfloat16()
            for _ in range(100):  # redraw tokens whose c FP32 noise could round the other way
                terms = x.double() * v0.double()
                near = spec.near_bf16_tie(terms.sum(-1), terms)
                if not bool(near.any()):
                    break
                x[near] = torch.randn(int(near.sum()), hidden, generator=g).bfloat16()
            else:
                raise AssertionError("could not draw tokens away from BF16 rounding boundaries")
        self.x = x.cuda()
        empty_rows = list(range(0, tokens, 3)) if pattern == "empty-rows" else ()
        empty_expert = self.offset + local // 2 if pattern == "empty-expert" else None
        top_k = min(top_k, experts - (empty_expert is not None))
        self.ids, self.probs = spec.make_routing(
            tokens,
            top_k,
            experts,
            generator=g,
            device_="cuda",
            empty_expert=empty_expert,
            empty_rows=empty_rows,
        )

    def check(self, out, activation, what):
        args = (self.x, self.w1, self.w2, self.ids, self.probs)
        ref = ou.fp64_reference(
            spec.moe_reference,
            *args,
            activation=activation,
            local_offset=self.offset,
            act_round=spec.round_to(torch.bfloat16),
        )
        with spec.exact_fp32_matmul():
            trusted = spec.moe_reference(
                self.x.float(),
                self.w1.float(),
                self.w2.float(),
                self.ids,
                self.probs,
                activation=activation,
                local_offset=self.offset,
                act_round=spec.round_to(torch.bfloat16),
                out_dtype=torch.bfloat16,
            )
        assert out.dtype == torch.bfloat16 and out.shape == ref.shape, (out.dtype, out.shape)
        return ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=what)


SHAPES = [(128, 128), (256, 384), (2688, 1856)]
EP = [(1, 0), (2, 1), (4, 2)]
FACTORS = {
    "tokens": [1, 64, 65, 128, 129, 1024],
    "shape": SHAPES,
    "local": [1, 8],
    "top_k": [1, 2, 6],
    "ep": EP,
    "finalize": ["fused", "unfused"],
    "pattern": ["random", "empty-expert", "empty-rows"],
}


def _case_id(c):
    return (
        f"T{c['tokens']}-H{c['shape'][0]}I{c['shape'][1]}-l{c['local']}-k{c['top_k']}"
        f"-ep{c['ep'][0]}r{c['ep'][1]}-{c['finalize']}-{c['pattern']}"
    )


def _run_bf16(case, monkeypatch, activation, clamp=None):
    _cutlass_module_ready()
    if case["finalize"] == "unfused":
        _unfused_finalize(monkeypatch)
    ep_size, ep_rank = case["ep"]
    problem = Bf16Problem(
        tokens=case["tokens"],
        hidden=case["shape"][0],
        inter=case["shape"][1],
        local=case["local"],
        ep_size=ep_size,
        ep_rank=ep_rank,
        top_k=case["top_k"],
        pattern=case["pattern"],
        rank1=clamp is None,  # unfused Relu2 stores the FC1 output in BF16
    )
    experts = _experts(
        problem.w1, problem.w2, ep_size=ep_size, ep_rank=ep_rank, activation=activation, clamp=clamp
    )
    out = _forward(experts, problem.x, problem.ids, problem.probs)
    act = BF16_RELU2 if clamp is None else spec.clamped_relu2(clamp)
    problem.check(out, act, what=f"{activation}{'' if clamp is None else f'(L={clamp})'}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=_case_id(c)) for c in ou.pairwise_cases(FACTORS)]
)
def test_bf16_relu2_matches_fp64_moe(case, monkeypatch):
    _run_bf16(case, monkeypatch, "Relu2")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"L{c['clamp']}-{_case_id(c)}")
        for c in ou.pairwise_cases(dict(FACTORS, clamp=[0.5, 8.0]))
    ],
)
def test_bf16_clamped_relu2_matches_fp64_moe(case, monkeypatch):
    _run_bf16(case, monkeypatch, "ClampedRelu2", clamp=case["clamp"])


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.sanitizer
def test_bf16_relu2_small_case():
    """A small representative case (for the compute-sanitizer pass)."""
    _cutlass_module_ready()
    problem = Bf16Problem(
        tokens=9,
        hidden=128,
        inter=128,
        local=4,
        ep_size=2,
        ep_rank=1,
        top_k=2,
        pattern="empty-rows",
        rank1=True,
    )
    experts = _experts(problem.w1, problem.w2, ep_size=2, ep_rank=1, activation="Relu2")
    problem.check(_forward(experts, problem.x, problem.ids, problem.probs), BF16_RELU2, "small")


_GEMM1, _GEMM2 = "trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("finalize", ["fused", "unfused"])
def test_bf16_relu2_every_gemm2_tactic(finalize, monkeypatch):
    """Each GEMM2 tactic FlashInfer's autotuner may select (its runner's valid tactics: CUTLASS
    tile and finalize variants) gives the reference result; GEMM1 keeps its default."""
    _cutlass_module_ready()
    from flashinfer.autotuner import AutoTuner

    if finalize == "unfused":
        _unfused_finalize(monkeypatch)
    choose_one = AutoTuner.choose_one
    pin = SimpleNamespace(tactic=None, valid=None)

    def pinned(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op not in (_GEMM1, _GEMM2):
            return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)
        gemm1_count = runners[0].fused_moe_runner.get_gemm1_tactic_count()
        if custom_op == _GEMM2 and pin.valid is None:
            pin.valid = sorted(
                t - gemm1_count
                for t in runners[0].get_valid_tactics(inputs, None)
                if t >= gemm1_count
            )
        if custom_op == _GEMM1 or pin.tactic is None:
            return runners[0], -1
        return runners[0], gemm1_count + pin.tactic

    monkeypatch.setattr(AutoTuner, "choose_one", pinned)
    problem = Bf16Problem(
        tokens=129,
        hidden=256,
        inter=384,
        local=8,
        ep_size=2,
        ep_rank=1,
        top_k=2,
        pattern="random",
        rank1=True,
    )
    experts = _experts(problem.w1, problem.w2, ep_size=2, ep_rank=1, activation="Relu2")
    problem.check(_forward(experts, problem.x, problem.ids, problem.probs), BF16_RELU2, "default")
    assert pin.valid, "FlashInfer reported no valid GEMM2 tactic"
    for tactic in pin.valid:
        pin.tactic = tactic
        out = _forward(experts, problem.x, problem.ids, problem.probs)
        problem.check(out, BF16_RELU2, what=f"GEMM2 tactic {tactic} ({finalize} finalize)")


def _cutlass_mxfp8(values):
    from megatron.core.inference.moe.flashinfer_mxfp8 import prepare_cutlass_mxfp8_weights

    return prepare_cutlass_mxfp8_weights(spec.to_cuda(spec.canonical_mxfp8(values)))


def test_saturating_inputs_give_exact_activations():
    """CPU check of the MXFP8 ClampedRelu2 inputs: every FC1 output is beyond 40 L, so the
    activation is L**2 or 0 in fp64; the weights are MXFP8-exact."""
    g = ou.make_generator()
    w1, w2, x, limit = _mxfp8_clamped_inputs(4, 256, 384, 64, g)
    z = torch.einsum("eih,th->eti", w1, x)
    assert bool((z.abs() >= 40 * limit).all())
    a = spec.clamped_relu2(limit)(z)
    assert bool(((a == limit**2) | (a == 0)).all())
    for e in range(4):
        ou.assert_exact(spec.mxfp8_roundtrip(w1[e]), w1[e], what="W1")
        ou.assert_exact(spec.mxfp8_roundtrip(w2[e]), w2[e], what="W2")


def _mxfp8_clamped_inputs(local, hidden, inter, tokens, generator, limit=0.5):
    u, v = spec.rank1_factors(local, inter, hidden, row_block=32, generator=generator)
    w1 = u[:, :, None] * v[:, None, :]
    w2 = spec.mx_exact_values(local * hidden, inter, generator=generator).reshape(
        local, hidden, inter
    )
    v0 = v[0] / v[0, 0]
    # Tokens aligned with v0 (v[e] = +-2**j v0), so |v[e] . x| >= sum |x| / 4: large.
    x = torch.rand(tokens, hidden, generator=generator, dtype=torch.float64) + 0.5
    x = (x * torch.sign(v0) * 2.0**6).to(torch.bfloat16).double()
    return w1, w2, x, limit


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=f"T{c['tokens']}-H{c['shape'][0]}I{c['shape'][1]}-k{c['top_k']}"
            f"-ep{c['ep'][0]}r{c['ep'][1]}-{c['finalize']}",
        )
        for c in ou.pairwise_cases(
            {
                "tokens": [1, 65, 128, 129, 1024],
                "shape": [(128, 128), (256, 384), (2688, 2048)],
                "top_k": [1, 2, 4],
                "ep": EP,
                "finalize": ["fused", "unfused"],
            }
        )
    ],
)
def test_mxfp8_clamped_relu2_matches_fp64_moe(case, monkeypatch):
    _run_mxfp8_clamped(case, monkeypatch)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_sanitizer_mxfp8_clamped_relu2_small_case(monkeypatch):
    """Small case for compute-sanitizer: MXFP8 ClampedRelu2 at the sweep's smallest shape with
    65 tokens (64 + 1, as in the sweep), top-2 over 8 experts at EP rank 1 of 2 (routes to the
    other rank's experts are skipped), fused finalize."""
    _run_mxfp8_clamped(
        dict(tokens=65, shape=(128, 128), top_k=2, ep=(2, 1), finalize="fused"), monkeypatch
    )


def _run_mxfp8_clamped(case, monkeypatch):
    _cutlass_module_ready()
    if case["finalize"] == "unfused":
        _unfused_finalize(monkeypatch)
    g = ou.make_generator()
    local, (hidden, inter) = 4, case["shape"]
    ep_size, ep_rank = case["ep"]
    w1, w2, x, limit = _mxfp8_clamped_inputs(local, hidden, inter, case["tokens"], g)
    ids, probs = spec.make_routing(
        case["tokens"],
        min(case["top_k"], local * ep_size),
        local * ep_size,
        generator=g,
        device_="cuda",
    )
    experts = _experts(
        _cutlass_mxfp8(w1),
        _cutlass_mxfp8(w2),
        ep_size=ep_size,
        ep_rank=ep_rank,
        activation="ClampedRelu2",
        clamp=limit,
        mxfp8=True,
    )
    xb = x.to(torch.bfloat16).cuda()
    out = _forward(experts, xb, ids, probs)
    act = spec.clamped_relu2(limit)
    offset = ep_rank * local
    ref = ou.fp64_reference(
        spec.moe_reference,
        xb,
        w1.cuda(),
        w2.cuda(),
        ids,
        probs,
        activation=act,
        local_offset=offset,
    )
    with spec.exact_fp32_matmul():
        trusted = spec.moe_reference(
            xb.float(),
            w1.float().cuda(),
            w2.float().cuda(),
            ids,
            probs,
            activation=act,
            local_offset=offset,
            out_dtype=torch.bfloat16,
        )
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    ou.assert_close_to_reference(out, ref, trusted, dim=-1, what="MXFP8 ClampedRelu2")


# ---------------------------------------------------------------------------------------------
# MXFP8 ClampedRelu2 outside saturation
# ---------------------------------------------------------------------------------------------

ACT_MARGIN = 2.0**-7
"""Relative distance every unsaturated activation keeps from an FC2-input rounding or scale
boundary. An admissible evaluation moves an activation by less than 2**-8.4 of itself
(``tanh.approx.f32``: about 2**-11 relative, 2**-10 after squaring; a BF16 store before the
requantization: 2**-9), so the margin is 2.6 times that."""

UNSATURATED_REGIMES = {
    # Range of log2(|z_max| / L) per token; its other FC1 outputs are 2, 4 and 8 times smaller.
    "near-zero": (-10.0, -4.0),  # tanh(z / L) ~ z / L: the activation is ~ z**2
    "knee": (-2.0, 2.0),  # z around L
    "transition": (0.5, 3.0),  # the activation reaches L**2 at FP8 precision (z ~ 2.4 L)
    "saturation": (3.0, 5.4),  # FP32 tanh reaches 1 (z ~ 9 L), up to the saturated test's 40 L
    "mixed": (-10.0, 5.4),
}

_MAX_REDRAWS = 400


def _floor_log2(t: torch.Tensor) -> torch.Tensor:
    """Exact ``floor(log2(t))`` of positive fp64 values (as fp64)."""
    return (torch.frexp(t)[1] - 1).double()


def fc2_input_is_fixed(act: torch.Tensor, margin: torch.Tensor, rule: str) -> torch.Tensor:
    """Whether moving each activation ``act [..., I]`` (fp64, >= 0) by up to ``margin`` of itself
    (relative, broadcastable to ``[..., 1]``) changes no MXFP8 block scale and no E4M3 code of the
    FC2 input. Returns ``[...]`` bool.

    The scale of a block depends only on its amax, which moves by at most ``margin`` too; with the
    scale fixed, each code depends only on its element and rounding is monotone, so checking both
    ends of the interval suffices. ``rule="agnostic"`` also requires the RCEIL scale to equal the
    OCP floor scale ``2**(floor(log2 amax) - 8)`` over the interval (amax mantissa in [1, 1.75]);
    ``"rceil"`` only keeps amax away from RCEIL's boundary ``448 * 2**e``.
    """
    blocks = act.reshape(*act.shape[:-1], -1, spec.MX_BLOCK)
    amax = blocks.amax(-1)
    m = margin.to(act.dtype)  # [..., 1] against the block amax [..., I // 32]
    lo, hi = amax * (1 - m), amax * (1 + m)
    exponent = spec.rceil_exponent(lo)
    scale_ok = exponent == spec.rceil_exponent(hi)
    if rule == "agnostic":
        scale_ok &= (_floor_log2(lo) - 8 == exponent) & (_floor_log2(hi) - 8 == exponent)
    else:
        assert rule == "rceil", rule
    scale = spec.pow2(exponent).unsqueeze(-1)
    me = m.unsqueeze(-1)
    codes_ok = (
        spec.e4m3_encode(blocks * (1 - me) / scale) == spec.e4m3_encode(blocks * (1 + me) / scale)
    ).all(-1)
    return ((scale_ok & codes_ok) | (amax == 0)).all(-1)


class ClampedMxfp8Problem:
    """MXFP8 ClampedRelu2 inputs outside saturation whose FC2 input is fixed (CPU, fp64).

    The weights and tokens are MXFP8-exact and BF16-exact, so the kernel's FC1 differs from the
    fp64 one only by its FP32 accumulation. ``W1[e] = u[e] v[e]^T`` (``spec.rank1_factors`` with
    two levels: ``u = +-2**-(n_b + d_i)``, ``n_b`` in {0, 1} per 32-row block, ``d_i`` in {0, 1};
    ``v[e] = +-2**j_e v0``, ``j_e`` in {-1, 0}); u's first 32-row block is all positive and its
    second all negative, so every token routed to an expert has an all-zero FC2-input block there.
    A token is a dense MX-exact vector times a power of two that puts ``log2(|v0 . x| / L)`` at a
    uniform draw from the regime's range, so its FC1 outputs take the magnitudes
    ``|v0 . x| 2**-m``, ``m`` in 0..3, with a random mantissa. Tokens are redrawn until, for each
    local expert they route to, ``fc2_input_is_fixed`` holds with ``ACT_MARGIN`` plus twice the
    FP32 noise of ``v0 . x`` (``spec.FP32_SUM_NOISE``; the activation's relative sensitivity to
    ``z`` is at most 2). ``zero_tokens`` makes every fourth token (from the third) zero.
    """

    def __init__(
        self,
        *,
        tokens,
        hidden,
        inter,
        local,
        ep_size,
        ep_rank,
        top_k,
        limit,
        regime,
        rule="agnostic",
        zero_tokens=False,
        seed=0,
    ):
        g = ou.make_generator(ou.DEFAULT_SEED + seed)
        self.limit, self.local, self.offset, self.rule = limit, local, ep_rank * local, rule
        u, v = spec.rank1_factors(
            local,
            inter,
            hidden,
            row_block=spec.MX_BLOCK,
            generator=g,
            levels=2,
            v_scale_exponents=(-1, 0),
        )
        u[:, : spec.MX_BLOCK] = u[:, : spec.MX_BLOCK].abs()
        u[:, spec.MX_BLOCK : 2 * spec.MX_BLOCK] = -u[:, spec.MX_BLOCK : 2 * spec.MX_BLOCK].abs()
        self.u, self.v, self.v0 = u, v, v[0] / v[0, 0]
        self.w1 = u[:, :, None] * v[:, None, :]
        self.w2 = spec.mx_exact_values(local * hidden, inter, generator=g).reshape(
            local, hidden, inter
        )
        experts = local * ep_size
        self.ids, self.probs = spec.make_routing(
            tokens, min(top_k, experts), experts, generator=g, device_="cpu"
        )
        self.routed = torch.stack(
            [(self.ids == self.offset + j).any(-1) for j in range(local)], dim=-1
        )
        lo, hi = UNSATURATED_REGIMES[regime]
        self.x = torch.zeros(tokens, hidden, dtype=torch.float64)
        todo = torch.ones(tokens, dtype=torch.bool)
        if zero_tokens:
            todo[2::4] = False
        for _ in range(_MAX_REDRAWS):
            rows = todo.nonzero().reshape(-1)
            if rows.numel() == 0:
                break
            base = spec.mx_exact_values(rows.numel(), hidden, generator=g, exponents=(0, 0))
            target = lo + (hi - lo) * torch.rand(rows.numel(), generator=g, dtype=torch.float64)
            c0 = base @ self.v0
            usable = c0 != 0
            shift = torch.round(target - torch.log2(c0.abs().clamp(min=1e-300) / limit))
            candidate = base * spec.pow2(torch.where(usable, shift, 0.0))[:, None]
            ok = usable & self.fixed(candidate, self.routed[rows])
            self.x[rows[ok]] = candidate[ok]
            todo[rows[ok]] = False
        else:
            raise AssertionError(f"could not draw {int(todo.sum())} tokens with a fixed FC2 input")

    def fc1(self, x):
        """The FC1 outputs ``z [T, local, I]`` (fp64, exact) and the per-token relative FP32
        noise bound of ``v0 . x``."""
        terms = x * self.v0
        noise = spec.FP32_SUM_NOISE * x.shape[-1] ** 0.5 * terms.norm(dim=-1)
        noise = noise / terms.sum(-1).abs().clamp(min=1e-300)
        return (x @ self.v.T)[:, :, None] * self.u[None], noise

    def activation(self, z):
        return spec.clamped_relu2(self.limit)(z)

    def fixed(self, x, routed):
        """Per token: is the FC2 input fixed on every local expert the token routes to?"""
        z, noise = self.fc1(x)
        ok = fc2_input_is_fixed(
            self.activation(z), (ACT_MARGIN + 2 * noise)[:, None, None], self.rule
        )
        return (ok | ~routed).all(-1)

    def fc2_input(self, x):
        """``[T, local, I]`` spec MXFP8 FC2 input of the fp64 activation."""
        a = self.activation(self.fc1(x)[0])
        return spec.mxfp8_roundtrip(a.reshape(-1, a.shape[-1])).reshape(a.shape)

    def run(self, monkeypatch, finalize, ep_size, ep_rank):
        """The call site's output on CUDA (BF16 ``[T, H]``)."""
        if finalize == "unfused":
            _unfused_finalize(monkeypatch)
        experts = _experts(
            _cutlass_mxfp8(self.w1),
            _cutlass_mxfp8(self.w2),
            ep_size=ep_size,
            ep_rank=ep_rank,
            activation="ClampedRelu2",
            clamp=self.limit,
            mxfp8=True,
        )
        return experts, self.x.to(torch.bfloat16).cuda(), self.ids.cuda(), self.probs.cuda()

    def check(self, out, what):
        """Kernel output against the fp64 reference, bounded per row by the trusted fp32
        implementation, after checking that both requantize the activation identically."""
        fc2_inputs = {}

        def requantize(key):
            def rnd(a):
                q = spec.mxfp8_roundtrip(a)
                fc2_inputs.setdefault(key, []).append(q)
                return q.to(a.dtype)

            return rnd

        xb = self.x.to(torch.bfloat16).cuda()
        args = (xb, self.w1.cuda(), self.w2.cuda(), self.ids.cuda(), self.probs.cuda())
        act = spec.clamped_relu2(self.limit)
        ref = ou.fp64_reference(
            spec.moe_reference,
            *args,
            activation=act,
            local_offset=self.offset,
            act_round=requantize("reference"),
        )
        with spec.exact_fp32_matmul():
            trusted = spec.moe_reference(
                xb.float(),
                args[1].float(),
                args[2].float(),
                args[3],
                args[4],
                activation=act,
                local_offset=self.offset,
                act_round=requantize("trusted"),
                out_dtype=torch.bfloat16,
            )
        pairs = zip(fc2_inputs.get("reference", []), fc2_inputs.get("trusted", []), strict=True)
        for r, t in pairs:
            ou.assert_exact(t, r, what="trusted FC2 input equals the reference's")
        assert out.shape == ref.shape and out.dtype == torch.bfloat16, (out.shape, out.dtype)
        return ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=what)


def _run_unsaturated(case, monkeypatch, rule="agnostic"):
    _cutlass_module_ready()
    ep_size, ep_rank = case["ep"]
    hidden, inter = case["shape"]
    problem = ClampedMxfp8Problem(
        tokens=case["tokens"],
        hidden=hidden,
        inter=inter,
        local=4,
        ep_size=ep_size,
        ep_rank=ep_rank,
        top_k=case["top_k"],
        limit=case["limit"],
        regime=case["regime"],
        rule=rule,
        zero_tokens=case.get("zero_tokens", False),
    )
    experts, xb, ids, probs = problem.run(monkeypatch, case["finalize"], ep_size, ep_rank)
    out = _forward(experts, xb, ids, probs)
    problem.check(out, what=f"MXFP8 ClampedRelu2 L={case['limit']} {case['regime']}")
    return problem, experts, xb, ids, probs


# Regime per L: with a power-of-two L, (L tanh)**2 approaches L**2 = 2**k from below, where
# the RCEIL and OCP floor scales differ, so the sweep (agnostic) uses those L below the knee and
# 3.5 and 6 (L**2 mantissas 1.53 and 1.125) through saturation.
CLAMP_REGIMES = [
    (0.5, "near-zero"),
    (8.0, "knee"),
    (16.0, "mixed"),
    (3.5, "knee"),
    (3.5, "transition"),
    (3.5, "saturation"),
    (6.0, "mixed"),
]


@pytest.mark.parametrize(
    "limit, regime, rule",
    [(3.5, "mixed", "agnostic"), (0.5, "knee", "agnostic"), (16.0, "transition", "rceil")],
)
def test_unsaturated_inputs_fix_the_fc2_input(limit, regime, rule):
    """CPU check of ``ClampedMxfp8Problem``: MX-exact BF16 operands; FC1 outputs spread over the
    regime; every routed (token, expert) FC2 input unchanged under the margin, equal to the one a
    float32 activation gives, with an all-zero block; zero tokens zero; and with ``"rceil"`` many
    blocks whose amax mantissa lies in (1.75, 2), where RCEIL and the OCP floor differ."""
    problem = ClampedMxfp8Problem(
        tokens=96,
        hidden=96,
        inter=128,
        local=4,
        ep_size=2,
        ep_rank=1,
        top_k=4,
        limit=limit,
        regime=regime,
        rule=rule,
        zero_tokens=True,
    )
    x = problem.x
    ou.assert_exact(x.to(torch.bfloat16).double(), x, what="tokens are BF16-exact")
    ou.assert_exact(spec.mxfp8_roundtrip(x), x, what="tokens are MXFP8-exact")
    for e in range(problem.local):
        ou.assert_exact(spec.mxfp8_roundtrip(problem.w1[e]), problem.w1[e], what="W1")
        ou.assert_exact(spec.mxfp8_roundtrip(problem.w2[e]), problem.w2[e], what="W2")
    live = x.abs().amax(-1) > 0
    assert int((~live).sum()) == 96 // 4 and not bool(x[2::4].any()), "zero tokens"

    z, _ = problem.fc1(x)
    routed = problem.routed & live[:, None]
    assert bool(problem.fixed(x, problem.routed).all())
    lo, hi = UNSATURATED_REGIMES[regime]
    reach = torch.log2((x[live] @ problem.v0).abs() / limit)  # log2(|z_max| / L) per token
    assert float(reach.min()) >= lo - 0.5 and float(reach.max()) <= hi + 0.5, reach.aminmax()
    assert float(reach.max() - reach.min()) >= 0.5 * (hi - lo), "the draws span the regime"

    reference = problem.fc2_input(x)
    a32 = spec.clamped_relu2(limit)(z.float())
    trusted = spec.mxfp8_roundtrip(a32.reshape(-1, a32.shape[-1]).double()).reshape(a32.shape)
    ou.assert_exact(trusted[routed], reference[routed], what="FP32 activation, same FC2 input")
    blocks = reference[routed].reshape(int(routed.sum()), -1, spec.MX_BLOCK)
    assert bool((blocks.abs().amax(-1) == 0).any(-1).all()), "an all-zero block per routed pair"
    assert bool((reference[~live] == 0).all())

    amax = problem.activation(z)[routed].reshape(int(routed.sum()), -1, spec.MX_BLOCK).amax(-1)
    amax = amax[amax > 0]
    mantissa = amax / spec.pow2(_floor_log2(amax))
    between = float(((mantissa > 1.75) & (mantissa < 2)).double().mean())
    if rule == "agnostic":
        assert between == 0.0
    else:
        assert between >= 0.2, between


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=f"L{c['clamp'][0]}-{c['clamp'][1]}-T{c['tokens']}-H{c['shape'][0]}I{c['shape'][1]}"
            f"-k{c['top_k']}-ep{c['ep'][0]}r{c['ep'][1]}-{c['finalize']}"
            f"{'-zero-tokens' if c['zero_tokens'] else ''}",
        )
        for c in ou.pairwise_cases(
            {
                "clamp": CLAMP_REGIMES,
                "tokens": [1, 65, 128, 129, 1024],
                "shape": [(128, 128), (256, 384), (2688, 2048)],
                "top_k": [1, 2, 4],
                "ep": EP,
                "finalize": ["fused", "unfused"],
                "zero_tokens": [False, True],
            }
        )
    ],
)
def test_mxfp8_clamped_relu2_unsaturated_matches_fp64_moe(case, monkeypatch):
    """MXFP8 ClampedRelu2 outside saturation (``ClampedMxfp8Problem``) against the fp64 MoE of
    the requantized activation, per row, factor 2 of the trusted fp32 implementation."""
    limit, regime = case["clamp"]
    _run_unsaturated(dict(case, limit=limit, regime=regime), monkeypatch)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("limit", [0.5, 8.0, 16.0])
def test_mxfp8_fc2_input_scale_is_rceil_near_power_of_two_activations(limit, monkeypatch):
    """With a power-of-two L, ``(L tanh(z / L))**2`` approaches ``L**2 = 2**k`` from below, so
    in the transition regime many FC2-input blocks have an amax ``m * 2**b`` with ``m`` in
    (1.75, 2). RCEIL scales such a block by ``2**(b - 7)`` and keeps every element; OCP's floor
    rule (``2**(b - 8)``, then saturate) would clip its largest element to ``1.75 * 2**b``, up to
    12.5% low. The FC2 input of the reference is the RCEIL one (contract above); the inputs keep
    the activations away from every other boundary."""
    _run_unsaturated(
        dict(
            tokens=129,
            shape=(256, 384),
            top_k=2,
            ep=(2, 1),
            finalize="fused",
            limit=limit,
            regime="transition",
        ),
        monkeypatch,
        rule="rceil",
    )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("finalize", ["fused", "unfused"])
@pytest.mark.parametrize("gemm", ["gemm1", "gemm2"])
def test_mxfp8_clamped_relu2_every_tactic(gemm, finalize, monkeypatch):
    """Each tactic FlashInfer reports valid for one GEMM of the MXFP8 ClampedRelu2 MoE (the other
    keeps its default) gives the reference result outside saturation. GEMM1's are the fused
    ClampedRelu2 + MXFP8-requantization epilogue variants (FlashInfer filters them to the ones
    with activation fusion); I = 384 ends half way through the 256-wide tile, T = 129 one row into
    a second 128-row tile."""
    _cutlass_module_ready()
    from flashinfer.autotuner import AutoTuner

    choose_one = AutoTuner.choose_one
    target = _GEMM1 if gemm == "gemm1" else _GEMM2
    pin = SimpleNamespace(tactic=None, valid=None)

    def pinned(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op not in (_GEMM1, _GEMM2):
            return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)
        if custom_op == target and pin.valid is None:
            # The runner's stage is set to this GEMM before choose_one; its tactic ids are
            # global (GEMM2's follow GEMM1's).
            pin.valid = sorted(t for t in runners[0].get_valid_tactics(inputs, None) if t >= 0)
        if custom_op != target or pin.tactic is None:
            return runners[0], -1
        return runners[0], pin.tactic

    monkeypatch.setattr(AutoTuner, "choose_one", pinned)
    problem, experts, xb, ids, probs = _run_unsaturated(
        dict(
            tokens=129,
            shape=(256, 384),
            top_k=2,
            ep=(2, 1),
            finalize=finalize,
            limit=3.5,
            regime="mixed",
        ),
        monkeypatch,
    )
    assert pin.valid, f"FlashInfer reported no valid {gemm} tactic"
    for tactic in pin.valid:
        pin.tactic = tactic
        out = _forward(experts, xb, ids, probs)
        problem.check(out, what=f"{gemm} tactic {tactic} ({finalize} finalize)")
