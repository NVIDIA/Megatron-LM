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
of a rounding boundary each implementation lands). MXFP8 ``ClampedRelu2``: a rank-1 ``W1`` and
tokens whose FC1 outputs are all beyond 40 L in magnitude, so the activation is exactly ``L**2``
or 0 whatever the kernel's tanh, and the FC2 input is MXFP8-exact; the FC1 numerics are then
only tested through the sign and the threshold (dense ClampedRelu2 MXFP8 inputs would make the
oracle depend on FP4/FP8 rounding of a transcendental).

Variants: activation {Relu2, ClampedRelu2 with L = 0.5 (mostly saturated) and 8 (mostly
linear)}, finalize {fused, unfused}, every GEMM2 tactic FlashInfer reports valid (Relu2), tokens
1, 64/65, 128/129 and 1024, H and I 128, 256/384 and production (2688, 1856), E_local 1 and 8,
top-k 1, 2, 6, EP (size, rank) (1, 0), (2, 1), (4, 2); an empty expert and all -1 rows.
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
