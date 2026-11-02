# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of the float64 GLM mHC emulator (glm_mhc_emulator.py).

They pin the TF32 bit helpers to an integer oracle, the exact modes to float64 autograd, the
accumulation models to an exact-rational oracle and to outputs measured on GB300 (sm_103) hardware,
the split plans to the P2 prototype's rule and to the shipped kernels' chains and per-column pieces,
and every mutation to its power threshold.
"""

import dataclasses
import functools
import hashlib
import math
import os
import random
import struct
import typing
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.transformer.glm_mhc_emulator import (
    ALIGN_K8_X2,
    CONFIG_ROUNDING_TO_MODE,
    DEFAULT_MAX_CHUNK_ELEMS,
    EPS_FP32,
    EXACT,
    IEEE_RN,
    OPERAND_MODES,
    OPERAND_SLOTS,
    P9_M3_BARS,
    P9_POWER_THRESHOLDS,
    PAIR_N,
    R01_ALIGN_K8_X0,
    RZ_K8,
    TF32_MODES,
    AccumModel,
    ChainPlan,
    ColumnChainPlans,
    GLMPairPlans,
    _accumulate_columns,
    _accumulate_split_loop,
    accumulate,
    emulate,
    glm_mappings_chain_f64,
    glm_pair_fused_plans,
    glm_pair_kernel_model,
    glm_pair_kernel_plans,
    glm_pair_reference_f64,
    gp1_s_model,
    gp2_plan,
    gp2_plan_nonempty,
    gp4_wgrad_column_plans,
    gp4_wgrad_plan,
    ordered_reduce,
    pair_data,
    rel_rms,
    round_operand,
    rz_to_fp32,
    single_chain,
    sinkhorn_f64,
    sinkhorn_vjp,
    tf32_edge_patterns,
    tf32_round,
    tf32_round_int_reference,
    ulp32_grid,
)

_N = 24  # n^2 + 2n mapping outputs at n = 4 residual streams
_U = 2.0**-23  # the fp32 ulp of 1.0
# The inputs of every pair case: x bf16 [T, K], W fp32 [24, K] and G_p fp32 [T, 24]. The kernels'
# GPU tests draw theirs with the same function.
_pair_data = pair_data


def _fl32(value: float) -> float:
    """The nearest fp32 value: the IEEE fp32 result of one operation computed in float64."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _patterns_to_fp32(patterns):
    return torch.tensor(
        [bits - (1 << 32) if bits >= 1 << 31 else bits for bits in patterns], dtype=torch.int32
    ).view(torch.float32)


def _hf_pair_autograd(x, W, G_p, G_s):
    """float64 autograd of the HF formula, in hyper_connection._proj_rms_impl's order with
    normalize_before_projection."""
    x64 = x.detach().to(torch.float64).requires_grad_(True)
    W64 = W.detach().to(torch.float64).requires_grad_(True)
    norm = torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + EPS_FP32)
    proj = F.linear(x64 * norm, W64)
    loss = 0.0
    if G_p is not None:
        loss = loss + (proj * G_p.to(torch.float64)).sum()
    if G_s is not None:
        loss = loss + (norm[:, 0] * G_s.to(torch.float64)).sum()
    g_x, g_W = torch.autograd.grad(loss, (x64, W64), allow_unused=True)
    return norm[:, 0].detach(), proj.detach(), g_x, g_W


def _hf_sinkhorn(logits, iters=20, eps=1e-6):
    # An inline copy of hyper_connection._sinkhorn_iterations, which is compiled.
    M = logits.softmax(dim=-1) + eps
    M = M / (M.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


@pytest.fixture
def heavy_cpu_case():
    """Heavy emulations run on rank 0 only, with up to 16 intra-op threads.

    CI runs every rank of a bucket at once, and torch.distributed.run sets OMP_NUM_THREADS=1.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        pytest.skip("heavy CPU emulation: rank 0 only")
    threads = torch.get_num_threads()
    # The CPUs this process may run on: os.cpu_count() counts the host's, also under a cpuset.
    if hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    torch.set_num_threads(min(16, cpus))
    yield
    torch.set_num_threads(threads)


# ============================================================================
# TF32 bit helpers
# ============================================================================

# Named TF32 edge cases: (pattern, RNE, RZ, RNA). Each also holds with the sign bit set.
_TF32_EDGE_TABLE = {
    "tie_even_kept_bit": (0x3F801000, 0x3F800000, 0x3F800000, 0x3F802000),
    "tie_odd_kept_bit": (0x3F803000, 0x3F804000, 0x3F802000, 0x3F804000),
    "above_half": (0x3F801001, 0x3F802000, 0x3F800000, 0x3F802000),
    "below_half": (0x3F800FFF, 0x3F800000, 0x3F800000, 0x3F800000),
    **{
        f"sticky_bit_{j}": (0x3F801000 | 1 << j, 0x3F802000, 0x3F800000, 0x3F802000)
        for j in range(12)
    },
    "carry_into_exponent": (0x3FFFF000, 0x40000000, 0x3FFFE000, 0x40000000),
    "max_finite": (0x7F7FFFFF, 0x7F800000, 0x7F7FE000, 0x7F800000),
    "largest_staying_finite": (0x7F7FEFFF, 0x7F7FE000, 0x7F7FE000, 0x7F7FE000),
    "inf": (0x7F800000,) * 4,
    "nan_low_payload": (0x7F800001,) * 4,
    "nan_quiet": (0x7FC00000,) * 4,
    "nan_negative_payload": (0xFFC00001,) * 4,
    "subnormal_below_half": (0x00000FFF, 0x00000000, 0x00000000, 0x00000000),
    "subnormal_tie_even": (0x00001000, 0x00000000, 0x00000000, 0x00002000),
    "subnormal_tie_odd": (0x00003000, 0x00004000, 0x00002000, 0x00004000),
    "subnormal_carry_to_min_normal": (0x007FF000, 0x00800000, 0x007FE000, 0x00800000),
    "zero": (0x00000000,) * 4,
    "negative_zero": (0x80000000,) * 4,
}


@pytest.mark.parametrize("mode", TF32_MODES)
def test_tf32_round_matches_int_reference(mode):
    """The torch twin equals the integer oracle on the edge cases and 65,536 random patterns."""
    patterns = tf32_edge_patterns()
    rounded = tf32_round(_patterns_to_fp32(patterns), mode).view(torch.int32).tolist()
    mismatches = [
        f"{bits:#010x}"
        for bits, got in zip(patterns, rounded)
        if got & 0xFFFFFFFF != tf32_round_int_reference(bits, mode)
    ]
    assert not mismatches, f"tf32_round({mode!r}) differs from the integer oracle: {mismatches[:8]}"


@pytest.mark.parametrize("case", list(_TF32_EDGE_TABLE))
def test_tf32_round_named_edges(case):
    """The oracle and the torch twin give the table's result in every mode, for both signs.

    The sticky and below-half rows pin a full-sticky RNE: crafted operands alone cannot tell it
    from a rounding that reads only a few bits below the guard.
    """
    bits, *expected = _TF32_EDGE_TABLE[case]
    patterns = set(tf32_edge_patterns())
    for sign in (0, 0x80000000):
        pattern = bits ^ sign
        assert pattern in patterns
        for mode, want in zip(("rne", "rz", "rna"), expected):
            want ^= sign
            assert tf32_round_int_reference(pattern, mode) == want, (case, mode, hex(pattern))
            got = tf32_round(_patterns_to_fp32([pattern]), mode).view(torch.int32).item()
            assert got & 0xFFFFFFFF == want, (case, mode, hex(pattern))


def test_config_rounding_names_map_to_modes():
    """TransformerConfig.mhc_fused_tf32_rounding's names map onto the emulator's TF32 modes.

    The one test that ties the emulator to TransformerConfig: it needs that field.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    field = {f.name: f for f in dataclasses.fields(TransformerConfig)}["mhc_fused_tf32_rounding"]
    assert set(typing.get_args(field.type)) == set(CONFIG_ROUNDING_TO_MODE)
    assert CONFIG_ROUNDING_TO_MODE[field.default] == "rne"
    assert set(CONFIG_ROUNDING_TO_MODE.values()) <= set(TF32_MODES)


# ============================================================================
# Exact modes (float64)
# ============================================================================

_PAIR_CASES = {
    "x1": dict(T=64, K=16384),
    "x1e-3": dict(T=64, K=16384, scale=1e-3),
    "x1e3": dict(T=64, K=16384, scale=1e3),
    "heavy_rows": dict(T=64, K=16384, rows="heavy"),
    "zero_rows": dict(T=64, K=16384, rows="zero_rows"),
    "T17_K384": dict(T=17, K=384),
    "G_s": dict(T=64, K=16384, with_G_s=True),
    "no_G_p": dict(T=64, K=16384, with_G_s=True, with_G_p=False),
}


@pytest.mark.parametrize("case", list(_PAIR_CASES))
def test_glm_pair_exact_matches_autograd(case):
    """The pair's float64 formulas equal float64 autograd of the HF formula: rel-RMS <= 1e-10.

    Both sides add eps = fl32(1e-5), the value that hyper_connection's fp32 "+ eps" adds.
    """
    spec = _PAIR_CASES[case]
    T = spec["T"]
    x, W, G_p = _pair_data(T, spec["K"], spec.get("scale", 1.0), 3, spec.get("rows", "randn"))
    if not spec.get("with_G_p", True):
        G_p = None
    G_s = torch.randn(T, generator=torch.Generator().manual_seed(4)) if "with_G_s" in spec else None
    assert EPS_FP32 == _fl32(1e-5)

    result = glm_pair_reference_f64(x, W, G_p, G_s)
    s_ref, p_ref, g_x_ref, g_W_ref = _hf_pair_autograd(x, W, G_p, G_s)
    assert rel_rms(result.s, s_ref) <= 1e-10
    assert rel_rms(result.p, p_ref) <= 1e-10
    assert rel_rms(result.g_x, g_x_ref) <= 1e-10
    if G_p is None:
        assert result.g_xn is None and result.g_W is None and g_W_ref is None
    else:
        assert rel_rms(result.g_W, g_W_ref) <= 1e-10


# (T, K, x scale). At T = 4096, d_alpha_res sums T * n^2 = 65,536 addends: the bound
# (n - 1) * u * kappa with kappa <= 606 is 4.4e-9, a 2.3 x margin under the 1e-8 scalar bar.
_CHAIN_CASES = {
    "T64_K16384_x1": (64, 16384, 1.0),
    "T64_K16384_x1e-3": (64, 16384, 1e-3),
    "T64_K16384_x1e3": (64, 16384, 1e3),
    "T17_K384": (17, 384, 1.0),
    "T4096_K384": (4096, 384, 1.0),
}


def _hf_mappings_autograd(x, W, alpha, bias, grads, n=4):
    """float64 autograd of the HF chain: the GLM projection, the normalize-before-projection
    branch of HyperConnectionModule._compute_h and the Sinkhorn."""
    x64 = x.detach().to(torch.float64).requires_grad_(True)
    W64 = W.detach().to(torch.float64).requires_grad_(True)
    alphas = [torch.tensor([a], dtype=torch.float64, requires_grad=True) for a in alpha]
    b64 = bias.detach().to(torch.float64).requires_grad_(True)
    norm = torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + EPS_FP32)
    proj = F.linear(x64 * norm, W64)
    alpha_ = torch.cat([alphas[0].expand(n), alphas[1].expand(n), alphas[2].expand(n * n)])
    h = proj * alpha_ + b64
    outputs = {
        "h_pre": h[:, :n].sigmoid() + 1e-6,  # hyper_connection's _MHC_COMPUTE_H_EPS
        "h_post": h[:, n : 2 * n].sigmoid() * 2,
        "H_res": _hf_sinkhorn(h[:, 2 * n :].reshape(-1, n, n)),
    }
    loss = sum((outputs[name] * grad).sum() for name, grad in zip(outputs, grads))
    names = ("g_x", "g_W", "d_alpha_pre", "d_alpha_post", "d_alpha_res", "d_bias")
    gradients = torch.autograd.grad(loss, (x64, W64, *alphas, b64))
    return {**{name: t.detach() for name, t in outputs.items()}, **dict(zip(names, gradients))}


def _chain_inputs(case, request):
    """x, W, alpha, bias and grads of a _CHAIN_CASES case (T = 4096 runs as a heavy CPU case)."""
    T, K, scale = _CHAIN_CASES[case]
    if T >= 4096:
        request.getfixturevalue("heavy_cpu_case")
    x, W, _ = _pair_data(T, K, scale, seed=11)
    g = torch.Generator().manual_seed(12)
    alpha = (0.7, 1.3, 0.9)
    bias = torch.randn(_N, generator=g, dtype=torch.float64) * 0.1
    grads = tuple(
        torch.randn(shape, generator=g, dtype=torch.float64)
        for shape in ((T, 4), (T, 4), (T, 4, 4))
    )
    return x, W, alpha, bias, grads


@pytest.mark.parametrize("case", list(_CHAIN_CASES))
def test_glm_mappings_chain_exact_matches_autograd(case, request):
    """The analytic chain equals float64 autograd: tensors <= 1e-10, the alpha scalars <= 1e-8."""
    x, W, alpha, bias, grads = _chain_inputs(case, request)

    result = glm_mappings_chain_f64(x, W, alpha, bias, grads)
    reference = _hf_mappings_autograd(x, W, alpha, bias, grads)
    for name in ("h_pre", "h_post", "H_res", "g_x", "g_W", "d_bias"):
        assert rel_rms(result[name], reference[name]) <= 1e-10, name
    for name in ("d_alpha_pre", "d_alpha_post", "d_alpha_res"):
        got, want = result[name].item(), reference[name].item()
        assert abs(got - want) <= 1e-8 * abs(want), (name, got, want)


@pytest.mark.parametrize("case", list(_CHAIN_CASES))
def test_glm_mappings_chain_returns_contributions(case, request):
    """The chain also returns p and G_h (float64 [T, 24]): G_h * p summed over each column group is
    d_alpha_pre, d_alpha_post and d_alpha_res bitwise, and G_h summed over T is d_bias, so a module
    test can score each alpha gradient through its contribution tensor instead of one scalar."""
    x, W, alpha, bias, grads = _chain_inputs(case, request)
    T = x.shape[0]

    result = glm_mappings_chain_f64(x, W, alpha, bias, grads)
    for name in ("p", "G_h"):
        assert result[name].dtype == torch.float64 and tuple(result[name].shape) == (T, _N), name
    assert torch.equal(result["p"], glm_pair_reference_f64(x, W).p)
    contributions = result["G_h"] * result["p"]
    for name, columns in (("pre", slice(0, 4)), ("post", slice(4, 8)), ("res", slice(8, _N))):
        assert torch.equal(contributions[:, columns].sum(), result[f"d_alpha_{name}"]), name
    assert torch.equal(result["G_h"].sum(dim=0), result["d_bias"])


def _sinkhorn_data(scale, T=4096, n=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, n, n, generator=g, dtype=torch.float64) * scale
    grad = torch.randn(T, n, n, generator=g, dtype=torch.float64)
    return logits, grad


def _sinkhorn_autograd(logits, grad):
    leaf = logits.clone().requires_grad_(True)
    (result,) = torch.autograd.grad(_hf_sinkhorn(leaf), leaf, grad)
    return result


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_sinkhorn_vjp_exact_matches_autograd(scale):
    """The float64 VJP equals float64 autograd of the HF Sinkhorn (20 iterations): <= 1e-10."""
    logits, grad = _sinkhorn_data(scale)
    reference = _sinkhorn_autograd(logits, grad)
    assert torch.equal(sinkhorn_f64(logits), _hf_sinkhorn(logits))
    assert rel_rms(sinkhorn_vjp(logits, grad), reference) <= 1e-10
    # The same algorithm in fp32 torch order is an error predictor for the Triton kernel, not a
    # gate: tl.exp2 and fp32 division lower to approximate instructions.
    fp32 = sinkhorn_vjp(logits, grad, dtype=torch.float32)
    assert fp32.dtype == torch.float32
    print(f"GLM_MHC_EMULATOR_SINKHORN_FP32 scale={scale:g} rel_rms={rel_rms(fp32, reference):.3e}")


# ============================================================================
# Rounding mode: the elementary pieces and their pins
# ============================================================================


def _one(products, model, plan=None):
    """One dot product of the given exact products with ones."""
    a = torch.tensor([products], dtype=torch.float64)
    plan = plan if plan is not None else single_chain(len(products))
    return float(accumulate(a, torch.ones_like(a), model, plan)[0, 0])


# Position 0 holds 1 (the first k-step leaves acc = 1); positions 8-15 form the second k-step.
_Q8 = [1.0] + [0.0] * 7 + [_U / 4] * 8
_NQ = [1.0] + [0.0] * 7 + [-_U / 4]
_CANCEL = [1.0] + [0.0] * 7 + [-1.0, 2.0**-30]
_K4_VS_K8 = [1.0, _U / 2, 0.0, 0.0, 0.0, _U / 2]
_TWO_SPLITS = [1.0] + [0.0] * 7 + [_U / 4] + [0.0] * 7  # partials 1 and u/4: fl32(1 + u/4) = 1
_BLOCKS = [1.0, _U / 2, _U / 2]  # blocks of 2: fl32(1 + u/2) = 1 (a tie, to even), then + u/2
_GRID_IN = torch.tensor([1.0, 1.5, 0.75, 0.0, 3.0e-30], dtype=torch.float64)
_RZ_IN = torch.tensor(
    [1 + 2.0**-30, 1 - 2.0**-30, -(1 - 2.0**-30), 2.0**200, 1.5, -0.0], dtype=torch.float64
)

# 22 hand-computed checks: name -> (computation, expected value).
_UNIT_CHECKS = {
    "rz_to_fp32": (
        lambda: rz_to_fp32(_RZ_IN).to(torch.float64).tolist(),
        [1.0, 1 - 2.0**-24, -(1 - 2.0**-24), 3.4028234663852886e38, 1.5, 0.0],
    ),
    "ulp32_grid": (
        lambda: ulp32_grid(_GRID_IN).tolist(),
        [_U, _U, _U / 2, 1.0, 2.0 ** (math.frexp(3.0e-30)[1] - 24)],
    ),
    "ulp32_grid_extra2": (lambda: ulp32_grid(_GRID_IN[:1], 2).tolist(), [_U / 4]),
    "ulp32_grid_inverse": (
        lambda: torch.mul(*ulp32_grid(_GRID_IN, with_inverse=True)).tolist(),
        [1.0] * 5,
    ),
    "q8_rz": (lambda: _one(_Q8, RZ_K8), 1 + 2 * _U),
    "q8_align": (lambda: _one(_Q8, R01_ALIGN_K8_X0), 1.0),
    "q8_align_extra2": (lambda: _one(_Q8, ALIGN_K8_X2), 1 + 2 * _U),
    "q8_rn": (lambda: _one(_Q8, IEEE_RN), 1.0),
    "q8_rnk": (lambda: _one(_Q8, AccumModel("rnk")), 1 + 2 * _U),
    "nq_rz": (lambda: _one(_NQ, RZ_K8), 1 - _U / 2),
    "nq_rnk": (lambda: _one(_NQ, AccumModel("rnk")), 1.0),  # a tie, to even
    "nq_align": (lambda: _one(_NQ, R01_ALIGN_K8_X0), 1.0),
    "cancel_rz": (lambda: _one(_CANCEL, RZ_K8), 2.0**-30),
    "cancel_align": (lambda: _one(_CANCEL, R01_ALIGN_K8_X0), 0.0),
    "cancel_rn": (lambda: _one(_CANCEL, IEEE_RN), 2.0**-30),
    "k4_vs_k8_rz_k8": (lambda: _one(_K4_VS_K8, RZ_K8), 1 + _U),
    "k4_vs_k8_rz_k4": (lambda: _one(_K4_VS_K8, AccumModel("rz", 4)), 1.0),
    "split2_rz": (lambda: _one(_TWO_SPLITS, RZ_K8, ChainPlan.equal(16, 2)), 1.0),
    "split1_rz": (lambda: _one(_TWO_SPLITS, RZ_K8), 1.0),  # RZ of 1 + u/4 is 1 as well
    "rn_block2": (lambda: _one(_BLOCKS, AccumModel("rn", rn_block=2)), 1.0),
    "exact_compensated": (lambda: _one([1.0, 2.0**-60, -1.0], EXACT), 2.0**-60),
    # (1 + u/2) ties to 1, twice: the order of the IEEE fp32 adds matters.
    "ordered_reduce": (
        lambda: ordered_reduce(torch.tensor([1.0, 2.0**-24, 2.0**-24], dtype=torch.float32)).item(),
        1.0,
    ),
}


@pytest.mark.parametrize("check", list(_UNIT_CHECKS))
def test_elementary_unit_checks(check):
    """The 22 hand-computed checks of rz_to_fp32, ulp32_grid, the chains and the reduction."""
    compute, expected = _UNIT_CHECKS[check]
    assert compute() == expected


# The errors that r01, an earlier model of the tensor-core accumulation, computed (rel-RMS against
# the exact sum; modeled, not measured) and printed with 4 significant digits.
_R01_PUBLISHED = {
    "p/rn_blk": 1.635e-07,
    "p/mma_rz": 3.437e-06,
    "p/mma_align": 1.108e-06,
    "p/a_order": 1.782e-04,
    "gw/rn_blk": 1.846e-07,
    "gw/mma_rz_1": 1.212e-05,
    "gw/mma_align_1": 1.890e-06,
    "gw/mma_rz_2": 6.049e-06,
    "gw/mma_align_2": 1.296e-06,
}


def _r01_p_data(rows=6, nout=8, seed=11, k_len=16384):
    """r01's projection data, drawn in its order: xn, the raw bf16 x, w, and s per pair."""
    rng = random.Random(seed)
    xn_rows, x_rows, w_rows, s_rows = [], [], [], []
    for _ in range(rows):
        gx = torch.tensor([rng.gauss(0.0, 1.0) for _ in range(k_len)], dtype=torch.float64)
        x = gx.to(torch.float32).to(torch.bfloat16).to(torch.float64)
        ms = math.fsum((x * x).tolist()) / k_len  # squares of bf16 values are exact
        s = _fl32(1.0 / math.sqrt(_fl32(_fl32(ms) + 1e-5)))
        xn = tf32_round((x * s).to(torch.float32), "rne")
        for _ in range(nout):
            gw = torch.tensor([rng.gauss(0.0, 1.0) for _ in range(k_len)], dtype=torch.float64)
            xn_rows.append(xn)
            x_rows.append(x.to(torch.float32))
            w_rows.append(tf32_round((gw / math.sqrt(k_len) * 4.0).to(torch.float32), "rne"))
            s_rows.append(s)
    xn, x, w = (torch.stack(tensors) for tensors in (xn_rows, x_rows, w_rows))
    return xn, x, w, torch.tensor(s_rows, dtype=torch.float64)


def _r01_gw_data(n_out=96, seed=23, t_len=4096):
    """r01's wgrad data: g [n_out, T] and xn [n_out, T], TF32-exact fp32."""
    rng = random.Random(seed)
    g_rows, x_rows = [], []
    for _ in range(n_out):
        g = torch.tensor([rng.gauss(0.0, 1.0) for _ in range(t_len)], dtype=torch.float64)
        g_rows.append(tf32_round((g * 1e-3).to(torch.float32), "rne"))
        xv = torch.tensor([rng.gauss(0.0, 1.0) for _ in range(t_len)], dtype=torch.float64)
        x_rows.append(tf32_round(xv.to(torch.float32), "rne"))
    return torch.stack(g_rows), torch.stack(x_rows)


def _pairwise(a, b, model, plan):
    """Row-wise dot products of a [P, K] and b [P, K] (TF32-exact operands) as [P]."""
    a64, b64 = a.to(torch.float64).unsqueeze(-2), b.to(torch.float64).unsqueeze(-2)
    return accumulate(a64, b64, model, plan)[..., 0, 0]


def test_r01_reproduction():
    """r01's two experiments, on p and on g_W: its 9 modeled errors to 4 significant digits.

    The data come from random.Random, so they are platform-stable.
    """
    rn_blk = AccumModel("rn", rn_block=64)
    xn, x, w, s = _r01_p_data()
    split16 = ChainPlan.equal(16384, 16)
    exact = _pairwise(xn, w, EXACT, single_chain(16384))
    measured = {
        "p/rn_blk": rel_rms(_pairwise(xn, w, rn_blk, split16), exact),
        "p/mma_rz": rel_rms(_pairwise(xn, w, RZ_K8, split16), exact),
        "p/mma_align": rel_rms(_pairwise(xn, w, R01_ALIGN_K8_X0, split16), exact),
        # project-first order: fl32(s * the blocked-RN chain of the raw x)
        "p/a_order": rel_rms((s * _pairwise(x, w, rn_blk, split16).double()).float(), exact),
    }
    g, xg = _r01_gw_data()
    chain, split2 = single_chain(4096), ChainPlan.equal(4096, 2)
    exact_gw = _pairwise(g, xg, EXACT, chain)
    for key, model, plan in (
        ("gw/rn_blk", rn_blk, chain),
        ("gw/mma_rz_1", RZ_K8, chain),
        ("gw/mma_align_1", R01_ALIGN_K8_X0, chain),
        ("gw/mma_rz_2", RZ_K8, split2),
        ("gw/mma_align_2", R01_ALIGN_K8_X0, split2),
    ):
        measured[key] = rel_rms(_pairwise(g, xg, model, plan), exact_gw)
    for key, published in _R01_PUBLISHED.items():
        assert f"{measured[key]:.3e}" == f"{published:.3e}", (key, measured[key], published)


# The crafted accumulation cases measured on GB300 hardware: (position, value) along the
# contraction; the other operand is 1 at every position. Positions 0-16 lie in the first split of
# every measured kernel variant, so each case is one chain of 17.
_CRAFTED_CASES = {
    "q8": [(0, 1.0)] + [(8 + i, _U / 4) for i in range(8)],
    "e8": [(0, 1.0)] + [(8 + i, _U / 8) for i in range(8)],
    "q3": [(0, 1.0), (8, _U / 4), (9, _U / 4), (10, _U / 4)],
    "t1": [(0, 1.0), (8, 0.75 * _U)],
    "nq": [(0, 1.0), (8, -_U / 4)],
    "ntiny": [(0, 1.0), (8, -(2.0**-40))],
    "cancel": [(0, 1.0), (8, -1.0), (9, 2.0**-30)],
    "cancel_rev": [(0, 1.0), (8, 2.0**-30), (9, -1.0)],
    "neg_q8": [(0, -1.0)] + [(8 + i, -_U / 4) for i in range(8)],
    "neg_nq": [(0, -1.0), (8, _U / 4)],
    "tie": [(0, 1.0), (8, _U / 2)],
    "tie_odd": [(0, 1.0), (1, _U), (8, _U / 2)],
    "acc_small": [(0, _U / 2), (8, 1.0), (9, _U / 4)],
    "k4_vs_k8": [(0, 1.0), (1, _U / 2), (5, _U / 2)],
    "k8_vs_k16": [(0, 1.0), (4, _U / 2), (8, _U / 2)],
    "k16_vs_k32": [(0, 1.0), (8, _U / 2), (16, _U / 2)],
    "ak4_vs_ak8": [(i, _U / 4) for i in range(4)] + [(4, 1.0)],
    "ak8_vs_ak16": [(i, _U / 4) for i in range(8)] + [(8, 1.0)],
}
# The outputs measured on GB300 (sm_103) hardware, identical in all 9 sources: the cuBLAS and the
# Triton TF32 kernels of p, g_xn and g_W.
_CRAFTED_GB300 = {
    "acc_small": "0x1.0000000000000p+0",
    "ak4_vs_ak8": "0x1.0000020000000p+0",
    "ak8_vs_ak16": "0x1.0000040000000p+0",
    "cancel": "0x0.0p+0",
    "cancel_rev": "0x0.0p+0",
    "e8": "0x1.0000000000000p+0",
    "k16_vs_k32": "0x1.0000000000000p+0",
    "k4_vs_k8": "0x1.0000020000000p+0",
    "k8_vs_k16": "0x1.0000000000000p+0",
    "neg_nq": "-0x1.fffffe0000000p-1",
    "neg_q8": "-0x1.0000040000000p+0",
    "nq": "0x1.fffffe0000000p-1",
    "ntiny": "0x1.0000000000000p+0",
    "q3": "0x1.0000000000000p+0",
    "q8": "0x1.0000040000000p+0",
    "t1": "0x1.0000000000000p+0",
    "tie": "0x1.0000000000000p+0",
    "tie_odd": "0x1.0000020000000p+0",
}


def _crafted_matches(model):
    """The names of the crafted cases whose output under model equals the GB300 output."""
    a = torch.zeros(len(_CRAFTED_CASES), 17, dtype=torch.float64)
    for row, entries in enumerate(_CRAFTED_CASES.values()):
        for position, value in entries:
            a[row, position] = value
    out = accumulate(a, torch.ones(1, 17, dtype=torch.float64), model, single_chain(17))[:, 0]
    # Compared as hex strings, which keep the sign of zero: GB300 gave +0 for cancel and cancel_rev.
    return {
        name
        for name, value in zip(_CRAFTED_CASES, out.tolist())
        if value.hex() == _CRAFTED_GB300[name]
    }


def test_crafted_cases_match_gb300_hardware():
    """ALIGN_K8_X2 reproduces all 18 crafted outputs that GB300's TF32 MMAs produced."""
    assert _crafted_matches(ALIGN_K8_X2) == set(_CRAFTED_CASES)


# The neighbours of ALIGN_K8_X2; the counts are the cases each reproduces.
_CRAFTED_NEIGHBOURS = {
    "align_k8_x0": R01_ALIGN_K8_X0,  # r01's member: 12 of 18
    "align_k8_x1": AccumModel("align", 8, 1),  # 13
    "align_k8_x3": AccumModel("align", 8, 3),  # 17 (fails e8)
    "align_k4_x2": AccumModel("align", 4, 2),  # fails k4_vs_k8
    "align_k16_x2": AccumModel("align", 16, 2),  # fails k8_vs_k16
    "rz_k8": RZ_K8,  # 14
    "ieee_rn": IEEE_RN,  # 10
}


@pytest.mark.parametrize("neighbour", list(_CRAFTED_NEIGHBOURS))
def test_crafted_neighbours_fail(neighbour):
    """Every neighbour of ALIGN_K8_X2 misses at least one crafted case: the cases pin the member."""
    matches = _crafted_matches(_CRAFTED_NEIGHBOURS[neighbour])
    assert len(matches) < len(_CRAFTED_CASES), f"{neighbour} reproduces every crafted case"


def _floor_log2(value):
    """e with 2**e <= value < 2**(e + 1), for a positive Fraction."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - 1 if Fraction(2) ** exponent > value else exponent


_FP32_MAX = Fraction(2**128 - 2**104)


def _round_fp32(value, mode):
    """A Fraction rounded to fp32 exactly: to nearest-even ("rne") or toward zero ("rz")."""
    if value == 0:
        return Fraction(0)
    magnitude = abs(value)
    quantum = Fraction(2) ** (max(_floor_log2(magnitude), -126) - 23)  # subnormals: 2**-149
    units, rest = divmod(magnitude, quantum)
    if mode == "rne" and (rest > quantum / 2 or (rest == quantum / 2 and units % 2 == 1)):
        units += 1
    if units * quantum > _FP32_MAX:
        raise OverflowError("the rational oracle covers the finite fp32 range only")
    return units * quantum if value > 0 else -units * quantum


def _align_grid(addends, extra_bits):
    # ulp32 of the largest addend, times 2**-extra_bits
    return Fraction(2) ** (_floor_log2(max(abs(a) for a in addends)) - 23 - extra_bits)


def _oracle_chain(products, model):
    """One chain of model over exact products, in rational arithmetic with exact fp32 roundings."""
    if model.kind == "exact":
        return sum(products, Fraction(0))
    acc = Fraction(0)
    if model.kind == "rn":
        total = Fraction(0)
        for position, product in enumerate(products, start=1):
            acc = _round_fp32(acc + product, "rne")
            if model.rn_block and (position % model.rn_block == 0 or position == len(products)):
                total, acc = _round_fp32(total + acc, "rne"), Fraction(0)
        return total if model.rn_block else acc
    for k0 in range(0, len(products), model.k_step):
        addends = [acc] + list(products[k0 : k0 + model.k_step])
        if model.kind in ("rz", "rnk"):
            acc = _round_fp32(sum(addends), "rz" if model.kind == "rz" else "rne")
        elif any(addends):  # align
            grid = _align_grid(addends, model.align_extra_bits)
            acc = _round_fp32(sum(math.trunc(a / grid) * grid for a in addends), "rz")
    return acc


def _oracle_accumulate(a, b, model, plan):
    """accumulate's specification: one chain per split, then fp32 RN adds of the partials."""
    partials = [
        _oracle_chain([a[k] * b[k] for k in range(start, stop)], model)
        for start, stop in plan.splits
    ]
    if model.kind == "exact":
        return sum(partials, Fraction(0))
    total = Fraction(0)
    for partial in partials:
        total = _round_fp32(total + partial, "rne")
    return total


def _oracle_dot_products(count=200, seed=20261101):
    """Dot products of length <= 64 at mixed scales (operands randn * 2**[-6, 6]), with 1-4
    splits (some empty), cycling through the operand modes and k-steps 4, 8 and 16.

    rz and rnk round a float64 k-step sum, which is inexact for raw operands and for TF32 addends
    that span more than about 29 binades. That rounding changes the fp32 result only next to an
    fp32 boundary; these seeded data never meet one, or the oracle test would fail.
    """
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        length = rng.randint(1, 64)
        a, b = (
            [_fl32(rng.gauss(0.0, 1.0) * 2.0 ** rng.randint(-6, 6)) for _ in range(length)]
            for _ in range(2)
        )
        bounds = [0] + sorted(rng.randint(0, length) for _ in range(rng.randint(0, 3))) + [length]
        plan = ChainPlan(length, tuple(zip(bounds[:-1], bounds[1:])))
        cases.append((a, b, OPERAND_MODES[index % 4], (4, 8, 16)[index % 3], plan))
    return cases


_ORACLE_MODELS = {
    "exact": EXACT,
    "rn": IEEE_RN,
    "rn_block16": AccumModel("rn", rn_block=16),
    "rnk": AccumModel("rnk"),
    "rz": RZ_K8,
    "align_x0": R01_ALIGN_K8_X0,
    "align_x2": ALIGN_K8_X2,
}


@pytest.mark.parametrize("family", list(_ORACLE_MODELS))
def test_accumulate_matches_exact_rational_oracle(family):
    """accumulate equals a rational-arithmetic oracle bitwise on 200 random dot products.

    TF32 and raw operands, k-steps 4, 8 and 16, random split plans. The exact model sums in
    float64, so it is held to the Neumaier bound instead: 2**-50 of the sum of |products|.
    """
    mismatches = []
    for a, b, mode, k_step, plan in _oracle_dot_products():
        model = dataclasses.replace(_ORACLE_MODELS[family], k_step=k_step)
        a64 = round_operand(torch.tensor([a], dtype=torch.float32), mode)
        b64 = round_operand(torch.tensor([b], dtype=torch.float32), mode)
        got = Fraction(accumulate(a64, b64, model, plan)[0, 0].item())
        exact_a, exact_b = (list(map(Fraction, t[0].tolist())) for t in (a64, b64))
        want = _oracle_accumulate(exact_a, exact_b, model, plan)
        if model.kind == "exact":
            bound = Fraction(2) ** -50 * sum(abs(u * v) for u, v in zip(exact_a, exact_b))
            failed = abs(got - want) > bound
        else:
            failed = got != want
        if failed:
            mismatches.append((mode, k_step, plan.splits, float(got), float(want)))
    assert not mismatches, f"{len(mismatches)} mismatches, e.g. {mismatches[:3]}"


def _nearest_even_twin(values, extra_bits=2):
    """align's twin that rounds each addend to nearest-even on the grid instead of truncating it."""
    addends = [Fraction(v) for v in values]
    grid = _align_grid(addends, extra_bits)
    return float(_round_fp32(sum(round(a / grid) * grid for a in addends), "rz"))


def test_align_truncates_addends_not_nearest_even():
    """One k-step that separates truncated addends from addends rounded to nearest-even.

    The 18 crafted cases cannot: the GB300 measurement settles truncation on random data only.
    q = 3 * 2**-27 is 0.75 of a grid unit at max = 1 (grid 2**-25), so truncation drops each q,
    while the twin rounds four of them up to 1 + 4 * 2**-25 = 1 + 2**-23.
    """
    q = 3 * 2.0**-27
    for sign in (1.0, -1.0):
        values = [sign * v for v in (1.0, q, q, q, q, 0.0, 0.0, 0.0)]
        a = torch.tensor([values], dtype=torch.float64)
        out = accumulate(a, torch.ones_like(a), ALIGN_K8_X2, single_chain(8))
        assert float(out[0, 0]) == sign * float.fromhex("0x1.0000000000000p+0")
        assert _nearest_even_twin(values) == sign * float.fromhex("0x1.0000020000000p+0")


# shape -> (plan function, length, the kernel's split lengths)
_PLAN_SHAPES = {
    "K16384": (gp2_plan, 16384, [1024] * 16),
    "K4096": (gp2_plan, 4096, [256] * 16),
    "K1024": (gp2_plan, 1024, [64] * 16),
    "K384": (gp2_plan, 384, [64] * 6 + [0] * 10),
    "T4096": (gp4_wgrad_plan, 4096, [2048, 2048]),
    "T4095": (gp4_wgrad_plan, 4095, [2048, 2047]),
    "T112": (gp4_wgrad_plan, 112, [64, 48]),
    "T17": (gp4_wgrad_plan, 17, [17, 0]),
    "T21": (gp4_wgrad_plan, 21, [21, 0]),
}


@pytest.mark.parametrize("shape", list(_PLAN_SHAPES))
def test_kernel_split_plans_follow_p2_rules(shape):
    """The plans are the fused kernels' splits: per = cdiv(cdiv(L, n), 64) * 64, masked at L.

    GP-2 splits K 16 ways and GP-4's wgrad splits T 2 ways, as the host wrappers of the prototype
    kernels (P2) do. Once the kernel module exists it exports these plans and its tests assert that
    they agree.
    """
    plan_function, length, split_lengths = _PLAN_SHAPES[shape]
    plan = plan_function(length)
    assert [stop - start for start, stop in plan.splits] == split_lengths
    n_split = 16 if plan_function is gp2_plan else 2
    per = math.ceil(math.ceil(length / n_split) / 64) * 64
    assert plan.splits == tuple(
        (min(i * per, length), min((i + 1) * per, length)) for i in range(n_split)
    )


# The shipped kernels' plans, pinned to the values the M3 registration recorded for its 15 shapes (T
# in {4096, 4095, 112, 21, 17}, K in {16384, 4096, 384}) and its fixture shapes.
# K -> gp2_plan_nonempty(K)'s chain lengths: the K values of both.
_NONEMPTY_PLAN_SHAPES = {
    "K16384": (16384, [1216] * 13 + [576]),
    "K4096": (4096, [320] * 12 + [256]),
    "K512": (512, [64] * 8),
    "K384": (384, [64] * 6),
    "K128": (128, [64] * 2),
}
# K -> the empty splits of gp2_plan(K, 14), which gp2_plan_nonempty drops: 14 minus its chains. The
# registration states 1 at K = 4096 and 8 at K = 384; 0, 6 and 12 follow from the chain counts
# above.
_EMPTY_SPLITS_DROPPED = {16384: 0, 4096: 1, 512: 6, 384: 8, 128: 12}


@pytest.mark.parametrize("shape", list(_NONEMPTY_PLAN_SHAPES))
def test_gp2_plan_nonempty_pins(shape):
    """GP-2's shipped chains: k = cdiv(cdiv(K, n), 64) * 64 with n = min(14, cdiv(K, 64)), chains
    [i * k, min((i + 1) * k, K)) for i < cdiv(K, k), none empty; gp2_plan(K, 14) has the same
    non-empty chains, followed by its empty splits (none at K = 16384, 1 at 4096, 6 at 512, 8 at
    384 and 12 at 128)."""
    K, chain_lengths = _NONEMPTY_PLAN_SHAPES[shape]
    plan = gp2_plan_nonempty(K)
    assert [stop - start for start, stop in plan.splits] == chain_lengths
    n = min(14, math.ceil(K / 64))
    per = math.ceil(math.ceil(K / n) / 64) * 64
    assert plan.splits == tuple((i * per, min((i + 1) * per, K)) for i in range(math.ceil(K / per)))
    p2_form = gp2_plan(K, 14)
    assert tuple((start, stop) for start, stop in p2_form.splits if stop > start) == plan.splits
    empty = sum(stop == start for start, stop in p2_form.splits)
    assert empty == _EMPTY_SPLITS_DROPPED[K] == 14 - len(chain_lengths)


# (T, K) -> (layouts, pieces per column tile (fewest, most), tokens per piece (fewest, most), NPROG,
# pieces in all or None where not recorded): gp4_wgrad_column_plans at the 15 M3 shapes and the
# fixture shapes. The registration derived all of them from the round-2 kernel's partition rule;
# at the 15 M3 shapes it checked them against round 2's recorded per-shape splits, while the 7
# fixture shapes were never run on a GPU (it records their layouts and pieces per tile; their
# tokens per piece and NPROG = min(304, NU) follow from the same rule).
_COLUMN_PLAN_SHAPES = {
    "T4096_K16384": (4096, 16384, 8, (3, 4), (192, 1728), 304, 416),
    "T4096_K4096": (4096, 4096, 2, (10, 10), (192, 448), 304, 320),
    "T4096_K384": (4096, 384, 1, (64, 64), (64, 64), 192, 192),
    "T4095_K16384": (4095, 16384, 8, (3, 4), (192, 1728), 304, 416),
    "T4095_K4096": (4095, 4096, 2, (10, 10), (192, 448), 304, 320),
    "T4095_K384": (4095, 384, 1, (64, 64), (63, 64), 192, 192),
    "T112_K16384": (112, 16384, 1, (2, 2), (48, 64), 256, 256),
    "T112_K4096": (112, 4096, 1, (2, 2), (48, 64), 64, 64),
    "T112_K384": (112, 384, 1, (2, 2), (48, 64), 6, 6),
    "T21_K16384": (21, 16384, 1, (1, 1), (21, 21), 128, 128),
    "T21_K4096": (21, 4096, 1, (1, 1), (21, 21), 32, 32),
    "T21_K384": (21, 384, 1, (1, 1), (21, 21), 3, 3),
    "T17_K16384": (17, 16384, 1, (1, 1), (17, 17), 128, 128),
    "T17_K4096": (17, 4096, 1, (1, 1), (17, 17), 32, 32),
    "T17_K384": (17, 384, 1, (1, 1), (17, 17), 3, 3),
    "T520_K16384": (520, 16384, 4, (3, 3), (8, 256), 304, None),
    "T256_K512": (256, 512, 1, (4, 4), (64, 64), 16, None),
    "T64_K16384": (64, 16384, 1, (1, 1), (64, 64), 128, None),
    "T64_K384": (64, 384, 1, (1, 1), (64, 64), 3, None),
    "T16_K16384": (16, 16384, 1, (1, 1), (16, 16), 128, None),
    "T16_K128": (16, 128, 1, (1, 1), (16, 16), 1, None),
    "T8_K512": (8, 512, 1, (1, 1), (8, 8), 4, None),
}


def _tile_pieces(plans, block_c=128):
    """The token pieces (splits) of every block_c-column tile, in column order."""
    tiles = {}
    for ranges, plan in plans.groups:
        for start, stop in ranges:
            assert start % block_c == 0 and (stop % block_c == 0 or stop == plans.width)
            for column in range(start, stop, block_c):
                tiles[column // block_c] = plan.splits
    return [tiles[tile] for tile in range(len(tiles))]


@pytest.mark.parametrize("shape", list(_COLUMN_PLAN_SHAPES))
def test_gp4_wgrad_column_plans_pins(shape):
    """GP-4's shipped wgrad pieces: per 128-column tile, the runs of the static partition of the
    NU = cdiv(T, 64) x cdiv(K, 128) column-major units over NPROG = min(304, NU) programs, in
    program order. Layouts, pieces and their lengths are the recorded values; the pieces start
    exactly at the programs' first units and at the tiles' first units, which pins NPROG wherever a
    tile has more than one token block (with one block per tile every unit starts a tile, and the
    pieces do not depend on NPROG). glm_pair_fused_plans is gp2_plan_nonempty, one dgrad chain over
    N and these pieces."""
    T, K, layouts, pieces, tokens, nprog, npieces = _COLUMN_PLAN_SHAPES[shape]
    plans = gp4_wgrad_column_plans(T, K)
    assert isinstance(plans, ColumnChainPlans) and (plans.length, plans.width) == (T, K)
    tiles = _tile_pieces(plans)
    assert len(tiles) == math.ceil(K / 128)
    assert len(plans.groups) == len(set(tiles)) == layouts
    assert (min(map(len, tiles)), max(map(len, tiles))) == pieces
    lengths = [stop - start for splits in tiles for start, stop in splits]
    assert (min(lengths), max(lengths)) == tokens
    assert npieces is None or len(lengths) == npieces
    n_blocks = math.ceil(T / 64)
    n_units = n_blocks * len(tiles)
    assert all(start % 64 == 0 for splits in tiles for start, _ in splits)
    piece_starts = {
        tile * n_blocks + start // 64 for tile, splits in enumerate(tiles) for start, _ in splits
    }
    program_starts = {q * n_units // nprog for q in range(nprog)}
    assert piece_starts == program_starts | {tile * n_blocks for tile in range(len(tiles))}
    assert glm_pair_fused_plans(T, K) == GLMPairPlans(
        proj=gp2_plan_nonempty(K), dgrad=single_chain(_N), wgrad=plans
    )
    assert PAIR_N == _N


# The 8 layouts at T = 4096, K = 16384 (piece boundaries; tile c takes layout c mod 8), checked by
# hand against the round-2 kernel's partition: tile 0 takes programs 0-2, whose first units are 0,
# 26 and 53 of 64 per tile (8192 units over 304 programs: 26.95 units per program).
_PRODUCTION_LAYOUTS = (
    (0, 1664, 3392, 4096),
    (0, 1024, 2752, 4096),
    (0, 384, 2112, 3840, 4096),
    (0, 1472, 3200, 4096),
    (0, 832, 2560, 4096),
    (0, 192, 1920, 3648, 4096),
    (0, 1280, 3008, 4096),
    (0, 640, 2368, 4096),
)


def test_gp4_wgrad_column_plans_production_layouts():
    """At production every 8th tile shares a layout: 8 groups of 16 tiles in the order of their
    first tile, with the pieces above. T = 4095 has the same partition, its last piece one token
    shorter."""
    for T in (4096, 4095):
        plans = gp4_wgrad_column_plans(T, 16384)
        assert len(plans.groups) == len(_PRODUCTION_LAYOUTS)
        for layout, ((ranges, plan), bounds) in enumerate(zip(plans.groups, _PRODUCTION_LAYOUTS)):
            assert ranges == tuple((c * 128, (c + 1) * 128) for c in range(layout, 128, 8))
            assert plan.splits == tuple(zip(bounds[:-1], bounds[1:-1] + (T,)))


def test_column_chain_plans_reject_inconsistent_groups():
    """ColumnChainPlans' columns must cover [0, width) once and its plans share one length; the
    kernel model takes one for wgrad only, and one that covers K."""
    pieces = ChainPlan(64, ((0, 32), (32, 64)))
    plans = ColumnChainPlans(
        64, ((((0, 128), (256, 384)), pieces), (((128, 256),), single_chain(64)))
    )
    assert plans.width == 384
    for groups in (
        ((((0, 128), (192, 384)), pieces), (((128, 256),), single_chain(64))),  # overlap
        ((((0, 128), (256, 384)), pieces),),  # gap
        ((((0, 128),), pieces), (((128, 256),), single_chain(32))),  # another length
        ((((0, 128), (128, 128)), pieces),),  # empty range
    ):
        with pytest.raises(ValueError):
            ColumnChainPlans(64, groups)
    x, W, G_p = _pair_data(64, 384, seed=9)
    fused = glm_pair_fused_plans(64, 384)
    for wrong in (  # column plans that would otherwise run, for p and for g_xn
        dataclasses.replace(fused, proj=ColumnChainPlans(384, ((((0, _N),), fused.proj),))),
        dataclasses.replace(fused, dgrad=ColumnChainPlans(_N, ((((0, 384),), fused.dgrad),))),
    ):
        with pytest.raises(TypeError):
            glm_pair_kernel_model(x, W, G_p, plans=wrong)
    narrow = dataclasses.replace(fused, wgrad=gp4_wgrad_column_plans(64, 256))
    with pytest.raises(ValueError):
        glm_pair_kernel_model(x, W, G_p, plans=narrow)


def _pad_splits(a64, bt64, plan):
    """Pad each split with zero products after its last real k-step, to one length."""
    width = max(stop - start for start, stop in plan.splits)

    def pad(t):
        parts = [t[..., start:stop] for start, stop in plan.splits]
        return torch.cat([F.pad(part, (0, width - part.shape[-1])) for part in parts], dim=-1)

    return pad(a64), pad(bt64), ChainPlan.equal(width * len(plan.splits), len(plan.splits))


def test_split_loop_equals_vectorized_form():
    """The per-split loop equals the vectorized form bitwise, for every model.

    For equal splits directly, and in row chunks of one row; for unequal and empty splits
    through operands padded with zero products after each split's last real k-step.
    """
    g = torch.Generator().manual_seed(10)
    a = round_operand(torch.randn(6, 384, generator=g), "rne")
    bt = round_operand(torch.randn(5, 384, generator=g) * 0.05, "rne")
    models = (
        EXACT,
        IEEE_RN,
        AccumModel("rn", rn_block=16),
        AccumModel("rnk"),
        RZ_K8,
        AccumModel("rz", 4),
        ALIGN_K8_X2,
        R01_ALIGN_K8_X0,
        AccumModel("align", 16, 2),
    )
    unequal_plans = (
        gp2_plan(384),
        gp4_wgrad_plan(112),
        gp4_wgrad_plan(17),
        ChainPlan(301, ((0, 3), (3, 3), (3, 200), (200, 301))),
    )
    for model in models:
        for n in (1, 4, 16):
            plan = ChainPlan.equal(384, n)
            vectorized = accumulate(a, bt, model, plan)
            loop = _accumulate_split_loop(a, bt, model, plan, DEFAULT_MAX_CHUNK_ELEMS)
            assert torch.equal(loop, vectorized), (model, n)
            assert torch.equal(accumulate(a, bt, model, plan, max_chunk_elems=1), vectorized)
        for plan in unequal_plans:
            a_cut, bt_cut = a[..., : plan.length], bt[..., : plan.length]
            a_padded, bt_padded, padded_plan = _pad_splits(a_cut, bt_cut, plan)
            padded = accumulate(a_padded, bt_padded, model, padded_plan)
            assert torch.equal(accumulate(a_cut, bt_cut, model, plan), padded), (model, plan)


# shape -> (T, K): GP-2's empty splits at K = 384, GP-4's unequal or empty wgrad splits
_UNEQUAL_SHAPES = {
    "K384": (64, 384),
    "T112": (112, 256),
    "T4095": (4095, 128),
    "T17": (17, 256),
    "T21": (21, 256),
}


@pytest.mark.parametrize("shape", list(_UNEQUAL_SHAPES))
def test_unequal_and_empty_splits_run(shape):
    """The kernel model runs on the kernels' unequal and empty splits, where equal splits are wrong
    or impossible, and each split is its own chain added in index order.

    It also pins the wiring that the power checks cannot see, since a mutation and its baseline
    share it: the default plans, xn = fl32(xf * s) (O1) and g_xn as one chain over N (O5).
    """
    T, K = _UNEQUAL_SHAPES[shape]
    x, W, G_p = _pair_data(T, K, seed=9)
    plans = GLMPairPlans(proj=gp2_plan(K), dgrad=single_chain(_N), wgrad=gp4_wgrad_plan(T))
    assert glm_pair_kernel_plans(T, K) == plans
    result = glm_pair_kernel_model(x, W, G_p)
    # The float64 product of a bf16 and an fp32 value is exact, so .float() rounds it once.
    assert torch.equal(result.xn, (x.double() * result.s.double()[:, None]).float())
    g_xn = emulate(G_p, W.t().contiguous(), ALIGN_K8_X2, "rne", single_chain(_N))
    assert torch.equal(result.g_xn, g_xn)
    for out, a, bt, plan in (
        (result.p, result.xn, W, gp2_plan(K)),
        (result.g_W, G_p.t(), result.xn.t(), gp4_wgrad_plan(T)),
    ):
        assert len({stop - start for start, stop in plan.splits}) > 1  # not an equal split
        total = torch.zeros_like(out)
        for start, stop in plan.splits:
            a_split, bt_split = a[:, start:stop].contiguous(), bt[:, start:stop].contiguous()
            total = total + emulate(
                a_split, bt_split, ALIGN_K8_X2, "rne", single_chain(stop - start)
            )
        assert torch.equal(out, total)
    if T % 2:
        with pytest.raises(ValueError):
            ChainPlan.equal(T, 2)
    reference = glm_pair_reference_f64(x, W, G_p)
    for name in ("p", "g_xn", "g_W"):
        assert rel_rms(getattr(result, name), getattr(reference, name)) <= 1e-3, name


def _wgrad_operands(T, K):
    """The rounded wgrad operands T(G_p^T) [24, T] and T(xn^T) [K, T] of pair_data (seed 0)."""
    x, _, G_p = _pair_data(T, K)
    xn = x.float() * gp1_s_model(x)[:, None]
    return round_operand(G_p.t().contiguous(), "rne"), round_operand(xn.t().contiguous(), "rne")


def _per_piece_wgrad(a64, bt64, plans, drop_last=False):
    """The definition of the per-column wgrad: for every 128-column tile, accumulate over each of
    its pieces as one chain, then ordered_reduce of the piece partials in index order from +0.0."""
    out = torch.empty(a64.shape[0], bt64.shape[0], dtype=torch.float32)
    for tile, pieces in enumerate(_tile_pieces(plans)):
        columns = slice(tile * 128, min((tile + 1) * 128, bt64.shape[0]))
        partials = []
        for index, (start, stop) in enumerate(pieces):
            a_piece = a64[:, start:stop].contiguous()
            if drop_last and index == len(pieces) - 1:
                a_piece = torch.zeros_like(a_piece)
            b_piece = bt64[columns, start:stop].contiguous()
            partials.append(accumulate(a_piece, b_piece, ALIGN_K8_X2, single_chain(stop - start)))
        out[:, columns] = ordered_reduce(torch.stack(partials))
    return out


@pytest.fixture(scope="module")
def production_wgrad():
    """g_W at T = 4096, K = 16384 (pair_data seed 0) on the shipped per-column plan ("fused") and as
    one chain over T per column ("one_chain"), computed once for this module's tests."""
    cache = {}

    def emulation(kind):
        if kind not in cache:
            a64, bt64 = _wgrad_operands(4096, 16384)
            if kind == "fused":
                plans = glm_pair_fused_plans(4096, 16384).wgrad
                cache[kind] = _accumulate_columns(a64, bt64, ALIGN_K8_X2, plans)
            else:
                cache[kind] = accumulate(a64, bt64, ALIGN_K8_X2, single_chain(4096))
        return cache[kind]

    yield emulation
    cache.clear()


def _sha256_16(tensor):
    """The first 16 hex digits of the sha256 of a tensor's bytes."""
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


# pair_data's draws (seed 0, randn, scale 1) at the shapes whose outputs this module pins: sha256_16
# of x, W and G_p as the M3 registration drew them (CPU torch 2.15.0 on x86_64, whose seeded randn
# takes the AVX2/AVX512 kernel). Seeded CPU randn gives other bytes on another kernel (the scalar
# one, ATEN_CPU_CAPABILITY=default, or aarch64's, as in the GB300 image: M3's D-job D2), and outputs
# pinned from one set of draws say nothing about another. So the per-column digests and the
# registered S8 effect below are compared only where pair_data reproduces these draws; the other
# checks of those tests run on every platform.
_REGISTERED_DRAWS = {
    (4096, 16384): ("156452d814073b60", "4071d7691a94ff0b", "d8c76c231d62ac7a"),
    (520, 16384): ("ebf07aab2f50406b", "e73cfc1449dd477a", "5cc5ced63e493f9a"),
    (112, 384): ("5a2e2828fc5896fa", "9b4c0cacd360a811", "339d45f86ce8824c"),
}


@functools.lru_cache(maxsize=None)
def _draw_digests(T, K):
    """sha256_16 of x, W and G_p of pair_data(T, K) (seed 0, randn, scale 1)."""
    return tuple(_sha256_16(tensor) for tensor in _pair_data(T, K))


def _compare_pinned(row, T, K, values):
    """Whether pair_data(T, K) reproduces the registration's draws, so that the outputs pinned from
    them are compared here. Prints the draws, this platform's values and the answer either way."""
    draws = _draw_digests(T, K)
    compared = draws == _REGISTERED_DRAWS[(T, K)]
    print(
        f"GLM_MHC_EMULATOR_PINNED row={row} draws={'/'.join(draws)} values={'/'.join(values)} "
        f"compared={'yes' if compared else 'no'}"
    )
    return compared


# shape -> (T, K, layouts, digest of g_W, digest of g_W under drop_kernel_split or None): production
# (8 layouts of 3 or 4 unequal pieces), T = 520 (4 layouts of 3 pieces in every tile) and T = 112
# (one layout of two unequal pieces, [0, 64) and [64, 112)). The digests (sha256_16 of the fp32
# g_W) are the M3 registration's, from its own per-column wgrad code on pair_data seed 0: its
# production baseline and its per-column checks at T = 520 and 112. drop_kernel_split runs off
# production only. They are compared where pair_data reproduces the registration's draws
# (_REGISTERED_DRAWS).
_PER_COLUMN_SHAPES = {
    "T4096_K16384": (4096, 16384, 8, "bc9294de06c6db33", None),
    "T520_K16384": (520, 16384, 4, "9d1aa0ab55a0a68c", "11b06695286a9c21"),
    "T112_K384": (112, 384, 1, "96318e856bcc16d0", "5cfe805ae3ad0f7e"),
}


@pytest.mark.parametrize("shape", list(_PER_COLUMN_SHAPES))
def test_per_column_wgrad_equals_accumulate_per_piece(shape, request, production_wgrad):
    """The per-column path (_accumulate_columns, which _contract runs for g_W) equals, for every
    128-column tile, accumulate over each piece plus ordered_reduce of the tile's partials in piece
    order, bitwise: each column group runs on its own pieces. Under drop_last_split it equals the
    same with the last piece of every tile zeroed. Where pair_data reproduces the registration's
    draws, both equal the registered digests."""
    T, K, layouts, digest, dropped_digest = _PER_COLUMN_SHAPES[shape]
    if T >= 4096:
        request.getfixturevalue("heavy_cpu_case")
    plans = glm_pair_fused_plans(T, K).wgrad
    assert len(plans.groups) == layouts
    assert len({stop - start for _, plan in plans.groups for start, stop in plan.splits}) > 1
    a64, bt64 = _wgrad_operands(T, K)
    if T >= 4096:
        per_column = production_wgrad("fused")
    else:
        per_column = _accumulate_columns(a64, bt64, ALIGN_K8_X2, plans)
    assert torch.equal(per_column, _per_piece_wgrad(a64, bt64, plans))
    registered, measured = [digest], [_sha256_16(per_column)]
    if dropped_digest is not None:
        dropped = _accumulate_columns(a64, bt64, ALIGN_K8_X2, plans, drop_last_split=True)
        assert torch.equal(dropped, _per_piece_wgrad(a64, bt64, plans, drop_last=True))
        assert not torch.equal(dropped, per_column)
        registered.append(dropped_digest)
        measured.append(_sha256_16(dropped))
    if _compare_pinned(f"per_column/{shape}", T, K, measured):
        assert measured == registered


def test_kernel_model_runs_the_shipped_plans():
    """glm_pair_kernel_model on glm_pair_fused_plans at (112, 384): g_W is the per-column path's,
    drop_kernel_split zeroes the last piece of every column tile and ignore_kernel_split runs one
    chain over T for every column (both bitwise against accumulate per piece), and p is the ordered
    reduction of GP-2's non-empty chains."""
    T, K = 112, 384
    x, W, G_p = _pair_data(T, K)
    plans = glm_pair_fused_plans(T, K)
    a64, bt64 = _wgrad_operands(T, K)
    result = glm_pair_kernel_model(x, W, G_p, plans=plans)
    assert torch.equal(result.xn, x.float() * gp1_s_model(x)[:, None])
    assert torch.equal(result.g_W, _per_piece_wgrad(a64, bt64, plans.wgrad))
    dropped = glm_pair_kernel_model(x, W, G_p, plans=plans, mutation="drop_kernel_split")
    assert torch.equal(dropped.g_W, _per_piece_wgrad(a64, bt64, plans.wgrad, drop_last=True))
    assert not torch.equal(dropped.g_W, result.g_W)
    ignored = glm_pair_kernel_model(x, W, G_p, plans=plans, mutation="ignore_kernel_split")
    assert torch.equal(ignored.g_W, accumulate(a64, bt64, ALIGN_K8_X2, single_chain(T)))
    assert not torch.equal(ignored.g_W, result.g_W)
    xn64, w64 = round_operand(result.xn, "rne"), round_operand(W, "rne")
    chains = [
        accumulate(xn64[:, start:stop], w64[:, start:stop], ALIGN_K8_X2, single_chain(stop - start))
        for start, stop in gp2_plan_nonempty(K).splits
    ]
    assert torch.equal(result.p, ordered_reduce(torch.stack(chains)))


@pytest.mark.parametrize("slot", OPERAND_SLOTS)
def test_operand_slots_route_to_their_contraction(slot):
    """A per-slot operand mode rounds only its own operand of its own contraction.

    With slot truncated ("rz") and every other slot rounded to nearest-even, the contraction that
    reads slot equals the emulation with that operand truncated, and the other two keep their "rne"
    result. The mutation dgrad_unrounded_w is exactly this case for W_dgrad.
    """
    T, K = 17, 384
    x, W, G_p = _pair_data(T, K, seed=14)
    modes = dict.fromkeys(OPERAND_SLOTS, "rne")
    modes[slot] = "rz"
    result = glm_pair_kernel_model(x, W, G_p, operand_mode=modes)
    baseline = glm_pair_kernel_model(x, W, G_p)
    # tensor -> ((slot, operand) of A, (slot, operand) of Bt), plan
    contractions = {
        "p": ((("xn", result.xn), ("W_fwd", W)), gp2_plan(K)),
        "g_xn": ((("G_p_dgrad", G_p), ("W_dgrad", W.t())), single_chain(_N)),
        "g_W": ((("G_p_wgrad", G_p.t()), ("xn_wgrad", result.xn.t())), gp4_wgrad_plan(T)),
    }
    for name, (operands, plan) in contractions.items():
        out, rne = getattr(result, name), getattr(baseline, name)
        if slot in dict(operands):
            a64, bt64 = (round_operand(t.contiguous(), modes[s]) for s, t in operands)
            assert torch.equal(out, accumulate(a64, bt64, ALIGN_K8_X2, plan)), name
            assert not torch.equal(out, rne), name
        else:
            assert torch.equal(out, rne), name
    if slot == "W_dgrad":
        mutated = glm_pair_kernel_model(x, W, G_p, mutation="dgrad_unrounded_w")
        for name in ("p", "g_xn", "g_W"):
            assert torch.equal(getattr(mutated, name), getattr(result, name)), name


def _bf16_rne_bits(value):
    bits = _fp32_bits(value)
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def _epilogue_replay(result, x, G_p, G_s, inv_k):
    """GP-4's fp32 epilogue (O6-O8) replayed with Python floats, one fp32 rounding per operation."""
    xs, ss = x.float().tolist(), result.s.tolist()
    g_xn = result.g_xn.tolist() if G_p is not None else [[0.0] * len(xs[0])] * len(xs)
    g_s_rows, g_x32_rows = [], []
    for t, s in enumerate(ss):
        if G_p is not None:
            acc = 0.0
            for g, p in zip(G_p[t].tolist(), result.p[t].tolist()):
                acc = _fl32(acc + _fl32(g * p))  # sequential, in index order
            g_s = _fl32(acc / s)
            g_s = _fl32(g_s + G_s[t].item()) if G_s is not None else g_s
        else:
            g_s = G_s[t].item()
        coef = _fl32(_fl32(_fl32(_fl32(s * s) * s) * inv_k) * g_s)
        g_s_rows.append(g_s)
        g_x32_rows.append([_fl32(_fl32(s * d) - _fl32(coef * v)) for d, v in zip(g_xn[t], xs[t])])
    return g_s_rows, g_x32_rows


def test_kernel_model_epilogue_order():
    """O6-O9 op by op: g_s summed sequentially, coef left to right with inv_k = fl32(1/384), g_x32
    without FMA, and g_x the bf16 RNE of g_x32."""
    T, K = 17, 384
    x, W, G_p = _pair_data(T, K, seed=5)
    G_s = torch.randn(T, generator=torch.Generator().manual_seed(6))
    result = glm_pair_kernel_model(x, W, G_p, G_s)
    inv_k = _fl32(1.0 / K)
    assert inv_k != 1.0 / K  # 1 / K is an fp32 value only at K = 2**n

    g_s, g_x32 = _epilogue_replay(result, x, G_p, G_s, inv_k)
    assert result.g_s.tolist() == g_s
    assert result.g_x32.tolist() == g_x32
    g_x_bits = (result.g_x.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    assert g_x_bits == [[_bf16_rne_bits(v) for v in row] for row in g_x32]
    # The replay can see the rounding of inv_k: the unrounded 1 / K changes g_x32.
    assert _epilogue_replay(result, x, G_p, G_s, 1.0 / K)[1] != g_x32


def test_glm_pair_without_g_p():
    """Without G_p (the norm-only case) g_xn and g_W are None and g_x comes from G_s alone."""
    T, K = 17, 384
    x, W, G_p = _pair_data(T, K, seed=8)
    G_s = torch.randn(T, generator=torch.Generator().manual_seed(9))
    result = glm_pair_kernel_model(x, W, None, G_s)
    assert result.g_xn is None and result.g_W is None
    assert torch.equal(result.g_s, G_s)
    assert torch.equal(result.p, glm_pair_kernel_model(x, W, G_p).p)
    assert result.g_x32.tolist() == _epilogue_replay(result, x, None, G_s, _fl32(1.0 / K))[1]
    reference = glm_pair_reference_f64(x, W, None, G_s)
    assert reference.g_xn is None and reference.g_W is None
    assert rel_rms(result.g_x32, reference.g_x) <= 1e-6


def _rsqrt_fp32(value):
    """value ** -0.5 for a positive Fraction, rounded once to the nearest fp32 value.

    The fp32 rounding midpoints are multiples of 2**-150, so they are integers at the scale 2**160:
    the integer square root and a sticky bit round as the exact value does.
    """
    scaled = value.denominator << 320
    units = math.isqrt(scaled // value.numerator)
    sticky = units * units * value.numerator != scaled
    return float(_round_fp32(Fraction(2 * units + sticky, 2**161), "rne"))


def _gp1_s_replay(row, K):
    """O11 with exact rationals and one fp32 rounding per step: ss = fl32(the exact sum of the
    squares), ms = fl32(ss * fl32(1 / K)) and s = fl32(fl32(ms + eps) ** -0.5)."""
    ss = _round_fp32(sum(map(Fraction, (v * v for v in row)), Fraction(0)), "rne")
    ms = _fl32(float(ss) * _fl32(1.0 / K))
    return _rsqrt_fp32(Fraction(_fl32(ms + EPS_FP32)))


@pytest.mark.parametrize("K", [384, 16384])
def test_gp1_s_model_rounds_each_step_to_fp32(K):
    """gp1_s_model (O11) equals its formula replayed with exact rationals, at x scales 1e-3, 1 and
    1e3 and on all-zero rows (s = fl32(eps ** -0.5)). inv_k = fl32(1 / 384) is inexact.

    The power checks cannot see a wrong s model: a mutation and its baseline share it.
    """
    T = 64 if K == 384 else 8
    for scale in (1e-3, 1.0, 1e3):
        x, _, _ = _pair_data(T, K, scale, seed=15, rows="zero_rows")
        replay = [_gp1_s_replay(row, K) for row in x.float().tolist()]
        assert gp1_s_model(x).tolist() == replay, scale
        assert replay[0] == 316.227783203125  # row 0 is all zero


# ============================================================================
# Mutation power (a mutation keeps its check only where its emulated effect is >= 10 x the bar of
# the fused kernels' GPU tests, per tensor, at a stated shape and x scale)
# ============================================================================

_POWER_T, _POWER_K = 512, 16384


def _power_record(row, effect, threshold):
    print(
        f"GLM_MHC_EMULATOR_POWER row={row} effect={effect:.3e} threshold={threshold:.3e} "
        f"ratio={effect / threshold:.1f}"
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_sinkhorn_t0_mutation_has_power(scale):
    """Treating t = 0 like the other Sinkhorn iterations moves the VJP by >= 1e-6 at every logit
    scale, 1e4 x the 1e-10 exact bar."""
    logits, grad = _sinkhorn_data(scale)
    effect = rel_rms(
        sinkhorn_vjp(logits, grad, mutation="t0_like_others"), _sinkhorn_autograd(logits, grad)
    )
    _power_record(f"sinkhorn_t0_like_others/x{scale:g}", effect, 1e-6)
    assert effect >= 1e-6


# It runs before the cached T = 512 emulations below (pytest keeps the file order, and the cache
# lives until the module ends), so the two peaks of memory do not add up.
@pytest.mark.usefixtures("heavy_cpu_case")
def test_emulator_split_power_on_g_w_at_production_t():
    """At T = 4096, one 4,096-token chain instead of GP-4's split-M 2 moves g_W by more than the
    threshold (5.3 x, modeled); at T = 512 it does not (0.67 x), so this row runs here only."""
    T = 4096
    x, _, G_p = _pair_data(T, _POWER_K)
    xn = x.float() * gp1_s_model(x)[:, None]
    a, bt = G_p.t().contiguous(), xn.t().contiguous()
    del x, xn
    kernel = emulate(a, bt, ALIGN_K8_X2, "rne", gp4_wgrad_plan(T))
    one_chain = emulate(a, bt, ALIGN_K8_X2, "rne", single_chain(T))
    effect = rel_rms(one_chain, kernel)
    _power_record("ignore_kernel_split/g_W/T4096", effect, P9_POWER_THRESHOLDS["g_W"])
    assert effect >= P9_POWER_THRESHOLDS["g_W"]


# The M3 registration's power table of the shipped plans (computed on CPU before any kernel output,
# with its own per-column wgrad code and plan formulas): its row ignore_kernel_split/g_W at
# T = 4096, K = 16384, pair_data seed 0, scale 1, covered (62.4 x the bar). It is compared where
# pair_data reproduces the registration's draws (_REGISTERED_DRAWS).
_REGISTERED_IGNORE_G_W_T4096 = 6.971007311506578e-06


@pytest.mark.usefixtures("heavy_cpu_case")
def test_fused_split_power_on_g_w_at_production_t(production_wgrad):
    """On the shipped per-column plan at T = 4096, one chain over T per column instead of its pieces
    moves g_W by 6.2 x the threshold: covered, as registered. Where pair_data reproduces the
    registration's draws, the effect is also the registered one to a relative 1e-9 (two float64
    computations of one formula)."""
    effect = rel_rms(production_wgrad("one_chain"), production_wgrad("fused"))
    _power_record("ignore_kernel_split/g_W/T4096/fused", effect, P9_POWER_THRESHOLDS["g_W"])
    if _compare_pinned("ignore_kernel_split/g_W/T4096/fused", 4096, 16384, [repr(effect)]):
        assert abs(effect - _REGISTERED_IGNORE_G_W_T4096) <= 1e-9 * _REGISTERED_IGNORE_G_W_T4096
    assert effect >= P9_POWER_THRESHOLDS["g_W"]


@pytest.fixture(scope="module")
def power_cases():
    """Kernel-model emulations at T = 512, K = 16384, shared by this module's power tests.

    Baseline: "rne" operands, ALIGN_K8_X2 and the kernels' plans, on _pair_data (seed 0).
    """
    cache = {}

    def emulation(scale, mutation=None):
        if (scale, mutation) not in cache:
            x, W, G_p = _pair_data(_POWER_T, _POWER_K, scale)
            result = glm_pair_kernel_model(x, W, G_p, mutation=mutation)
            # Only the dropped radial term is scored on g_x32; keep the cache small.
            names = ("p", "g_xn", "g_W") + (("g_x32",) if mutation == "drop_radial" else ())
            cache[scale, mutation] = {name: getattr(result, name) for name in names}
        return cache[scale, mutation]

    yield emulation
    cache.clear()


# row -> (mutation, tensor, x scale): the checks with power. The g_W thresholds are established
# for a tcgen05 wgrad only; an mma.sync wgrad needs its g_W bar re-established first.
_POWER_ROWS = {
    "eps_outside_sqrt/p/x1": ("eps_outside_sqrt", "p", 1.0),
    "eps_outside_sqrt/g_W/x1": ("eps_outside_sqrt", "g_W", 1.0),
    "eps_outside_sqrt/p/x1e-3": ("eps_outside_sqrt", "p", 1e-3),
    "eps_outside_sqrt/g_W/x1e-3": ("eps_outside_sqrt", "g_W", 1e-3),
    "project_first/p/x1": ("project_first", "p", 1.0),
    "rz_for_rn/p/x1": ("rz_for_rn", "p", 1.0),
    "rz_for_rn/g_xn/x1": ("rz_for_rn", "g_xn", 1.0),
    "rz_for_rn/g_W/x1": ("rz_for_rn", "g_W", 1.0),
    "bf16_p/p/x1": ("bf16_p", "p", 1.0),
    "drop_kernel_split/p/x1": ("drop_kernel_split", "p", 1.0),
    "drop_kernel_split/g_W/x1": ("drop_kernel_split", "g_W", 1.0),
    "ignore_kernel_split/p/x1": ("ignore_kernel_split", "p", 1.0),
    "dgrad_unrounded_w/g_xn/x1": ("dgrad_unrounded_w", "g_xn", 1.0),
}


@pytest.mark.usefixtures("heavy_cpu_case")
@pytest.mark.parametrize("row", list(_POWER_ROWS))
def test_mutation_power_against_p9_thresholds(power_cases, row):
    """Each mutation row with power moves its tensor by at least its P9_POWER_THRESHOLDS value."""
    mutation, tensor, scale = _POWER_ROWS[row]
    effect = rel_rms(power_cases(scale, mutation)[tensor], power_cases(scale)[tensor])
    threshold = P9_POWER_THRESHOLDS[tensor]
    assert threshold == pytest.approx(10 * P9_M3_BARS[tensor], rel=1e-15)
    _power_record(row, effect, threshold)
    assert effect >= threshold, f"{row}: {effect:.3e} < {threshold:.3e}"


@pytest.mark.usefixtures("heavy_cpu_case")
@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_dropped_radial_term_has_power_on_fp32_g_x(power_cases, scale):
    """Dropping the radial term (coef) moves only g_x, which no bar of P9_M3_BARS covers.

    Its check is the GPU tests' float64 bar on the fp32 g_x: fused <= 1.1 x unfused and <= 1e-3,
    the smaller clause. The unfused error is modeled: fp32 autograd of the HF norm, fed the emulated
    TF32 dgrad product g_xn. Not covered at x scale 1e-3, where eps dominates the mean square and
    the effect is only 2.4 x the bar.
    """
    x, W, G_p = _pair_data(_POWER_T, _POWER_K, scale)
    reference = glm_pair_reference_f64(x, W, G_p)
    xf = x.float().requires_grad_(True)
    (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS_FP32)).backward(
        power_cases(scale)["g_xn"]
    )
    bar = min(1.1 * rel_rms(xf.grad, reference.g_x), 1e-3)
    effect = rel_rms(power_cases(scale, "drop_radial")["g_x32"], reference.g_x)
    _power_record(f"drop_radial/g_x32/x{scale:g}", effect, 10 * bar)
    assert effect >= 10 * bar


# mutation -> the tensors it cannot change, by construction
_UNCHANGED_BY_CONSTRUCTION = {
    "eps_outside_sqrt": ("g_xn",),
    "project_first": ("g_xn", "g_W"),
    "bf16_p": ("g_xn", "g_W"),
    "drop_radial": ("p", "g_xn", "g_W"),
    "drop_kernel_split": ("g_xn",),
    "ignore_kernel_split": ("g_xn",),
    "dgrad_unrounded_w": ("p", "g_W"),
}
# row -> (mutation, tensor, x scale): rows that move their tensor by less than the threshold
_NOT_COVERED_ROWS = {
    "eps_outside_sqrt/p/x1e3": ("eps_outside_sqrt", "p", 1e3),
    "eps_outside_sqrt/g_W/x1e3": ("eps_outside_sqrt", "g_W", 1e3),
    "ignore_kernel_split/g_W/x1": ("ignore_kernel_split", "g_W", 1.0),
}


@pytest.mark.usefixtures("heavy_cpu_case")
def test_mutations_without_emulator_power_are_recorded(power_cases):
    """The rows that are 0 by construction stay exactly 0, and the rows below the threshold stay
    below it, so the record of the table executes.

    Below the threshold: eps outside the sqrt at x scale 1e3, where s moves 1 ulp on 49 of 512
    tokens and no TF32 operand flips, so p and g_W do not move; the emulator ignoring the kernel's
    wgrad split at T = 512 (0.67 x the g_W threshold;
    test_emulator_split_power_on_g_w_at_production_t covers T = 4096). Also not covered: the
    dropped radial term at x scale 1e-3. Once the GPU tests pass GP-1's s to the emulator, no
    agreement bar sees an s-side mutation such as eps outside the sqrt: only their s bar at x scale
    1e-3 and on all-zero rows does.
    """
    for mutation, tensors in _UNCHANGED_BY_CONSTRUCTION.items():
        for tensor in tensors:
            assert torch.equal(power_cases(1.0, mutation)[tensor], power_cases(1.0)[tensor])
    for scale in (1e-3, 1e3):
        assert torch.equal(
            power_cases(scale, "eps_outside_sqrt")["g_xn"], power_cases(scale)["g_xn"]
        )
    for row, (mutation, tensor, scale) in _NOT_COVERED_ROWS.items():
        effect = rel_rms(power_cases(scale, mutation)[tensor], power_cases(scale)[tensor])
        _power_record(row, effect, P9_POWER_THRESHOLDS[tensor])
        assert effect < P9_POWER_THRESHOLDS[tensor], f"{row} is covered now: {effect:.3e}"
    # The wgrad half of ignore_kernel_split does change g_W, by less than the threshold at T = 512.
    ignored = power_cases(1.0, "ignore_kernel_split")["g_W"]
    assert not torch.equal(ignored, power_cases(1.0)["g_W"])


@pytest.mark.usefixtures("heavy_cpu_case")
def test_order_and_rn_power_minima(power_cases):
    """The emulator sides of the GPU tests' order and RN checks: rel(emu_A, emu_HF) >= 5e-5 on p,
    and rel(emu_RZ, emu_RN) >= 1e-4 on p, g_xn and g_W (42 x and 84 x the p threshold), with emu_A
    the project-first emulation and emu_RZ the one with "rz" operands."""
    base = power_cases(1.0)
    assert rel_rms(power_cases(1.0, "project_first")["p"], base["p"]) >= 5e-5
    for tensor in ("p", "g_xn", "g_W"):
        assert rel_rms(power_cases(1.0, "rz_for_rn")[tensor], base[tensor]) >= 1e-4, tensor


@pytest.fixture(scope="module")
def fused_power_cases():
    """g_W of kernel-model emulations at T = 512, K = 16384 on the shipped plans: "rne" operands,
    ALIGN_K8_X2 and glm_pair_fused_plans (7 wgrad layouts of 3 pieces of 64-256 tokens), on
    _pair_data (seed 0, scale 1)."""
    cache = {}

    def emulation(mutation=None):
        if mutation not in cache:
            x, W, G_p = _pair_data(_POWER_T, _POWER_K)
            plans = glm_pair_fused_plans(_POWER_T, _POWER_K)
            cache[mutation] = glm_pair_kernel_model(x, W, G_p, plans=plans, mutation=mutation).g_W
        return cache[mutation]

    yield emulation
    cache.clear()


# row -> mutation: the shipped plans' g_W rows at T = 512, split by their power against the g_W
# threshold (computed on CPU, MODELED, before any kernel output). Dropping the last piece of every
# column tile moves g_W by 4.9e5 x the threshold; one chain over T per column only by 0.78 x (P2's
# split-M 2 row: 0.67 x), so ignore_kernel_split keeps its g_W check at T = 4096 alone.
_FUSED_POWER_ROWS = {"drop_kernel_split/g_W/T512": "drop_kernel_split"}
_FUSED_NOT_COVERED_ROWS = {"ignore_kernel_split/g_W/T512": "ignore_kernel_split"}


@pytest.mark.usefixtures("heavy_cpu_case")
@pytest.mark.parametrize("row", list(_FUSED_POWER_ROWS))
def test_fused_plan_mutation_power(fused_power_cases, row):
    """Each shipped-plan g_W row with power at T = 512 moves g_W by at least the g_W threshold."""
    effect = rel_rms(fused_power_cases(_FUSED_POWER_ROWS[row]), fused_power_cases())
    _power_record(f"{row}/fused", effect, P9_POWER_THRESHOLDS["g_W"])
    assert effect >= P9_POWER_THRESHOLDS["g_W"], f"{row}: {effect:.3e}"


@pytest.mark.usefixtures("heavy_cpu_case")
def test_fused_plan_mutations_without_power_are_recorded(fused_power_cases):
    """The shipped-plan g_W rows below the threshold at T = 512 change g_W, by less than the
    threshold (test_fused_split_power_on_g_w_at_production_t covers ignore_kernel_split there)."""
    for row, mutation in _FUSED_NOT_COVERED_ROWS.items():
        mutated = fused_power_cases(mutation)
        effect = rel_rms(mutated, fused_power_cases())
        _power_record(f"{row}/fused", effect, P9_POWER_THRESHOLDS["g_W"])
        assert not torch.equal(mutated, fused_power_cases()), row
        assert effect < P9_POWER_THRESHOLDS["g_W"], f"{row} is covered now: {effect:.3e}"
