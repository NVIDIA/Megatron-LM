# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `recompute_w_u_fwd_kernel` (megatron/core/ssm/ops/gdp/wy_fast.py), launched by
`recompute_w_u_fwd`: the WY factors w and u of each chunk.

Reached in inference only through `chunk_gated_delta_product_varlen` (GDP and GDN prefill), on the expanded
stream: k [1, T*M, H, K] (L2-normalised), v [1, T*M, H, V], beta [1, T*M, H] (GDP: activation dtype; GDN:
fp32), A = (I + beta K K^T)^-1 [1, T*M, H, 64] in the activation dtype, g the within-chunk base-2 cumulative
decays (fp32), the padded expanded-stream descriptors. w and u are internal to the entry point, and the
grouped-head (HV > H) and no-decay variants are not selectable there, so it gets this direct test.

Contract (from the launcher's documented arguments and the WY math, before reading the kernel): per chunk
(L rows) and value head hv (key head hv // (HV / H)), with A_c = A[c0:c0+L, hv, :L]:
  u = A_c (beta * v),     w = A_c (beta * 2^g * k)    (no 2^g without g),
w [B, T, HV, K] in k's dtype, u like v; rows outside every sequence are not written; inputs read-only.

Numerics: the two products are tensor-core products whose operands are in the activation dtype, so beta * v
and beta * 2^g * k are materialized in that dtype before them, with fp32 accumulation. Tolerance class:
floating point against the fp64 definition (no intermediate rounding); trusted = the definition in torch
fp32 on the same device with those two operands rounded to the activation dtype, then the result rounded
(factor 2). Without the operand rounding the trusted implementation is 1.4-3.1x more accurate than any
bf16-operand product (measured in the Triton interpreter), which no tolerance policy should absorb.

Variants and dispatch (read from the kernel after the contract): heuristics USE_G, IS_VARLEN; constexprs H,
HV, K, V, BT, BK = BV = 64 (the K and V loops; masked tails); autotune over num_warps {2, 4, 8} x num_stages
{2, 3, 4}. Covered: K in {32, 64, 96, 128, 256}, V in {16, 64, 96, 128}, heads (1, 1), (4, 4) and grouped
(2, 4), beta in the activation dtype and fp32, bf16 and fp16, the three decay regimes, no decay, every
num_warps pinned in turn, a fixed-length batch; chunk boundaries from `delta_rule.STAGE_PACKS`.
Production: GDP 800m (32 heads, K 128, V 64, M 3) on a 4096-token pack.

Memory: a direct launch into guarded NaN-poisoned w and u: rows outside every sequence stay poisoned;
`sanitizer` on a small case. Offsets: the head index is int64 in the kernel; no `large` case.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.wy_fast import recompute_w_u_fwd, recompute_w_u_fwd_kernel
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _inputs(pack_name, *, H, HV, K, V, dtype, beta_fp32, regime, g0, use_g=True):
    lens, M = dr.STAGE_PACKS[pack_name]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=HV,
        head_k=K,
        head_v=V,
        dtype=DTYPES[dtype],
        beta_dtype=torch.float32 if beta_fp32 else None,
        regime=regime,
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    chain = dr.build_chain(pack, use_g=use_g, device=DEVICE)
    k = chain.k[:, :, :: HV // H].contiguous()
    return pack, chain, k


def _check(w, u, k, v, beta, Ai, G, cu_dp, *, what):
    rows = dr.owned_rows(cu_dp, v.shape[1]).to(v.device)
    ref_w, ref_u = ou.fp64_reference(dr.wy_factors, k, v, beta, Ai, G, cu_dp)
    tr_w, tr_u = dr.trusted_fp32(
        dr.wy_factors, k, v, beta, Ai, G, cu_dp, operand=dr.round_to(k.dtype)
    )
    ou.assert_close_to_reference(
        u[:, rows], ref_u[:, rows], tr_u[:, rows].to(u.dtype), what=f"u {what}"
    )
    ou.assert_close_to_reference(
        w[:, rows], ref_w[:, rows], tr_w[:, rows].to(w.dtype), what=f"w {what}"
    )


CASES = [
    pytest.param(
        f, id="{pack}-K{K}-V{V}-H{heads[0]}HV{heads[1]}-{dtype}-beta{beta}-{regime}".format(**f)
    )
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "K": [32, 64, 96, 128, 256],
            "V": [16, 64, 96, 128],
            "heads": [(1, 1), (4, 4), (2, 4)],
            "dtype": list(DTYPES),
            "beta": ["act", "fp32"],
            "regime": list(dr.G_REGIMES) + ["no_decay"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_wy_factors_match_fp64(case):
    """The prefill call (varlen, production descriptors), pairwise over packs, K, V, heads, dtypes and
    decay regimes (and no decay)."""
    g0 = ou.make_generator(device=DEVICE)
    H, HV = case["heads"]
    use_g = case["regime"] != "no_decay"
    pack, chain, k = _inputs(
        case["pack"],
        H=H,
        HV=HV,
        K=case["K"],
        V=case["V"],
        dtype=case["dtype"],
        beta_fp32=case["beta"] == "fp32",
        regime=case["regime"] if use_g else "production",
        g0=g0,
        use_g=use_g,
    )
    inputs = [x for x in (k, pack.v, pack.beta, chain.Ai, chain.Gdp) if x is not None]
    before = [x.clone() for x in inputs]
    with dr.pinned_autotuning():
        w, u = recompute_w_u_fwd(
            k=k,
            v=pack.v,
            beta=pack.beta,
            A=chain.Ai,
            g=chain.Gdp,
            cu_seqlens=chain.cu_dp,
            chunk_indices=chain.desc["chunk_indices_dp"],
        )
    assert (
        w.shape == (1, k.shape[1], HV, case["K"]) and w.dtype == k.dtype and u.dtype == pack.v.dtype
    )
    for x, b in zip(inputs, before):
        ou.assert_exact(x, b, what="inputs are read-only")
    _check(w, u, k, pack.v, pack.beta, chain.Ai, chain.Gdp, chain.cu_dp, what=f"({case})")


@ou.requires_sm(ou.HOPPER)
def test_every_wy_config():
    """Each num_warps pinned in turn (no block-size candidates: BK = BV = 64 are fixed)."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain, k = _inputs(
        "M3", H=2, HV=2, K=96, V=96, dtype="bf16", beta_fp32=False, regime="production", g0=g0
    )
    warps = {}
    for config in dr.autotuner(recompute_w_u_fwd_kernel).configs:
        if config.num_warps not in warps or config.num_stages < warps[config.num_warps].num_stages:
            warps[config.num_warps] = config
    assert set(warps) == {2, 4, 8}
    for config in warps.values():
        with dr.pinned(recompute_w_u_fwd_kernel, config):
            w, u = recompute_w_u_fwd(
                k=k,
                v=pack.v,
                beta=pack.beta,
                A=chain.Ai,
                g=chain.Gdp,
                cu_seqlens=chain.cu_dp,
                chunk_indices=chain.desc["chunk_indices_dp"],
            )
        _check(
            w, u, k, pack.v, pack.beta, chain.Ai, chain.Gdp, chain.cu_dp, what=dr.config_id(config)
        )


@ou.requires_sm(ou.HOPPER)
def test_fixed_length_batch():
    """No `cu_seqlens` (IS_VARLEN off): B = 2 rows of 2 chunks and a partial one."""
    g0 = ou.make_generator(device=DEVICE)
    T, H, K, V = 2 * CHUNK + 33, 2, 64, 32
    chains = []
    for _ in range(2):
        pack = dr.make_pack(
            (T,), num_householder=1, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
        )
        chains.append((pack, dr.build_chain(pack, device=DEVICE)))
    cat = lambda f: torch.cat([f(p, c) for p, c in chains])  # noqa: E731
    k, v, beta = cat(lambda p, c: c.k), cat(lambda p, c: p.v), cat(lambda p, c: p.beta)
    Ai, G = cat(lambda p, c: c.Ai), cat(lambda p, c: c.Gdp)
    with dr.pinned_autotuning():
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, g=G)
    cu = torch.tensor([0, T], device=DEVICE)
    for b in range(2):
        s = slice(b, b + 1)
        _check(w[s], u[s], k[s], v[s], beta[s], Ai[s], G[s], cu, what=f"batch row {b}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_wy_writes_only_owned_rows():
    """A direct launch, as the launcher makes it, into guarded NaN-poisoned w and u: rows outside every
    sequence stay poisoned; guards intact."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain, k = _inputs(
        "M3", H=2, HV=4, K=96, V=96, dtype="bf16", beta_fp32=True, regime="clamp_off", g0=g0
    )
    B, T, H, K = k.shape
    HV, V = pack.v.shape[2], pack.v.shape[3]
    w = ou.guarded_empty((B, T, HV, K), k.dtype, DEVICE)
    u = ou.guarded_empty((B, T, HV, V), pack.v.dtype, DEVICE)
    ci = chain.desc["chunk_indices_dp"]
    with dr.pinned_autotuning():
        recompute_w_u_fwd_kernel[(ci.shape[0], B * HV)](
            k=k,
            v=pack.v,
            beta=pack.beta,
            w=w.tensor,
            u=u.tensor,
            A=chain.Ai,
            g=chain.Gdp,
            cu_seqlens=chain.cu_dp,
            chunk_indices=ci,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=CHUNK,
            BK=64,
            BV=64,
        )
    ou.check_guards(w=w, u=u)
    rows = dr.owned_rows(chain.cu_dp, T).to(DEVICE)
    ou.assert_poisoned(w.tensor[:, ~rows], what="w rows no sequence owns")
    ou.assert_poisoned(u.tensor[:, ~rows], what="u rows no sequence owns")
    _check(
        w.tensor,
        u.tensor,
        k,
        pack.v,
        pack.beta,
        chain.Ai,
        chain.Gdp,
        chain.cu_dp,
        what="(direct launch)",
    )


PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)


@ou.requires_sm(ou.HOPPER)
def test_wy_production():
    """GDP 800m prefill: 32 heads, K 128, V 64, M 3, a 4096-token pack, timed autotuning."""
    g0 = ou.make_generator(device=DEVICE)
    pack = dr.make_pack(
        PRODUCTION_LENS,
        num_householder=3,
        heads=32,
        head_k=128,
        head_v=64,
        generator=g0,
        device=DEVICE,
    )
    chain = dr.build_chain(pack, device=DEVICE)
    w, u = recompute_w_u_fwd(
        k=chain.k,
        v=pack.v,
        beta=pack.beta,
        A=chain.Ai,
        g=chain.Gdp,
        cu_seqlens=chain.cu_dp,
        chunk_indices=chain.desc["chunk_indices_dp"],
    )
    _check(w, u, chain.k, pack.v, pack.beta, chain.Ai, chain.Gdp, chain.cu_dp, what="production")
