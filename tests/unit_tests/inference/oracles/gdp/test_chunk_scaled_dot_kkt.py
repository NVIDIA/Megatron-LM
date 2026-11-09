# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `chunk_scaled_dot_kkt_fwd_kernel` (megatron/core/ssm/ops/gdp/scaled_dot_kkt.py), launched by
`chunk_scaled_dot_kkt_fwd`: the strictly lower-triangular, decayed beta K K^T of each chunk, the first half
of the WY representation.

Reached in inference only through `chunk_gated_delta_product_varlen` (GDP and GDN prefill), on the
Householder-expanded stream: k [1, T*M, H, K] (L2-normalised), beta [1, T*M, H] (GDP: activation dtype; GDN:
fp32), g the within-chunk base-2 cumulative decays of the expanded stream (fp32), `cu_seqlens * M`, chunk 64,
fp32 output and the padded expanded-stream descriptors. A is internal to the entry point, and the entry
cannot select the grouped-head (HV > H) or no-decay variants, so it gets this direct test.

Contract (from the launcher's documented arguments and the WY math, before reading the kernel): for every
chunk (64 rows of a sequence, the last possibly shorter, L rows) and value head hv (key head hv // (HV / H)):
  A[t, hv, j - c0] = beta[t, hv] * (k_t . k_j) * 2^(g[t, hv] - g[j, hv])   for c0 <= j < t,
and 0 for j >= t and for the columns j - c0 >= L; without g no decay. Rows outside every sequence are not
written. Output [B, T, HV, 64] in `output_dtype`; k, g, beta read-only.

Tolerance class: floating point against the fp64 definition, trusted = the definition in torch fp32 on the
same device (factor 2); the zeros on and above the diagonal and past a short chunk are exact (the
clamp-off decays make 2^(g_t - g_j) overflow there, which must not leak); exact by construction with dyadic
keys and betas and no decay (the dot products are exact in fp32).

Variants and dispatch (read from the kernel after the contract): heuristics USE_G, IS_VARLEN; constexprs H,
HV (grouped heads), K, BT; autotune over BK in {32, 64, 128} (the K loop) x num_warps {2, 4, 8} x num_stages
{2, 3, 4}, key (H, HV, K, BT, IS_VARLEN). Covered: K in {16, 32, 64, 96, 128, 256} (one and several BK
blocks, a masked tail at 96 and 16), heads (1, 1), (4, 4) and grouped (2, 4), beta in the activation dtype
and fp32, bf16 and fp16 keys, the three decay regimes, every BK tile pinned in turn, no decay, fixed-length
batches (no `cu_seqlens`). Chunk boundaries from `delta_rule.STAGE_PACKS` (last-chunk lengths 1 to 64
around the 16-row blocks, M 1/2/3, padding requests, the padded descriptor rows past sequence 0, NaN tail
tokens). Production: GDP 800m (32 heads, K 128, M 3) on a 4096-token pack.

Memory: a direct launch into a guarded NaN-poisoned A: rows outside every sequence stay poisoned;
`sanitizer` on a small case. Offsets: the head index is int64 in the kernel, so (bos * H + h) * K is 64-bit;
no `large` case.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
    chunk_scaled_dot_kkt_fwd_kernel,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _inputs(pack_name, *, H, HV, K, dtype, beta_fp32, regime, g0, use_g=True):
    """k [1, T*M, H, K], beta and Gdp [1, T*M, HV] from a pack with HV heads (k: every (HV/H)-th head)."""
    lens, M = dr.STAGE_PACKS[pack_name]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=HV,
        head_k=K,
        head_v=16,
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


def _check(A, k, beta, G, cu_dp, *, what):
    T = A.shape[1]
    rows = dr.owned_rows(cu_dp, T).to(A.device)
    ref = ou.fp64_reference(dr.scaled_dot_kkt, k, beta, G, cu_dp)
    trusted = dr.trusted_fp32(dr.scaled_dot_kkt, k, beta, G, cu_dp)
    # The structural zeros are exact (clamp-off decays overflow 2^(g_t - g_j) there).
    zeros = A[0].transpose(1, 2)[dr.structural_zeros(cu_dp, T).to(A.device)]
    ou.assert_exact(
        zeros.abs(), torch.zeros_like(zeros), what=f"{what}: zeros on/above the diagonal"
    )
    ou.assert_close_to_reference(A[:, rows], ref[:, rows], trusted[:, rows], what=what)


CASES = [
    pytest.param(
        f, id="{pack}-K{K}-H{heads[0]}HV{heads[1]}-{dtype}-beta{beta}-{regime}".format(**f)
    )
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "K": [16, 32, 64, 96, 128, 256],
            "heads": [(1, 1), (4, 4), (2, 4)],
            "dtype": list(DTYPES),
            "beta": ["act", "fp32"],
            "regime": list(dr.G_REGIMES),
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_kkt_matches_fp64(case):
    """The prefill call (varlen, production expanded-stream descriptors, fp32 A), pairwise over packs, K,
    heads, dtypes and decay regimes."""
    g0 = ou.make_generator(device=DEVICE)
    H, HV = case["heads"]
    pack, chain, k = _inputs(
        case["pack"],
        H=H,
        HV=HV,
        K=case["K"],
        dtype=case["dtype"],
        beta_fp32=case["beta"] == "fp32",
        regime=case["regime"],
        g0=g0,
    )
    inputs = (k, chain.Gdp, pack.beta)
    before = [x.clone() for x in inputs]
    with dr.pinned_autotuning():
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            g=chain.Gdp,
            beta=pack.beta,
            cu_seqlens=chain.cu_dp,
            chunk_size=CHUNK,
            output_dtype=torch.float32,
            chunk_indices=chain.desc["chunk_indices_dp"],
        )
    assert A.shape == (1, k.shape[1], HV, CHUNK) and A.dtype == torch.float32
    for x, b in zip(inputs, before):
        ou.assert_exact(x, b, what="inputs are read-only")
    _check(A, k, pack.beta, chain.Gdp, chain.cu_dp, what=f"A ({case})")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("K", [96, 256])
def test_every_kkt_tile(K):
    """Each BK tile (32, 64, 128) pinned in turn: one, several and masked K blocks."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain, k = _inputs(
        "M3", H=2, HV=2, K=K, dtype="bf16", beta_fp32=False, regime="production", g0=g0
    )
    tiles = dr.distinct_tiles(chunk_scaled_dot_kkt_fwd_kernel, ["BK"])
    assert {c.kwargs["BK"] for c in tiles} == {32, 64, 128}
    for config in tiles:
        with dr.pinned(chunk_scaled_dot_kkt_fwd_kernel, config):
            A = chunk_scaled_dot_kkt_fwd(
                k=k,
                g=chain.Gdp,
                beta=pack.beta,
                cu_seqlens=chain.cu_dp,
                chunk_indices=chain.desc["chunk_indices_dp"],
            )
        _check(A, k, pack.beta, chain.Gdp, chain.cu_dp, what=f"A K={K} {dr.config_id(config)}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("K", [64, 256])
@pytest.mark.parametrize("varlen", [True, False], ids=["varlen", "fixed"])
def test_dyadic_kkt_without_decay_is_exact(K, varlen):
    """No decay (USE_G off), dyadic keys and betas: beta (k_t . k_j) is exact in fp32, so A must equal
    the fp64 definition bit for bit; grouped heads (2 -> 4); varlen pack and fixed-length batch."""
    g0 = ou.make_generator(device=DEVICE)
    H, HV = 2, 4
    if varlen:
        lens = (1, 17, 64, 65, 130)
        T, B = sum(lens), 1
        cu = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
        )
    else:
        B, T, cu = 3, 2 * CHUNK + 7, None
    k = ou.dyadic_tensor(
        (B, T, H, K),
        mantissa_bits=3,
        exponents=(-2, 0),
        dtype=torch.bfloat16,
        device=DEVICE,
        generator=g0,
    )
    beta = ou.dyadic_tensor(
        (B, T, HV),
        mantissa_bits=3,
        exponents=(-2, 0),
        dtype=torch.bfloat16,
        device=DEVICE,
        generator=g0,
    ).abs()
    with dr.pinned_autotuning():
        A = chunk_scaled_dot_kkt_fwd(k=k, g=None, beta=beta, cu_seqlens=cu, chunk_size=CHUNK)
    if varlen:
        expected = dr.scaled_dot_kkt(k.double(), beta.double(), None, cu)
    else:
        expected = torch.cat(
            [
                dr.scaled_dot_kkt(
                    k[b : b + 1].double(),
                    beta[b : b + 1].double(),
                    None,
                    torch.tensor([0, T], device=DEVICE),
                )
                for b in range(B)
            ]
        )
    ou.assert_exact(
        A, expected.to(torch.float32), what=f"dyadic A (K={K}, {'varlen' if varlen else 'fixed'})"
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_kkt_writes_only_owned_rows():
    """A direct launch, as the launcher makes it, into a guarded NaN-poisoned A: rows outside every
    sequence (padded tail, padded descriptors) stay poisoned, guards intact."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain, k = _inputs(
        "M3", H=2, HV=4, K=96, dtype="bf16", beta_fp32=True, regime="clamp_off", g0=g0
    )
    B, T, H, K = k.shape
    HV = pack.beta.shape[2]
    A = ou.guarded_empty((B, T, HV, CHUNK), torch.float32, DEVICE)
    ci = chain.desc["chunk_indices_dp"]
    with dr.pinned_autotuning():
        chunk_scaled_dot_kkt_fwd_kernel[(ci.shape[0], B * HV)](
            k=k,
            g=chain.Gdp,
            beta=pack.beta,
            A=A.tensor,
            cu_seqlens=chain.cu_dp,
            chunk_indices=ci,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=CHUNK,
        )
    ou.check_guards(A=A)
    rows = dr.owned_rows(chain.cu_dp, T).to(DEVICE)
    ou.assert_poisoned(A.tensor[:, ~rows], what="rows no sequence owns")
    _check(A.tensor, k, pack.beta, chain.Gdp, chain.cu_dp, what="A (direct launch)")


PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)


@ou.requires_sm(ou.HOPPER)
def test_kkt_production():
    """GDP 800m prefill: 32 heads, K 128, M 3, a 4096-token pack (12288 expanded rows)."""
    g0 = ou.make_generator(device=DEVICE)
    pack = dr.make_pack(
        PRODUCTION_LENS,
        num_householder=3,
        heads=32,
        head_k=128,
        head_v=16,
        generator=g0,
        device=DEVICE,
    )
    k = dr.l2norm(pack.k.double()).bfloat16()
    cu_dp = pack.cu_seqlens * 3
    desc = dr.descriptors(pack, device=DEVICE)
    from megatron.core.ssm.ops.gdp.common import RCP_LN2

    G = dr.local_cumsum(dr.interleave_decays(pack.g, 3).double(), cu_dp, scale=RCP_LN2).float()
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=G, beta=pack.beta, cu_seqlens=cu_dp, chunk_indices=desc["chunk_indices_dp"]
    )
    _check(A, k, pack.beta, G, cu_dp, what="A production")
