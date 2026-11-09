# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `chunk_fwd_kernel_o` (megatron/core/ssm/ops/gdp/chunk_o.py), launched by
`chunk_gated_delta_product_fwd_o`: the outputs of the GDP/GDN chunked prefill from the chunk-entry states
and the intra-chunk attention.

Reached in inference only through `chunk_gated_delta_product_varlen` (GDP and GDN prefill): q [1, T, H, K]
(L2-normalised), k [1, T*M, H, K] and v_new [1, T*M, H, V] on the expanded stream, the chunk-entry states h
(rows by `chunk_offsets`, in q's dtype; the padded descriptor rows are never written and hold garbage), the
token-stream within-chunk base-2 decays G, `cu_seqlens`, the padded token-stream descriptors, M. The entry
point's output is this kernel's, but its inputs there are the other stages' outputs; this direct test drives
it with exact definitions of those inputs and covers the fixed-length and no-decay variants.

Contract (from the launcher's documented arguments and the chunked delta-rule math, before reading the
kernel): for token t in chunk j (first token c0) of sequence n, head h:
  o_t = scale * ( 2^G_t q_t^T h_j + sum_{s = c0 .. t} sum_{m < M} 2^(G_t - G_s) (q_t . k_{s,m}) v_new_{s,m} ),
with k_{s,m}, v_new_{s,m} the expanded-stream rows s * M + m (no decay factors without G); o [B, T, H, V]
in v_new's dtype, zero at tokens outside every sequence (the launcher zero-initialises it). Inputs read-only.

Numerics: the masked, decayed score matrix is a tensor-core operand in the activation dtype before its
product with v_new (FLA's design), accumulation in fp32. Tolerance class: floating point against the fp64
definition; trusted = the definition in torch fp32 on the same device with the score matrix rounded to the
activation dtype, then the output rounded (factor 2; with fp32 scores the trusted implementation is ~1.4x
more accurate, Triton interpreter). Zero outputs outside every sequence: exact.

Variants and dispatch (read from the kernel after the contract): heuristics USE_G, IS_VARLEN; constexprs H,
K, V, BT, num_householder; autotune over BK, BV in {64, 128} (H100 and GB200: `check_shared_mem()` holds on
both; [32, 64] elsewhere) x num_warps {2, 4} (`IS_NVIDIA_HOPPER`, true for every cc >= 9) x num_stages
{2, 3, 4}; grid (cdiv(V, BV), NT, B*H) with the K loop. No Blackwell-specific variant. Covered: K in {32,
64, 96, 128, 256}, V in {16, 64, 96, 128, 256}, H 1 and 4, M 1/2/3, bf16 and fp16, the three decay regimes
and no decay, each (BK, BV) tile pinned in turn, a fixed-length batch, NaN in the never-written h rows of
the padded descriptors; chunk boundaries from `delta_rule.STAGE_PACKS`. Production: GDP 800m (32 heads,
K 128, V 64, M 3) and GDN (32 heads, K = V = 128) on a 4096-token pack.

Memory: a direct launch into a guarded NaN-poisoned o: rows outside every sequence stay poisoned; NaN in
the tail tokens' inputs and the padded h rows must not reach owned rows; `sanitizer` on a small case.
int32: h's row offset is int64, but the q/k/v/o base offsets (bos * [M *] H + h) * K / V are int32
products; `test_key_offsets_past_int32` (`large`) starts a sequence where bos * M * H * K crosses 2^31.
Suspected finding O7b-M2 (by reading): expected to fail on main there; a GDP 800m prefill step (M 3, 32
heads, K 128) reaches it at 174763 tokens.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.chunk_o import chunk_fwd_kernel_o, chunk_gated_delta_product_fwd_o
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _outputs(dtype, M, scale):
    def fn(q, k, v_new, h, G, cu, *, rounded):
        return dr.chunk_outputs(
            q,
            k,
            v_new,
            h,
            G,
            cu,
            num_householder=M,
            scale=scale,
            score_operand=dr.round_to(dtype) if rounded else None,
        )

    return fn


def _check(o, chain, M, scale, *, what, h_rows=None):
    q, k, v_new, G, cu = chain.q, chain.k, chain.v_new, chain.G, chain.cu
    n = dr.chunk_offsets(cu)[-1]
    h = chain.h[0, :n] if h_rows is None else h_rows
    rows = dr.owned_rows(cu, q.shape[1]).to(q.device)
    fn = _outputs(q.dtype, M, scale)
    ref = ou.fp64_reference(fn, q, k, v_new, h, G, cu, rounded=False)
    trusted = dr.trusted_fp32(fn, q, k, v_new, h, G, cu, rounded=True).to(o.dtype)
    ou.assert_close_to_reference(o[:, rows], ref[:, rows], trusted[:, rows], what=what)
    return rows


def _chain(pack_name, *, H, K, V, dtype, regime, g0):
    lens, M = dr.STAGE_PACKS[pack_name]
    use_g = regime != "no_decay"
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=H,
        head_k=K,
        head_v=V,
        dtype=DTYPES[dtype],
        regime=regime if use_g else "production",
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    h0 = torch.randn(len(lens), H, K, V, generator=g0, device=DEVICE)
    return pack, dr.build_chain(pack, h0=h0, use_g=use_g, device=DEVICE)


CASES = [
    pytest.param(f, id="{pack}-K{K}-V{V}-H{H}-{dtype}-{regime}".format(**f))
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "K": [32, 64, 96, 128, 256],
            "V": [16, 64, 96, 128, 256],
            "H": [1, 4],
            "dtype": list(DTYPES),
            "regime": list(dr.G_REGIMES) + ["no_decay"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_outputs_match_fp64(case):
    """The prefill call (varlen, production descriptors, NaN in the padded h rows), pairwise over packs,
    K, V, heads, dtypes and decay regimes (and no decay)."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain(
        case["pack"],
        H=case["H"],
        K=case["K"],
        V=case["V"],
        dtype=case["dtype"],
        regime=case["regime"],
        g0=g0,
    )
    M, scale = pack.M, case["K"] ** -0.5
    inputs = [x for x in (chain.q, chain.k, chain.v_new, chain.h, chain.G) if x is not None]
    before = [x.clone() for x in inputs]
    with dr.pinned_autotuning():
        o = chunk_gated_delta_product_fwd_o(
            q=chain.q,
            k=chain.k,
            v=chain.v_new,
            h=chain.h,
            g=chain.G,
            scale=scale,
            cu_seqlens=chain.cu,
            chunk_size=CHUNK,
            num_householder=M,
            chunk_indices=chain.desc["chunk_indices"],
        )
    assert o.shape == (1, chain.q.shape[1], case["H"], case["V"]) and o.dtype == chain.v_new.dtype
    for x, b in zip(inputs, before):
        ou.assert_exact(x, b, what="inputs are read-only")
    rows = _check(o, chain, M, scale, what=f"o ({case})")
    ou.assert_exact(
        o[:, ~rows], torch.zeros_like(o[:, ~rows]), what="tokens no sequence owns read 0"
    )


@ou.requires_sm(ou.HOPPER)
def test_every_chunk_o_tile():
    """Each (BK, BV) tile pinned in turn, at K 256 and V 192 (K loops and a masked V tile)."""
    tiles = dr.distinct_tiles(chunk_fwd_kernel_o, ["BK", "BV"])
    assert len(tiles) == 4
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain("M2", H=2, K=256, V=192, dtype="bf16", regime="production", g0=g0)
    for config in tiles:
        with dr.pinned(chunk_fwd_kernel_o, config):
            o = chunk_gated_delta_product_fwd_o(
                q=chain.q,
                k=chain.k,
                v=chain.v_new,
                h=chain.h,
                g=chain.G,
                scale=0.0625,
                cu_seqlens=chain.cu,
                num_householder=pack.M,
                chunk_indices=chain.desc["chunk_indices"],
            )
        _check(o, chain, pack.M, 0.0625, what=f"o {dr.config_id(config)}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("M", [1, 3])
def test_fixed_length_batch(M):
    """No `cu_seqlens` (IS_VARLEN off): B = 2 rows of 2 chunks and a partial one."""
    g0 = ou.make_generator(device=DEVICE)
    T, H, K, V = 2 * CHUNK + 9, 2, 64, 32
    chains = []
    for _ in range(2):
        pack = dr.make_pack(
            (T,), num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
        )
        h0 = torch.randn(1, H, K, V, generator=g0, device=DEVICE)
        chains.append(dr.build_chain(pack, h0=h0, device=DEVICE))
    n = -(-T // CHUNK)
    q, k, v_new, G = (torch.cat([getattr(c, a) for c in chains]) for a in ("q", "k", "v_new", "G"))
    h = torch.cat([c.h[:, :n] for c in chains])  # [B, NT, H, K, V]
    with dr.pinned_autotuning():
        o = chunk_gated_delta_product_fwd_o(
            q=q, k=k, v=v_new, h=h, g=G, scale=K**-0.5, num_householder=M
        )
    for b, c in enumerate(chains):
        _check(o[b : b + 1], c, M, K**-0.5, what=f"o batch row {b}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_chunk_o_writes_only_owned_rows():
    """A direct launch, as the launcher makes it, into a guarded NaN-poisoned o: rows outside every
    sequence stay poisoned (the launcher zero-fills them); guards intact."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain("M3", H=2, K=96, V=96, dtype="bf16", regime="clamp_off", g0=g0)
    B, T, H, K = chain.q.shape
    V = chain.v_new.shape[-1]
    o = ou.guarded_empty((B, T, H, V), chain.v_new.dtype, DEVICE)
    ci = chain.desc["chunk_indices"]
    scale = K**-0.5
    with dr.pinned_autotuning():
        chunk_fwd_kernel_o[lambda meta: (-(-V // meta["BV"]), ci.shape[0], B * H)](
            chain.q,
            chain.k,
            chain.v_new,
            chain.h,
            chain.G,
            o.tensor,
            chain.cu,
            ci,
            scale,
            T=T,
            num_householder=pack.M,
            H=H,
            K=K,
            V=V,
            BT=CHUNK,
        )
    ou.check_guards(o=o)
    rows = _check(o.tensor, chain, pack.M, scale, what="o (direct launch)")
    ou.assert_poisoned(o.tensor[:, ~rows], what="rows no sequence owns")


PRODUCTION = {
    # name: (H, K, V, M)
    "gdp_800m": (32, 128, 64, 3),
    "gdn_default": (32, 128, 128, 1),
}
PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_chunk_o_production(name):
    """A 4096-token prefill at the production shapes, timed autotuning."""
    H, K, V, M = PRODUCTION[name]
    g0 = ou.make_generator(device=DEVICE)
    pack = dr.make_pack(
        PRODUCTION_LENS, num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    h0 = torch.randn(len(PRODUCTION_LENS), H, K, V, generator=g0, device=DEVICE)
    chain = dr.build_chain(pack, h0=h0, device=DEVICE)
    o = chunk_gated_delta_product_fwd_o(
        q=chain.q,
        k=chain.k,
        v=chain.v_new,
        h=chain.h,
        g=chain.G,
        scale=K**-0.5,
        cu_seqlens=chain.cu,
        num_householder=M,
        chunk_indices=chain.desc["chunk_indices"],
    )
    _check(o, chain, M, K**-0.5, what=f"o {name}")


LARGE_M, LARGE_K = 3, 256
KEY_ROW = LARGE_M * LARGE_K
"""Elements of k per token at H 1, K 256, M 3: the k base offset of a sequence is bos * KEY_ROW."""


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "bos",
    [
        pytest.param((2**31 - 1) // KEY_ROW, id="base-fits", marks=pytest.mark.large),
        pytest.param(-(-(2**31) // KEY_ROW), id="base-past-2^31", marks=pytest.mark.large),
    ],
)
def test_key_offsets_past_int32(bos):
    """A 65-token sequence starting at token `bos`, after a sequence of zeros: its k/v base offset
    bos * M * H * K is just below or just past 2^31 elements (H 1, K = V = 256, M 3). The contract holds
    at any offset; `[base-past-2^31]` is expected to fail on main (suspected O7b-M2)."""
    K = V = LARGE_K
    M = LARGE_M
    L_b = CHUNK + 1
    T = bos + L_b
    n_a = -(-bos // CHUNK)
    ou.require_free_memory(T * (K + 2 * M * K + V) * 2 + (n_a + 2) * K * V * 2 + 2**30)
    g0 = ou.make_generator(device=DEVICE)
    small = dr.make_pack(
        (L_b,), num_householder=M, heads=1, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    h0 = torch.randn(1, 1, K, V, generator=g0, device=DEVICE)
    chain = dr.build_chain(small, h0=h0, device=DEVICE)

    def embed(x, per):
        out = torch.zeros(1, T * per, *x.shape[2:], dtype=x.dtype, device=DEVICE)
        out[:, bos * per :] = x
        return out

    q, k, v_new, G = embed(chain.q, 1), embed(chain.k, M), embed(chain.v_new, M), embed(chain.G, 1)
    h = torch.zeros(1, n_a + 2, 1, K, V, dtype=torch.bfloat16, device=DEVICE)
    h[0, n_a:] = chain.h[0, :2]
    idx = [(0, j) for j in range(n_a)] + [(1, 0), (1, 1)]
    ci = torch.tensor(idx, dtype=torch.int32, device=DEVICE)
    cu = torch.tensor([0, bos, T], dtype=torch.int32, device=DEVICE)
    o = chunk_gated_delta_product_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=G,
        scale=K**-0.5,
        cu_seqlens=cu,
        num_householder=M,
        chunk_indices=ci,
    )
    out = o[:, bos:]
    fn = _outputs(torch.bfloat16, M, K**-0.5)
    args = (chain.q, chain.k, chain.v_new, chain.h[0, :2], chain.G, chain.cu)
    ref = ou.fp64_reference(fn, *args, rounded=False)
    trusted = dr.trusted_fp32(fn, *args, rounded=True).bfloat16()
    ou.assert_close_to_reference(out, ref, trusted, what=f"o of the sequence at token {bos}")
