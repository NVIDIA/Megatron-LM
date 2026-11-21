# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `chunk_local_cumsum_scalar_kernel` (megatron/core/ssm/ops/gdp/cumsum.py), launched by
`chunk_local_cumsum`: the within-chunk cumulative sum of the GDP/GDN log decays.

Reached in inference only through `chunk_gated_delta_product_varlen` (megatron/core/ssm/ops/gdp/chunk.py),
from `GatedDeltaProductMixer.ssm_prefill` and `GatedDeltaNet.ssm_prefill`: twice per prefill, on the token
stream (g [1, T, H], `cu_seqlens`) and on the Householder-expanded stream (g interleaved with M - 1 zero
decays per token, `cu_seqlens * M`), with `scale = RCP_LN2` (base-2 decays), fp32 output, chunk 64 and the
precomputed padded descriptors. Its output is internal to the entry point (the decays only reach the
outputs through exp2 of differences), so it gets this direct test.

Contract (from the launcher's documented arguments, before reading the kernel): for `g` [B, T, H] (or
[B, H, T] with `head_first`) and each sequence (row b, or each `cu_seqlens` segment of a B = 1 pack) split
into chunks of `chunk_size` tokens from its start:
  o[t] = scale * sum_{s = chunk start .. t} g[s]        (reverse: s = t .. chunk end - 1)
computed in fp32, stored in `output_dtype` (None keeps g's dtype); `scale` None means 1. Positions outside
every sequence are not written. `chunk_size` must be a power of two (asserted). g is read-only.

Tolerance class: floating point against the fp64 sum, trusted = torch's fp32 cumsum per chunk on the same
device times the same fp32 scale (factor 2); exact by construction for dyadic decays, whose sums are exact in
fp32: o = fp32(sum * fp32(scale)), one rounding. Unwritten positions and read-only input: exact.

Variants and dispatch (read from the kernel after the contract): heuristics HAS_SCALE, IS_VARLEN; constexprs
REVERSE, HEAD_FIRST, BT (the chunk size), B, H; autotune over num_warps 1/2/4/8 (key B, H, BT, IS_VARLEN,
REVERSE). Inference selects only (varlen, scale, not reversed, not head-first, BT 64, fp32 out); the others
are covered in `test_cumsum_variants`. Sizes: sequence lengths around 64 and 128 (STAGE_PACKS), the
expanded stream of M 2 and 3, H 1 to 32, zero-length padding requests, the padded descriptor rows that
address past sequence 0 (`build_gdp_chunk_descriptors`), padded tail tokens (NaN); every num_warps config;
decays in the production, clamp-on and clamp-off regimes. Production: the GDP 800m prefill (H 32, M 3) on a
4096-token pack of six sequences.

Memory: a direct launch into a guarded, NaN-poisoned output: positions outside every sequence must stay
poisoned and the guards intact; `sanitizer` on a small padded pack. Offsets (bos * H + h) stay far below
2^31 for any prefill, so there is no `large` case.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.common import RCP_LN2
from megatron.core.ssm.ops.gdp.cumsum import chunk_local_cumsum, chunk_local_cumsum_scalar_kernel
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK


def _stream(pack, stream):
    """The decays and boundaries of one stream: ('token', g, cu) or ('expanded', interleaved g, cu * M)."""
    if stream == "token":
        return pack.g, pack.cu_seqlens, "chunk_indices"
    g = dr.interleave_decays(pack.g, pack.M)
    ou.poison_(g[:, pack.tokens * pack.M :])
    return g.contiguous(), pack.cu_seqlens * pack.M, "chunk_indices_dp"


def _check(out, g, cu, scale, *, what):
    """o against the fp64 sum on the positions some sequence owns."""
    rows = dr.owned_rows(cu, g.shape[1]).to(out.device)
    ref = ou.fp64_reference(dr.local_cumsum, g, cu, scale=scale)
    trusted = dr.trusted_fp32(dr.local_cumsum, g, cu, scale=scale, sequential=True)
    ou.assert_close_to_reference(out[:, rows], ref[:, rows], trusted[:, rows], what=what)


CASES = [
    pytest.param(f, id="{pack}-{stream}-H{H}-{regime}".format(**f))
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "stream": ["token", "expanded"],
            "H": [1, 4, 32],
            "regime": list(dr.G_REGIMES),
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_cumsum_matches_fp64(case):
    """The prefill call: varlen pack, production descriptors, scale 1/ln 2, fp32 out."""
    g0 = ou.make_generator(device=DEVICE)
    lens, M = dr.STAGE_PACKS[case["pack"]]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=case["H"],
        head_k=16,
        head_v=16,
        regime=case["regime"],
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    desc = dr.descriptors(pack, device=DEVICE)
    g, cu, key = _stream(pack, case["stream"])
    before = g.clone()
    with dr.pinned_autotuning():
        out = chunk_local_cumsum(
            g,
            chunk_size=CHUNK,
            scale=RCP_LN2,
            cu_seqlens=cu,
            output_dtype=torch.float32,
            chunk_indices=desc[key],
        )
    assert out.dtype == torch.float32
    ou.assert_exact(g, before, what="g is read-only")
    _check(out, g, cu, RCP_LN2, what=f"cumsum ({case})")


def _negative_dyadic(shape, g0):
    """Decays -(|m| 2^e + 1/4), |m| < 16, e in [-2, 2]: multiples of 1/4 below 61 in magnitude (exact in
    bf16), whose sums over 128 terms are exact in fp32; never zero, so no signed zeros arise."""
    d = ou.dyadic_tensor(
        shape, mantissa_bits=4, exponents=(-2, 2), dtype=torch.float32, device=DEVICE, generator=g0
    )
    return -(d.abs() + 0.25)


def _dyadic_pack(lens, H, g0):
    """A pack whose decays are dyadic (exact sums in fp32), with NaN tail tokens."""
    T = sum(lens)
    g = _negative_dyadic((1, T + dr.TAIL_TOKENS, H), g0)
    ou.poison_(g[:, T:])
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    return g, cu


def _exact_expected(g, cu, scale, out_dtype, reverse=False, chunk=CHUNK):
    """fp32(fp32 sum * fp32(scale)): the sum is exact, so this is the contract's single rounding."""
    s = dr.local_cumsum(g.double(), cu, reverse=reverse, chunk=chunk).to(torch.float32)
    if scale is not None:
        s = s * torch.tensor(scale, dtype=torch.float32, device=s.device)
    return s.to(out_dtype)


@ou.requires_sm(ou.HOPPER)
def test_dyadic_cumsum_is_exact_for_every_config():
    """Every autotune candidate (num_warps 1/2/4/8), pinned in turn: with dyadic decays the sums are
    exact in fp32 in any order, so each config must return fp32(sum * fp32(1/ln 2)) bit for bit."""
    g0 = ou.make_generator(device=DEVICE)
    lens = (1, 63, 64, 65, 129, 0)
    g, cu = _dyadic_pack(lens, 4, g0)
    from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors

    cu_list = [int(x) for x in cu]
    idx, _, _, n, _ = build_gdp_chunk_descriptors(cu_list, len(lens), 1, g.shape[1])
    ci = torch.tensor(idx, dtype=torch.int32, device=DEVICE).view(n, 2)
    rows = dr.owned_rows(cu, g.shape[1]).to(DEVICE)
    expected = _exact_expected(g, cu, RCP_LN2, torch.float32)
    configs = dr.autotuner(chunk_local_cumsum_scalar_kernel).configs
    assert len(configs) >= 1
    for config in list(configs):
        with dr.pinned(chunk_local_cumsum_scalar_kernel, config):
            out = chunk_local_cumsum(
                g,
                chunk_size=CHUNK,
                scale=RCP_LN2,
                cu_seqlens=cu,
                output_dtype=torch.float32,
                chunk_indices=ci,
            )
        ou.assert_exact(out[:, rows], expected[:, rows], what=f"cumsum, {dr.config_id(config)}")


VARIANTS = {
    # name: (reverse, head_first, varlen, scale, chunk, in dtype, out dtype)
    "reverse": (True, False, True, RCP_LN2, 64, torch.float32, torch.float32),
    "head_first_fixed": (False, True, False, RCP_LN2, 64, torch.float32, torch.float32),
    "fixed_length": (False, False, False, RCP_LN2, 64, torch.float32, torch.float32),
    "no_scale": (False, False, True, None, 64, torch.float32, torch.float32),
    "chunk16": (False, False, True, RCP_LN2, 16, torch.float32, torch.float32),
    "chunk32_reverse": (True, False, True, None, 32, torch.float32, torch.float32),
    "chunk128": (False, False, True, RCP_LN2, 128, torch.float32, torch.float32),
    "bf16_in_keep_dtype": (False, False, True, None, 64, torch.bfloat16, None),
    "bf16_in_fp32_out": (False, False, True, RCP_LN2, 64, torch.bfloat16, torch.float32),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(VARIANTS))
def test_cumsum_variants(name):
    """The launcher's variants that prefill never selects, exact on dyadic decays: reversed sums,
    head-first layout, fixed-length batches (B = 3), no scale, other chunk sizes, bf16 input."""
    reverse, head_first, varlen, scale, chunk, in_dtype, out_dtype = VARIANTS[name]
    g0 = ou.make_generator(device=DEVICE)
    H = 3
    if varlen:
        lens = (chunk - 1, chunk, chunk + 1, 1, 2 * chunk + 1)
        g, cu = _dyadic_pack(lens, H, g0)
        g = g[:, : sum(lens)].to(in_dtype).contiguous()
        cu_list = cu
        x = g
    else:
        B, T = 3, 2 * chunk + 5
        g = _negative_dyadic((B, T, H), g0).to(in_dtype)
        x = g.transpose(1, 2).contiguous() if head_first else g
        cu_list = None
    with dr.pinned_autotuning():
        out = chunk_local_cumsum(
            x,
            chunk_size=chunk,
            reverse=reverse,
            scale=scale,
            cu_seqlens=None if not varlen else cu_list,
            head_first=head_first,
            output_dtype=out_dtype,
        )
    out_dtype = in_dtype if out_dtype is None else out_dtype
    assert out.dtype == out_dtype
    if head_first:
        out = out.transpose(1, 2)
    if varlen:
        expected = _exact_expected(g.to(torch.float32), cu_list, scale, out_dtype, reverse, chunk)
    else:
        rows = []
        for b in range(g.shape[0]):
            cu_b = torch.tensor([0, g.shape[1]], device=DEVICE)
            rows.append(
                _exact_expected(
                    g[b : b + 1].to(torch.float32), cu_b, scale, out_dtype, reverse, chunk
                )
            )
        expected = torch.cat(rows)
    ou.assert_exact(out, expected, what=f"cumsum variant {name}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
@pytest.mark.parametrize("stream", ["token", "expanded"])
def test_cumsum_writes_only_owned_positions(stream):
    """A direct launch, as the launcher makes it, into a guarded NaN-poisoned output: positions outside
    every sequence (padded tail tokens, past the padded descriptors) stay poisoned, guards intact.
    """
    g0 = ou.make_generator(device=DEVICE)
    lens, M = dr.STAGE_PACKS["M3"]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=2,
        head_k=16,
        head_v=16,
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    desc = dr.descriptors(pack, device=DEVICE)
    g, cu, key = _stream(pack, stream)
    B, T, H = g.shape
    out = ou.guarded_empty((B, T, H), torch.float32, DEVICE)
    with dr.pinned_autotuning():
        chunk_local_cumsum_scalar_kernel[(desc[key].shape[0], B * H)](
            s=g,
            o=out.tensor,
            scale=RCP_LN2,
            cu_seqlens=cu,
            chunk_indices=desc[key],
            T=T,
            B=B,
            H=H,
            BT=CHUNK,
            HEAD_FIRST=False,
            REVERSE=False,
        )
    ou.check_guards(out=out)
    rows = dr.owned_rows(cu, T).to(DEVICE)
    ou.assert_poisoned(out.tensor[:, ~rows], what="positions no sequence owns")
    _check(out.tensor, g, cu, RCP_LN2, what=f"cumsum ({stream}, direct launch)")


PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)
"""A 4096-token prefill of six sequences, boundaries at chunk offsets 0, 1 and 63."""


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("stream", ["token", "expanded"])
def test_cumsum_production(stream):
    """GDP 800m prefill (32 heads, M 3): a 4096-token pack, both streams."""
    g0 = ou.make_generator(device=DEVICE)
    pack = dr.make_pack(
        PRODUCTION_LENS,
        num_householder=3,
        heads=32,
        head_k=16,
        head_v=16,
        generator=g0,
        device=DEVICE,
    )
    desc = dr.descriptors(pack, device=DEVICE)
    g, cu, key = _stream(pack, stream)
    out = chunk_local_cumsum(
        g,
        chunk_size=CHUNK,
        scale=RCP_LN2,
        cu_seqlens=cu,
        output_dtype=torch.float32,
        chunk_indices=desc[key],
    )
    _check(out, g, cu, RCP_LN2, what=f"cumsum production ({stream})")
