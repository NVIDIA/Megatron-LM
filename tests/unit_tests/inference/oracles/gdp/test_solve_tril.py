# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `merge_16x16_to_64x64_inverse_kernel` (megatron/core/ssm/ops/gdp/solve_tril.py), launched by
`solve_tril`: (I + A)^-1 of each chunk's strictly lower-triangular A, the second step of the WY
representation.

Reached in inference only through `chunk_gated_delta_product_varlen` (GDP and GDN prefill), on the expanded
stream: A [1, T*M, H, 64] fp32 from the kkt kernel, `cu_seqlens * M`, the padded expanded-stream descriptors,
output in the activation dtype. The inverse is internal to the entry point, so it gets this direct test.

Contract (from the launcher's documented arguments and linear algebra, before reading the kernel): for every
chunk of L <= 64 rows of a sequence and head h, with A_c = A[c0:c0+L, h, :L] strictly lower triangular
(assumed, not checked; the rest of the chunk's columns zero):
  Ai[c0:c0+L, h, :L] = (I + A_c)^-1     (unit lower triangular),
zero above the diagonal and in columns >= L, in `output_dtype` (None keeps A's); rows outside every
sequence are not written. The launcher zero-initialises Ai. A is read-only.

Tolerance class: floating point against the fp64 inverse (`torch.linalg.solve_triangular`), trusted = the
same in torch fp32 on the same device, rounded to the output dtype (factor 2); the zeros above the diagonal
are exact (as values: the kernel may store -0). Exact by construction for structured A whose inverse is
small integers or I - A: the sub-diagonal -1 (inverse all ones), +1 (inverse +-1), A = 0, and dyadic A
confined to one off-diagonal 16 x 16 block (A^2 = 0, so the inverse is I - A); these pin each of the four
diagonal 16-row blocks (forward substitution) and each of the six off-diagonal merges to its place.

Variants and dispatch (read from the kernel after the contract): heuristics IS_VARLEN; constexprs H, BT
(64 only, asserted), USE_TMA (`FLA_USE_TMA=1` on sm90+, read at import: not covered, no caller sets it);
autotune over num_warps {2, 4, 8} x num_stages {2, 3, 4, 5} x DOT_PRECISION (`ieee` only without TMA). The
16-row blocks make the valid rows of a chunk the boundary: last-chunk lengths 1, 2, 15, 16, 17, 31, 32, 33,
47, 48, 49, 62, 63, 64 (`delta_rule.STAGE_PACKS`), with M 1/2/3, padding requests and the padded descriptor
rows, NaN tail rows; output bf16, fp16 and fp32; heads 1 and 4; every num_warps pinned in turn; a
fixed-length batch. Production: GDP 800m (32 heads, M 3) on a 4096-token pack.

Memory: a direct launch into a guarded NaN-poisoned Ai: in each chunk's rows the kernel writes the ten
16 x 16 blocks on and below the block diagonal and nothing else (the six above stay poisoned; the launcher's
zero fill supplies them), rows outside every sequence stay poisoned; `sanitizer` on a small case. Offsets:
the head index is int64 in the kernel; the per-row `(i_t * BT + i) * H * BT` is int32 but needs a single
sequence of ~2^25 / H expanded tokens; no `large` case.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.common import IS_TMA_SUPPORTED
from megatron.core.ssm.ops.gdp.solve_tril import merge_16x16_to_64x64_inverse_kernel, solve_tril
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
BLOCK = 16
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _check(Ai, A, cu, *, what):
    T = A.shape[1]
    rows = dr.owned_rows(cu, T).to(A.device)
    ref = ou.fp64_reference(dr.tril_inverse, A, cu)
    trusted = dr.trusted_fp32(dr.tril_inverse, A, cu).to(Ai.dtype)
    zeros = Ai[0].transpose(1, 2)[dr.structural_zeros(cu, T, diagonal=False).to(A.device)]
    ou.assert_exact(zeros.abs(), torch.zeros_like(zeros), what=f"{what}: zeros above the diagonal")
    ou.assert_close_to_reference(Ai[:, rows], ref[:, rows], trusted[:, rows], what=what)


def _chain_A(pack_name, *, H, regime, g0, dtype=torch.bfloat16):
    lens, M = dr.STAGE_PACKS[pack_name]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=H,
        head_k=64,
        head_v=16,
        dtype=dtype,
        regime=regime,
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    return pack, dr.build_chain(pack, device=DEVICE)


CASES = [
    pytest.param(f, id="{pack}-H{H}-{out}-{regime}".format(**f))
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "H": [1, 4],
            "out": list(DTYPES),
            "regime": list(dr.G_REGIMES),
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_tril_inverse_matches_fp64(case):
    """The prefill call on realistic A (from L2-normalised keys, betas and decays), pairwise over the
    boundary packs, heads, output dtypes and decay regimes."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain_A(case["pack"], H=case["H"], regime=case["regime"], g0=g0)
    A = chain.A
    before = A.clone()
    with dr.pinned_autotuning():
        Ai = solve_tril(
            A=A,
            cu_seqlens=chain.cu_dp,
            chunk_indices=chain.desc["chunk_indices_dp"],
            output_dtype=DTYPES[case["out"]],
        )
    assert Ai.dtype == DTYPES[case["out"]] and Ai.shape == A.shape
    ou.assert_exact(A, before, what="A is read-only")
    _check(Ai, A, chain.cu_dp, what=f"(I + A)^-1 ({case})")


@ou.requires_sm(ou.HOPPER)
def test_every_tril_config():
    """Each num_warps (and, with TMA off, the one DOT_PRECISION) pinned in turn."""
    assert not IS_TMA_SUPPORTED, "FLA_USE_TMA is set: the TMA path is not what inference runs"
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain_A("M1b", H=2, regime="production", g0=g0)
    warps = {}
    for config in dr.autotuner(merge_16x16_to_64x64_inverse_kernel).configs:
        cost = (config.num_stages,)
        if config.num_warps not in warps or cost < (warps[config.num_warps].num_stages,):
            warps[config.num_warps] = config
    assert set(warps) == {2, 4, 8}
    for config in warps.values():
        with dr.pinned(merge_16x16_to_64x64_inverse_kernel, config):
            Ai = solve_tril(
                A=chain.A,
                cu_seqlens=chain.cu_dp,
                chunk_indices=chain.desc["chunk_indices_dp"],
                output_dtype=torch.bfloat16,
            )
        _check(Ai, chain.A, chain.cu_dp, what=f"(I + A)^-1 {dr.config_id(config)}")


def _structured(kind, lens, H, g0):
    """A [1, T, H, 64] fp32 and the exact inverse, for `kind` (see the module docstring)."""
    T = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE)
    A = torch.zeros(1, T, H, CHUNK, dtype=torch.float64, device=DEVICE)
    expected = torch.zeros_like(A)
    for _, start, end in dr.chunks(cu):
        L = end - start
        a = torch.zeros(H, L, L, dtype=torch.float64, device=DEVICE)
        if kind == "minus_subdiag":
            a += torch.diag_embed(-torch.ones(H, L - 1, dtype=torch.float64, device=DEVICE), -1)
        elif kind == "plus_subdiag":
            a += torch.diag_embed(torch.ones(H, L - 1, dtype=torch.float64, device=DEVICE), -1)
        elif kind.startswith("block"):
            r, c = int(kind[5]), int(kind[6])
            rows, cols = slice(BLOCK * r, min(BLOCK * (r + 1), L)), slice(
                BLOCK * c, min(BLOCK * (c + 1), L)
            )
            if rows.start < L and cols.start < L:
                n_r, n_c = rows.stop - rows.start, cols.stop - cols.start
                a[:, rows, cols] = ou.dyadic_tensor(
                    (H, n_r, n_c),
                    mantissa_bits=3,
                    exponents=(-3, 0),
                    dtype=torch.bfloat16,
                    device=DEVICE,
                    generator=g0,
                ).double()
        eye = torch.eye(L, dtype=torch.float64, device=DEVICE).expand(H, L, L)
        inv = torch.linalg.solve_triangular(eye + a, eye, upper=False)
        A[0, start:end, :, :L] = a.permute(1, 0, 2)
        expected[0, start:end, :, :L] = inv.permute(1, 0, 2)
    return A.to(torch.float32), expected, cu


STRUCTURED = ["zero", "minus_subdiag", "plus_subdiag"] + [
    f"block{r}{c}" for r in range(4) for c in range(r)
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("out", ["bf16", "fp32"])
@pytest.mark.parametrize("kind", STRUCTURED)
def test_structured_inverse_is_exact(kind, out):
    """Inverses that are exact in any output dtype: each diagonal block's forward substitution and each
    off-diagonal merge must land its entries in place, bit for bit (as values), on a full chunk and on
    partial chunks of 40 and 17 rows."""
    g0 = ou.make_generator(device=DEVICE)
    A, expected, cu = _structured(kind, (64, 40, 17), 2, g0)
    if kind == "plus_subdiag" or kind == "minus_subdiag":
        assert expected.abs().max() == 1, "the structured inverse must be +-1 entries"
    with dr.pinned_autotuning():
        Ai = solve_tril(A=A, cu_seqlens=cu, output_dtype=DTYPES[out])
    # "+ 0.0" maps -0 to +0: the contract fixes the values, not the sign of the zeros.
    ou.assert_exact(Ai + 0.0, expected.to(DTYPES[out]) + 0.0, what=f"inverse of {kind}")


@ou.requires_sm(ou.HOPPER)
def test_fixed_length_batch():
    """No `cu_seqlens` (IS_VARLEN off): B = 2 rows of 2 chunks and a partial one."""
    g0 = ou.make_generator(device=DEVICE)
    B, T, H = 2, 2 * CHUNK + 33, 2
    lens, M = (T,), 1
    A_rows, refs = [], []
    for _ in range(B):
        pack = dr.make_pack(
            lens, num_householder=M, heads=H, head_k=32, head_v=16, generator=g0, device=DEVICE
        )
        A_rows.append(dr.build_chain(pack, device=DEVICE).A)
    A = torch.cat(A_rows)
    with dr.pinned_autotuning():
        Ai = solve_tril(A=A, output_dtype=torch.bfloat16)
    cu = torch.tensor([0, T], device=DEVICE)
    for b in range(B):
        _check(Ai[b : b + 1], A[b : b + 1], cu, what=f"(I + A)^-1 batch row {b}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_tril_writes_only_its_blocks():
    """A direct launch, as the launcher makes it, into a guarded NaN-poisoned Ai: each owned row gets the
    16-column blocks up to its own block (checked against the inverse), the blocks above stay poisoned,
    and rows outside every sequence stay poisoned; guards intact."""
    g0 = ou.make_generator(device=DEVICE)
    pack, chain = _chain_A("M3", H=2, regime="clamp_off", g0=g0)
    A = chain.A
    B, T, H, BT = A.shape
    Ai = ou.guarded_empty((B, T, H, BT), torch.bfloat16, DEVICE)
    ci = chain.desc["chunk_indices_dp"]
    with dr.pinned_autotuning():
        merge_16x16_to_64x64_inverse_kernel[ci.shape[0], B * H](
            A=A,
            Ai=Ai.tensor,
            cu_seqlens=chain.cu_dp,
            chunk_indices=ci,
            T=T,
            H=H,
            BT=BT,
            USE_TMA=IS_TMA_SUPPORTED,
        )
    ou.check_guards(Ai=Ai)
    written = torch.zeros(T, BT, dtype=torch.bool)
    for _, start, end in dr.chunks(chain.cu_dp):
        for r in range(end - start):
            written[start + r, : BLOCK * (r // BLOCK + 1)] = True
    written = written.to(DEVICE)
    out = Ai.tensor[0].transpose(1, 2)  # [T, BT, H]
    ou.assert_poisoned(
        out[~written], what="blocks above the block diagonal and rows no sequence owns"
    )
    ref = ou.fp64_reference(dr.tril_inverse, A, chain.cu_dp)[0].transpose(1, 2)
    trusted = dr.trusted_fp32(dr.tril_inverse, A, chain.cu_dp)[0].transpose(1, 2).bfloat16()
    ou.assert_close_to_reference(
        out[written], ref[written], trusted[written], what="written blocks"
    )


PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)


@ou.requires_sm(ou.HOPPER)
def test_tril_production():
    """GDP 800m prefill: 32 heads, M 3, a 4096-token pack (12288 expanded rows), bf16 output, with the
    autotuner's own (timed) choice."""
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
    chain = dr.build_chain(pack, device=DEVICE)
    Ai = solve_tril(
        A=chain.A,
        cu_seqlens=chain.cu_dp,
        chunk_indices=chain.desc["chunk_indices_dp"],
        output_dtype=torch.bfloat16,
    )
    _check(Ai, chain.A, chain.cu_dp, what="(I + A)^-1 production")
