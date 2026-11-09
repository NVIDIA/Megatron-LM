# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `l2norm_fwd_kernel` and `l2norm_fwd_kernel1` (megatron/core/ssm/ops/gdp/common.py), launched by
`l2norm_fwd`: the q/k L2 normalisation of the GDP/GDN chunked prefill.

Reached in inference only through `chunk_gated_delta_product_varlen(use_qk_l2norm_in_kernel=True)`
(megatron/core/ssm/ops/gdp/chunk.py), which normalises q [1, T, H, K] and k [1, T*M, H, K] before the
chunked stages: always for GDP (`GatedDeltaProductMixer.ssm_prefill`), when `use_qk_l2norm` for GDN
(`GatedDeltaNet.ssm_prefill`). The normalised tensors are internal to the entry point, and its second
kernel (D > 512) is unreachable there (K <= 256 is asserted by the state kernel), so it gets this direct test.

Contract (from the launcher's documented behaviour and the GDP/GDN math, before reading the kernels): for x
[..., D], contiguous, every row r over the last dimension:
  y[r] = x[r] / sqrt(sum_d x[r, d]^2 + eps)        (eps = 1e-6 by default)
computed in fp32, y in x's dtype and shape. The launcher also writes rstd[r] = 1 / sqrt(sum + eps) (fp32,
for a backward pass; not returned). A feature dimension of 64 KiB or more raises RuntimeError. x is
read-only.

Tolerance class: floating point against the fp64 formula, trusted = the same formula in torch fp32 on the
same device, cast to x's dtype (factor 2). Special values follow the formula: a zero row gives 0, a NaN
spreads only within its row, an inf row gives NaN. Rows whose sum of squares overflows fp32 (|x| above
~1e19) are outside the documented fp32 numerics and are not tested.

Variants and dispatch (read from the kernels after the contract): D <= 512 -> `l2norm_fwd_kernel`, a block
of BT rows per program, BD = next_pow2(D) (masked), autotuned over BT in {8, 16, 32, 64, 128} x num_warps
in {1, 2, 4, 8, 16} with key (D, NB), NB = cdiv(T, 65536); D > 512 -> `l2norm_fwd_kernel1`, one row per
program, autotuned over num_warps {1, ..., 32}. Covered: D in {1, 15, 16, 17, 63, 64, 65, 96, 127, 128, 129,
255, 256, 257, 511, 512, 513, 1024, 1025} (both kernels and every BD class), rows around every BT (1, 7, 8,
9, 127, 128, 129) and around the NB step (65536, 65537), bf16 and fp16 (q is asserted non-fp32 by the
entry point; k shares its dtype in every caller), every BT tile pinned in turn, the autotuner's timed choice
at the production shape, special values, the 64 KiB error. Production: GDP 800m prefill k on a 4096-token
pack (32 heads, M 3, K 128: 393216 rows) and the GDN shape (32 heads, K 128).

Memory: a direct launch into guarded NaN-poisoned y and rstd for both kernels; `sanitizer` on a small case.
Offsets: the block kernel addresses through block pointers (64-bit strides); the row kernel's `i_t * D` is
int32 but only runs for D > 512, which inference never selects; not covered at 2^31.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.common import l2norm_fwd, l2norm_fwd_kernel, l2norm_fwd_kernel1
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
BLOCK_KERNEL_MAX_D = 512
"""`l2norm_fwd` uses the block kernel for D <= 512 and the row kernel above."""
BT_TILES = (8, 16, 32, 64, 128)
NB_ROWS = 2048 * 32
"""NB = cdiv(T, 65536) is part of the block kernel's autotune key."""

DIMS = sorted(
    set(ou.boundary_sizes(16, 64, 128, 256, BLOCK_KERNEL_MAX_D, multiples=(), degenerate=False))
    | {1, 96, 1024, 1025}
)
ROWS = (1, 7, 8, 9, 127, 128, 129)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _reference(x, eps=dr.L2NORM_EPS):
    return dr.l2norm(x, eps)


def _rstd(x, eps=dr.L2NORM_EPS):
    return 1 / torch.sqrt((x * x).sum(-1) + eps)


def _check(y, x, *, what):
    ref = ou.fp64_reference(_reference, x)
    trusted = dr.trusted_fp32(_reference, x).to(x.dtype)
    ou.assert_close_to_reference(y, ref, trusted, what=what)


def _tile_config(index):
    """The block kernel's BT tiles in turn, at their cheapest warps."""
    tiles = dr.distinct_tiles(l2norm_fwd_kernel, ["BT"])
    return tiles[index % len(tiles)]


def _cheapest(kernel):
    from megatron.core.tuning.selection import cheapest

    return cheapest(dr.autotuner(kernel).configs)


def _run(x, config_index):
    """l2norm_fwd with one candidate pinned (rotating over BT for the block kernel)."""
    D = x.shape[-1]
    kernel = l2norm_fwd_kernel if D <= BLOCK_KERNEL_MAX_D else l2norm_fwd_kernel1
    config = _tile_config(config_index) if D <= BLOCK_KERNEL_MAX_D else _cheapest(kernel)
    with dr.pinned(kernel, config):
        return l2norm_fwd(x), dr.config_id(config)


CASES = [
    pytest.param(
        D,
        ROWS[i % len(ROWS)],
        list(DTYPES)[i % 2],
        id=f"D{D}-rows{ROWS[i % len(ROWS)]}-{list(DTYPES)[i % 2]}",
    )
    for i, D in enumerate(DIMS)
] + [pytest.param(128, rows, dt, id=f"D128-rows{rows}-{dt}") for rows in ROWS for dt in DTYPES]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("D,rows,dtype", CASES)
def test_l2norm_matches_fp64(D, rows, dtype):
    """Every feature size class (both kernels, masked BD tails) and rows around every BT tile."""
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(rows, 2, D, generator=g, device=DEVICE).to(DTYPES[dtype])
    before = x.clone()
    y, cfg = _run(x, D + rows)
    assert y.shape == x.shape and y.dtype == x.dtype
    ou.assert_exact(x, before, what="x is read-only")
    _check(y, x, what=f"l2norm D={D} rows={rows} {dtype} ({cfg})")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("D", [96, 128])
def test_every_l2norm_tile(D):
    """Each BT tile of the block kernel pinned in turn, on rows that leave a partial last tile."""
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(2 * 128 + 3, 4, D, generator=g, device=DEVICE).bfloat16()
    for config in dr.distinct_tiles(l2norm_fwd_kernel, ["BT"]):
        with dr.pinned(l2norm_fwd_kernel, config):
            y = l2norm_fwd(x)
        _check(y, x, what=f"l2norm D={D} {dr.config_id(config)}")
    assert {c.kwargs["BT"] for c in dr.distinct_tiles(l2norm_fwd_kernel, ["BT"])} == set(BT_TILES)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("rows", [NB_ROWS, NB_ROWS + 1], ids=["NB1", "NB2"])
def test_rows_around_the_nb_key_step(rows):
    """65536 and 65537 rows: NB (an autotune key and constexpr) steps from 1 to 2."""
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(rows, 128, generator=g, device=DEVICE).bfloat16()
    y, cfg = _run(x, rows)
    _check(y, x, what=f"l2norm rows={rows} ({cfg})")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("D", [64, 1024], ids=["block", "row"])
def test_special_values_stay_in_their_row(D):
    """Zero rows give 0; a NaN spreads over its row; an inf gives NaN there and 0 beside it (x / inf);
    tiny and large finite values normalise; the other rows are unaffected."""
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(9, D, generator=g, device=DEVICE)
    x[1] = 0
    x[2, 3] = float("nan")
    x[3, 0] = float("inf")
    x[4] *= 1e-30
    x[5] *= 1e15
    x[6, : D // 2] = 0
    x = x.bfloat16()
    y, cfg = _run(x, 0)
    ou.assert_exact(y[1], torch.zeros_like(y[1]), what="zero row")
    nan_rows = torch.isnan(y).any(-1).nonzero().flatten().tolist()
    assert nan_rows == [2, 3], f"NaN must stay in rows 2 and 3, found it in rows {nan_rows}"
    _check(y, x, what=f"rows with special values (D={D}, {cfg})")


@ou.requires_sm(ou.HOPPER)
def test_feature_dim_of_64_kib_raises():
    """D * itemsize >= 64 KiB is rejected before any launch."""
    x = torch.zeros(1, 32769, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError, match="64KB"):
        l2norm_fwd(x)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
@pytest.mark.parametrize("D", [96, 600], ids=["block", "row"])
def test_l2norm_memory(D):
    """Direct launches as `l2norm_fwd` makes them, into guarded NaN-poisoned y and rstd: every row is
    written once (y and rstd checked against the formula), nothing outside."""
    g = ou.make_generator(device=DEVICE)
    T = 129
    x = torch.randn(T, D, generator=g, device=DEVICE).bfloat16()
    y = ou.guarded_empty((T, D), torch.bfloat16, DEVICE)
    rstd = ou.guarded_empty((T,), torch.float32, DEVICE)
    BD = 1 << (D - 1).bit_length()
    if D <= BLOCK_KERNEL_MAX_D:
        config = _tile_config(3)
        with dr.pinned(l2norm_fwd_kernel, config):
            l2norm_fwd_kernel[lambda meta: (triton_cdiv(T, meta["BT"]),)](
                x=x,
                y=y.tensor,
                rstd=rstd.tensor,
                eps=dr.L2NORM_EPS,
                T=T,
                D=D,
                BD=BD,
                NB=triton_cdiv(T, NB_ROWS),
            )
    else:
        with dr.pinned(l2norm_fwd_kernel1, _cheapest(l2norm_fwd_kernel1)):
            l2norm_fwd_kernel1[(T,)](
                x=x, y=y.tensor, rstd=rstd.tensor, eps=dr.L2NORM_EPS, D=D, BD=BD
            )
    ou.check_guards(y=y, rstd=rstd)
    _check(y.tensor, x, what=f"y (direct launch, D={D})")
    ref = ou.fp64_reference(_rstd, x)
    ou.assert_close_to_reference(rstd.tensor, ref, dr.trusted_fp32(_rstd, x), what="rstd")


def triton_cdiv(a, b):
    return -(-a // b)


PRODUCTION = {
    # GDP 800m prefill k: 4096 tokens x M 3 x 32 heads, K 128.
    "gdp_k": (4096 * 3 * 32, 128),
    # GDN prefill q/k after the GQA repeat: 4096 tokens x 32 heads, K 128.
    "gdn_qk": (4096 * 32, 128),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_l2norm_production(name):
    """Production prefill shapes; `gdn_qk` with the autotuner's own (timed) choice over all 25
    candidates, `gdp_k` pinned to the cheapest (deterministic mode)."""
    rows, D = PRODUCTION[name]
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(1, rows, D, generator=g, device=DEVICE).bfloat16()
    if name == "gdn_qk":
        y = l2norm_fwd(x)
    else:
        with dr.pinned_autotuning():
            y = l2norm_fwd(x)
    _check(y, x, what=f"l2norm production {name}")
