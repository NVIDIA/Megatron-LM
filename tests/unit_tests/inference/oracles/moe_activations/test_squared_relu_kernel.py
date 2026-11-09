# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_squared_relu_kernel`` / ``padded_squared_relu``
(megatron/core/inference/moe/activations.py).

Contract (from the non-batch-invariant squared-ReLU call site in ``mcore_fused_moe``
(``activation_func(fc1_output, permutation_map, n_used)``, the BF16 experts and the MXFP8 experts
with ``inference_moe_disable_fused_quant_kernels``), training's squared ReLU and the launcher's
documented API; written before reading the kernel):

* **Computes** the squared ReLU of the expert-grouped FC1 output: for every row
  ``r < min(n_used, M)`` with ``permutation_map[r] >= 0``, ``out[r] = bf16(c(x[r])**2)``,
  evaluated in fp32 and rounded once (``_squared_relu``: ``c`` = ReLU, or with ``clamp_scale =
  s`` the tanh soft clamp ``s * tanh(relu(x) / s)``, the fp32 value squared directly as training's
  fused ``weighted_clamped_squared_relu`` does).
* **Inputs:** ``x`` [M, N] bf16, contiguous; ``permutation_map`` [M] integer (int32 in
  production), -1 marks an alignment-padding row; ``n_used`` a 1-element device int32 tensor
  (``inclusive_expert_offsets[-1:]``), read on device (CUDA-graph safe); ``clamp_scale``;
  ``zero_padding``.
* **Outputs:** a new [M, N] bf16 tensor; ``x`` is not modified. Rows ``>= n_used`` are not
  written. Padding rows below ``n_used`` are +0.0 with ``zero_padding`` and not written
  otherwise. ``n_used > M`` processes the M rows and nothing beyond; ``n_used = 0`` writes
  nothing.
* **Special values:** as training's ``relu(x)**2``: NaN propagates, ``-inf`` gives +0, ``+inf``
  gives ``+inf`` unclamped and ``s**2`` clamped; a square past the bf16 range overflows to
  ``+inf``; where the fp32 evaluation passes through a subnormal, ±0 is also accepted (GPU
  flush-to-zero).
* **Tolerance class:** unclamped, exact (``x * x`` of a bf16 value is exact in fp32, so the one
  bf16 rounding is determined; bitwise training's ``squared_relu``). Clamped, per element the
  single bf16 rounding of an fp32 evaluation (``_common.assert_rounded_once``) plus the harness
  policy per row against torch's fp32 ops rounded once; bitwise the live compiled training op on
  a GPU. Padding zeros and untouched rows are exact (bitwise / still poisoned).

Structure (read from the launcher after writing the contract): ``BLOCK_N = min(next_pow2(N),
1024)`` column tiles; a persistent grid of ``NUM_BLOCKS = min(M, 512)`` programs strides over
rows to the buffer height, skipping rows ``>= n_used``; ``CLAMP`` and ``ZERO_PADDING`` are
compile-time variants (production passes ``ZERO_PADDING=False``); int64 row offsets. Covered:
both variants of each; N around the 1024 tile cap and inside one tile, production widths; M and
``n_used`` around the 512-program grid; padding at program starts and wraps; int32 and int64
maps holding token indices in expert-grouped order; special values; CUDA-graph replay with a
changed ``n_used``; offsets past 2**31 (``large``).

Runs on H100 (the torch-backend BF16 squared-ReLU experts).
"""

import pytest
import torch

from megatron.core.inference.moe import activations
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations import _squared_relu as sr
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = _common.hardware(ou.HOPPER)

BLOCK_N_CAP = 1024  # activations.py: BLOCK_N = min(triton.next_power_of_2(N), 1024)
PROGRAMS_CAP = 512  # activations.py: NUM_BLOCKS = min(M, 512)
PRODUCTION_FFN = (1856, 2688)  # moe_ffn_hidden_size / hidden size of the nanov3-style layers

WIDTHS = ou.boundary_sizes(32, BLOCK_N_CAP, multiples=(2,), minimum=1, production=PRODUCTION_FFN)
ROWS = (1, 2, PROGRAMS_CAP - 1, PROGRAMS_CAP, PROGRAMS_CAP + 1, 2 * PROGRAMS_CAP + 1)

CASES = ou.pairwise_cases(
    {
        "width": [w for w in WIDTHS if w not in (63, 64, 65, 2047, 2048)],
        "live": list(_glu.LIVE_COUNTS),
        "padding": list(_glu.PADDING_PATTERNS),
        "rows": list(ROWS),
        "clamp": list(sr.CLAMP_SCALES),
        "zero_padding": [False, True],
        "perm_dtype": [torch.int32, torch.int64],
    }
)


def _case_id(case):
    dtype = str(case["perm_dtype"]).replace("torch.", "")
    zp = "zero" if case["zero_padding"] else "skip"
    return (
        f"N{case['width']}-M{case['rows']}-n{case['live']}-{case['padding']}"
        f"-clamp{case['clamp']}-{zp}-{dtype}"
    )


def run_padded_squared_relu(x, perm, n_used, clamp=None, zero_padding=False):
    """``padded_squared_relu`` with its output guarded and NaN-poisoned; checks the guard bands
    and that the inputs are read-only."""
    before = [x.clone(), perm.clone()]
    with _common.captured_allocations(activations) as allocs:
        out = activations.padded_squared_relu(
            x, perm, n_used, clamp_scale=clamp, zero_padding=zero_padding
        )
    assert len(allocs.buffers) == 1 and out.data_ptr() == allocs.buffers[0].tensor.data_ptr()
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    allocs.check("padded_squared_relu output")
    ou.assert_exact(x, before[0], what="x is read-only")
    ou.assert_exact(perm, before[1], what="permutation_map is read-only")
    return out


def check_rows(out, x, perm, n_used, clamp, zero_padding, what):
    """Live rows against the contract; padding rows zero or untouched; rows past n_used
    untouched."""
    rows = x.shape[0]
    index = torch.arange(rows, device=out.device)
    below = index < min(n_used, rows)
    live = below & (perm.to(out.device) >= 0)
    padding = below & ~live
    sr.assert_squared_relu_values(out[live], x[live], clamp, what=f"{what}: live rows")
    if zero_padding:
        ou.assert_exact(
            out[padding], torch.zeros_like(out[padding]), what=f"{what}: padding rows are +0"
        )
    else:
        ou.assert_poisoned(out[padding], what=f"{what}: padding rows")
    ou.assert_poisoned(out[~below], what=f"{what}: rows past n_used")


def squared_relu_inputs(shape, clamp, generator):
    """FC1-like bf16 inputs; with the clamp, wide enough that tanh bends and saturates."""
    scale = 4.0 if clamp is None else 1.5 * clamp
    return (torch.randn(shape, generator=generator, device=DEVICE) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_values_and_live_rows(case):
    """Pairwise over width, grid size, live count, padding layout, clamp, ZERO_PADDING and map
    dtype."""
    g = ou.make_generator(device=DEVICE)
    rows, width, clamp = case["rows"], case["width"], case["clamp"]
    n_used = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    x = squared_relu_inputs((rows, width), clamp, g)
    perm = _glu.permutation_map(rows, case["padding"], case["perm_dtype"], g, DEVICE)
    out = run_padded_squared_relu(
        x, perm, _common.device_scalar(n_used), clamp, case["zero_padding"]
    )
    check_rows(out, x, perm, n_used, clamp, case["zero_padding"], what=_case_id(case))


@pytest.mark.parametrize("clamp", [None, 10.0])
@pytest.mark.parametrize("width", PRODUCTION_FFN)
def test_production_widths(width, clamp):
    """A production buffer: 8192 expert-grouped rows of a production width, 6000 used, 10%
    alignment padding."""
    g = ou.make_generator(device=DEVICE)
    rows, n_used = 8192, 6000
    x = squared_relu_inputs((rows, width), clamp, g)
    perm = _glu.permutation_map(rows, "random", torch.int32, g, DEVICE)
    out = run_padded_squared_relu(x, perm, _common.device_scalar(n_used), clamp)
    check_rows(out, x, perm, n_used, clamp, False, what=f"N{width} clamp {clamp}")


SPECIAL_INPUTS = (
    0.0,
    -0.0,
    1e-39,  # bf16 subnormal
    -1e-39,
    1e-20,  # square below the fp32 normal range
    1e-19,
    0.5,
    -3.0,
    1.5e19,  # square near the top of the fp32 / bf16 range
    2e19,  # square overflows to inf
    3e38,
    -3e38,
    float("inf"),
    float("-inf"),
)


@pytest.mark.parametrize("zero_padding", [False, True])
@pytest.mark.parametrize("clamp", [None, 8.0])
def test_special_values(clamp, zero_padding):
    """Zeros, subnormals, squares under and over the fp32 range, infinities: the same
    infinities as the fp64 model math rounded to bf16, every finite element its single
    rounding (or ±0 where the fp32 evaluation passes through a subnormal)."""
    x = torch.tensor(SPECIAL_INPUTS, dtype=torch.float64).repeat(2, 1)
    x[1] *= 0.75
    x = x.to(torch.bfloat16)
    if _common.INTERPRET:
        # The Triton 3.8 interpreter converts bf16 subnormals to fp32 wrongly.
        x[(x.double().abs() < _common.FP32_TINY) & (x != 0)] = 0.0
    x = x.to(DEVICE)
    rows = x.shape[0]
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    out = run_padded_squared_relu(x, perm, _common.device_scalar(rows), clamp, zero_padding)
    exact = ou.fp64_reference(sr.squared_relu_exact, x, clamp)
    expected = _common.rne_bf16(exact)
    _common.assert_same_nonfinite(out, expected, what="special values")
    finite = torch.isfinite(expected)
    flush = sr.may_flush(x, clamp).to(finite.device)
    _common.assert_rounded_once(
        out[finite], exact[finite], what="special values", may_flush=flush[finite]
    )
    if clamp is None:
        ok = (out.double() == expected) | (flush & (out == 0))
        assert bool(ok[finite].all()), "unclamped squares are not exact"


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_nan_propagates(clamp):
    """A NaN FC1 output stays NaN, as training's ``relu(NaN) = NaN``; its neighbours are
    unaffected.

    Fails on main (finding O2-N3): ``_clamped_relu`` computes the ReLU as ``tl.maximum(x, 0.0)``
    with Triton's default ``propagate_nan=NONE`` (``maxnumf``), so NaN becomes 0. Fixed by the
    open PR #8955 (s4f-nan-relu), which changes ``_clamped_relu``.
    """
    g = ou.make_generator(device=DEVICE)
    x = squared_relu_inputs((3, 160), clamp, g).cpu()
    x[0, 3] = float("nan")
    x[2, 100:110] = float("nan")
    x = x.to(DEVICE)
    perm = torch.arange(3, dtype=torch.int32, device=DEVICE)
    out = run_padded_squared_relu(x, perm, _common.device_scalar(3), clamp)
    nan = torch.isnan(x)
    ou.assert_exact(torch.isnan(out), nan, what="NaN positions")
    exact = ou.fp64_reference(sr.squared_relu_exact, x, clamp)
    _common.assert_rounded_once(out[~nan], exact[~nan], what="finite neighbours")


def test_n_used_zero_writes_nothing():
    """A batch with no live rows leaves the whole output untouched."""
    g = ou.make_generator(device=DEVICE)
    x = squared_relu_inputs((PROGRAMS_CAP + 1, 96), None, g)
    perm = _glu.permutation_map(x.shape[0], "runs", torch.int32, g, DEVICE)
    for zero_padding in (False, True):
        out = run_padded_squared_relu(x, perm, _common.device_scalar(0), None, zero_padding)
        ou.assert_poisoned(out, what=f"output with n_used = 0 (zero_padding={zero_padding})")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass: two column tiles, padding runs,
    a live count inside the buffer, the clamp and zeroed padding."""
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = 130, 1025, 100
    x = squared_relu_inputs((rows, width), 8.0, g)
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = run_padded_squared_relu(x, perm, _common.device_scalar(n_used), 8.0, True)
    check_rows(out, x, perm, n_used, 8.0, True, what="sanitizer case")


@_common.needs_cuda
@pytest.mark.parametrize("clamp,width", [(None, PRODUCTION_FFN[0]), (8.0, PRODUCTION_FFN[1])])
def test_matches_training_bitwise(clamp, width):
    """Live rows are bitwise training's compiled fused op with unit weights
    (``weighted_squared_relu`` = ``squared_relu``, or ``weighted_clamped_squared_relu``)."""
    from megatron.core.fusions.fused_weighted_squared_relu import (
        weighted_clamped_squared_relu,
        weighted_squared_relu,
    )

    g = ou.make_generator(device=DEVICE)
    rows = 513
    x = squared_relu_inputs((rows, width), clamp, g)
    perm = _glu.permutation_map(rows, "none", torch.int32, g, DEVICE)
    out = run_padded_squared_relu(x, perm, _common.device_scalar(rows), clamp)
    ones = torch.ones(rows, 1, dtype=torch.float32, device=DEVICE)
    if clamp is None:
        expected = weighted_squared_relu(x, ones)
    else:
        expected = weighted_clamped_squared_relu(x, ones, clamp)
    ou.assert_exact(out, expected, what="padded_squared_relu vs training")


@_common.needs_cuda
@pytest.mark.parametrize("zero_padding", [False, True])
def test_cuda_graph_replay_follows_device_count(zero_padding):
    """Captured once, replayed with other live counts: the kernel reads ``n_used`` on device.
    The static output is poisoned before each replay; rows at or past the new count must stay
    poisoned."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 2 * PROGRAMS_CAP, 256
    x = squared_relu_inputs((rows, width), 8.0, g)
    perm = _glu.permutation_map(rows, "program-stride", torch.int32, g, DEVICE)
    n_used = _common.device_scalar(rows)
    kwargs = dict(clamp_scale=8.0, zero_padding=zero_padding)
    activations.padded_squared_relu(x, perm, n_used, **kwargs)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = activations.padded_squared_relu(x, perm, n_used, **kwargs)
    for count in (1, PROGRAMS_CAP + 1, 0, rows, PROGRAMS_CAP - 1):
        ou.poison_(out)
        n_used.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        check_rows(out, x, perm, count, 8.0, zero_padding, what=f"replay with n_used={count}")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(BLOCK_N_CAP))
def test_offsets_past_int32(rows):
    """Input and output offsets ``row * N`` reach 2**31: rows around the crossing and the last
    rows."""
    width = BLOCK_N_CAP
    ou.require_free_memory(rows * width * 2 * 2 + 2**30)
    x = torch.ones(rows, width, dtype=torch.bfloat16, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    probe = [0, 2**31 // width - 1, 2**31 // width, rows - 2, rows - 1]
    probe = sorted({r for r in probe if 0 <= r < rows})
    x[probe] = squared_relu_inputs((len(probe), width), None, g)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    out = activations.padded_squared_relu(x, perm, _common.device_scalar(rows))
    sr.assert_squared_relu_values(out[probe], x[probe], what=f"rows {probe}")
