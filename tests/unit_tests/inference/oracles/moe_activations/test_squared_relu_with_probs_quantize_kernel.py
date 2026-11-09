# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_squared_relu_with_probs_quantize_kernel`` / ``squared_relu_with_probs_quantize_mxfp8``
(megatron/core/inference/moe/batch_invariant.py, added by #8645).

Contract (from the batch-invariant MXFP8 squared-ReLU call site in ``mcore_fused_moe``, where this
launcher replaces ``squared_relu_with_probs(..., zero_padding=True)`` followed by
``mxfp8_quantize_rows(.., n_used)``, from training's fused weighted squared ReLU, which batch
invariance must match, and from the OCP MX spec):

* **Computes**, for every row ``r < min(n_used, M)``:
  - a live row (``permutation_map[r] >= 0``): the MXFP8 encoding (``_mx_spec``: E4M3 elements,
    round-up E8M0 scales over 32-element blocks, cuBLAS 128 x 4 scale tiles) of the bf16
    activation ``a = bf16(c(x)**2 * probs[r])``, evaluated in fp32 and rounded once, bitwise
    training's fused ``weighted_squared_relu`` / ``weighted_clamped_squared_relu``
    (``_squared_relu``: ``c`` = ReLU, or the tanh soft clamp with ``clamp_scale``);
  - an alignment-padding row (``permutation_map[r] < 0``): the encoding of a zero row, codes 0x00
    and scale bytes 0x00 (the spec's all-zero block, ``X = 2**-127``), because the grouped GEMM
    reads it.
* **Inputs:** ``x`` [M, K] bf16, K a multiple of 32; ``permutation_map`` [M] int32 (int64
  accepted), the source token of each row; ``n_used`` a 1-element device int32 tensor
  (``inclusive_expert_offsets[-1:]``), read on device (CUDA-graph safe); ``probs`` [M] fp32, the
  routing probability of the permuted row (``permuted_probs``); ``clamp_scale`` float or None.
* **Outputs:** an ``MXFP8Tensor`` (data [M, K] E4M3, swizzled E8M0 scales, backend ``triton``,
  logical dtype bf16). Rows at or past ``n_used``: data not written, scale bytes 0x00; every tile
  padding byte 0x00; ``n_used > M`` quantizes the M rows and writes nothing past them. Inputs are
  not modified. A strided ``x`` is quantized by its values or rejected.
* **Special values:** NaN propagates (training's ``relu(NaN) = NaN``) and decodes to NaN; ``+inf``
  (unclamped) decodes to NaN or saturates; ``-inf`` gives 0; a zero probability gives a zero row;
  blocks without such values are unaffected. Blocks whose amax is at most ``448 * 2**-127`` take
  E8M0's smallest scale, 0x00 (the round-up rule within E8M0's range).
* **Tolerance class:** bit-exact against the spec quantization of the contract's activation.
  Unclamped, the activation is determined bit for bit (``x * x`` exact in fp32, one fp32 product
  with the probability, one bf16 rounding). Clamped, either bf16 neighbour within the fp32
  evaluation window of a tie is admissible, and the codes and scales must be the spec
  quantization of an admissible activation. On a GPU, also exact against the spec quantization of
  the live compiled training op's output.

Structure (from the launcher): a persistent grid of ``NUM_BLOCKS = max(1, min(M, 512))`` programs
strides over rows ``< min(n_used, M)``; the whole row is in registers (``BLOCK_K =
next_pow2(K)``, ``K / 32`` real groups of ``BLOCK_K / 32``); ``CLAMP`` is a compile-time variant.
Covered: both ``CLAMP`` variants; K around powers of two and with 1-3 padded scale columns,
production widths; M and ``n_used`` around the 512-program grid and the 128-row scale tile
(``n_used = 0`` and ``> M``); padding layouts with int32 and int64 maps holding token indices in
expert-grouped order; round-up scale ties, E4M3 midpoints and wide-range blocks; special values;
CUDA-graph replay with changing ``n_used``; a production buffer; offsets past 2**31 (``large``).

Runs on H100 and on GB200 (``launch_on_gb200``), where the MXFP8 expert path runs it.
"""

import pytest
import torch

from megatron.core.inference.moe import batch_invariant
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx
from tests.unit_tests.inference.oracles.moe_activations import _squared_relu as sr
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = [pytest.mark.launch_on_gb200, _common.hardware(ou.HOPPER, ou.BLACKWELL)]

PROGRAMS_CAP = 512  # batch_invariant.py: num_blocks = max(1, min(num_rows, 512))
PRODUCTION_FFN = (1856, 2688)  # moe_ffn_hidden_size / hidden size of the nanov3-style layers

CASES = ou.pairwise_cases(
    {
        "width": [32, 96, 160, 992, 1024, 1056, *PRODUCTION_FFN],
        "rows": [1, 127, 128, 129, PROGRAMS_CAP - 1, PROGRAMS_CAP, PROGRAMS_CAP + 1, 1025],
        "live": list(_glu.LIVE_COUNTS),
        "padding": list(_glu.PADDING_PATTERNS),
        "clamp": list(sr.CLAMP_SCALES),
        "perm_dtype": [torch.int32, torch.int64],
    }
)


def _case_id(c):
    dtype = str(c["perm_dtype"]).replace("torch.", "")
    return f"K{c['width']}-M{c['rows']}-n{c['live']}-{c['padding']}-clamp{c['clamp']}-{dtype}"


def run_fused(x, perm, n_used, probs, clamp):
    """The launcher with its data and scale storage guarded (data NaN-poisoned, scale storage
    zeroed as the launcher asks); checks the guards, the result's metadata and read-only inputs."""
    before = [t.clone() for t in (x, perm, probs)]
    with _common.captured_allocations(batch_invariant) as allocs:
        t = batch_invariant.squared_relu_with_probs_quantize_mxfp8(x, perm, n_used, probs, clamp)
    assert len(allocs.buffers) == 2
    allocs.check("squared_relu_with_probs_quantize_mxfp8 outputs")
    for name, now, then in zip(("x", "permutation_map", "probs"), (x, perm, probs), before):
        ou.assert_exact(now, then, what=f"{name} is read-only")
    assert t.data.dtype == torch.float8_e4m3fn and t.scale.dtype == torch.float8_e8m0fnu
    assert t.backend == "triton" and t.dtype == x.dtype and t.data.shape == x.shape
    assert t.scale.numel() == mx.scale_bytes(*x.shape)
    return t


def poison_byte(device):
    """The byte an untouched (NaN-poisoned) E4M3 element holds."""
    poison = ou.poison_(torch.empty(1, dtype=torch.float8_e4m3fn, device=device))
    return poison.cpu().view(torch.uint8)


def check(t, x, perm, probs, n_used, clamp, what):
    """Live rows against the spec quantization of an admissible activation; padding rows below
    ``n_used`` zero blocks; rows past ``n_used`` untouched with zero scales; tile padding zero."""
    rows, k = x.shape
    index = torch.arange(rows)
    below = index < min(n_used, rows)
    live = below & (perm.cpu() >= 0)
    padding = below & ~live
    scales, pad_bytes = mx.unswizzle(t.scale.cpu(), rows, k // mx.BLOCK)
    data = t.data.cpu().view(torch.uint8)
    ou.assert_exact(pad_bytes, torch.zeros_like(pad_bytes), what=f"{what}: tile padding bytes")
    ou.assert_exact(
        scales[~below], torch.zeros_like(scales[~below]), what=f"{what}: scales past n_used"
    )
    ou.assert_exact(
        data[~below],
        poison_byte(t.data.device).expand_as(data[~below]),
        what=f"{what}: data past n_used untouched",
    )
    for name, got in (("codes", data[padding]), ("scales", scales[padding])):
        ou.assert_exact(got, torch.zeros_like(got), what=f"{what}: padding rows, {name} zero")
    if bool(live.any()):
        lo, hi = sr.activation_bounds(x.cpu()[live], clamp, probs.cpu()[live])
        sr.assert_quantized_activation(data[live], scales[live], lo, hi, what=f"{what}: live")


def inputs(rows, width, clamp, generator):
    """FC1-like bf16 inputs (the clamp bites at |x| ~ s) and routing probabilities in [0, 1)."""
    scale = 4.0 if clamp is None else 1.5 * clamp
    x = (torch.randn(rows, width, generator=generator, device=DEVICE) * scale).to(torch.bfloat16)
    probs = torch.rand(rows, generator=generator, device=DEVICE, dtype=torch.float32)
    return x, probs


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_matches_spec(case):
    """Pairwise over K, M, live count, padding layout, clamp and map dtype."""
    g = ou.make_generator(device=DEVICE)
    rows, width, clamp = case["rows"], case["width"], case["clamp"]
    n_used = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    perm = _glu.permutation_map(rows, case["padding"], case["perm_dtype"], g, DEVICE)
    x, probs = inputs(rows, width, clamp, g)
    t = run_fused(x, perm, _common.device_scalar(n_used), probs, clamp)
    check(t, x, perm, probs, n_used, clamp, _case_id(case))


def _midpoint_fractions():
    """bf16 values ``y`` in [2**-10, 1) whose activation ``bf16(448 * y**2)`` is an E4M3 rounding
    midpoint (normal or subnormal): scaled by a block whose amax is ``448 * X``, those elements
    land exactly between two E4M3 codes."""
    exps = torch.arange(-10, 0, dtype=torch.float64)
    mantissas = 1.0 + torch.arange(128, dtype=torch.float64) / 128
    y = (mantissas[None, :] * torch.exp2(exps)[:, None]).reshape(-1)
    q = _common.rne_bf16(448.0 * y * y)
    quantum = _common.format_quantum(q, mx.E4M3_MANTISSA_BITS, mx.E4M3_EMIN)
    steps = q / quantum
    return y[(steps - steps.floor()) == 0.5]


def boundary_inputs(rows, width, generator):
    """Inputs whose activations sit on MXFP8 boundaries, built from exact bf16 arithmetic.

    Each row's probability is ``1.75 * 2**m``. Each 32-block has one amax element ``2**j * t``
    (``j`` per block, so blocks span many binades) with ``t`` = 1, giving ``amax = 448 * 2**(m + 2j
    - 8)``, exactly the round-up scale tie, or ``t`` = the bf16 value above or below 1, giving
    the next activation above or below it. The other elements are ``2**j * y`` with ``y`` from
    ``_midpoint_fractions`` (half of them: E4M3 rounding midpoints, normal and subnormal, under
    the tie's scale), any fraction below 1, zero or negative (ReLU to 0).
    """
    g = generator
    blocks = rows * (width // mx.BLOCK)
    pool = _midpoint_fractions()
    m = torch.randint(-8, 0, (rows,), generator=g)
    probs = (1.75 * torch.exp2(m.double())).float()
    j = torch.randint(-20, 21, (blocks, 1), generator=g).double()
    pick = torch.randint(0, 8, (blocks, mx.BLOCK), generator=g)
    mid = pool[torch.randint(0, pool.numel(), (blocks, mx.BLOCK), generator=g)]
    anywhere = 1.0 - torch.rand(blocks, mx.BLOCK, generator=g, dtype=torch.float64)
    y = torch.where(pick < 4, mid, anywhere).to(torch.bfloat16).double()
    y = torch.where(pick == 6, torch.zeros_like(y), y)
    y = torch.where(pick == 7, -y, y)
    tops = torch.tensor([1.0, 1.0 + 2.0**-7, 1.0 - 2.0**-8], dtype=torch.float64)
    t = tops[torch.randint(0, 3, (blocks,), generator=g)]
    y = y.clamp(max=1.0 - 2.0**-8)  # strictly below the amax element
    col = torch.randint(0, mx.BLOCK, (blocks,), generator=g)
    y[torch.arange(blocks), col] = t
    x = (y * torch.exp2(j)).to(torch.bfloat16).reshape(rows, width)
    return x.to(DEVICE), probs.to(DEVICE)


@pytest.mark.parametrize(
    "rows,width,clamp", [(128, 160, None), (129, 1856, None), (128, 160, 8.0), (257, 1056, 16.0)]
)
def test_mxfp8_boundaries(rows, width, clamp):
    """Round-up scale ties and their neighbours, E4M3 midpoints, blocks across many binades,
    zeros and negative inputs, with probabilities that are not powers of two."""
    g = ou.make_generator()
    x, probs = boundary_inputs(rows, width, g)
    perm = _glu.permutation_map(rows, "none", torch.int32, g, DEVICE)
    t = run_fused(x, perm, _common.device_scalar(rows), probs, clamp)
    check(t, x, perm, probs, rows, clamp, f"boundaries M{rows} K{width} clamp {clamp}")
    if clamp is None:
        # The inputs reach the boundaries they are built for (a check of the construction).
        a, _ = sr.activation_bounds(x.cpu(), None, probs.cpu())
        blocks = a.reshape(-1, mx.BLOCK)
        x_scale = mx.e8m0_decode(mx.quantize(a)[1]).reshape(-1, 1)
        ties = blocks.amax(-1) == mx.E4M3_MAX * x_scale.reshape(-1)
        q = blocks / x_scale
        step = q / _common.format_quantum(q, mx.E4M3_MANTISSA_BITS, mx.E4M3_EMIN)
        midpoints = (step - step.floor()) == 0.5
        assert int(ties.sum()) >= blocks.shape[0] // 6, "too few round-up scale ties"
        assert int(midpoints.sum()) >= blocks.numel() // 10, "too few E4M3 midpoints"


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_special_values(clamp):
    """Infinities, fp32 overflow of the square, signed zeros, subnormal and tiny inputs next to
    ordinary values, a zero probability and a saturated clamp: every block without an infinite
    activation is the spec quantization of its activation; an infinite activation decodes to NaN
    or saturates."""
    rows, width = 6, 160
    g = ou.make_generator()
    x = torch.randn(rows, width, generator=g).to(torch.bfloat16)
    probs = torch.rand(rows, generator=g, dtype=torch.float32) * 0.9 + 0.05
    special = [(0, 3, float("inf")), (0, 40, float("-inf")), (3, 70, 3e38), (5, 100, 1e4)]
    for r, c, v in special:
        x[r, c] = v
    x[1, 32:64] = 0.0
    x[1, 64:96:2] = -0.0
    x[1, 65:96:2] = -1.5
    x[2] = x[2].abs() + 0.5
    probs[2] = 0.0
    # Next to ordinary values in the same block (so the block scale is ordinary and both an
    # exact and a flushed-to-zero evaluation encode as code 0): a bf16 subnormal input, and an
    # input whose square is below the fp32 normal range.
    x[4, 0:8] = 1e-39
    x[4, 8:16] = 1e-20
    x = x.to(DEVICE)
    probs = probs.to(DEVICE)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = run_fused(x, perm, _common.device_scalar(rows), probs, clamp)
    scales, _ = mx.unswizzle(t.scale.cpu(), rows, width // mx.BLOCK)
    codes = t.data.cpu().view(torch.uint8)
    xc, pc = x.cpu(), probs.cpu()
    infinite = {(0, 3), (3, 70)} if clamp is None else set()
    for r in range(rows):
        for b in range(width // mx.BLOCK):
            cols = slice(b * mx.BLOCK, (b + 1) * mx.BLOCK)
            if any(rr == r and cols.start <= cc < cols.stop for rr, cc in infinite):
                continue
            lo, hi = sr.activation_bounds(xc[r : r + 1, cols], clamp, pc[r : r + 1])
            sr.assert_quantized_activation(
                codes[r : r + 1, cols], scales[r : r + 1, b : b + 1], lo, hi, what=f"block {r},{b}"
            )
    ou.assert_exact(codes[2], torch.zeros_like(codes[2]), what="zero-probability row codes")
    decoded = mx.dequantize(codes, scales)
    for r, c in infinite:
        got = decoded[r, c].item()
        sat = 448.0 * mx.e8m0_decode(scales[r, c // mx.BLOCK]).item()
        assert got != got or got == sat, f"infinite activation at ({r}, {c}) decodes to {got}"


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_nan_propagates(clamp):
    """A NaN FC1 output decodes to NaN, as training's ``relu(NaN) = NaN``; other blocks are
    unaffected.

    Fails on main (suspected finding O2-M2): the ReLU is ``tl.maximum(value, 0.0)`` with Triton's
    default ``propagate_nan=NONE`` (``maxnumf``), so NaN becomes 0, as in O2-N3 and O3-N1. The
    open fix PR #8955 (s4f-nan-relu) predates this kernel and does not change it.
    """
    rows, width = 3, 160
    g = ou.make_generator()
    x = torch.randn(rows, width, generator=g).to(torch.bfloat16)
    x[0, 3] = float("nan")
    x[2, 32:64] = float("nan")
    probs = torch.rand(rows, generator=g, dtype=torch.float32) * 0.9 + 0.05
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = run_fused(x.to(DEVICE), perm, _common.device_scalar(rows), probs.to(DEVICE), clamp)
    scales, _ = mx.unswizzle(t.scale.cpu(), rows, width // mx.BLOCK)
    codes = t.data.cpu().view(torch.uint8)
    for r, b in ((0, 1), (0, 4), (1, 0), (1, 3), (2, 0), (2, 2)):
        cols = slice(b * mx.BLOCK, (b + 1) * mx.BLOCK)
        lo, hi = sr.activation_bounds(x[r : r + 1, cols], clamp, probs[r : r + 1])
        sr.assert_quantized_activation(
            codes[r : r + 1, cols], scales[r : r + 1, b : b + 1], lo, hi, what=f"block {r},{b}"
        )
    decoded = mx.dequantize(codes, scales)
    for r, c in ((0, 3), (2, 32), (2, 63)):
        got = decoded[r, c].item()
        assert got != got, f"relu(NaN)**2 * p at ({r}, {c}) decodes to {got}, not NaN"


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_tiny_blocks_take_the_smallest_scale(clamp):
    """Blocks whose activation amax is at most ``448 * 2**-127`` (normal fp32 and bf16 values)
    take E8M0's smallest scale, 0x00: the smallest power of two with ``amax / X <= 448``.

    Fails on main (suspected finding O2-M1): the kernel's quantization rounds ``amax / 448`` up
    by adding 0x7FFFFF to its fp32 bits, which for a subnormal quotient gives 2**-126 (byte
    0x01), as in O2-N1. The open fix PR #8956 (s4f-mxfp8-quant) predates this kernel and does not
    change its copy of the quantization.
    """
    # (x, p): activations 448 * 2**-127 exactly, 2**-120, 2**-123 and 2**-126.
    tiny = [(2.0**-60, 3.5), (2.0**-60, 1.0), (2.0**-62, 2.0), (2.0**-63, 1.0)]
    rows, width = len(tiny), 64
    x = torch.zeros(rows, width, dtype=torch.float64)
    probs = torch.zeros(rows, dtype=torch.float32)
    for r, (v, p) in enumerate(tiny):
        x[r, 0] = v
        x[r, 1:32:3] = -v  # ReLU to zero
        x[r, 2:32:3] = v
        x[r, 32:] = 1.5  # an ordinary block next to it
        probs[r] = p
    x = x.to(torch.bfloat16)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = run_fused(x.to(DEVICE), perm, _common.device_scalar(rows), probs.to(DEVICE), clamp)
    check(t, x, perm, probs, rows, clamp, f"tiny blocks, clamp {clamp}")


def test_strided_input_is_quantized_or_rejected():
    """A strided [M, K] ``x`` (transpose, column slice, row step, broadcast row) is quantized by
    its values or rejected, never misread. The launcher asserts a contiguous ``x``."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 128, 160
    base, probs = inputs(2 * rows, 2 * width, None, g)
    probs = probs[:rows].contiguous()
    views = {
        "transpose": base[:width, :rows].t(),
        "column-slice": base[:rows, :width],
        "row-step": base[::2, :width],
        "broadcast": base[0, :width].expand(rows, width),
    }
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    for name, x in views.items():
        assert x.shape == (rows, width) and not x.is_contiguous()
        before = x.clone()
        try:
            with _common.captured_allocations(batch_invariant) as allocs:
                t = batch_invariant.squared_relu_with_probs_quantize_mxfp8(
                    x, perm, _common.device_scalar(rows), probs
                )
        except (AssertionError, ValueError, RuntimeError):
            continue  # rejected before any launch
        allocs.check(f"outputs for a strided input ({name})")
        ou.assert_exact(x, before, what="x is read-only")
        check(t, before, perm, probs, rows, None, f"strided input ({name})")


@pytest.mark.parametrize("clamp", [None, 10.0])
def test_production_buffer(clamp):
    """A production buffer: 16384 expert-grouped rows of the nanov3 expert width, 12000 used,
    10% alignment padding."""
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = 16384, PRODUCTION_FFN[0], 12000
    perm = _glu.permutation_map(rows, "random", torch.int32, g, DEVICE)
    x, probs = inputs(rows, width, clamp, g)
    t = run_fused(x, perm, _common.device_scalar(n_used), probs, clamp)
    check(t, x, perm, probs, n_used, clamp, f"production buffer, clamp {clamp}")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass: padded scale columns, padding
    runs, a live count inside the buffer, the clamp."""
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = 130, 160, 100
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    x, probs = inputs(rows, width, 8.0, g)
    t = run_fused(x, perm, _common.device_scalar(n_used), probs, 8.0)
    check(t, x, perm, probs, n_used, 8.0, "sanitizer case")


@_common.needs_cuda
@pytest.mark.parametrize("clamp,width", [(None, PRODUCTION_FFN[0]), (8.0, PRODUCTION_FFN[1])])
def test_matches_training_bitwise(clamp, width):
    """Live rows are bitwise the spec quantization of training's compiled fused op
    (``weighted_squared_relu`` / ``weighted_clamped_squared_relu``); padding rows are zero."""
    from megatron.core.fusions.fused_weighted_squared_relu import (
        weighted_clamped_squared_relu,
        weighted_squared_relu,
    )

    g = ou.make_generator(device=DEVICE)
    rows, n_used = 513, 400
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    x, probs = inputs(rows, width, clamp, g)
    t = run_fused(x, perm, _common.device_scalar(n_used), probs, clamp)
    if clamp is None:
        expected = weighted_squared_relu(x, probs[:, None])
    else:
        expected = weighted_clamped_squared_relu(x, probs[:, None], clamp)
    expected = torch.where((perm >= 0)[:, None], expected, torch.zeros_like(expected))
    below = torch.arange(rows) < n_used
    mx.assert_matches_spec(t.data, t.scale, expected, rows=below, what="vs training")
    ou.assert_poisoned(t.data[n_used:], what="data past n_used")


@_common.needs_cuda
def test_cuda_graph_replay_follows_device_count():
    """Captured once, replayed with other ``n_used`` values: the data is poisoned and the scales
    zeroed before each replay, and rows past each count must stay so."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 2 * PROGRAMS_CAP, 160
    perm = _glu.permutation_map(rows, "program-stride", torch.int32, g, DEVICE)
    x, probs = inputs(rows, width, None, g)
    n_used = _common.device_scalar(rows)
    batch_invariant.squared_relu_with_probs_quantize_mxfp8(x, perm, n_used, probs)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t = batch_invariant.squared_relu_with_probs_quantize_mxfp8(x, perm, n_used, probs)
    for count in (1, PROGRAMS_CAP + 1, 0, rows, PROGRAMS_CAP - 1):
        ou.poison_(t.data)
        t.scale.view(torch.uint8).zero_()
        n_used.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        check(t, x, perm, probs, count, None, f"replay with n_used={count}")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(PRODUCTION_FFN[1]))
def test_offsets_past_int32(rows):
    """Row offsets ``row * K`` reach 2**31 at a production width: rows around the crossing and
    the last rows."""
    width = PRODUCTION_FFN[1]
    ou.require_free_memory(rows * width * 3 + 2**30)
    x = torch.ones(rows, width, dtype=torch.bfloat16, device=DEVICE)
    probs = torch.full((rows,), 0.5, dtype=torch.float32, device=DEVICE)
    probe = sorted({0, 2**31 // width - 1, min(2**31 // width, rows - 1), rows - 2, rows - 1})
    g = ou.make_generator(device=DEVICE)
    x[probe], probs[probe] = inputs(len(probe), width, None, g)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = batch_invariant.squared_relu_with_probs_quantize_mxfp8(
        x, perm, _common.device_scalar(rows), probs
    )
    scales, _ = mx.unswizzle(t.scale.cpu(), rows, width // mx.BLOCK)
    lo, hi = sr.activation_bounds(x[probe].cpu(), None, probs[probe].cpu())
    sr.assert_quantized_activation(t.data[probe].cpu(), scales[probe], lo, hi, what=f"rows {probe}")
