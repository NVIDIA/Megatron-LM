# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_squared_relu_quantize_kernel`` / ``squared_relu_and_quantize_mxfp8``
(megatron/core/inference/moe/activations.py).

Contract (from the ``mcore_fused_moe`` call site, where this kernel replaces the unfused
``padded_squared_relu`` followed by ``mxfp8_quantize``, training's squared ReLU and the OCP MX
spec; written before reading the kernel):

* **Computes**, for every row ``r < min(n_used, M)`` with ``permutation_map[r] >= 0``, the MXFP8
  encoding (``_mx_spec``: E4M3 elements, round-up E8M0 scales over 32-element blocks, cuBLAS
  128 x 4 scale tiles) of the bf16 activation ``a = bf16(c(x)**2)``: ``c(x) = relu(x)``, or with
  ``clamp_scale = s`` the tanh soft clamp ``s * tanh(relu(x) / s)`` (training applies
  ``s * tanh(x / s)`` before the ReLU; the two commute). The square is materialized in bf16
  before quantizing, as the unfused path does, so the MXFP8 bins are those of the bf16
  activation.
* **Inputs:** ``x`` [M, K] bf16 contiguous, K a multiple of 32 (asserted); ``permutation_map``
  [M] int (-1 = padding); ``n_used`` 1-element device int32; ``clamp_scale`` float or None.
* **Outputs:** an ``MXFP8Tensor`` (data [M, K] E4M3, swizzled E8M0 scales). Rows past
  ``n_used`` and padding rows: data not written, scale bytes 0x00 (the scale storage starts
  zeroed); every tile padding byte 0x00. ``x`` is not modified.
* **Special values:** as training's squared ReLU, NaN propagates (``relu(NaN) = NaN``) and must
  decode to NaN (OCP FP8 Table 3); ``+inf`` decodes to NaN or saturates; ``-inf`` is 0.
* **Tolerance class:** bit-exact against the spec. Unclamped, ``a`` is exact (``x**2`` of a bf16
  value rounded once), so codes and scales are bit-exact. Clamped, ``a`` is the single bf16
  rounding of an fp32 tanh evaluation: either bf16 neighbour within the fp32 evaluation window
  of a tie (``_common.FP32_EVALUATION_WINDOW``); each block's scale and code must then be the
  spec quantization of one of the admissible activations.

Structure: persistent grid ``NUM_BLOCKS = min(M, 512)`` over rows; the whole row in registers,
``BLOCK_K = next_pow2(K)`` with ``K / 32`` real groups; ``CLAMP`` compile-time variant. Covered:
both ``CLAMP`` variants, K around powers of two and with padded scale columns, production
widths, M and ``n_used`` around the grid and the 128-row tile, padding layouts.

Runs on H100 and on GB200 (``launch_on_gb200``), where production uses it.
"""

import pytest
import torch

from megatron.core.inference.moe import activations
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = [pytest.mark.launch_on_gb200, _common.hardware(ou.HOPPER, ou.BLACKWELL)]

PROGRAMS_CAP = 512  # activations.py: NUM_BLOCKS = min(M, 512)

CASES = ou.pairwise_cases(
    {
        "width": [32, 96, 160, 992, 1024, 1056, 1856, 2688],
        "rows": [1, 127, 128, 129, 511, 512, 513, 1025],
        "live": list(_glu.LIVE_COUNTS),
        "padding": list(_glu.PADDING_PATTERNS),
        "clamp": [None, 8.0, 16.0],
    }
)


def _case_id(c):
    return f"K{c['width']}-M{c['rows']}-n{c['live']}-{c['padding']}-clamp{c['clamp']}"


def activation_bounds(x, clamp):
    """The admissible bf16 activations of ``x`` (fp64 in): equal unless the clamped value is
    within the fp32 evaluation window of a bf16 rounding midpoint."""
    x = x.to(torch.float64)
    r = x.clamp(min=0.0)  # NaN stays NaN
    if clamp is None:
        a = _common.rne_bf16(r * r)
        return a, a
    c = clamp * torch.tanh(r / clamp)
    exact = c * c
    nearest = _common.rne_bf16(exact)
    lo, hi = _common.bf16_neighbours(exact)
    near_tie = (exact - (lo + hi) / 2).abs() <= _common.FP32_EVALUATION_WINDOW * exact.abs()
    return torch.where(near_tie, lo, nearest), torch.where(near_tie, hi, nearest)


def assert_quantized_activation(codes, scales, x, clamp, what):
    """Rows of kernel codes [n, K] and unswizzled scales [n, K/32] against the spec quantization
    of an admissible activation of ``x`` [n, K]."""
    lo, hi = activation_bounds(x.cpu(), clamp)
    codes, scales = codes.cpu().view(torch.uint8), scales.cpu()
    s_lo, s_hi = mx.quantize(lo)[1], mx.quantize(hi)[1]
    scale_ok = (scales == s_lo) | (scales == s_hi)
    n, k = codes.shape
    x_scale = mx.e8m0_decode(scales).repeat_interleave(mx.BLOCK, dim=-1)
    c_lo = mx.e4m3_encode(mx.e4m3_round(lo / x_scale))
    c_hi = mx.e4m3_encode(mx.e4m3_round(hi / x_scale))
    code_ok = (codes == c_lo) | (codes == c_hi)
    if not (bool(scale_ok.all()) and bool(code_ok.all())):
        ref_codes, ref_scales = mx.quantize(_common.rne_bf16(lo))
        ou.assert_quantized_bit_exact(
            codes, scales, ref_codes, ref_scales, elems_per_scale=mx.BLOCK, what=what
        )
        raise AssertionError(f"{what}: {int((~code_ok).sum())} codes inadmissible")


def run_fused(x, perm, n_used, clamp):
    before = x.clone()
    with _common.captured_allocations(activations) as allocs:
        t = activations.squared_relu_and_quantize_mxfp8(x, perm, n_used, clamp_scale=clamp)
    assert len(allocs.buffers) == 2
    allocs.check("squared_relu_and_quantize_mxfp8 outputs")
    ou.assert_exact(x, before, what="x is read-only")
    assert t.data.dtype == torch.float8_e4m3fn and t.scale.dtype == torch.float8_e8m0fnu
    assert t.backend == "triton" and t.data.shape == x.shape
    return t


def check(t, x, perm, n_used, clamp, what):
    rows, k = x.shape
    index = torch.arange(rows)
    live = (index < min(n_used, rows)) & (perm.cpu() >= 0)
    scales, padding = mx.unswizzle(t.scale.cpu(), rows, k // mx.BLOCK)
    ou.assert_exact(padding, torch.zeros_like(padding), what=f"{what}: tile padding bytes")
    ou.assert_exact(
        scales[~live], torch.zeros_like(scales[~live]), what=f"{what}: scales of skipped rows"
    )
    data = t.data.cpu().view(torch.uint8)
    poison = ou.poison_(torch.empty(1, dtype=torch.float8_e4m3fn, device=t.data.device))
    poison = poison.cpu().view(torch.uint8)
    ou.assert_exact(
        data[~live], poison.expand_as(data[~live]), what=f"{what}: data of skipped rows untouched"
    )
    if bool(live.any()):
        assert_quantized_activation(data[live], scales[live], x.cpu()[live], clamp, what)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_matches_spec(case):
    """Pairwise over K, M, live count, padding layout and the clamp."""
    g = ou.make_generator(device=DEVICE)
    rows, width = case["rows"], case["width"]
    n_used = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    scale = 4.0 if case["clamp"] is None else 12.0  # the clamp bites at |x| ~ s
    x = (torch.randn(rows, width, generator=g, device=DEVICE) * scale).to(torch.bfloat16)
    perm = _glu.permutation_map(rows, case["padding"], torch.int32, g, DEVICE)
    t = run_fused(x, perm, _common.device_scalar(n_used), case["clamp"])
    check(t, x, perm, n_used, case["clamp"], _case_id(case))


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_mxfp8_boundaries(clamp):
    """Activations spread over many binades within each block, zeros, negative inputs and
    saturated tanh (``a = s**2``)."""
    rows, width = 128, 256
    g = ou.make_generator()
    q = mx.mx_inputs(rows, width, torch.float32, "boundary", g).double()
    # x = sqrt(a) for the boundary activations a = |q| when representable after squaring.
    x = q.abs().sqrt().to(torch.bfloat16)
    x[:, ::7] = -x[:, ::7]
    x[::5, :32] = 0.0
    x[3::5, 40:72] = 1e4  # tanh saturates: a = s**2
    x = x.to(DEVICE)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = run_fused(x, perm, _common.device_scalar(rows), clamp)
    check(t, x, perm, rows, clamp, f"boundaries, clamp {clamp}")


@pytest.mark.parametrize("clamp", [None, 8.0])
def test_special_values(clamp):
    """NaN propagates through the squared ReLU and decodes to NaN; +inf decodes to NaN or
    saturates; -inf gives 0; blocks without them are unaffected.

    Fails on main (suspected finding O2-N3): the ReLU is ``tl.maximum(x, 0.0)`` with Triton's
    default ``propagate_nan=NONE``, which lowers to ``maxnumf`` and returns 0 for a NaN input
    (the interpreter's ``np.fmax`` does the same), so an upstream NaN is quantized to 0 instead
    of surfacing as training's ``relu(NaN) = NaN`` would.
    """
    rows, width = 4, 160
    g = ou.make_generator()
    x = torch.randn(rows, width, generator=g).to(torch.bfloat16)
    special = [(0, 3, float("nan")), (1, 40, float("inf")), (2, 70, float("-inf"))]
    for r, c, v in special:
        x[r, c] = v
    x = x.to(DEVICE)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    t = run_fused(x, perm, _common.device_scalar(rows), clamp)
    scales, _ = mx.unswizzle(t.scale.cpu(), rows, width // mx.BLOCK)
    codes = t.data.cpu().view(torch.uint8)
    # NaN blocks, and +inf blocks unless the clamp maps +inf to s (tanh saturates).
    bad = {(r, c // mx.BLOCK) for r, c, v in special if v != v or (v > 0 and clamp is None)}
    for r in range(rows):
        for b in range(width // mx.BLOCK):
            if (r, b) in bad:
                continue
            cols = slice(b * mx.BLOCK, (b + 1) * mx.BLOCK)
            assert_quantized_activation(
                codes[r : r + 1, cols],
                scales[r : r + 1, b : b + 1],
                x.cpu()[r : r + 1, cols],
                clamp,
                what=f"block {r},{b}",
            )
    decoded = mx.dequantize(codes, scales)
    nan_value = decoded[0, 3].item()
    assert nan_value != nan_value, f"relu(NaN)**2 decodes to {nan_value}, not NaN"
    inf_value = decoded[1, 40].item()
    sat = 448.0 * mx.e8m0_decode(scales[1, 40 // mx.BLOCK]).item()
    if clamp is None:
        assert inf_value != inf_value or inf_value == sat, f"relu(inf)**2 decodes to {inf_value}"


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass."""
    g = ou.make_generator(device=DEVICE)
    x = (torch.randn(130, 160, generator=g, device=DEVICE) * 4).to(torch.bfloat16)
    perm = _glu.permutation_map(130, "runs", torch.int32, g, DEVICE)
    t = run_fused(x, perm, _common.device_scalar(100), 8.0)
    check(t, x, perm, 100, 8.0, "sanitizer case")
