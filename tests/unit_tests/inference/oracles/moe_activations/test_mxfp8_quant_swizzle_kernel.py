# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_mxfp8_quant_swizzle_kernel`` (and its row helper ``_mxfp8_quant_swizzle_row``)
through ``mxfp8_quantize``, ``mxfp8_quantize_into`` and ``MXFP8Tensor.from_bf16`` / ``quantize_``
(megatron/core/inference/quantization/mxfp8_quantize.py, mxfp8_tensor.py).

Contract (from the OCP MX v1.0 and OCP FP8 specifications, the launcher's documented API and its
consumers, ``torch.nn.functional.scaled_mm`` / ``scaled_grouped_mm`` with ``BlockWise1x32`` and
``SWIZZLE_32_4_4``; written before reading the kernel; the spec details are in ``_mx_spec``):

* **Computes** the MXFP8 encoding of every row of ``x`` [M, K]: per 32-element block along K, an
  E8M0 scale ``X`` and E4M3 elements ``P_i = RNE(x_i / X)``. ``X`` is the round-up scale of the
  training recipe (the kernel docstring cites Mishra et al.): the smallest power of two with
  ``amax / X <= 448``, clamped to E8M0's range [2**-127, 2**127]; 2**-127 (byte 0x00) for an
  all-zero block. OCP MX §6.3 allows this rule ("other algorithms may be supported"); its
  suggested floor rule differs whenever amax's significand exceeds 1.75 and saturates there.
* **Inputs:** ``x`` [M, K], bf16, fp16 or fp32 (asserted), K a multiple of 32 (asserted).
  ``mxfp8_quantize_into`` writes into caller storage: data [M, K] ``float8_e4m3fn`` contiguous,
  scales 1-D of exactly the tile layout's size, ``uint8`` or ``float8_e8m0fnu``.
* **Outputs:** data [M, K] E4M3 codes; scales in the cuBLAS 128 x 4 tile layout, every padding
  byte (rows past M up to a multiple of 128, columns past K/32 up to a multiple of 4) 0x00, even
  in reused storage. Nothing outside the two outputs is written; ``x`` is not modified.
* **Strided input:** ``x`` is any [M, K] tensor; a launcher that cannot quantize a strided view
  must reject it rather than read the wrong elements.
* **Special values** are implementation-defined by the spec (§4 leaves the scale algorithm out of
  scope; §5.1 decodes a NaN scale as an all-NaN block). Asserted: blocks without NaN or Inf are
  unaffected by a neighbour that has them; a NaN input decodes to NaN; an Inf input decodes to
  NaN or saturates to ±448 * X (Table 3, OVF or SAT).
* **Tolerance class:** bit-exact against the spec, codes and scales reported apart
  (``ou.assert_quantized_bit_exact``), the swizzled buffer byte for byte.

Structure (read after writing the contract): one program per row (grid ``(M,)``); the whole row
in registers, ``BLOCK_K = next_pow2(K)`` with ``BLOCK_GROUPS = BLOCK_K / 32`` blocks of which
``K / 32`` are real; scale tiles of 128 rows by 4 columns. Covered: K around powers of two and
with 1-3 padded scale columns, production widths; M around the 128-row tile; the three input
dtypes; random, wide-range, round-up-tie and E4M3-midpoint, and zero/sign-of-zero inputs; reused
storage; the ``scaled_mm`` consumer (Blackwell); row offsets past 2**31 (``large``).

Runs on H100 and on GB200 (``launch_on_gb200``), where production uses it.
"""

import pytest
import torch

from megatron.core.inference.quantization import mxfp8_quantize as mq
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = [pytest.mark.launch_on_gb200, _common.hardware(ou.HOPPER, ou.BLACKWELL)]

WIDTHS = (32, 64, 96, 128, 160, 224, 992, 1024, 1056, 1856, 2688, 4096, 7168)
ROWS = (1, 2, 127, 128, 129, 256, 257)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)

CASES = ou.pairwise_cases(
    {"width": list(WIDTHS), "kind": list(mx.VALUE_KINDS), "rows": list(ROWS), "dtype": list(DTYPES)}
)


def _case_id(case):
    return f"K{case['width']}-M{case['rows']}-{case['kind']}-{str(case['dtype'])[6:]}"


def quantize_guarded(x):
    """``mxfp8_quantize(x)`` with its data and scale storage guarded (data NaN-poisoned, scale
    storage 0xFF before the launcher clears it)."""
    before = x.clone()
    with _common.launcher_on_cpu(), _common.captured_allocations(mq) as allocs:
        data, scale = mq.mxfp8_quantize(x)
    assert len(allocs.buffers) == 2
    allocs.check("mxfp8_quantize outputs")
    ou.assert_exact(x, before, what="x is read-only")
    assert data.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
    return data, scale


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_quantize_matches_spec(case):
    """Pairwise over K, M, value kind and input dtype."""
    g = ou.make_generator()
    x = mx.mx_inputs(case["rows"], case["width"], case["dtype"], case["kind"], g).to(DEVICE)
    data, scale = quantize_guarded(x)
    mx.assert_matches_spec(data, scale, x, what=_case_id(case))


@pytest.mark.parametrize("width", [160, 1856])
def test_quantize_into_reused_storage(width):
    """``mxfp8_quantize_into`` over storage that held other bytes: every data byte and every
    scale byte, padding included, ends up as the spec says; nothing outside is written."""
    g = ou.make_generator()
    rows = 129
    x = mx.mx_inputs(rows, width, torch.bfloat16, "wide", g).to(DEVICE)
    data = ou.guarded_empty((rows, width), torch.float8_e4m3fn, DEVICE)
    scale = ou.guarded_empty((mx.scale_bytes(rows, width),), torch.uint8, DEVICE)
    scale.tensor.fill_(0xA7)
    with _common.launcher_on_cpu():
        mq.mxfp8_quantize_into(x, data.tensor, scale.tensor)
    ou.check_guards(data=data, scale=scale)
    mx.assert_matches_spec(data.tensor, scale.tensor, x, what="quantize_into")
    # And again with e8m0 storage holding the previous result of different values.
    y = mx.mx_inputs(rows, width, torch.bfloat16, "randn", g).to(DEVICE)
    with _common.launcher_on_cpu():
        mq.mxfp8_quantize_into(y, data.tensor, scale.tensor.view(torch.float8_e8m0fnu))
    ou.check_guards(data=data, scale=scale)
    mx.assert_matches_spec(data.tensor, scale.tensor, y, what="quantize_into, second write")


def test_mxfp8_tensor_entry_points():
    """``MXFP8Tensor.from_bf16(backend="triton")`` and ``quantize_`` (which takes any layout of
    the logical values) produce the spec encoding."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    g = ou.make_generator()
    x = mx.mx_inputs(200, 2688, torch.bfloat16, "wide", g).to(DEVICE)
    with _common.launcher_on_cpu():
        t = MXFP8Tensor.from_bf16(x, backend="triton")
    mx.assert_matches_spec(t.data, t.scale, x, what="from_bf16")
    y = mx.mx_inputs(2688, 200, torch.bfloat16, "randn", g).to(DEVICE).t()  # [200, 2688]
    with _common.launcher_on_cpu():
        t.quantize_(y)
    mx.assert_matches_spec(t.data, t.scale, y.contiguous(), what="quantize_ of a strided view")


def test_strided_input_is_quantized_or_rejected():
    """A strided [M, K] view is quantized by its values, or rejected; never misread.

    Fails on main: ``mxfp8_quantize`` does not check contiguity and its kernel addresses row r
    at ``r * K``, so a transposed view quantizes the wrong elements (suspected finding O2-N2).
    """
    g = ou.make_generator()
    base = mx.mx_inputs(256, 128, torch.bfloat16, "wide", g).to(DEVICE)
    x = base.t()  # [128, 256], strides (1, 128)
    try:
        with _common.launcher_on_cpu():
            data, scale = mq.mxfp8_quantize(x)
    except (AssertionError, ValueError, RuntimeError):
        return
    mx.assert_matches_spec(data, scale, x.contiguous(), what="strided input")


TINY_AMAX = (448.0 * 2.0**-127, 2.0**-126, 2.0**-127, 2.0**-130, 2.0**-133)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tiny_blocks_take_the_smallest_scale(dtype):
    """Blocks whose ``amax / 448`` is at most 2**-127 take E8M0's smallest scale, 0x00
    (2**-127): it is the smallest power of two with ``amax / X <= 448``.

    Fails on main (suspected finding O2-N1): the kernel rounds ``amax / 448`` up by adding
    0x7FFFFF to its fp32 bits, which for an fp32 subnormal quotient yields 2**-126 (byte 0x01).
    The codes then differ too (half the spec's), although both encodings decode to nearly the
    same values. A round-up E8M0 conversion such as ``cvt.rp.satfinite.ue8m0x2.f32`` gives 0x00
    (recalled from the PTX semantics, not run here).
    """
    rows = len(TINY_AMAX)
    x = torch.zeros(rows, 64, dtype=torch.float64)
    for r, amax in enumerate(TINY_AMAX):
        x[r, 0] = amax
        x[r, 1:32] = torch.linspace(-amax, amax, 31, dtype=torch.float64)
        x[r, 32:] = 1.5  # an ordinary block next to it
    x = x.to(dtype).to(DEVICE)
    data, scale = quantize_guarded(x)
    mx.assert_matches_spec(data, scale, x, what=f"tiny blocks, {dtype}")


def test_special_values():
    """NaN and Inf only affect their own block: NaN decodes to NaN, Inf to NaN or ±448 * X."""
    g = ou.make_generator()
    rows, k = 4, 160
    x = mx.mx_inputs(rows, k, torch.bfloat16, "randn", g)
    special = [(0, 3, float("nan")), (1, 40, float("inf")), (2, 70, float("-inf"))]
    special += [(3, 100, float("nan")), (3, 101, float("inf"))]
    for r, c, v in special:
        x[r, c] = v
    x = x.to(DEVICE)
    data, scale = quantize_guarded(x)
    scales, padding = mx.unswizzle(scale.cpu(), rows, k // mx.BLOCK)
    codes = data.cpu().view(torch.uint8)
    bad_blocks = {(r, c // mx.BLOCK) for r, c, _ in special}
    finite = torch.where(torch.isfinite(x.cpu()), x.cpu(), torch.zeros_like(x.cpu()))
    ref_codes, ref_scales = mx.quantize(finite)
    for r in range(rows):
        for b in range(k // mx.BLOCK):
            if (r, b) in bad_blocks:
                continue
            cols = slice(b * mx.BLOCK, (b + 1) * mx.BLOCK)
            ou.assert_exact(codes[r, cols], ref_codes[r, cols], what=f"codes of block {r},{b}")
            ou.assert_exact(scales[r, b], ref_scales[r, b], what=f"scale of block {r},{b}")
    ou.assert_exact(padding, torch.zeros_like(padding), what="tile padding bytes")
    decoded = mx.dequantize(codes, scales)
    for r, c, v in special:
        got = decoded[r, c].item()
        if v != v:
            assert got != got, f"NaN at ({r}, {c}) decodes to {got}"
        else:
            x_scale = mx.e8m0_decode(scales[r, c // mx.BLOCK]).item()
            saturated = got == (448.0 if v > 0 else -448.0) * x_scale
            assert got != got or saturated, f"{v} at ({r}, {c}) decodes to {got}"


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass: a padded scale tile column."""
    g = ou.make_generator()
    x = mx.mx_inputs(129, 160, torch.bfloat16, "boundary", g).to(DEVICE)
    data, scale = quantize_guarded(x)
    mx.assert_matches_spec(data, scale, x, what="sanitizer case")


def identity_weight(k):
    """The [k, k] identity as an MXFP8 weight built from the spec: E4M3 1.0 is 0x38, and every
    block scale is 1.0 (0x7F)."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    codes = torch.zeros(k, k, dtype=torch.uint8)
    codes.fill_diagonal_(0x38)
    scales = torch.full((k, k // mx.BLOCK), mx.E8M0_BIAS, dtype=torch.uint8)
    return MXFP8Tensor(
        data=codes.view(torch.float8_e4m3fn).to(DEVICE),
        scale=mx.swizzle(scales).view(torch.float8_e8m0fnu).to(DEVICE),
        backend="triton",
    )


@_common.needs_cuda
@_common.hardware(ou.BLACKWELL)
@pytest.mark.parametrize("rows,width", [(1, 128), (129, 160), (256, 1856)])
def test_scaled_mm_reads_the_scales(rows, width):
    """The production consumer reads the kernel's scales where the kernel put them.

    ``x_q @ I`` through ``_mm_mxfp8_torch`` (``torch.nn.functional.scaled_mm``, ``BlockWise1x32``,
    ``SWIZZLE_32_4_4``) with an identity weight encoded from the spec equals the spec decode of
    ``x`` exactly (one nonzero product per output, representable in bf16).
    """
    from megatron.core.inference.quantization import utils as qutils
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    assert qutils.HAVE_TORCH_SCALED_MM, "torch.nn.functional.scaled_mm with MX scaling missing"
    g = ou.make_generator()
    x = mx.mx_inputs(rows, width, torch.bfloat16, "wide", g).clamp(-1e30, 1e30).to(DEVICE)
    data, scale = mq.mxfp8_quantize(x)
    out = qutils._mm_mxfp8_torch(
        MXFP8Tensor(data=data, scale=scale, backend="triton"), identity_weight(width)
    )
    expected = mx.dequantize(*mx.quantize(x.cpu()))
    ou.assert_exact(
        out.float().cpu() + 0.0,
        (expected + 0.0).to(torch.bfloat16).float(),
        what="scaled_mm(x_q, I) vs spec decode",
    )


@pytest.mark.parametrize("rows", ou.int32_overflow_params(8192))
def test_offsets_past_int32(rows):
    """Row offsets ``row * K`` reach 2**31 (bf16 source, fp8 output): rows around the crossing
    and the last rows."""
    width = 8192
    ou.require_free_memory(rows * width * 3 + 2**30)
    x = torch.ones(rows, width, dtype=torch.bfloat16, device=DEVICE)
    probe = sorted({0, 2**31 // width - 1, min(2**31 // width, rows - 1), rows - 1})
    g = ou.make_generator()
    x[probe] = mx.mx_inputs(len(probe), width, torch.bfloat16, "wide", g).to(DEVICE)
    data, scale = mq.mxfp8_quantize(x)
    ref_codes, ref_scales = mx.quantize(x[probe].cpu())
    scales, _ = mx.unswizzle(scale.cpu(), rows, width // mx.BLOCK)
    ou.assert_quantized_bit_exact(
        data[probe].cpu().view(torch.uint8),
        scales[probe],
        ref_codes,
        ref_scales,
        elems_per_scale=mx.BLOCK,
        what=f"rows {probe}",
    )
