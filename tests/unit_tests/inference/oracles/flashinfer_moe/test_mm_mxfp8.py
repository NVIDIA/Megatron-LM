# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``mm_mxfp8`` through Megatron's ``_mm_mxfp8_flashinfer`` and
``mm_mxfp8`` (``megatron/core/inference/quantization/utils.py``), the MXFP8 GEMM of the
inference linear layers (``tensor_parallel/inference_layers.py:_apply_linear``) when the weight
was quantized with ``backend="flashinfer"`` (the default of ``MXFP8Tensor.from_bf16``).

Contract (from the call site and FlashInfer's documented API):

* ``mm_mxfp8(x [M, 1, K] BF16, W: MXFP8Tensor [N, K], out=None)`` quantizes ``x`` to MXFP8
  (``from_bf16(x, backend=W.backend)``) and returns ``[M, 1, N]`` BF16 with
  ``out[m, n] = sum_k dequant(x)[m, k] * dequant(W)[n, k]``: both operands decoded from their
  E4M3 codes and 128x4-swizzled E8M0 scales, products accumulated in FP32, one rounding to BF16.
  With ``out [M, N]`` the result is written there (and only there) and returned as a
  ``[M, 1, N]`` view of it.
* FlashInfer requires ``K >= 128`` and ``N >= 128`` (and ``K % 32 == 0``): smaller problems
  raise ValueError instead of computing (Megatron does not check). A wrong ``out`` shape raises
  ValueError. A weight without a backend fails Megatron's assertion. Inputs are read only.

Oracle: fp64 ``x @ W.T`` of MXFP8-exact operands (``spec.mx_exact_values``; block maxima in
[288, 416] times a power of two, so any RCEIL quantizer is lossless) - the weight's bytes are
first checked to decode, by the spec, to exactly those values. Tolerance class: floating point,
per row, bounded by torch's FP32 GEMM on the same device (TF32 off) rounded once to BF16.

Variants: FlashInfer's ``auto`` backend (CUTLASS on SM100; tactics chosen by M), so M covers 1,
8/9 (the low-latency M <= 8 candidate), 127/128/129 (the 128-row tile and scale tile) and 4096;
K covers 128, 160 (not a multiple of 128) and 2688 (production); N 128, 160 and 4096; out
``None`` and preallocated. Hardware: Blackwell.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec


def _weight(n: int, k: int, generator):
    """An MXFP8Tensor (FlashInfer backend) holding the MX-exact ``[n, k]`` values exactly."""
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    values = spec.mx_exact_values(n, k, generator=generator)
    w = MXFP8Tensor.from_bf16(values.to(torch.bfloat16).cuda(), backend="flashinfer")
    decoded = spec.mxfp8_decode(
        spec.as_bytes(w.data).cpu(), spec.unswizzle_128x4(w.scale.cpu(), n, k // 32)
    )
    ou.assert_exact(decoded, values, what="FlashInfer-quantized weight decodes to its values")
    return w, values


def test_operands_are_mx_exact():
    """CPU check of the oracle's operands: the spec quantizer is lossless on them."""
    values = spec.mx_exact_values(129, 160, generator=ou.make_generator())
    ou.assert_exact(spec.mxfp8_roundtrip(values), values, what="operands")
    assert torch.equal(values.to(torch.bfloat16).double(), values), "not exact in BF16"


CASES = ou.pairwise_cases(
    {
        "m": [1, 8, 9, 127, 128, 129, 4096],
        "k": [128, 160, 2688],
        "n": [128, 160, 4096],
        "out": [False, True],
    }
)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"M{c['m']}-K{c['k']}-N{c['n']}-{'out' if c['out'] else 'alloc'}")
        for c in CASES
    ],
)
def test_matches_fp64_gemm(case):
    spec.require_flashinfer()
    from megatron.core.inference.quantization.utils import mm_mxfp8

    m, k, n = case["m"], case["k"], case["n"]
    g = ou.make_generator()
    weight, w_values = _weight(n, k, g)
    x_values = spec.mx_exact_values(m, k, generator=g)
    x = x_values.to(torch.bfloat16).cuda().unsqueeze(1)
    before = (x.clone(), weight.data.clone(), weight.scale.clone())
    out = ou.guarded_empty((m, n), torch.bfloat16) if case["out"] else None
    result = mm_mxfp8(x, weight, out=None if out is None else out.tensor)
    for t, b, name in zip((x, weight.data, weight.scale), before, ("x", "W data", "W scales")):
        ou.assert_exact(t, b, what=f"{name} is read-only")
    assert result.shape == (m, 1, n) and result.dtype == torch.bfloat16
    if out is not None:
        ou.check_guards(out=out)
        assert result.data_ptr() == out.tensor.data_ptr(), "result is not a view of out"
    ref = ou.fp64_reference(torch.matmul, x_values.cuda(), w_values.cuda().T)
    with spec.exact_fp32_matmul():
        trusted = (x_values.float().cuda() @ w_values.float().cuda().T).to(torch.bfloat16)
    ou.assert_close_to_reference(result.squeeze(1), ref, trusted, dim=-1, what=str(case))


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_small_problem_through_flashinfer_entry():
    """``_mm_mxfp8_flashinfer`` on already-quantized operands (the inner call site)."""
    spec.require_flashinfer()
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
    from megatron.core.inference.quantization.utils import _mm_mxfp8_flashinfer

    g = ou.make_generator()
    weight, w_values = _weight(128, 128, g)
    x_values = spec.mx_exact_values(9, 128, generator=g)
    x_mx = MXFP8Tensor.from_bf16(x_values.to(torch.bfloat16).cuda(), backend="flashinfer")
    out = ou.guarded_empty((9, 128), torch.bfloat16)
    _mm_mxfp8_flashinfer(x_mx, weight, out=out.tensor)
    ou.check_guards(out=out)
    ref = ou.fp64_reference(torch.matmul, x_values.cuda(), w_values.cuda().T)
    with spec.exact_fp32_matmul():
        trusted = (x_values.float().cuda() @ w_values.float().cuda().T).to(torch.bfloat16)
    ou.assert_close_to_reference(out.tensor, ref, trusted, dim=-1, what="_mm_mxfp8_flashinfer")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("bad", ["k<128", "n<128", "out-shape"])
def test_unsupported_problems_raise(bad):
    spec.require_flashinfer()
    from megatron.core.inference.quantization.utils import mm_mxfp8

    g = ou.make_generator()
    k = 96 if bad == "k<128" else 128
    n = 96 if bad == "n<128" else 128
    weight, _ = _weight(n, k, g)
    x = spec.mx_exact_values(4, k, generator=g).to(torch.bfloat16).cuda().unsqueeze(1)
    out = torch.empty(4, n + 1, dtype=torch.bfloat16, device="cuda") if bad == "out-shape" else None
    with pytest.raises(ValueError):
        mm_mxfp8(x, weight, out=out)


def test_weight_without_backend_is_rejected():
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
    from megatron.core.inference.quantization.utils import mm_mxfp8

    weight = MXFP8Tensor(data=torch.empty(128, 128), scale=torch.empty(512))
    with pytest.raises(AssertionError, match="backend is None"):
        mm_mxfp8(torch.empty(1, 1, 128, dtype=torch.bfloat16), weight)
