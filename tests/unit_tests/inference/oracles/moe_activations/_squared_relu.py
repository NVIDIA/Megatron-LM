# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The squared-ReLU contract shared by the squared-ReLU kernel oracles (not a test module).

Model math (training: ``megatron/core/activations.py`` ``squared_relu`` and the fused
``weighted_squared_relu`` / ``weighted_clamped_squared_relu`` of
``megatron/core/fusions/fused_weighted_squared_relu.py``):

    a = c(x)**2 [* p]        c(x) = relu(x), or s * tanh(relu(x) / s) with the tanh soft clamp

where ``s`` is ``activation_func_tanh_clamp_scale`` and ``p`` the routing probability of the row
(the batch-invariant kernels apply it at the activation). Training applies ``s * tanh(x / s)``
before the ReLU; the soft clamp is non-decreasing and maps 0 to 0, so the two orders agree. The
fused training ops evaluate this in fp32 (bf16 inputs upcast) and round once to bf16.

* **Unclamped**, that fp32 evaluation is exact IEEE arithmetic with no freedom left: ``x`` is a
  bf16 value, so ``x * x`` is exact in fp32 (unless it overflows or falls below the normal
  range), and the probability multiply is one correctly rounded fp32 product. The bf16 result is
  therefore determined bit for bit (``unclamped_fp32``): the tolerance class is exact.
* **Clamped**, ``tanh`` is not correctly rounded in fp32, so the result is the single bf16
  rounding of an fp32 evaluation: the correctly rounded value of the fp64 model math, or either
  bf16 neighbour where that value lies within ``_common.FP32_EVALUATION_WINDOW`` of a rounding
  midpoint (the fp32 evaluation errs by a few fp32 ulps, about 2**-21 relative).

NaN propagates (training's ``relu(NaN) = NaN``), ``-inf`` gives 0, ``+inf`` gives ``+inf``
unclamped and ``s**2 [* p]`` clamped.
"""

from typing import Optional

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx

CLAMP_SCALES = (None, 8.0, 16.0)
"""Unclamped, and two tanh soft-clamp scales (``activation_func_tanh_clamp_scale``)."""


def squared_relu_exact(
    x: torch.Tensor, clamp: Optional[float] = None, probs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """fp64 model math ``c(x)**2 [* p]`` (``probs`` [rows], one per row); NaN stays NaN."""
    r = torch.where(torch.isnan(x), x, x.clamp(min=0.0))  # relu, NaN kept
    if clamp is not None:
        s = _glu.fp32_bound(clamp)
        r = s * torch.tanh(r / s)
    out = r * r
    if probs is not None:
        out = out * probs.reshape(-1, 1)
    return out


def unclamped_fp32(x: torch.Tensor, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The fp32 evaluation of ``relu(x)**2 [* p]`` for bf16 ``x`` and fp32 ``p``, as fp64.

    ``relu(x)**2`` of a bf16 value has at most 16 significant bits, so it is exact in fp64 and
    the fp64 -> fp32 cast reproduces the fp32 square exactly (or its overflow to inf); the
    product of that with an fp32 probability has at most 48 significant bits, exact in fp64, so
    the second cast is the correctly rounded fp32 product.
    """
    x = x.to(torch.float64)
    r = torch.where(torch.isnan(x), x, x.clamp(min=0.0))
    square = (r * r).to(torch.float32).to(torch.float64)
    if probs is None:
        return square
    product = square * probs.to(torch.float64).reshape(-1, 1)
    return product.to(torch.float32).to(torch.float64)


def activation_bounds(
    x: torch.Tensor, clamp: Optional[float] = None, probs: Optional[torch.Tensor] = None
) -> tuple:
    """The admissible bf16 activations ``(lo, hi)`` of bf16 ``x`` (fp64 tensors of bf16 values).

    Unclamped, both are the one bf16 value of the exact fp32 evaluation. Clamped, they are the
    correctly rounded value, or the two bf16 neighbours where the fp64 model math lies within the
    fp32 evaluation window of a rounding midpoint.
    """
    x = x.cpu()
    p = None if probs is None else probs.cpu()
    if clamp is None:
        a = _common.rne_bf16(unclamped_fp32(x, p))
        return a, a
    exact = squared_relu_exact(x.to(torch.float64), clamp, None if p is None else p.double())
    nearest = _common.rne_bf16(exact)
    lo, hi = _common.bf16_neighbours(exact)
    near_tie = (exact - (lo + hi) / 2).abs() <= _common.FP32_EVALUATION_WINDOW * exact.abs()
    return torch.where(near_tie, lo, nearest), torch.where(near_tie, hi, nearest)


def squared_relu_trusted(
    x: torch.Tensor, clamp: Optional[float] = None, probs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The trusted implementation: torch's fp32 ops on the same device, rounded once to bf16.
    Independent of the kernels (no Triton)."""
    r = torch.relu(x.float())
    if clamp is not None:
        r = clamp * torch.tanh(r / clamp)
    out = r * r
    if probs is not None:
        out = out * probs.float().reshape(-1, 1)
    return out.to(torch.bfloat16)


def may_flush(x: torch.Tensor, clamp: Optional[float] = None, probs=None) -> torch.Tensor:
    """Elements whose fp32 evaluation passes through a subnormal (a subnormal input, square or
    result): GPU fp32 arithmetic may flush those to zero, so ``±0`` is admissible there too."""
    xd = x.cpu().to(torch.float64)
    exact = squared_relu_exact(xd, clamp, None if probs is None else probs.cpu().double())
    tiny = _common.FP32_TINY
    sub_input = (xd.abs() < tiny) & (xd != 0)
    sub_square = (xd > 0) & (xd * xd < tiny)
    return sub_input | sub_square | ((exact.abs() < tiny) & (exact != 0))


def assert_squared_relu_values(
    out: torch.Tensor,
    x: torch.Tensor,
    clamp: Optional[float] = None,
    probs: Optional[torch.Tensor] = None,
    *,
    what: str,
) -> None:
    """Rows of a bf16 squared-ReLU output against the contract, for finite normal-range inputs.

    Unclamped: bitwise the exact fp32 evaluation rounded once. Clamped: every element the single
    bf16 rounding of an fp32 evaluation (``_common.assert_rounded_once``), and the harness policy
    per row against torch's fp32 ops rounded once (``squared_relu_trusted``).
    """
    if out.numel() == 0:
        return
    if clamp is None:
        lo, _ = activation_bounds(x, None, probs)
        ou.assert_exact(out, lo.to(torch.bfloat16).to(out.device), what=what)
        return
    exact = ou.fp64_reference(squared_relu_exact, x, clamp, probs)
    _common.assert_rounded_once(out, exact, what=what)
    trusted = squared_relu_trusted(x, clamp, probs)
    ou.assert_close_to_reference(out, exact, trusted, dim=-1, what=what)


def assert_quantized_activation(
    codes: torch.Tensor, scales: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, *, what: str
) -> None:
    """Kernel codes [n, K] and unswizzled E8M0 scales [n, K/32] against the spec quantization
    (``_mx_spec``) of an admissible activation in ``[lo, hi]`` (``activation_bounds``).

    Where ``lo == hi`` everywhere this is bit-exactness against ``mx.quantize(lo)``. Otherwise
    each block's scale must be the spec scale of ``lo`` or of ``hi``, and each code the spec
    encoding, under the kernel's scale, of the element's ``lo`` or ``hi``.
    """
    codes, scales = codes.cpu().view(torch.uint8), scales.cpu().view(torch.uint8)
    ref_codes, ref_scales = mx.quantize(lo)
    if bool((lo == hi).all()):
        ou.assert_quantized_bit_exact(
            codes, scales, ref_codes, ref_scales, elems_per_scale=mx.BLOCK, what=what
        )
        return
    s_hi = mx.quantize(hi)[1]
    scale_ok = (scales == ref_scales) | (scales == s_hi)
    x_scale = mx.e8m0_decode(scales).repeat_interleave(mx.BLOCK, dim=-1)
    c_lo = mx.e4m3_encode(mx.e4m3_round(lo / x_scale))
    c_hi = mx.e4m3_encode(mx.e4m3_round(hi / x_scale))
    code_ok = (codes == c_lo) | (codes == c_hi)
    if not (bool(scale_ok.all()) and bool(code_ok.all())):
        ou.assert_quantized_bit_exact(
            codes, scales, ref_codes, ref_scales, elems_per_scale=mx.BLOCK, what=what
        )
        raise AssertionError(f"{what}: {int((~code_ok).sum())} codes inadmissible")
