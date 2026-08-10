# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization-aware training (QAT) primitive for Megatron Lite.

Scope (see ``docs/qat_cross_framework_design.md``): weight-only QAT. Two family
of formats share one STE + three-state skeleton:

* **Integer** (validated skeleton, kept): ``int8`` (W8A16) and non-NVFP4
  ``int4`` (W4A16-int) — max-calibrated affine/symmetric integer Q/DQ.
* **Floating-point** (the shipping target): ``fp8_e4m3`` (W8A16, E4M3, aligned
  with verl/ModelOpt fp8 weight fake-quant) and ``mxfp4`` (OCP microscaling
  float4: E2M1 element + E8M0 power-of-two block scale, block=32). The float
  encodings reuse torch's native ``float8_e4m3fn`` RNE cast and the OCP E2M1
  grid — they are *not* home-grown bit layouts.

NVFP4 (W4A16 / W4A4, NVIDIA per-block FP8 scale) stays deferred as a second
FP4 option and is rejected loudly, never aliased onto MXFP4.

The primitive enforces the three-state separation mandated by the design:

1. **Master weight** — the trainable parameter stays the original (BF16) weight.
   Fake quantization is applied through ``torch.nn.utils.parametrize`` so the
   raw parameter survives untouched as ``...parametrizations.weight.original``
   and remains what the optimizer updates. QAT never registers ``W_hat`` as a
   parameter, and never forces an FP32 master.
2. **fake-quant / STE** — the forward path uses ``W_hat = dequant(quant(W))``;
   the backward path is a straight-through estimator (optionally clipped to the
   representable range). Quantization ``scale``/``amax`` are *statistics*, held
   in non-trainable buffers, never weight copies.
3. **Deployment representation** — packed integer tensors + scales are produced
   only on demand (export / rollout refit) by :func:`quantize_weight` /
   :func:`pack_int4`; the training step, optimizer and checkpoint never consume
   packed weights.

This module is model-agnostic on purpose (``primitive.design`` replaceability):
it knows nothing about Qwen/GLM/Kimi names and never calls into rollout. Models
opt in from their ``protocol.build_model`` via :func:`apply_qat_to_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

# MXFP4 numerics live in one place (see that module's docstring): the scale rule
# and the element rounding are defined to be bit-identical to the ModelOpt
# quantizer the rollout actually runs, so the error QAT compensates during
# training is the error deployment actually makes.
from megatron.lite.primitive.quantization.mxfp4 import (
    E2M1_LEVELS as _E2M1_LEVELS,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    E2M1_MAX as _E2M1_MAX,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    E8M0_BIAS as _E8M0_BIAS,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE as _MXFP4_BLOCK,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    e2m1_round_index as _e2m1_round_index,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    mx_shared_scale as _mx_shared_scale,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    mx_shared_scale_exponent as _mx_shared_scale_exponent,
)

# Supported formats -> nominal bit-width. Free-form strings are rejected; every
# enum must map to an exact quant/dequant contract.
_FORMAT_BITS: dict[str, int] = {
    "int8": 8,
    "int4": 4,
    "fp8_e4m3": 8,
    "mxfp4": 4,
}

# Canonicalising aliases accepted from configs.
_FORMAT_ALIASES: dict[str, str] = {"fp8": "fp8_e4m3"}

# MXFP4 is intrinsically a microscaling block format; OCP fixes the block to 32
# elements sharing one E8M0 scale. The block size, the E2M1 grid, the E8M0
# shared-scale rule and the element rounding all live in
# ``primitive.quantization.mxfp4`` — the single source of truth, defined to be
# bit-identical to the ModelOpt quantizer the rollout actually runs. They are
# imported above under the module-private aliases this file already used.

# Formats recognised as future work but explicitly not implemented here.
# Selecting one is a loud error naming the deferral, never a silent fallback to
# a different scale layout. NVFP4 uses a per-16 FP8(E4M3) block scale (not the
# E8M0 power-of-two of MXFP4) and needs its own serializer + validation.
_DEFERRED_FORMATS: frozenset[str] = frozenset({"nvfp4_w4a16", "nvfp4_w4a4"})

# Module leaf-names that must never be weight-quantized (numerically fragile /
# tiny). These are generic Megatron surface names, not model names.
_DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    "lm_head",
    "head",  # VocabParallelOutput (qwen3_moe/lite uses ``head.col.linear``)
    "output_layer",
    "gate",  # MoE router gate
    "router",
    "embedding",
    "embed",  # VocabParallelEmbedding wrapper (weight lives at ``embed.embedding``)
    "word_embeddings",
)


def canonical_state_key(key: str) -> str:
    """Map a QAT-parametrized state key back to its logical checkpoint name.

    ``torch.nn.utils.parametrize`` renames ``mod.weight`` to
    ``mod.parametrizations.weight.original`` (the surviving BF16 master), while
    HF checkpoints still reference the logical ``mod.weight``. Strip the
    parametrization wrapper so loaded tensors resolve onto the master weight
    instead of being silently dropped, which would train on random initialized
    weights.

    Only the ``.original`` master is rewritten; quantizer buffers such as
    ``...parametrizations.weight.0.amax`` are left untouched. This is a load-side
    mapping from an HF checkpoint into an already parametrized QAT model. Every
    QAT-enabled model needs it regardless of whether that model can natively
    export MXFP4; deployment export is a separate, opposite-direction contract.
    """
    marker = ".parametrizations."
    if marker not in key or not key.endswith(".original"):
        return key
    head, rest = key.split(marker, 1)
    attr = rest.split(".", 1)[0]
    return f"{head}.{attr}"


@dataclass(frozen=True)
class QATSpec:
    """Typed, explicit opt-in QAT configuration.

    Defaults are inert: ``enabled=False`` means callers get a bit-identical
    model with no quantizer nodes inserted.
    """

    enabled: bool = False
    format: str = "int8"
    # None derives the format default: MXFP4's fixed OCP block is 32; other
    # formats retain per-tensor (0). Explicit 0/-1/N integer grouping is kept.
    group_size: int | None = None
    symmetric: bool = True
    ste_clip: bool = (
        True  # zero grad outside representable range; False = pure pass-through
    )
    ignore_patterns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_IGNORE_PATTERNS
    )
    activation_bits: int | None = (
        None  # weight-only in phase 1; W*A* is gated separately
    )
    learnable_scales: bool = False  # LSQ future work; must be False in phase 1

    def __post_init__(self) -> None:
        # Canonicalise aliases (e.g. "fp8" -> "fp8_e4m3") even when disabled so
        # ``.format`` is always the exact contract key.
        canonical = _FORMAT_ALIASES.get(self.format, self.format)
        if canonical != self.format:
            object.__setattr__(self, "format", canonical)
        if self.group_size is None:
            object.__setattr__(
                self,
                "group_size",
                _MXFP4_BLOCK if canonical == "mxfp4" else 0,
            )
        if not self.enabled:
            return
        if self.format in _DEFERRED_FORMATS:
            raise ValueError(
                f"QAT format {self.format!r} is deferred and not implemented here. NVFP4 uses "
                "a per-16 FP8(E4M3) block scale with its own serializer/validation; do not "
                "alias it onto MXFP4's E8M0 scale."
            )
        if self.format not in _FORMAT_BITS:
            raise ValueError(
                f"Unknown QAT format {self.format!r}; supported: {sorted(_FORMAT_BITS)}."
            )
        if self.activation_bits is not None:
            raise ValueError(
                "activation quantization (W*A*) is not supported here; it needs a "
                "calibration/observer-freeze protocol and cross-DP amax sync before enabling."
            )
        if self.learnable_scales:
            raise ValueError(
                "learnable_scales (LSQ) is deferred; this path uses max calibration."
            )
        assert self.group_size is not None
        if self.group_size < -1:
            raise ValueError(f"group_size must be >= -1, got {self.group_size}.")
        if self.format == "mxfp4" and self.group_size != _MXFP4_BLOCK:
            raise ValueError(
                f"mxfp4 is a microscaling block format: group_size must be {_MXFP4_BLOCK} "
                f"(OCP block), got {self.group_size}. Set group_size={_MXFP4_BLOCK}."
            )

    @property
    def num_bits(self) -> int:
        return _FORMAT_BITS[self.format]

    def targets_module(self, name: str) -> bool:
        """True if a module should be quantized (no path component is on the ignore list).

        Matching is on dotted path *components* (exact, case-insensitive) so that
        e.g. the router leaf ``gate`` is skipped while the MLP linear ``gate_up``
        (a different component) is still quantized.
        """
        components = {c.lower() for c in name.split(".")}
        return not any(pat.lower() in components for pat in self.ignore_patterns)


def normalize_qat_spec(config: QATSpec | dict[str, Any] | None) -> QATSpec:
    if config is None:
        return QATSpec()
    if isinstance(config, QATSpec):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            f"QAT config must be QATSpec, dict, or None, got {type(config)!r}."
        )
    values = dict(config)
    if "ignore_patterns" in values and not isinstance(values["ignore_patterns"], tuple):
        values["ignore_patterns"] = tuple(values["ignore_patterns"])
    return QATSpec(**values)


# ---------------------------------------------------------------------------
# Integer quant/dequant numerics
# ---------------------------------------------------------------------------


def _int_qrange(num_bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer code range.

    Symmetric uses the restricted range ``[-(2^(b-1)-1), 2^(b-1)-1]`` (matches
    TensorRT/ModelOpt symmetric weight quant): scale = amax / (2^(b-1)-1) so the
    max magnitude maps exactly to the top code. Asymmetric (affine) uses the full
    unsigned range ``[0, 2^b-1]`` with a zero-point.
    """
    if symmetric:
        qmax = (1 << (num_bits - 1)) - 1
        return -qmax, qmax
    return 0, (1 << num_bits) - 1


def _reshape_for_groups(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, int]:
    """Reshape a 2D ``[out, in]`` weight so the reduction dim is last.

    Returns ``(view, reduce_dim)``. ``reduce_dim`` is the axis over which amax is
    taken (kept as size-1 for broadcasting).
    """
    if group_size == 0:  # per-tensor
        return weight, -1  # sentinel: reduce over all elements
    if group_size == -1:  # per-output-channel: one scale per row
        return weight, 1
    # block along in-features
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not divide in_features={in_features}."
        )
    view = weight.reshape(out_features, in_features // group_size, group_size)
    return view, 2


def compute_amax(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group max-abs statistic (detached; calibration is not differentiated)."""
    view, reduce_dim = _reshape_for_groups(weight.detach(), group_size)
    if reduce_dim == -1:
        return view.abs().amax()
    return view.abs().amax(dim=reduce_dim, keepdim=True)


def _compute_qparams(
    weight: torch.Tensor, num_bits: int, group_size: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """Return ``(scale, zero_point, qmin, qmax)`` broadcastable to the grouped view."""
    qmin, qmax = _int_qrange(num_bits, symmetric)
    view, reduce_dim = _reshape_for_groups(weight.detach(), group_size)
    eps = torch.finfo(torch.float32).tiny
    if symmetric:
        amax = compute_amax(weight, group_size).float()
        scale = (amax / qmax).clamp_min(eps)
        return scale, None, qmin, qmax
    # affine
    if reduce_dim == -1:
        wmin = view.float().amin()
        wmax = view.float().amax()
    else:
        wmin = view.float().amin(dim=reduce_dim, keepdim=True)
        wmax = view.float().amax(dim=reduce_dim, keepdim=True)
    scale = ((wmax - wmin) / (qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(qmin - wmin / scale)
    return scale, zero_point, qmin, qmax


class _FakeQuantizeSTE(torch.autograd.Function):
    """Q/DQ in forward, straight-through estimator in backward.

    ``scale``/``zero_point`` are treated as constants (calibration statistics),
    so no gradient flows to them. With ``clip=True`` the STE zeroes gradient for
    weights whose (pre-clamp) code falls outside ``[qmin, qmax]``.
    """

    @staticmethod
    def forward(ctx, weight, scale, zero_point, qmin, qmax, clip):  # type: ignore[override]
        orig_dtype = weight.dtype
        w = weight.float()
        if zero_point is None:
            q = torch.round(w / scale)
            q_clamped = q.clamp(qmin, qmax)
            w_hat = q_clamped * scale
        else:
            q = torch.round(w / scale + zero_point)
            q_clamped = q.clamp(qmin, qmax)
            w_hat = (q_clamped - zero_point) * scale
        if clip:
            ctx.save_for_backward((q >= qmin) & (q <= qmax))
            ctx.clip = True
        else:
            ctx.clip = False
        return w_hat.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        if ctx.clip:
            (mask,) = ctx.saved_tensors
            grad_output = grad_output * mask.to(grad_output.dtype)
        return grad_output, None, None, None, None, None


def fake_quantize_weight(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """Differentiable (STE) fake-quantization of a 2D weight per ``spec``."""
    if weight.dim() != 2:
        raise ValueError(
            f"fake_quantize_weight expects a 2D [out, in] weight, got {tuple(weight.shape)}."
        )
    if spec.format == "fp8_e4m3":
        return _fp8_fake_quantize_weight(weight, spec)
    if spec.format == "mxfp4":
        return _mxfp4_fake_quantize_weight(weight, spec)
    scale, zero_point, qmin, qmax = _compute_qparams(
        weight, spec.num_bits, spec.group_size, spec.symmetric
    )
    if spec.group_size > 0:
        out_features, in_features = weight.shape
        view = weight.reshape(
            out_features, in_features // spec.group_size, spec.group_size
        )
        w_hat = _FakeQuantizeSTE.apply(
            view, scale, zero_point, qmin, qmax, spec.ste_clip
        )
        return w_hat.reshape(out_features, in_features)
    return _FakeQuantizeSTE.apply(weight, scale, zero_point, qmin, qmax, spec.ste_clip)


# ---------------------------------------------------------------------------
# Floating-point quant/dequant numerics (fp8 E4M3; MXFP4 = E2M1 + E8M0)
# ---------------------------------------------------------------------------


def _fp8_e4m3_max() -> float:
    return float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


def _fp8_e4m3_qdq(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to E4M3 and back via torch's native RNE cast.

    Saturates to +-448 (the E4M3 finite max) so out-of-range values do not
    become NaN. This is exactly the encoding TE/ModelOpt use for fp8 weights.
    """
    fmax = _fp8_e4m3_max()
    clamped = x.float().clamp(-fmax, fmax)
    return clamped.to(torch.float8_e4m3fn).to(torch.float32)


def _e2m1_qdq(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor onto the OCP E2M1 grid, signed.

    Rounding (ties down) comes from ``mxfp4.e2m1_round_index`` so the training
    graph sees exactly the grid the rollout quantizer produces.
    """
    sign = torch.sign(x)
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=x.device)
    return sign * levels[_e2m1_round_index(x.float().abs())]


class _FloatFakeQuantSTE(torch.autograd.Function):
    """Straight-through estimator for the float fake-quant paths.

    ``w_hat`` is precomputed (under ``no_grad``) by the caller; forward returns
    it and backward passes the gradient straight through to ``weight``, optionally
    masked to the non-saturated region (matching the integer path's ``ste_clip``).
    """

    @staticmethod
    def forward(ctx, weight, w_hat, sat_mask):  # type: ignore[override]
        if sat_mask is not None:
            ctx.save_for_backward(sat_mask)
            ctx.clip = True
        else:
            ctx.clip = False
        return w_hat

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        if ctx.clip:
            (mask,) = ctx.saved_tensors
            grad_output = grad_output * mask.to(grad_output.dtype)
        return grad_output, None, None


def _grouped_view(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """2D->grouped view whose scale broadcasts along the reduction axis."""
    if group_size <= 0:  # per-tensor / per-channel keep 2D
        return weight
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not divide in_features={in_features}."
        )
    return weight.reshape(out_features, in_features // group_size, group_size)


def _fp8_fake_quantize_weight(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """STE fake-quant to E4M3 with an amax/448 float scale (per group)."""
    fmax = _fp8_e4m3_max()
    eps = torch.finfo(torch.float32).tiny
    amax = compute_amax(weight, spec.group_size).float()
    scale = (amax / fmax).clamp_min(eps)
    view = _grouped_view(weight, spec.group_size)
    with torch.no_grad():
        w_scaled = view.float() / scale
        w_hat = (_fp8_e4m3_qdq(w_scaled) * scale).to(weight.dtype)
        mask = (w_scaled.abs() <= fmax) if spec.ste_clip else None
    out = _FloatFakeQuantSTE.apply(view, w_hat, mask)
    return out.reshape(weight.shape)


def _mxfp4_fake_quantize_weight(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """STE fake-quant to MXFP4 (E2M1 element, E8M0 block scale, block=32).

    Bit-identical to what the rollout serves: the scale and the element rounding
    both come from ``primitive.quantization.mxfp4`` (see that module's docstring),
    which mirrors ModelOpt ``MXFP4QTensor.quantize``. Locked by
    ``tests/unit/primitive/test_mxfp4_modelopt_parity_unit.py``.

    Note that with the ModelOpt scale rule ``amax / scale <= 6`` holds by
    construction, so the ``ste_clip`` saturation mask is all-true for MXFP4. That
    is correct rather than vestigial: the deployed quantizer does not saturate
    either, so there is no out-of-range region whose gradient should be dropped.
    The mask is kept so the code path stays uniform with fp8/int.
    """
    view = _grouped_view(weight, _MXFP4_BLOCK)
    with torch.no_grad():
        block_amax = view.float().abs().amax(dim=2, keepdim=True)
        scale = _mx_shared_scale(block_amax)
        w_scaled = view.float() / scale
        w_hat = (_e2m1_qdq(w_scaled) * scale).to(weight.dtype)
        mask = (w_scaled.abs() <= _E2M1_MAX) if spec.ste_clip else None
    out = _FloatFakeQuantSTE.apply(view, w_hat, mask)
    return out.reshape(weight.shape)


# ---------------------------------------------------------------------------
# Deployment representation (packed integers + scales) — export only
# ---------------------------------------------------------------------------


def quantize_weight(weight: torch.Tensor, spec: QATSpec) -> dict[str, torch.Tensor]:
    """Produce the packed deployment snapshot for a BF16 weight.

    Returns a dict with the integer codes (``qweight``), ``scale`` and, for
    affine, ``zero_point``. The training step never calls this; it exists for
    export / rollout refit and for the round-trip validation contract.
    """
    if spec.format == "fp8_e4m3":
        return _quantize_weight_fp8(weight, spec)
    if spec.format == "mxfp4":
        return _quantize_weight_mxfp4(weight)
    scale, zero_point, qmin, qmax = _compute_qparams(
        weight, spec.num_bits, spec.group_size, spec.symmetric
    )
    view, _ = _reshape_for_groups(weight.detach(), spec.group_size)
    w = view.float()
    if zero_point is None:
        codes = torch.round(w / scale).clamp(qmin, qmax)
    else:
        codes = torch.round(w / scale + zero_point).clamp(qmin, qmax)
    codes = codes.reshape(weight.shape).to(torch.int8)
    out = {"qweight": codes, "scale": scale}
    if zero_point is not None:
        out["zero_point"] = zero_point
    return out


def _quantize_weight_fp8(
    weight: torch.Tensor, spec: QATSpec
) -> dict[str, torch.Tensor]:
    """Packed E4M3 deployment snapshot: fp8 codes + fp32 (amax/448) scale."""
    fmax = _fp8_e4m3_max()
    eps = torch.finfo(torch.float32).tiny
    amax = compute_amax(weight, spec.group_size).float()
    scale = (amax / fmax).clamp_min(eps)
    view = _grouped_view(weight.detach(), spec.group_size)
    codes = (view.float() / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    return {
        "qweight": codes.reshape(weight.shape),
        "scale": scale,
        "format": "fp8_e4m3",
    }


def _quantize_weight_mxfp4(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Packed MXFP4 snapshot: E2M1 nibbles (2/byte) + E8M0 scale byte per block.

    ``qweight`` is ``uint8`` packed nibbles of the 4-bit codes ``sign<<3 | mag``;
    ``scale`` is the biased E8M0 exponent (``uint8``) per 32-element block.
    """
    view = _grouped_view(weight.detach(), _MXFP4_BLOCK)  # [out, nblk, 32]
    block_amax = view.float().abs().amax(dim=2, keepdim=True)
    exponent = _mx_shared_scale_exponent(block_amax)  # e, so the scale is 2^e
    scale = torch.exp2(exponent)
    w_scaled = view.float() / scale
    mag = _e2m1_round_index(w_scaled.abs()).to(torch.int32)
    sign_bit = (w_scaled < 0).to(torch.int32)
    codes = ((sign_bit << 3) | mag).to(torch.int32)  # [out, nblk, 32], 0..15
    packed = pack_int4(codes)  # [out, nblk, 16] uint8
    # Encode from the exponent directly: at e = -127 the scale is a float32
    # subnormal, so round-tripping it through log2 would be needlessly fragile.
    e8m0 = (exponent.to(torch.int32) + _E8M0_BIAS).to(torch.uint8)
    return {"qweight": packed, "scale": e8m0, "format": "mxfp4"}


def _dequantize_weight_fp8(
    packed: dict[str, torch.Tensor], spec: QATSpec
) -> torch.Tensor:
    codes = packed["qweight"]
    scale = packed["scale"]
    if spec.group_size > 0:
        out_features, in_features = codes.shape
        view = codes.reshape(
            out_features, in_features // spec.group_size, spec.group_size
        ).float()
        return (view * scale).reshape(out_features, in_features)
    return codes.float() * scale


def _dequantize_weight_mxfp4(packed: dict[str, torch.Tensor]) -> torch.Tensor:
    packed_codes = packed["qweight"]  # [out, nblk, 16] uint8
    e8m0 = packed["scale"]  # [out, nblk, 1] uint8
    codes = unpack_int4(packed_codes, signed=False)  # [out, nblk, 32] int8, 0..15
    mag = (codes & 0x7).to(torch.long)
    sign = ((codes >> 3) & 0x1).float()
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=codes.device)
    scale = torch.exp2(e8m0.float() - _E8M0_BIAS)
    values = (1.0 - 2.0 * sign) * levels[mag] * scale
    out_features, n_blk, block = values.shape
    return values.reshape(out_features, n_blk * block)


def dequantize_weight(packed: dict[str, torch.Tensor], spec: QATSpec) -> torch.Tensor:
    """Inverse of :func:`quantize_weight` — reconstruct the fake-quantized BF16 weight."""
    if spec.format == "fp8_e4m3":
        return _dequantize_weight_fp8(packed, spec)
    if spec.format == "mxfp4":
        return _dequantize_weight_mxfp4(packed)
    codes = packed["qweight"]
    scale = packed["scale"]
    out_features, in_features = codes.shape
    if spec.group_size > 0:
        codes_v = codes.reshape(
            out_features, in_features // spec.group_size, spec.group_size
        ).float()
    else:
        codes_v = codes.float()
    if "zero_point" in packed:
        w = (codes_v - packed["zero_point"]) * scale
    else:
        w = codes_v * scale
    return w.reshape(out_features, in_features)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack a 2D int4 code tensor (last dim even) into ``uint8`` (two codes/byte).

    Codes are the signed range ``[-7, 7]`` (or ``[0, 15]`` affine); they are
    offset into ``[0, 15]`` nibbles. Low nibble = even index, high nibble = odd.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"pack_int4 needs an even last dim, got {codes.shape[-1]}.")
    ints = codes.to(torch.int32)
    lo = ints[..., 0::2] & 0x0F
    hi = ints[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, *, signed: bool = True) -> torch.Tensor:
    """Inverse of :func:`pack_int4`. Returns int8 codes; sign-extends if ``signed``."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    if signed:
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.stack([lo, hi], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    return out.to(torch.int8)


# ---------------------------------------------------------------------------
# Parametrization: keep master weight, fake-quant on access
# ---------------------------------------------------------------------------


def _compute_amax_tensor(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-tensor or per-expert amax for 2D ``[out, in]`` or 3D ``[E, out, in]`` weights."""
    if weight.dim() == 3:
        return torch.stack(
            [compute_amax(weight[i], group_size) for i in range(weight.shape[0])],
            dim=0,
        )
    if weight.dim() != 2:
        raise ValueError(
            f"QAT weight fake-quant expects 2D or 3D stacked weights, got {tuple(weight.shape)}."
        )
    return compute_amax(weight, group_size)


def _fake_quant_weight_tensor(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """Fake-quant a 2D linear weight or a 3D MoE stacked expert weight."""
    if weight.dim() == 3:
        return torch.stack(
            [fake_quantize_weight(weight[i], spec) for i in range(weight.shape[0])],
            dim=0,
        )
    if weight.dim() != 2:
        raise ValueError(
            f"QAT weight fake-quant expects 2D or 3D stacked weights, got {tuple(weight.shape)}."
        )
    return fake_quantize_weight(weight, spec)


class WeightFakeQuant(nn.Module):
    """``torch.nn.utils.parametrize`` module that fake-quantizes a weight.

    Registered on a linear's ``weight``, it moves the trainable master weight to
    ``parametrizations.weight.original`` and returns ``W_hat`` on every access.
    A persistent ``amax`` buffer carries the calibration statistic into the
    checkpoint so quantizer state round-trips (design checkpoint contract).
    """

    def __init__(self, spec: QATSpec, weight_shape: torch.Size):
        super().__init__()
        self.spec = spec
        amax = _compute_amax_tensor(torch.zeros(weight_shape), spec.group_size)
        # persistent so distckpt's named_buffers() loop saves/restores it.
        self.register_buffer("amax", amax.clone(), persistent=True)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.amax.copy_(
                _compute_amax_tensor(weight, self.spec.group_size).to(self.amax.dtype)
            )
        return _fake_quant_weight_tensor(weight, self.spec)


# ---------------------------------------------------------------------------
# Apply to modules / chunks (model-agnostic opt-in surface)
# ---------------------------------------------------------------------------


def _quantizable_weight_owner(module: nn.Module) -> nn.Module | None:
    """Return the sub-object that owns a quantizable ``weight`` Parameter, if any.

    MLite parallel linears wrap the real GEMM as ``module.linear`` (TE) whose
    ``.weight`` is the parameter; plain ``nn.Linear`` owns ``.weight`` directly.
    MoE ``te.GroupedLinear`` experts expose a 3D stacked ``[E, out, in]`` weight
    on the module itself and are fake-quantized per expert slice.
    """
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Module) and isinstance(
        getattr(inner, "weight", None), nn.Parameter
    ):
        if inner.weight.dim() in (2, 3):
            return inner
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter) and weight.dim() in (2, 3):
        return module
    return None


def apply_qat_to_module(module: nn.Module, spec: QATSpec) -> bool:
    """Register weight fake-quant on a module's 2D/3D weight. Returns applied."""
    owner = _quantizable_weight_owner(module)
    if owner is None:
        return False
    if parametrize.is_parametrized(owner, "weight"):
        return False
    parametrize.register_parametrization(
        owner, "weight", WeightFakeQuant(spec, owner.weight.shape), unsafe=True
    )
    return True


def apply_qat_to_chunks(
    chunks, spec: QATSpec | dict[str, Any] | None
) -> dict[str, int]:
    """Apply weight-only QAT to every eligible linear in the model chunks.

    Opt-in and inert by default: with a disabled spec nothing is registered and
    the model stays bit-identical. Must be called *before* optimizer
    construction so the optimizer captures the master ``weight.original``
    parameter. It may run either before or after HF weight load: the
    parametrization renames ``mod.weight`` to ``mod.parametrizations.weight.original``,
    so the checkpoint loader must map the logical ``….weight`` name onto the
    ``.original`` master (see ``_canonical_state_key`` in the model checkpoint
    module) and — when the optimizer is built pre-load — resync its fp32 master
    after load (``reload_model_params``). The persistent ``amax`` buffer is
    recomputed from the real weight on the first forward, so ordering never
    poisons the quantizer statistic. Routers / lm_head / embeddings are skipped.
    """
    spec = normalize_qat_spec(spec)
    stats = {"quantized_modules": 0, "skipped_ignored": 0, "skipped_no_weight": 0}
    if not spec.enabled:
        return stats
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if _quantizable_weight_owner(module) is None:
                continue
            if not spec.targets_module(name):
                stats["skipped_ignored"] += 1
                continue
            if apply_qat_to_module(module, spec):
                stats["quantized_modules"] += 1
    return stats


def qat_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect quantizer buffers (amax) for inspection / explicit persistence."""
    out = {}
    for name, buf in model.named_buffers():
        if name.endswith(".amax") and "parametrizations" in name:
            out[name] = buf.detach().clone()
    return out


__all__ = [
    "QATSpec",
    "WeightFakeQuant",
    "apply_qat_to_chunks",
    "apply_qat_to_module",
    "canonical_state_key",
    "compute_amax",
    "dequantize_weight",
    "fake_quantize_weight",
    "normalize_qat_spec",
    "pack_int4",
    "qat_state_dict",
    "quantize_weight",
    "unpack_int4",
]
