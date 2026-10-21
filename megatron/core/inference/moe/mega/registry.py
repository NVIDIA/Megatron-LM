# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Precision registry for FlashInfer mega MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import flashinfer.moe_ep.backends  # noqa: F401 — register mega kernels
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl import (
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl import (
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    _HAVE_MEGA_CONFIG = True
except ImportError:
    _HAVE_MEGA_CONFIG = False
    Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig = None  # type: ignore[misc, assignment]
    Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig = None  # type: ignore[misc, assignment]

# Imported on its own: flashinfer-python 0.6.x ships the kernels above and not
# this one, and folding it into that import would disable every mega precision
# on those builds. It first shipped in 0.7.1 (BF16 activations, NVFP4 weights).
try:
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl import (
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    _HAVE_BF16_NVFP4 = True
except ImportError:
    _HAVE_BF16_NVFP4 = False
    Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class MegaPrecisionSpec:
    """Maps a precision name to a FlashInfer megakernel config factory."""

    name: str
    build_config: Callable[["TransformerConfig"], Any]


def _require_mega_kernels(kernel_dir: str) -> None:
    """Fail with an actionable message when the mega kernels are missing."""
    if not _HAVE_MEGA_CONFIG:
        raise RuntimeError(
            "FlashInfer moe_ep sm100 mega kernels are not available. They require "
            "flashinfer-python >= 0.7 with "
            f"flashinfer/moe_ep/backends/mega/kernel/sm100/{kernel_dir}."
        )


# All builders leave gate_up_clamp/activation_clamp unset: FlashInfer hard-clamps
# the FC1 output, whereas activation_func_tanh_clamp_scale is a soft tanh clamp
# that replaces the swish gate entirely (SiTU-GLU). Config validation rejects the
# clamp for this backend rather than silently mismapping it.
#
# Every kernel takes intermediate_size as the post-SwiGLU width and derives the
# FC1 gate+up width internally as 2 * intermediate_size.
#
# The quantized kernels need no Megatron-side quantization: MegaConfig
# .preprocess_weights defaults to True, so FlashInfer quantizes the bf16
# MoEWeightPack to the kernel's format at warmup, and MegaConfig.quantize_input
# (also default True) quantizes activations in-kernel.


def _build_bf16_cutedsl_config(config: "TransformerConfig") -> Any:
    _require_mega_kernels("bf16_bf16_bf16_cutedsl")
    return Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=config.moe_ffn_hidden_size, top_k=config.moe_router_topk
    )


def _build_mxfp8_cutedsl_config(config: "TransformerConfig") -> Any:
    _require_mega_kernels("mxfp8_mxfp8_bf16_cutedsl")
    # kind defaults to mxfp8_e4m3; e5m2 trades mantissa for range and is not
    # exposed until a model needs it.
    return Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=config.moe_ffn_hidden_size, top_k=config.moe_router_topk
    )


def _build_bf16_nvfp4_cutedsl_config(config: "TransformerConfig") -> Any:
    _require_mega_kernels("bf16_nvfp4_bf16_cutedsl")
    if not _HAVE_BF16_NVFP4:
        raise RuntimeError(
            "inference_mega_precision='bf16_nvfp4' requires flashinfer-python>=0.7.1, "
            "which adds the BF16-activation NVFP4-weight megakernel "
            "(flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_nvfp4_bf16_cutedsl). "
            "Older flashinfer builds ship the other mega kernels but not this one."
        )
    # Activations, the FC1 handoff, and the output stay BF16; only the expert
    # weights are NVFP4. in_kernel_fc2_reduce stays off: its BF16 atomics are
    # nondeterministic, and the default combines BF16 partials in FP32.
    return Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=config.moe_ffn_hidden_size, top_k=config.moe_router_topk
    )


MEGA_PRECISION_REGISTRY: dict[str, MegaPrecisionSpec] = {
    "bf16": MegaPrecisionSpec(name="bf16", build_config=_build_bf16_cutedsl_config),
    "mxfp8": MegaPrecisionSpec(name="mxfp8", build_config=_build_mxfp8_cutedsl_config),
    # BF16 activations times NVFP4 weights. Needs flashinfer-python >= 0.7.1;
    # the builder raises if this install predates that kernel. NVFP4xNVFP4 and
    # the DeepGEMM fp8 x fp4 kernel are intentionally not registered.
    "bf16_nvfp4": MegaPrecisionSpec(
        name="bf16_nvfp4", build_config=_build_bf16_nvfp4_cutedsl_config
    ),
}


def get_mega_precision_spec(precision: str) -> MegaPrecisionSpec:
    try:
        return MEGA_PRECISION_REGISTRY[precision]
    except KeyError as exc:
        supported = ", ".join(sorted(MEGA_PRECISION_REGISTRY))
        raise ValueError(
            f"Unknown inference_mega_precision={precision!r}; supported: {supported}"
        ) from exc


def build_megakernel_config(config: "TransformerConfig") -> Any:
    """Build the FlashInfer megakernel config object for the model."""
    spec = get_mega_precision_spec(config.inference_mega_precision)
    return spec.build_config(config)
