# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization primitives for Megatron Lite (weight-only QAT: fp8/mxfp4 + int8/int4)."""

from __future__ import annotations

from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    BLOCK_SHAPE,
    CanonicalBlockFP8Weight,
    DeploymentBlockFP8Adapter,
    DeploymentFusedBlockFP8Adapter,
    PackedBlockFP8Activation,
    PackedBlockFP8Weight,
    PackedGroupedBlockFP8Weight,
    fp8_gemm_nt,
    pack_block_fp8_activation,
    pack_block_fp8_weight,
    pack_grouped_block_fp8_weight,
    quantize_block_fp8_weight,
    requantize_block_fp8_weight,
)
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    WeightFakeQuant,
    apply_qat_to_chunks,
    apply_qat_to_module,
    compute_amax,
    dequantize_weight,
    fake_quantize_weight,
    normalize_qat_spec,
    pack_int4,
    qat_state_dict,
    quantize_weight,
    unpack_int4,
)

__all__ = [
    "BLOCK_SHAPE",
    "CanonicalBlockFP8Weight",
    "DeploymentBlockFP8Adapter",
    "DeploymentFusedBlockFP8Adapter",
    "PackedBlockFP8Activation",
    "PackedBlockFP8Weight",
    "PackedGroupedBlockFP8Weight",
    "QATSpec",
    "WeightFakeQuant",
    "apply_qat_to_chunks",
    "apply_qat_to_module",
    "compute_amax",
    "dequantize_weight",
    "fake_quantize_weight",
    "fp8_gemm_nt",
    "normalize_qat_spec",
    "pack_int4",
    "pack_block_fp8_activation",
    "pack_block_fp8_weight",
    "pack_grouped_block_fp8_weight",
    "quantize_block_fp8_weight",
    "requantize_block_fp8_weight",
    "qat_state_dict",
    "quantize_weight",
    "unpack_int4",
]
