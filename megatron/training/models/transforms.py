# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class ModelTransformConfig:
    """Serializable ModelOpt setup used to construct and transform a model."""

    default_te_spec: bool = False
    """Use the stock Transformer Engine layer spec instead of the ModelOpt-local spec.

    The ModelOpt-local spec keeps layernorms as separate modules, but always loads and saves
    them under the fused Transformer Engine keys (e.g. ``linear_qkv.layer_norm_weight``), so
    its checkpoints are interchangeable with Transformer Engine spec checkpoints."""

    force_local_attention: bool = False
    """Use Megatron Core attention in the ModelOpt-local GPT layer spec."""

    real_quant_cfg: Literal["None", "fp8_real_quant", "fp8_blockwise_real_quant"] = "None"
    """Select the real-quantized transformer layer implementation."""

    moe_grouped_gemm: bool = True
    """Use grouped MoE experts; disable for transforms that address individual experts."""

    qad_train_target: Literal["mtp", "base", "both"] | None = None
    """Select the trainable portion of an MTP model."""

    lsq_scale_lr: float | None = None
    """Enable optimizer overrides for learned quantization scales."""

    def validate(self) -> None:
        """Validate transformation settings before allocating models or loading checkpoints."""
        if self.qad_train_target not in (None, "mtp", "base", "both"):
            raise ValueError(f"Unsupported qad_train_target: {self.qad_train_target}")
        if self.real_quant_cfg not in ("None", "fp8_real_quant", "fp8_blockwise_real_quant"):
            raise ValueError(f"Unsupported real_quant_cfg: {self.real_quant_cfg}")
