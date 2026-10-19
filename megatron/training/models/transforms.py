# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class ModelTransformConfig:
    """Serializable setup applied before wrapping a ModelOpt model for distributed execution."""

    qad_train_target: Literal["mtp", "base", "both"] | None = None
    """Select the trainable portion of an MTP model."""

    lsq_scale_lr: float | None = None
    """Enable optimizer overrides for learned quantization scales."""

    te_mcore_model: bool = False
    """Remap Transformer Engine checkpoint layernorm keys on load."""
