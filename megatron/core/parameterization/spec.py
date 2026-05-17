# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ResolvedScalingContext:
    """Resolved internal scaling context for existing Megatron parameterization knobs.

    PR1 intentionally does not add a new public recipe surface. It wraps the existing
    ``use_mup`` fields so model and optimizer code can route through a shared policy seam
    while preserving current behavior.
    """

    use_mup: bool
    width_mult: float = 1.0
    embedding_mult: float = 1.0
    output_mult: float = 1.0
    base_head_dim: Optional[float] = None
    attention_scale_power: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.use_mup

    @property
    def uses_width_mup(self) -> bool:
        return self.use_mup


def build_resolved_scaling_context(config: Any) -> ResolvedScalingContext:
    return ResolvedScalingContext(
        use_mup=bool(getattr(config, 'use_mup', False)),
        width_mult=getattr(config, 'mup_width_mult', 1.0),
        embedding_mult=getattr(config, 'mup_embedding_mult', 1.0),
        output_mult=getattr(config, 'mup_output_mult', 1.0),
        base_head_dim=getattr(config, 'mup_base_head_dim', None),
        attention_scale_power=getattr(config, 'mup_attn_scale_power', 1.0),
    )
