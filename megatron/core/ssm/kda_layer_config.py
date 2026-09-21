# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(kw_only=True)
class KDALayerConfig(TransformerConfig):
    """Configuration for a KDA (Kimi Delta Attention) layer in a hybrid stack."""

    kda_disable_fp8: bool = False
    """Force KDA projections to BF16 even under FP8 training,
    (KDA projections are BF16 in the checkpoint)."""

    kda_safe_gate: bool = False
    """Whether the KDA kernel should use bounded gate values."""

    kda_lower_bound: Optional[float] = None
    """Optional lower bound for KDA's bounded gate values."""

    kda_two_stage_gates: bool = False
    """Use low-rank f_b(f_a(x)) and g_b(g_a(x)) gates with a QKV-only input projection."""
