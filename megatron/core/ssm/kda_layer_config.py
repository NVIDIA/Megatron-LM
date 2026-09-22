# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(kw_only=True)
class KDALayerConfig(TransformerConfig):
    """Configuration for a KDA (Kimi Delta Attention) layer in a hybrid stack."""

    linear_num_value_heads: Optional[int] = 16
    """KDA uses equal key and value head counts by default."""

    kda_disable_fp8: bool = False
    """Force KDA projections to BF16 even under FP8 training,
    (KDA projections are BF16 in the checkpoint)."""

    kda_safe_gate: bool = False
    """Whether the KDA kernel should use bounded gate values."""

    kda_lower_bound: Optional[float] = None
    """Optional lower bound for KDA's bounded gate values."""

    kda_two_stage_gates: bool = False
    """Use low-rank f_b(f_a(x)) and g_b(g_a(x)) gates with a QKV-only input projection."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_kda()

    def validate_kda(self) -> None:
        """Validate KDA-specific dimensions, including configs copied with from_config."""
        if self.linear_num_key_heads != self.linear_num_value_heads:
            raise ValueError("KDA requires equal key and value head counts.")
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("KDA requires equal key and value head dimensions.")
