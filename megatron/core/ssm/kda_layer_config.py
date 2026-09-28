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
        if self.gdn_conv_pad_alignment is not None and self.gdn_conv_pad_alignment <= 0:
            raise ValueError(
                "gdn_conv_pad_alignment must be positive when set for KDA, got "
                f"{self.gdn_conv_pad_alignment}."
            )
        if self.kda_safe_gate and (
            self.kda_lower_bound is None or not (-5 <= self.kda_lower_bound < 0)
        ):
            raise ValueError(
                "kda_lower_bound must be in the safe range [-5, 0) when kda_safe_gate=True, "
                f"got {self.kda_lower_bound}."
            )
        required_positive = (
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        )
        for name in required_positive:
            value = getattr(self, name)
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive for KDA, got {value}.")
        if self.linear_num_key_heads != self.linear_num_value_heads:
            raise ValueError("KDA requires equal key and value head counts.")
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("KDA requires equal key and value head dimensions.")
        if (
            self.context_parallel_size > 1
            and self.linear_cp_mode == "chunkwise"
            and self.deterministic_mode
        ):
            raise ValueError(
                "KDA chunkwise context parallelism is incompatible with deterministic_mode "
                "because the fallback convolution cannot exchange history across ranks."
            )
        if self.linear_num_key_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                "KDA key/value head count must be divisible by tensor parallel size; "
                f"got heads={self.linear_num_key_heads}, tp={self.tensor_model_parallel_size}."
            )
        if self.linear_cp_mode == "headwise":
            tp_cp_size = self.tensor_model_parallel_size * self.context_parallel_size
            if self.linear_num_key_heads % tp_cp_size != 0:
                raise ValueError(
                    "KDA headwise context parallelism requires head count divisible by TP*CP; "
                    f"got heads={self.linear_num_key_heads}, tp_cp={tp_cp_size}."
                )
