# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""User settings for native LoRA training; no model or hook state is stored here."""

from dataclasses import dataclass, field
from math import isfinite

import torch


@dataclass
class LoRA:
    """Configure low-rank adapters on selected linear projections."""

    target_modules: list[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    """Module names or full-path wildcard patterns to adapt."""
    exclude_modules: list[str] = field(default_factory=list)
    """Module paths or names to leave frozen; exclusions take precedence."""
    dim: int = 32
    """Low-rank dimension."""
    alpha: float = 32.0
    """Scale adapter output by alpha / dim."""
    dropout: float = 0.0
    """Dropout probability on the adapter branch."""
    dropout_position: str = "pre"
    """Apply dropout before or after the low-rank projections."""
    lora_A_init_method: str = "xavier"
    """Initialization for the input projection: xavier, normal, kaiming or zero."""
    lora_B_init_method: str = "zero"
    """Initialization for the output projection."""
    lora_dtype: torch.dtype | None = None
    """Adapter parameter dtype; None follows the base layer."""
    canonical_qkv_projections: bool = False
    """Use independent Q, K and V adapters while keeping the base projection fused."""
    canonical_mlp_projections: bool = False
    """Use independent gate/up adapters on dense gated MLP projections."""
    share_expert_adapters: bool = True
    """Share adapters across experts; False creates a pair per local expert."""
    normalize_moe_lora: bool = False
    """Divide expert rank by router top-k, retaining the full rank elsewhere."""

    def validate(self) -> None:
        """Check settings without changing them or requiring an initialized runtime."""
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("LoRA dim must be a positive integer")
        if not isfinite(self.alpha):
            raise ValueError("LoRA alpha must be finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.dropout_position not in ("pre", "post"):
            raise ValueError("LoRA dropout_position must be pre or post")
        methods = {"xavier", "normal", "kaiming", "zero"}
        if self.lora_A_init_method not in methods or self.lora_B_init_method not in methods:
            raise ValueError(f"LoRA initialization must be one of {sorted(methods)}")
        if not self.target_modules or any(not name for name in self.target_modules):
            raise ValueError("LoRA requires at least one nonempty target pattern")
        if self.lora_dtype not in (None, torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("LoRA dtype must be float32, float16 or bfloat16")
        for name in self.target_modules:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("linear_q", "linear_k", "linear_v") and not self.canonical_qkv_projections:
                raise ValueError("Q/K/V targets require canonical_qkv_projections=True")
            if leaf in ("linear_fc1_gate", "linear_fc1_up") and not self.canonical_mlp_projections:
                raise ValueError("Gate/up targets require canonical_mlp_projections=True")
