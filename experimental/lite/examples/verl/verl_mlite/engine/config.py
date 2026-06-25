# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Config objects for the Verl MLite Megatron Lite engine."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from verl.workers.config.engine import EngineConfig


@dataclass
class MegatronLiteEngineConfig(EngineConfig):
    """Minimal VERL-facing config for the external Megatron Lite engine."""

    strategy: str = "mlite"
    custom_backend_module: str | None = "verl_mlite.engine.mlite_engine"
    model_name: str = "auto"
    impl: str = "lite"

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1

    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    cross_entropy_fusion: bool | None = None
    export_dtype: str | None = "bfloat16"
    load_hf_weights: bool = True
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy != "mlite":
            raise ValueError(
                f"MegatronLiteEngineConfig expects strategy='mlite', got {self.strategy!r}"
            )
        if self.custom_backend_module:
            importlib.import_module(self.custom_backend_module)
