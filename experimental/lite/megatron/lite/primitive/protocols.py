# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Protocol definitions for Lite model and primitive integration."""

from __future__ import annotations

from typing import Any, Protocol

from megatron.lite.primitive.bundle import ModelBundle


class ModelBuildProtocol(Protocol):
    """Protocol implemented by registered model implementation modules."""

    def build_model_config(self, hf_path: str = "") -> Any:
        """Build or load a model config object."""

    def build_model(self, model_cfg: Any, impl_cfg: Any | None = None) -> ModelBundle:
        """Build model chunks and return a runtime-consumable bundle."""


__all__ = ["ModelBuildProtocol"]
