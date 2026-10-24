# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Primitive interface contracts for Megatron Lite."""

from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config import PrecisionConfig, PrimitiveConfig
from megatron.lite.primitive.protocols import ModelBuildProtocol

__all__ = ["ModelBuildProtocol", "ModelBundle", "PrecisionConfig", "PrimitiveConfig"]
