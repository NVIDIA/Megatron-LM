# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Default local Megatron Lite backend."""

from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime, create

__all__ = ["MegatronLiteConfig", "MegatronLiteRuntime", "create"]
