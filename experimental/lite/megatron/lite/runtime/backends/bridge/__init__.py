# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Bridge backend for the Megatron Lite runtime API."""

from __future__ import annotations

from typing import Any

from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.backends.bridge.runtime import BridgeRuntime


def create(hf_path: str, cfg: BridgeConfig | dict[str, Any]) -> RuntimeBase:
    """Factory called by ``megatron.lite.runtime.create_runtime``."""
    return BridgeRuntime(hf_path, cfg)


__all__ = ["BridgeConfig", "BridgeRuntime", "create"]
