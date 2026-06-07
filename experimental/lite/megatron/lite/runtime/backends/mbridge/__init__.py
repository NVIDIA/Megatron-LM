"""mbridge backend for the Megatron Lite runtime API."""

from __future__ import annotations

from typing import Any

from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.backends.mbridge.runtime import MBridgeRuntime


def create(hf_path: str, cfg: BridgeConfig | dict[str, Any]) -> RuntimeBase:
    """Factory called by ``megatron.lite.runtime.create_runtime``."""
    return MBridgeRuntime(hf_path, cfg)


__all__ = ["BridgeConfig", "MBridgeRuntime", "create"]
