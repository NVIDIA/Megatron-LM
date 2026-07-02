"""Megatron Lite runtime backend."""

from __future__ import annotations

from typing import Any

from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.lite.config import LiteConfig
from megatron.lite.runtime.backends.lite.runtime import MegatronLiteRuntime


def create(hf_path: str, cfg: LiteConfig | dict[str, Any]) -> RuntimeBase:
    """Factory called by create_runtime.

    create_runtime escape hatch is checked inside MegatronLiteRuntime.build_model().
    """
    return MegatronLiteRuntime(hf_path, cfg)
