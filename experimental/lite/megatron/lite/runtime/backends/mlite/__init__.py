"""Megatron Lite backend — Megatron Lite's own training engine."""

from __future__ import annotations

from typing import Any

from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime


def create(hf_path: str, cfg: MegatronLiteConfig | dict[str, Any]) -> RuntimeBase:
    """Factory called by create_runtime.

    create_runtime escape hatch is checked inside MegatronLiteRuntime.build_model().
    """
    return MegatronLiteRuntime(hf_path, cfg)
