"""Top-level Megatron Lite package exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.runtime.backends.mlite.config import DebugConfig, MegatronLiteConfig
    from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig, RuntimeConfig

__all__ = [
    "DebugConfig",
    "MegatronLiteConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "RuntimeConfig",
]


def __getattr__(name: str):
    _lazy = {
        "DebugConfig": "megatron.lite.runtime.backends.mlite.config",
        "MegatronLiteConfig": "megatron.lite.runtime.backends.mlite.config",
        "OptimizerConfig": "megatron.lite.runtime.contracts",
        "ParallelConfig": "megatron.lite.runtime.contracts",
        "RuntimeConfig": "megatron.lite.runtime.contracts",
    }
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
