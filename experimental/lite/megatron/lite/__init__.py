"""Top-level Megatron Lite package exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.runtime.backends.lite.config import DebugConfig, LiteConfig
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

__all__ = [
    "DebugConfig",
    "LiteConfig",
    "OptimizerConfig",
    "ParallelConfig",
]


def __getattr__(name: str):
    _lazy = {
        "LiteConfig": "megatron.lite.runtime.backends.lite.config",
        "DebugConfig": "megatron.lite.runtime.backends.lite.config",
        "OptimizerConfig": "megatron.lite.runtime.contracts.config",
        "ParallelConfig": "megatron.lite.runtime.contracts.config",
    }
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
