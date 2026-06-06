"""Lazy re-export hub for ``megatron.lite.runtime.contracts``.

This preserves imports like ``from megatron.lite.runtime.contracts import X`` while
avoiding eager imports of heavyweight modules (for example ``torch`` from
``contracts.data``) during package import.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig, DebugConfig
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, RuntimeConfig
    from megatron.lite.runtime.contracts.data import (
        Batch,
        ForwardResult,
        ModelOutputs,
        PackedBatch,
        TrainBatch,
    )
    from megatron.lite.runtime.contracts.handle import ModelHandle

__all__ = [
    "MegatronLiteConfig",
    "Batch",
    "DebugConfig",
    "ForwardResult",
    "ModelHandle",
    "ModelOutputs",
    "OptimizerConfig",
    "PackedBatch",
    "ParallelConfig",
    "RuntimeConfig",
    "TrainBatch",
]


def __getattr__(name: str):
    _lazy = {
        "Batch": "megatron.lite.runtime.contracts.data",
        "DebugConfig": "megatron.lite.runtime.backends.mlite.config",
        "MegatronLiteConfig": "megatron.lite.runtime.backends.mlite.config",
        "ForwardResult": "megatron.lite.runtime.contracts.data",
        "ModelHandle": "megatron.lite.runtime.contracts.handle",
        "ModelOutputs": "megatron.lite.runtime.contracts.data",
        "OptimizerConfig": "megatron.lite.runtime.contracts.config",
        "PackedBatch": "megatron.lite.runtime.contracts.data",
        "ParallelConfig": "megatron.lite.runtime.contracts.config",
        "RuntimeConfig": "megatron.lite.runtime.contracts.config",
        "TrainBatch": "megatron.lite.runtime.contracts.data",
    }
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
