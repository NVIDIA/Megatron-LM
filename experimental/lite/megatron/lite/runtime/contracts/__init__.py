# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy re-export hub for ``megatron.lite.runtime.contracts``.

This preserves imports like ``from megatron.lite.runtime.contracts import X`` while
avoiding eager imports of heavyweight modules (for example ``torch`` from
``contracts.data``) during package import.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.mlite.config import DebugConfig, MegatronLiteConfig
    from megatron.lite.runtime.contracts.config import (
        OptimizerConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from megatron.lite.runtime.contracts.data import (
        Batch,
        ForwardResult,
        ModelOutputs,
        PackedBatch,
        TrainBatch,
    )
    from megatron.lite.runtime.contracts.handle import ModelHandle
    from megatron.lite.runtime.contracts.loss import LossContext

__all__ = [
    "MegatronLiteConfig",
    "Batch",
    "BridgeConfig",
    "DebugConfig",
    "ForwardResult",
    "LossContext",
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
        "BridgeConfig": "megatron.lite.runtime.backends.bridge.config",
        "DebugConfig": "megatron.lite.runtime.backends.mlite.config",
        "MegatronLiteConfig": "megatron.lite.runtime.backends.mlite.config",
        "ForwardResult": "megatron.lite.runtime.contracts.data",
        "LossContext": "megatron.lite.runtime.contracts.loss",
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
