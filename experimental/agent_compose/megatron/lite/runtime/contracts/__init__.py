# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy public exports for shared runtime contracts."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Batch": "megatron.lite.runtime.contracts.data",
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


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
