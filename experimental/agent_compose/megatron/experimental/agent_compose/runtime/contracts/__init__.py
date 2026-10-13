# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy public exports for shared runtime contracts."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Batch": "megatron.experimental.agent_compose.runtime.contracts.data",
    "ForwardResult": "megatron.experimental.agent_compose.runtime.contracts.data",
    "LossContext": "megatron.experimental.agent_compose.runtime.contracts.loss",
    "ModelHandle": "megatron.experimental.agent_compose.runtime.contracts.handle",
    "ModelOutputs": "megatron.experimental.agent_compose.runtime.contracts.data",
    "OptimizerConfig": "megatron.experimental.agent_compose.runtime.contracts.config",
    "PackedBatch": "megatron.experimental.agent_compose.runtime.contracts.data",
    "ParallelConfig": "megatron.experimental.agent_compose.runtime.contracts.config",
    "RuntimeConfig": "megatron.experimental.agent_compose.runtime.contracts.config",
    "TrainBatch": "megatron.experimental.agent_compose.runtime.contracts.data",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
