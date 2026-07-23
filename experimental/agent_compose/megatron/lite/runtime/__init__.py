# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public runtime interface for Megatron Lite."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from megatron.lite.runtime.contracts.config import RuntimeConfig

if TYPE_CHECKING:
    from megatron.lite.runtime.backends import Runtime
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
    from megatron.lite.runtime.contracts.data import (
        Batch,
        ForwardResult,
        ModelOutputs,
        PackedBatch,
        TrainBatch,
    )
    from megatron.lite.runtime.contracts.handle import ModelHandle
    from megatron.lite.runtime.contracts.loss import LossContext

_RUNTIME_REGISTRY: dict[str, str] = {}


def register_runtime(name: str, module_path: str) -> None:
    """Register a module that provides ``create(hf_path, backend_cfg)``."""
    if not name or not module_path:
        raise ValueError("runtime name and module path must be non-empty")
    _RUNTIME_REGISTRY[name] = module_path


def create_runtime(cfg: RuntimeConfig) -> Runtime:
    """Create a registered runtime backend for ``cfg``."""
    if cfg.backend not in _RUNTIME_REGISTRY:
        raise ValueError(
            f"No runtime backend registered for {cfg.backend!r}. "
            f"Available: {sorted(_RUNTIME_REGISTRY)}"
        )
    module = importlib.import_module(_RUNTIME_REGISTRY[cfg.backend])
    return module.create(cfg.hf_path, cfg.backend_cfg)


def __getattr__(name: str):
    lazy = {
        "Batch": "megatron.lite.runtime.contracts.data",
        "ForwardResult": "megatron.lite.runtime.contracts.data",
        "LossContext": "megatron.lite.runtime.contracts.loss",
        "ModelHandle": "megatron.lite.runtime.contracts.handle",
        "ModelOutputs": "megatron.lite.runtime.contracts.data",
        "OptimizerConfig": "megatron.lite.runtime.contracts.config",
        "PackedBatch": "megatron.lite.runtime.contracts.data",
        "ParallelConfig": "megatron.lite.runtime.contracts.config",
        "Runtime": "megatron.lite.runtime.backends",
        "TrainBatch": "megatron.lite.runtime.contracts.data",
    }
    if name in lazy:
        module = importlib.import_module(lazy[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Batch",
    "ForwardResult",
    "LossContext",
    "ModelHandle",
    "ModelOutputs",
    "OptimizerConfig",
    "PackedBatch",
    "ParallelConfig",
    "Runtime",
    "RuntimeConfig",
    "TrainBatch",
    "create_runtime",
    "register_runtime",
]
