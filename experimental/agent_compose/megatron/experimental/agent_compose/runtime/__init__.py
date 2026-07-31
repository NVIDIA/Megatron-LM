# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public runtime interface for Agent Compose."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from megatron.experimental.agent_compose.runtime.contracts.config import RuntimeConfig

if TYPE_CHECKING:
    from megatron.experimental.agent_compose.runtime.backends import Runtime
    from megatron.experimental.agent_compose.runtime.contracts.config import (
        OptimizerConfig,
        ParallelConfig,
    )
    from megatron.experimental.agent_compose.runtime.contracts.data import (
        Batch,
        ForwardResult,
        ModelOutputs,
        PackedBatch,
        TrainBatch,
    )
    from megatron.experimental.agent_compose.runtime.contracts.handle import ModelHandle
    from megatron.experimental.agent_compose.runtime.contracts.loss import LossContext

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
        "Batch": "megatron.experimental.agent_compose.runtime.contracts.data",
        "ForwardResult": "megatron.experimental.agent_compose.runtime.contracts.data",
        "LossContext": "megatron.experimental.agent_compose.runtime.contracts.loss",
        "ModelHandle": "megatron.experimental.agent_compose.runtime.contracts.handle",
        "ModelOutputs": "megatron.experimental.agent_compose.runtime.contracts.data",
        "OptimizerConfig": "megatron.experimental.agent_compose.runtime.contracts.config",
        "PackedBatch": "megatron.experimental.agent_compose.runtime.contracts.data",
        "ParallelConfig": "megatron.experimental.agent_compose.runtime.contracts.config",
        "Runtime": "megatron.experimental.agent_compose.runtime.backends",
        "TrainBatch": "megatron.experimental.agent_compose.runtime.contracts.data",
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
