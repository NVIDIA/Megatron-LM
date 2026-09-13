# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime entrypoints for Megatron Lite."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from megatron.lite.runtime.contracts.config import RuntimeConfig

if TYPE_CHECKING:
    from megatron.lite.runtime.backends import Runtime
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts.data import (
        Batch,
        ForwardResult,
        ModelOutputs,
        PackedBatch,
        TrainBatch,
    )
    from megatron.lite.runtime.contracts.handle import ModelHandle


def _runtime_registry() -> dict[str, str]:
    from megatron.lite.runtime.backends import RUNTIME_REGISTRY

    return RUNTIME_REGISTRY


def register_runtime(name: str, module_path: str) -> None:
    """Register a custom runtime backend.

    Args:
        name: Backend name (used in ``RuntimeConfig.backend``).
        module_path: Dotted module path providing a ``create(hf_path, cfg)`` function.

    Example::

        from megatron.lite.runtime import register_runtime
        register_runtime("my_backend", "my_package.my_runtime")
    """
    _runtime_registry()[name] = module_path


def create_runtime(cfg: RuntimeConfig) -> Runtime:
    """Create a Runtime instance for the given config."""
    mod = importlib.import_module(_runtime_registry()[cfg.backend])
    return mod.create(cfg.hf_path, cfg.backend_cfg)


def __getattr__(name: str):
    _lazy = {
        "Batch": "megatron.lite.runtime.contracts.data",
        "BridgeConfig": "megatron.lite.runtime.backends.bridge.config",
        "MegatronLiteConfig": "megatron.lite.runtime.backends.mlite.config",
        "ForwardResult": "megatron.lite.runtime.contracts.data",
        "ModelHandle": "megatron.lite.runtime.contracts.handle",
        "ModelOutputs": "megatron.lite.runtime.contracts.data",
        "PackedBatch": "megatron.lite.runtime.contracts.data",
        "Runtime": "megatron.lite.runtime.backends",
        "TrainBatch": "megatron.lite.runtime.contracts.data",
    }
    if name in _lazy:
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Batch",
    "BridgeConfig",
    "ForwardResult",
    "MegatronLiteConfig",
    "ModelHandle",
    "ModelOutputs",
    "PackedBatch",
    "Runtime",
    "RuntimeConfig",
    "TrainBatch",
    "create_runtime",
    "register_runtime",
]
