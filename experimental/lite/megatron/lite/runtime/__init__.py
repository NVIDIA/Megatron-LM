# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime entrypoints for Megatron Lite."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from megatron.lite.runtime.contracts.config import RuntimeConfig

if TYPE_CHECKING:
    from megatron.lite.runtime.backends import Runtime


def _runtime_registry() -> dict[str, str]:
    from megatron.lite.runtime.backends import RUNTIME_REGISTRY

    return RUNTIME_REGISTRY


def register_runtime(name: str, module_path: str) -> None:
    """Register a runtime backend module.

    The module must provide a ``create(hf_path, cfg)`` function returning a
    ``Runtime`` instance.
    """

    if not name:
        raise ValueError("Runtime name must be non-empty.")
    _runtime_registry()[name] = module_path


def create_runtime(cfg: RuntimeConfig) -> Runtime:
    """Create a runtime from a public runtime config."""

    registry = _runtime_registry()
    if cfg.backend not in registry:
        raise ValueError(f"Unknown runtime backend {cfg.backend!r}. Known: {sorted(registry)}")
    mod = importlib.import_module(registry[cfg.backend])
    return mod.create(cfg.hf_path, cfg.backend_cfg)


def __getattr__(name: str):
    lazy = {
        "ForwardResult": "megatron.lite.runtime.contracts.data",
        "ModelHandle": "megatron.lite.runtime.contracts.handle",
        "Runtime": "megatron.lite.runtime.backends",
        "TrainBatch": "megatron.lite.runtime.contracts.data",
    }
    if name in lazy:
        mod = importlib.import_module(lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ForwardResult",
    "ModelHandle",
    "Runtime",
    "RuntimeConfig",
    "TrainBatch",
    "create_runtime",
    "register_runtime",
]
