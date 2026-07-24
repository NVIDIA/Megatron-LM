"""Optimizer backend registry."""

from __future__ import annotations

import importlib

BACKENDS = {
    "mc": "megatron.lite.primitive.optimizers.megatron_wrap",
}


def get_optimizer_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown Megatron Lite optimizer backend: {name!r}.")
    return importlib.import_module(BACKENDS[name]).BACKEND


__all__ = ["get_optimizer_backend"]
