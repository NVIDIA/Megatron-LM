# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron RL utilities with independently importable packing primitives.

Packing modules do not load Pydantic generation APIs or the model runtime.
Historical generation request types remain available here on first access.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.rl.generation_api import GenericGenerationArgs, Request, TypeLookupable

__all__ = ["GenericGenerationArgs", "Request", "TypeLookupable"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module("megatron.rl.generation_api"), name)
    globals()[name] = value
    return value
