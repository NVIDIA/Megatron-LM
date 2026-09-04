# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Centralized Transformer Engine construction for parameterized modules."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import transformer_engine.pytorch as _TE  # pyright: ignore[reportMissingImports]

_DEVICE_MODULES = {"GroupedLinear", "LayerNormLinear", "Linear"}


def _accepts_device(constructor: Any) -> bool:
    try:
        return "device" in inspect.signature(constructor).parameters
    except (TypeError, ValueError):
        return False


def _parameter_module(name: str, *args: Any, **kwargs: Any) -> torch.nn.Module:
    device = torch.get_default_device()
    constructor = getattr(_TE, name)
    if name in _DEVICE_MODULES or _accepts_device(constructor):
        kwargs.setdefault(
            "device", device if device.type == "meta" else torch.device("cuda")
        )
    module = constructor(*args, **kwargs)
    return module.to_empty(device=device) if device.type == "meta" else module


def Linear(*args: Any, **kwargs: Any) -> torch.nn.Module:
    return _parameter_module("Linear", *args, **kwargs)


def LayerNormLinear(*args: Any, **kwargs: Any) -> torch.nn.Module:
    return _parameter_module("LayerNormLinear", *args, **kwargs)


def GroupedLinear(*args: Any, **kwargs: Any) -> torch.nn.Module:
    return _parameter_module("GroupedLinear", *args, **kwargs)


def RMSNorm(*args: Any, **kwargs: Any) -> torch.nn.Module:
    return _parameter_module("RMSNorm", *args, **kwargs)


def __getattr__(name: str) -> Any:
    attribute = getattr(_TE, name)
    if isinstance(attribute, type) and issubclass(attribute, torch.nn.Module):
        return lambda *args, **kwargs: _parameter_module(name, *args, **kwargs)
    return attribute
