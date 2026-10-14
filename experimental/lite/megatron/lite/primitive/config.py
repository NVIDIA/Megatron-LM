# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared primitive configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrecisionConfig:
    """Precision requested by a primitive implementation."""

    dtype: str = "float32"
    params_dtype: str | None = None
    reduce_dtype: str | None = None


@dataclass(frozen=True)
class PrimitiveConfig:
    """Generic primitive descriptor used before concrete primitives exist."""

    name: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


__all__ = ["PrecisionConfig", "PrimitiveConfig"]
