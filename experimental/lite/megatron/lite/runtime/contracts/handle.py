# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opaque model handle returned by runtime backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelHandle:
    """Runtime-owned model state with a small public surface."""

    model: Any
    optimizer: Any = None
    lr_scheduler: Any = None
    config: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def dp_rank(self) -> int:
        return 0

    @property
    def dp_size(self) -> int:
        return 1

    @property
    def dp_group(self) -> None:
        return None


__all__ = ["ModelHandle"]
