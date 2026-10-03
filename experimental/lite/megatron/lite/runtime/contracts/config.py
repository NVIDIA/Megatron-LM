# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared runtime configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParallelConfig:
    """Parallel dimensions requested by a model/runtime pair."""

    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1


@dataclass
class OptimizerConfig:
    """Optimizer options shared by lightweight runtime backends."""

    optimizer: str = "sgd"
    lr: float = 1e-2
    weight_decay: float = 0.0
    clip_grad: float | None = None


@dataclass
class RuntimeConfig:
    """Top-level runtime creation config."""

    backend: str = "mlite"
    hf_path: str = ""
    backend_cfg: Any = field(default_factory=dict)


__all__ = ["OptimizerConfig", "ParallelConfig", "RuntimeConfig"]
