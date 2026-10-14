# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Data contracts for runtime calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainBatch:
    """Minimal tensor batch used by local contract checks."""

    inputs: torch.Tensor
    targets: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardResult:
    """Result returned by ``Runtime.forward_backward``."""

    loss: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: Any = None


__all__ = ["ForwardResult", "TrainBatch"]
