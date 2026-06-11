# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model bundle returned by Lite model protocol modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class ModelBundle:
    """Everything a runtime needs after model construction."""

    chunks: list[nn.Module]
    optimizer: Any | None = None
    finalize_grads: Callable[[], None] | None = None
    forward_step: Callable[[nn.Module, dict[str, Any]], Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


__all__ = ["ModelBundle"]
