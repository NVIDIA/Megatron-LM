# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelBundle — return type of protocol.build_model()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class ModelBundle:
    """Everything runtime needs to run a training loop.

    Returned by protocol.build_model(). Model owns the construction
    of all fields — runtime just consumes them.
    """

    chunks: list[nn.Module]
    parallel_state: ParallelState
    optimizer: Any | None = None
    finalize_grads: Callable[[], None] | None = None
    forward_step: Callable[..., dict] | None = None
    # extra metadata (expert_classifier, model_cfg, etc.)
    extras: dict[str, Any] = field(default_factory=dict)
