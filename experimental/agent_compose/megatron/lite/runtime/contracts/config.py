# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backend-neutral runtime configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParallelConfig:
    """Parallel dimensions shared by runtime backends."""

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1
    pp_layout: str | list | None = None


@dataclass
class OptimizerConfig:
    """Optimizer and learning-rate scheduler settings."""

    optimizer: str = "adam"
    lr: float = 1e-3
    min_lr: float = 0.0
    clip_grad: float = 1.0
    weight_decay: float = 0.01
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    lr_warmup_steps: int = -1
    lr_warmup_init: float = 0.0
    lr_decay_steps: int | None = None
    lr_decay_style: str = "linear"
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: int | None = None
    use_checkpoint_opt_param_scheduler: bool = False

    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    offload_fraction: float | None = None
    use_precision_aware_optimizer: bool | None = None
    decoupled_weight_decay: bool | None = None


@dataclass
class RuntimeConfig:
    """Select a runtime backend and provide its configuration."""

    backend: str = "mlite"
    hf_path: str = ""
    backend_cfg: Any = field(default_factory=dict)


__all__ = ["OptimizerConfig", "ParallelConfig", "RuntimeConfig"]
