# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration for the local Megatron Lite backend."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig


def _pick_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {name: value for name, value in values.items() if name in names}


@dataclass
class MegatronLiteConfig:
    """Backend config used by the local ``mlite`` runtime."""

    model_name: str = "toy_dense"
    impl: str = "torch"
    device: str = "cpu"
    hf_path: str = ""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, hf_path: str, values: dict[str, Any]) -> "MegatronLiteConfig":
        values = dict(values)
        if "parallel" in values and isinstance(values["parallel"], dict):
            values["parallel"] = ParallelConfig(**_pick_fields(ParallelConfig, values["parallel"]))
        if "optimizer" in values and isinstance(values["optimizer"], dict):
            values["optimizer"] = OptimizerConfig(**_pick_fields(OptimizerConfig, values["optimizer"]))
        values.setdefault("hf_path", hf_path)
        return cls(**_pick_fields(cls, values))


__all__ = ["MegatronLiteConfig"]
