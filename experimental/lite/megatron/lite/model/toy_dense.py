# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tiny two-layer dense model used for contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from megatron.lite.primitive.bundle import ModelBundle


@dataclass
class ToyDenseConfig:
    """Shape config for the toy dense model."""

    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 2


@dataclass
class ImplConfig:
    """Implementation options for the pure PyTorch toy model."""

    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 2


class ToyDenseModel(nn.Module):
    """Two linear layers with one ReLU activation."""

    def __init__(self, config: ToyDenseConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def build_model_config(hf_path: str = "") -> ToyDenseConfig:
    """Return the default toy model config.

    ``hf_path`` is accepted to match the model protocol. The toy model does not
    read external config files.
    """

    _ = hf_path
    return ToyDenseConfig()


def build_model(model_cfg: ToyDenseConfig, impl_cfg: ImplConfig | None = None) -> ModelBundle:
    """Build the toy model and return a contract bundle."""

    if impl_cfg is not None:
        model_cfg = ToyDenseConfig(
            input_dim=impl_cfg.input_dim,
            hidden_dim=impl_cfg.hidden_dim,
            output_dim=impl_cfg.output_dim,
        )

    return ModelBundle(chunks=[ToyDenseModel(model_cfg)])


__all__ = ["ImplConfig", "ToyDenseConfig", "ToyDenseModel", "build_model", "build_model_config"]
