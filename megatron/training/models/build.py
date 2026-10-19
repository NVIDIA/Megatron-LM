# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass


@dataclass(kw_only=True)
class ModelBuildConfig:
    """Execution policy, independent of parameter initialization and model architecture."""

    device: str | None = None
    """Final device; None preserves the distributed builder's placement policy."""

    wrap_with_ddp: bool | None = None
    """None lets the training/inference entry point select wrapping."""

    mixed_precision: bool = True
    """Apply the standard precision wrapper before distributed wrapping."""

    def validate(self) -> None:
        if self.device is not None:
            import torch

            device = torch.device(self.device)
            if device.type not in ("cpu", "cuda"):
                raise ValueError("Model build device must be CPU or CUDA.")
