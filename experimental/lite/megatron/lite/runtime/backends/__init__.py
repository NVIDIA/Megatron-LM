# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime interface and backend registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from megatron.lite.runtime.contracts.data import ForwardResult
from megatron.lite.runtime.contracts.handle import ModelHandle


class Runtime(ABC):
    """Base class implemented by Megatron Lite runtime backends."""

    @abstractmethod
    def build_model(self, hf_path: str | None = None, cfg: Any = None, **kwargs) -> ModelHandle:
        """Build runtime-owned model state and return an opaque handle."""

    @abstractmethod
    def train_mode(self, handle: ModelHandle) -> Any:
        """Put the handled model in training mode."""

    @abstractmethod
    def eval_mode(self, handle: ModelHandle) -> Any:
        """Put the handled model in evaluation mode."""

    @abstractmethod
    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn: Callable[[Any, Any], tuple[Any, dict[str, Any]]] | None = None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        """Run a local forward/backward atom for one logical step."""

    @abstractmethod
    def zero_grad(self, handle: ModelHandle) -> None:
        """Clear gradients for the handled optimizer/model."""

    @abstractmethod
    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        """Run the optimizer step and return ``(ok, grad_norm, zero_grad_count)``."""

    @abstractmethod
    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]:
        """Advance or query the runtime learning-rate scheduler."""


RUNTIME_REGISTRY: dict[str, str] = {
    "mlite": "megatron.lite.runtime.backends.mlite",
}


__all__ = ["RUNTIME_REGISTRY", "Runtime"]
