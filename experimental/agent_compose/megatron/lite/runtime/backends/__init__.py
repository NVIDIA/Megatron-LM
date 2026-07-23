# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime interface implemented by Megatron Lite backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

    from megatron.lite.runtime.contracts.data import ForwardResult
    from megatron.lite.runtime.contracts.handle import ModelHandle


class Runtime(ABC):
    """Backend-neutral lifecycle and training interface.

    The required methods form the pretraining tier. Implementing
    :meth:`export_weights` adds the RL-ready tier; implementing both
    :meth:`export_weights` and :meth:`to` adds the RL-best tier.
    """

    @abstractmethod
    def build_model(
        self, hf_path: str | None = None, cfg: Any = None, **kwargs
    ) -> ModelHandle:
        """Build model state and return an opaque handle."""
        ...

    @abstractmethod
    def save_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None: ...

    @abstractmethod
    def load_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> int: ...

    @abstractmethod
    def train_mode(self, handle: ModelHandle) -> Any: ...

    @abstractmethod
    def eval_mode(self, handle: ModelHandle) -> Any: ...

    @abstractmethod
    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn: Callable | None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
        router_replay: Any = None,
    ) -> ForwardResult:
        """Run a logical forward/backward step over one or more microbatches."""
        ...

    @abstractmethod
    def zero_grad(self, handle: ModelHandle) -> None: ...

    @abstractmethod
    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        """Return ``(update_successful, grad_norm, num_zeros_in_grad)``."""
        ...

    @abstractmethod
    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]: ...

    def is_mp_src_rank_with_outputs(self, handle: ModelHandle) -> bool:
        """Return whether this rank owns complete model outputs."""
        return True

    def export_weights(
        self, handle: ModelHandle, **kwargs
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over inference-compatible ``(name, tensor)`` pairs."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement export_weights. "
            "Implement it to unlock the RL Ready tier."
        )

    def to(
        self,
        handle: ModelHandle,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        """Move selected model state between devices."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to(). "
            "Implement it to unlock the RL Best tier."
        )

    @property
    def tier(self) -> Literal["pretrain", "rl_ready", "rl_best"]:
        """Report the highest runtime API tier implemented by this backend."""
        cls = type(self)
        has_export = cls.export_weights is not Runtime.export_weights
        has_to = cls.to is not Runtime.to
        if has_export and has_to:
            return "rl_best"
        if has_export:
            return "rl_ready"
        return "pretrain"


__all__ = ["Runtime"]
