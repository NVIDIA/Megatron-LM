# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Loss-side runtime context, kept separate from batch data contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LossContext:
    """Per-microbatch loss/output policy for model-owned loss helpers."""

    temperature: float = 1.0
    calculate_entropy: bool = False
    return_log_probs: bool = True
    loss_scale: float = 1.0
    source_batch: Any | None = None


_CURRENT_LOSS_CONTEXT: ContextVar[LossContext | None] = ContextVar(
    "megatron_lite_loss_context", default=None
)


def get_loss_context() -> LossContext | None:
    return _CURRENT_LOSS_CONTEXT.get()


@contextmanager
def use_loss_context(loss_context: LossContext | None) -> Iterator[None]:
    token = _CURRENT_LOSS_CONTEXT.set(loss_context)
    try:
        yield
    finally:
        _CURRENT_LOSS_CONTEXT.reset(token)


def split_loss_context(item):
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and (item[1] is None or isinstance(item[1], LossContext))
    ):
        return item
    return item, None


__all__ = ["LossContext", "get_loss_context", "split_loss_context", "use_loss_context"]
