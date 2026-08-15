# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scoped notifications for observing short-lived tensors in Megatron Core."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

TensorObservationCallback = Callable[[object, str, str, torch.Tensor, int | None], None]

_ACTIVE_OBSERVER: ContextVar[tuple[TensorObservationCallback, frozenset[str]] | None] = ContextVar(
    "tensor_observation", default=None
)


@contextmanager
def capture_tensor_observations(
    observer: TensorObservationCallback, source_kinds: frozenset[str]
) -> Iterator[None]:
    """Notify ``observer`` about the selected tensor source kinds within this scope.

    Args:
        observer: Callback receiving the owner, local name, source kind, tensor, and optional
            tensor-parallel shard dimension of each observation.
        source_kinds: Source kinds to notify. Other observation sites remain no-ops.
    """
    token = _ACTIVE_OBSERVER.set((observer, source_kinds))
    try:
        yield
    finally:
        _ACTIVE_OBSERVER.reset(token)


def is_observing_tensor(source_kind: str) -> bool:
    """Return whether ``source_kind`` has an active observer."""
    active = _ACTIVE_OBSERVER.get()
    return active is not None and source_kind in active[1]


def observe_tensor(
    owner: object,
    name: str,
    source_kind: str,
    tensor: torch.Tensor,
    *,
    tp_shard_dim: int | None = None,
) -> None:
    """Notify the active observer about one tensor produced with autograd enabled.

    The notification is skipped during the no-grad forward of activation checkpointing. The
    grad-enabled recomputation is observed instead, producing one contribution per execution.

    Args:
        owner: Model object used to resolve a canonical site name.
        name: Name local to ``owner``.
        source_kind: Stable source category used by metric selection.
        tensor: Short-lived observed tensor.
        tp_shard_dim: Tensor dimension sharded over tensor parallel ranks, or ``None`` when the
            tensor is replicated over tensor parallel ranks.
    """
    if not torch.is_grad_enabled():
        return
    _notify_observer(owner, name, source_kind, tensor, tp_shard_dim)


def observe_layer_residuals(
    layer: object, accumulator: torch.Tensor, output: torch.Tensor
) -> None:
    """Observe a layer's incoming residual accumulator and net contribution.

    The contribution is reconstructed as ``output - accumulator`` outside the layer's local CUDA
    graph. It is computed without autograd only when a contribution metric is active.

    Args:
        layer: Layer owning the residual sites.
        accumulator: Residual stream entering the layer.
        output: Residual stream leaving the layer.
    """
    if not torch.is_grad_enabled():
        return
    config = getattr(layer, "config", None)
    tp_shard_dim = 0 if getattr(config, "sequence_parallel", False) else None
    if is_observing_tensor("residual_accumulator"):
        _notify_observer(
            layer,
            "residual_accumulator",
            "residual_accumulator",
            accumulator,
            tp_shard_dim,
        )
    if is_observing_tensor("residual_contribution"):
        with torch.no_grad():
            contribution = output.detach() - accumulator.detach()
        _notify_observer(
            layer,
            "residual_contribution",
            "residual_contribution",
            contribution,
            tp_shard_dim,
        )


def _notify_observer(
    owner: object,
    name: str,
    source_kind: str,
    tensor: torch.Tensor,
    tp_shard_dim: int | None,
) -> None:
    active = _ACTIVE_OBSERVER.get()
    if active is not None and source_kind in active[1]:
        active[0](owner, name, source_kind, tensor, tp_shard_dim)
