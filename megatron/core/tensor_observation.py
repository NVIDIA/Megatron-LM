# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scoped notifications for observing short-lived tensors in Megatron Core."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

TensorObservationCallback = Callable[
    [object, str, str, torch.Tensor, int | None, int | None, int | None], None
]

_ACTIVE_OBSERVER: ContextVar[tuple[TensorObservationCallback, frozenset[str]] | None] = ContextVar(
    "tensor_observation", default=None
)
_OBSERVATION_SUSPENDED: ContextVar[bool] = ContextVar("tensor_observation_suspended", default=False)


@contextmanager
def capture_tensor_observations(
    observer: TensorObservationCallback, source_kinds: frozenset[str]
) -> Iterator[None]:
    """Notify ``observer`` about the selected tensor source kinds within this scope.

    Args:
        observer: Callback receiving the owner, local name, source kind, tensor, and its optional
            tensor-parallel, sequence, and batch dimensions.
        source_kinds: Source kinds to notify. Other observation sites remain no-ops.
    """
    token = _ACTIVE_OBSERVER.set((observer, source_kinds))
    try:
        yield
    finally:
        _ACTIVE_OBSERVER.reset(token)


@contextmanager
def suspend_tensor_observations() -> Iterator[None]:
    """Temporarily suppress notifications to the active tensor observer.

    Activation checkpoint implementations use this around their backward recomputation so the
    original forward is observed exactly once.
    """
    token = _OBSERVATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _OBSERVATION_SUSPENDED.reset(token)


def is_observing_tensor(source_kind: str) -> bool:
    """Return whether ``source_kind`` has an active observer."""
    if _OBSERVATION_SUSPENDED.get():
        return False
    active = _ACTIVE_OBSERVER.get()
    return active is not None and source_kind in active[1]


def observe_tensor(
    owner: object,
    name: str,
    source_kind: str,
    tensor: torch.Tensor,
    *,
    tp_shard_dim: int | None = None,
    sequence_dim: int | None = None,
    batch_dim: int | None = None,
) -> None:
    """Notify the active observer about one tensor produced during the logical forward.

    Activation checkpoint implementations suppress notifications during their backward
    recomputation, so the original no-grad forward remains the single observed execution.

    Args:
        owner: Model object used to resolve a canonical site name.
        name: Name local to ``owner``.
        source_kind: Stable source category used by metric selection.
        tensor: Short-lived observed tensor.
        tp_shard_dim: Tensor dimension sharded over tensor parallel ranks, or ``None`` when the
            tensor is replicated over tensor parallel ranks.
        sequence_dim: Tensor dimension identifying the context-parallel sequence population, or
            ``None`` when that population is not represented by one tensor dimension.
        batch_dim: Tensor dimension identifying the data- and gradient-tensor-parallel batch
            population, or ``None`` when it is not represented by one tensor dimension.
    """
    _notify_observer(owner, name, source_kind, tensor, tp_shard_dim, sequence_dim, batch_dim)


def observe_layer_residuals(layer: object, accumulator: torch.Tensor, output: torch.Tensor) -> None:
    """Observe a layer's incoming residual accumulator and net contribution.

    With per-layer CUDA graphs, the contribution is reconstructed as ``output - accumulator``
    outside the captured layer. Full-iteration CUDA graphs include this call and do not replay
    Python observation notifications. The contribution is computed without autograd only when a
    contribution metric is active.

    Args:
        layer: Layer owning the residual sites.
        accumulator: Residual stream entering the layer.
        output: Residual stream leaving the layer.
    """
    config = getattr(layer, "config", None)
    tp_shard_dim = 0 if getattr(config, "sequence_parallel", False) else None
    if is_observing_tensor("residual_accumulator"):
        _notify_observer(
            layer, "residual_accumulator", "residual_accumulator", accumulator, tp_shard_dim, 0, 1
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
            0,
            1,
        )


def _notify_observer(
    owner: object,
    name: str,
    source_kind: str,
    tensor: torch.Tensor,
    tp_shard_dim: int | None,
    sequence_dim: int | None,
    batch_dim: int | None,
) -> None:
    if _OBSERVATION_SUSPENDED.get():
        return
    active = _ACTIVE_OBSERVER.get()
    if active is not None and source_kind in active[1]:
        active[0](owner, name, source_kind, tensor, tp_shard_dim, sequence_dim, batch_dim)
