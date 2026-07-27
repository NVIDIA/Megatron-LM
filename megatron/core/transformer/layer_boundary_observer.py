# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Scoped observation of transformer-layer input and output boundaries."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from torch import Tensor, nn

LayerBoundary = Literal["input", "output"]
LayerBoundaryObserver = Callable[[nn.Module, nn.Module, LayerBoundary, Tensor], None]

_LAYER_BOUNDARY_OBSERVER: ContextVar[LayerBoundaryObserver | None] = ContextVar(
    "layer_boundary_observer", default=None
)


@contextmanager
def observe_transformer_layer_boundaries(observer: LayerBoundaryObserver) -> Iterator[None]:
    """Make ``observer`` active for transformer-layer boundary notifications."""
    token = _LAYER_BOUNDARY_OBSERVER.set(observer)
    try:
        yield
    finally:
        _LAYER_BOUNDARY_OBSERVER.reset(token)


def observe_transformer_layer_input(
    stack: nn.Module, layer: nn.Module, hidden_states: Tensor
) -> None:
    """Notify the active observer of the decoder's initial residual stream."""
    if getattr(layer, "layer_number", None) != 1:
        return
    _observe_transformer_layer_boundary(stack, layer, "input", hidden_states)


def observe_transformer_layer_output(
    stack: nn.Module, layer: nn.Module, hidden_states: Tensor
) -> None:
    """Notify the active observer of a post-layer residual stream."""
    _observe_transformer_layer_boundary(stack, layer, "output", hidden_states)


def _observe_transformer_layer_boundary(
    stack: nn.Module, layer: nn.Module, boundary: LayerBoundary, hidden_states: Tensor
) -> None:
    observer = _LAYER_BOUNDARY_OBSERVER.get()
    if observer is not None:
        observer(stack, layer, boundary, hidden_states)
