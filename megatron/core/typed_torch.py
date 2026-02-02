# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utilities for improved type hinting with torch interfaces."""
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, ParamSpec, Protocol, TypeVar

import torch

P = ParamSpec('P')
R_co = TypeVar('R_co', covariant=True)
T = TypeVar('T')


class _Module(Generic[P, R_co], Protocol):
    """Protocol allowing us to unwrap `forward`."""

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        """Forward method of the matching torch.nn.Module."""
        ...


def apply_module(m: _Module[P, R_co], *, check_subclass: bool = True) -> Callable[P, R_co]:
    """Returns the provided module unchanged, but with correct type hints.

    Args:
      m: An instance of a subclass of `torch.nn.Module`.
      check_subclass: If `True`, checks that `m` is a subclass of
            `torch.nn.Module` and raises a `TypeError` if not.

    Returns:
      That module unchanged, but with correct type hints.
    """
    if check_subclass and not issubclass(type(m), torch.nn.Module):
        raise TypeError(f'{type(m)} is not a subclass of torch.nn.Module')
    return m  # type: ignore


def not_none(value: T | None) -> T:
    """Asserts that the provided value is not None and returns it.

    Args:
        value: An optional value.

    Returns:
        The provided value, guaranteed to be not None.
    """
    if value is None:
        raise ValueError('Expected value to be not None')
    return value
