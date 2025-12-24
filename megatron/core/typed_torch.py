# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utilities for improved type hinting with torch interfaces."""

from collections.abc import Callable
from typing import Generic, ParamSpec, Protocol, TypeVar

import torch

P = ParamSpec('P')
R_co = TypeVar('R_co', covariant=True)


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
