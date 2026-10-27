# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Utilities for improved type hinting with torch interfaces."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Concatenate, Generic, Literal, ParamSpec, Protocol, TypeVar, overload

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


R_src = TypeVar('R_src')
R_dst = TypeVar('R_dst')
P_src = ParamSpec('P_src')
P_dst = ParamSpec('P_dst')
First_dst = TypeVar('First_dst')


@overload
def copy_signature(
    source: Callable[P_src, Any],
    /,
    *,
    handle_return_type: Literal['preserve'] = 'preserve',
    handle_first_src_param: Literal['copy'] = 'copy',
    handle_first_dst_param: Literal['drop'] = 'drop',
) -> Callable[[Callable[..., R_dst]], Callable[P_src, R_dst]]: ...


@overload
def copy_signature(
    source: Callable[P_src, R_src],
    /,
    *,
    handle_return_type: Literal['overwrite'],
    handle_first_src_param: Literal['copy'] = 'copy',
    handle_first_dst_param: Literal['drop'] = 'drop',
) -> Callable[[Callable[..., Any]], Callable[P_src, R_src]]: ...


@overload
def copy_signature(
    source: Callable[Concatenate[Any, P_src], Any],
    /,
    *,
    handle_return_type: Literal['preserve'] = 'preserve',
    handle_first_src_param: Literal['skip'],
    handle_first_dst_param: Literal['drop'] = 'drop',
) -> Callable[[Callable[..., R_dst]], Callable[P_src, R_dst]]: ...


@overload
def copy_signature(
    source: Callable[Concatenate[Any, P_src], R_src],
    /,
    *,
    handle_return_type: Literal['overwrite'],
    handle_first_src_param: Literal['skip'],
    handle_first_dst_param: Literal['drop'] = 'drop',
) -> Callable[[Callable[..., Any]], Callable[P_src, R_src]]: ...


@overload
def copy_signature(
    source: Callable[P_src, Any],
    /,
    *,
    handle_return_type: Literal['preserve'] = 'preserve',
    handle_first_src_param: Literal['copy'] = 'copy',
    handle_first_dst_param: Literal['preserve'],
) -> Callable[
    [Callable[Concatenate[First_dst, ...], R_dst]], Callable[Concatenate[First_dst, P_src], R_dst]
]: ...


@overload
def copy_signature(
    source: Callable[P_src, R_src],
    /,
    *,
    handle_return_type: Literal['overwrite'],
    handle_first_src_param: Literal['copy'] = 'copy',
    handle_first_dst_param: Literal['preserve'],
) -> Callable[
    [Callable[Concatenate[First_dst, ...], Any]], Callable[Concatenate[First_dst, P_src], R_src]
]: ...


@overload
def copy_signature(
    source: Callable[Concatenate[Any, P_src], Any],
    /,
    *,
    handle_return_type: Literal['preserve'] = 'preserve',
    handle_first_src_param: Literal['skip'],
    handle_first_dst_param: Literal['preserve'],
) -> Callable[
    [Callable[Concatenate[First_dst, ...], R_dst]], Callable[Concatenate[First_dst, P_src], R_dst]
]: ...


@overload
def copy_signature(
    source: Callable[Concatenate[Any, P_src], R_src],
    /,
    *,
    handle_return_type: Literal['overwrite'],
    handle_first_src_param: Literal['skip'],
    handle_first_dst_param: Literal['preserve'],
) -> Callable[
    [Callable[Concatenate[First_dst, ...], Any]], Callable[Concatenate[First_dst, P_src], R_src]
]: ...


def copy_signature(
    source: Callable[..., Any],
    /,
    *,
    handle_return_type: Literal['preserve', 'overwrite'] = 'preserve',
    handle_first_src_param: Literal['copy', 'skip'] = 'copy',
    handle_first_dst_param: Literal['preserve', 'drop'] = 'drop',
):
    """Decorator to copy the signature from one function to another.

      Similar to `functools.wraps`, but preserves the signature instead of the
      metadata. Useful when writing adapter/wrapper functions that forward arguments
      to another function, as in:

          def function_with_lots_of_args(
              a: int,
              b: str,
              c: float,
              ...
          ) -> BigObject:
              ...

          @copy_signature(function_with_lots_of_args)
          def convenient_wrapper(*args: Any, **kwargs: Any) -> str:
              return function_with_lots_of_args(*args, **kwargs).to_string()

    Args:
        source: The function or callable from which to copy the signature.
        handle_return_type: How to handle the return type of the decorated
          function. 'preserve' to keep the decorated function's return type
          (the default, since many wrappers are specifically written to return a
          different type), or 'overwrite' to copy the source function's return
          type as well.
        handle_first_src_param: Whether to include the first parameter of the
          source function. 'copy' to include it in the decorated function's
          signature (the default), 'skip' to exclude it (useful for removing
          'self' or 'cls').
        handle_first_dst_param: Whether to keep the first parameter of the
          decorated function. 'drop' to overwrite it just like any other parameter
          (the default), or 'preserve' to keep it in the decorated function's
          signature (useful for preserving 'self' or 'cls').

      Returns:
          A decorator that copies the signature from `source` to the decorated function.
    """
    source_signature = inspect.signature(source)

    def decorator(decorated: Callable[..., Any], /) -> Callable[..., Any]:
        dest_signature = inspect.signature(decorated)
        new_params = []
        if handle_first_dst_param == 'preserve':
            new_params.append(next(iter(dest_signature.parameters.values())))
        src_params_iter = iter(source_signature.parameters.values())
        if handle_first_src_param == 'skip':
            next(src_params_iter)
        new_params.extend(src_params_iter)
        new_signature = dest_signature.replace(parameters=new_params)
        if handle_return_type == 'overwrite':
            new_signature = new_signature.replace(
                return_annotation=source_signature.return_annotation
            )

        decorated.__signature__ = new_signature  # type: ignore
        return decorated

    return decorator
