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

    Args:
        source: The function or callable from which to copy the signature.
        handle_return_type: How to handle the return type annotation.
            'preserve' to keep the decorated function's return type,
            'overwrite' to use the source function's return type.
        handle_first_src_param: How to handle the first parameter of the source function.
            'copy' to include it in the decorated function's signature,
            'skip' to exclude it. Useful for removing 'self' or 'cls'.
        handle_first_dst_param: How to handle the first parameter of the decorated function.
            'preserve' to keep it in the decorated function's signature,
            'drop' to exclude it. Useful for preserving 'self' or 'cls'.

    Returns:
        A decorator that copies the signature from `func` to the decorated function.
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
