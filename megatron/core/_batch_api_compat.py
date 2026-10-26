# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility adapters for the deprecated Hybrid-CP batch-helper keywords."""

import inspect
import warnings
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")
_DEPRECATED_KEYWORDS = (
    ("is_hybrid_cp", "dynamic_cp"),
    ("hybrid_cp_group_func", "dynamic_cp_group_func"),
)


def deprecated_hybrid_cp_kwargs(
    function: Callable[_Parameters, _Result],
) -> Callable[_Parameters, _Result]:
    """Accept old keyword names without changing positional arguments or new signatures."""

    @wraps(function)
    def wrapped(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
        for old_name, new_name in _DEPRECATED_KEYWORDS:
            if old_name in kwargs:
                if new_name in kwargs:
                    raise TypeError(f"Cannot pass both {old_name} and {new_name}")
                warnings.warn(
                    f"{old_name} is deprecated; use {new_name} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new_name] = kwargs.pop(old_name)
        return function(*args, **kwargs)

    # MBridge detects THD CP support by inspecting the deprecated flag's name.
    # Keep keyword aliases discoverable, not just accepted at call time.
    signature = inspect.signature(function)
    aliases = [
        inspect.Parameter(
            old_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=signature.parameters[new_name].annotation,
        )
        for old_name, new_name in _DEPRECATED_KEYWORDS
        if new_name in signature.parameters
    ]
    wrapped.__signature__ = signature.replace(parameters=[*signature.parameters.values(), *aliases])
    return wrapped
