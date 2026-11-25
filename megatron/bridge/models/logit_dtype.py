# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/logit_dtype.py
# Source license: Apache-2.0.

"""Compatibility helpers for MCore's optional output-logit dtype API."""

import inspect
from collections.abc import Callable
from typing import Any

import torch

_SUPPORTED_LOGIT_DTYPES = (torch.bfloat16, torch.float32)


def optional_dtype_kwarg(
    constructor: Callable[..., Any], keyword: str, dtype: torch.dtype | None
) -> dict[str, torch.dtype]:
    """Return an optional dtype keyword when the installed constructor supports it.

    Omitting a ``None`` value preserves the exact behavior of older MCore
    constructors. A configured value must never be silently dropped when the
    installed MCore predates the corresponding API.

    Args:
        constructor: Model or layer constructor that should receive the keyword.
        keyword: Constructor keyword, either ``logit_dtype`` or ``output_dtype``.
        dtype: Requested output dtype, or ``None`` to preserve the input dtype.

    Returns:
        An empty mapping for the default behavior, otherwise the supported
        constructor keyword.

    Raises:
        ValueError: If the requested dtype is outside MCore's supported contract.
        RuntimeError: If the installed constructor does not support the keyword.
    """
    if dtype is None:
        return {}
    if dtype not in _SUPPORTED_LOGIT_DTYPES:
        supported = ", ".join(str(item) for item in _SUPPORTED_LOGIT_DTYPES)
        raise ValueError(f"logit_dtype must be one of ({supported}) or None, got {dtype}.")

    parameters = inspect.signature(constructor).parameters
    if keyword not in parameters:
        constructor_name = getattr(
            constructor, "__qualname__", getattr(constructor, "__name__", str(constructor))
        )
        raise RuntimeError(
            f"{constructor_name} does not support {keyword}, so the installed MCore cannot honor "
            "the requested FP32/BF16 output logits. The base GPT/Hybrid API is tracked by "
            "Megatron-LM PR #6252; this constructor needs a compatible implementation of that API."
        )
    return {keyword: dtype}


def logit_dtype_kwarg(
    constructor: Callable[..., Any], dtype: torch.dtype | None
) -> dict[str, torch.dtype]:
    """Return a guarded ``logit_dtype`` constructor keyword."""
    return optional_dtype_kwarg(constructor, "logit_dtype", dtype)


def output_dtype_kwarg(
    constructor: Callable[..., Any], dtype: torch.dtype | None
) -> dict[str, torch.dtype]:
    """Return a guarded ``output_dtype`` constructor keyword."""
    return optional_dtype_kwarg(constructor, "output_dtype", dtype)


__all__ = ["logit_dtype_kwarg", "optional_dtype_kwarg", "output_dtype_kwarg"]
