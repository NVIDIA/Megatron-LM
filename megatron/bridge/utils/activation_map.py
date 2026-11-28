# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Compatibility imports for the shared Core config normalization utilities."""

from megatron.core.transformer.config_utils import (
    ACTIVATION_FUNC_MAP,
    DTYPE_MAP,
    callable_to_str,
    str_to_callable,
    str_to_dtype,
)

__all__ = [
    "ACTIVATION_FUNC_MAP",
    "DTYPE_MAP",
    "callable_to_str",
    "str_to_callable",
    "str_to_dtype",
]
