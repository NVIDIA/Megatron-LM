# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Compatibility imports for backend selection.

Backend selection now lives in :mod:`megatron.core.ops`. This module re-exports the names it
used to define so existing imports keep working.
"""

from megatron.core.ops import (
    BackendOptions,
    BackendSpecProvider,
    Operation,
    get_backend,
    get_backend_spec_provider,
)
from megatron.core.ops.providers.inference import InferenceSpecProvider
from megatron.core.ops.providers.local import LocalSpecProvider

__all__ = [
    "BackendOptions",
    "BackendSpecProvider",
    "InferenceSpecProvider",
    "LocalSpecProvider",
    "Operation",
    "get_backend",
    "get_backend_spec_provider",
]
