# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Compatibility imports for backend selection.

Backend selection lives in :mod:`megatron.core.ops`. This module keeps the names it used to
define working. The three ``*SpecProvider`` names are now factories rather than classes: there
is one provider type, and a preset selects which backend fills each operation. Calling them
still returns something with the same methods, so ``TESpecProvider().layer_norm()`` is
unchanged, but ``isinstance`` against them no longer makes sense.
"""

from megatron.core.ops import (
    BackendOptions,
    BackendSpecProvider,
    Operation,
    get_backend,
    get_backend_spec_provider,
)


def LocalSpecProvider() -> BackendSpecProvider:  # pylint: disable=invalid-name
    """Deprecated: use ``get_backend("local")``."""
    return get_backend("local")


def InferenceSpecProvider() -> BackendSpecProvider:  # pylint: disable=invalid-name
    """Deprecated: use ``get_backend("inference_optimized")``."""
    return get_backend("inference_optimized")


__all__ = [
    "BackendOptions",
    "BackendSpecProvider",
    "InferenceSpecProvider",
    "LocalSpecProvider",
    "Operation",
    "get_backend",
    "get_backend_spec_provider",
]
