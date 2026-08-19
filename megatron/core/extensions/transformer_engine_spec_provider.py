# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility import for the Transformer Engine backend.

Selection moved to :mod:`megatron.core.ops`. ``TESpecProvider`` is now a factory rather than a
class: there is one provider type, assembled from the per-operation backends the Transformer
Engine preset selects. Calling it still returns something with the same methods.
"""

from megatron.core.ops import BackendSpecProvider, get_backend

__all__ = ["TESpecProvider"]


def TESpecProvider() -> BackendSpecProvider:  # pylint: disable=invalid-name
    """Deprecated: use ``get_backend("transformer_engine")``."""
    return get_backend("transformer_engine")
