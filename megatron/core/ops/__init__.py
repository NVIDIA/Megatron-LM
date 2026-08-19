# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Operation implementations and the construction-time API that selects between them.

Model code asks a :class:`BackendSpecProvider` for the implementation of an operation while it
builds a spec, and calls the returned target directly afterwards. There is no registry lookup
and no dispatch in the forward path::

    provider = get_backend_spec_provider(config)   # honours --transformer-impl and --op-backend
    norm_cls = provider.layer_norm(rms_norm=True)  # returns the class, nothing else happens

A *backend* is any object implementing one or more provider methods. Each operation family
under ``megatron/core/ops/<family>/`` declares its own operations, contract, and backend table,
so adding a backend touches one family and nothing central. See ``README.md``.
"""

from megatron.core.ops.operations import Operation
from megatron.core.ops.options import BackendOptions
from megatron.core.ops.resolve import (
    PRESETS,
    backends_for,
    build_spec_provider,
    find_operation,
    get_backend,
    get_backend_spec_provider,
    operations,
    validate_backend,
)
from megatron.core.ops.spec_provider import BackendSpecProvider

__all__ = [
    "PRESETS",
    "BackendOptions",
    "BackendSpecProvider",
    "Operation",
    "backends_for",
    "build_spec_provider",
    "find_operation",
    "get_backend",
    "get_backend_spec_provider",
    "operations",
    "validate_backend",
]
