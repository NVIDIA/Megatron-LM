# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Operation implementations and the construction-time API that selects between them.

Model code asks a :class:`BackendSpecProvider` for the implementation of an operation while it
builds a spec, and calls the returned target directly afterwards. There is no registry lookup
and no dispatch in the forward path.

A *backend* is any object implementing one or more provider methods. Backends compose: a base
backend supplies everything, and any operation can be handed to another backend::

    provider = get_backend_spec_provider(config)   # honours --transformer-impl and --op-backend
    norm_cls = provider.layer_norm(rms_norm=True)  # returns the class, nothing else happens

Implementations live under ``megatron/core/ops/<operation>/<backend>.py``, next to the contract
their family has to meet.
"""

from megatron.core.ops.operations import Operation, parse_operation
from megatron.core.ops.options import BackendOptions
from megatron.core.ops.resolve import (
    available_backends,
    build_spec_provider,
    get_backend,
    get_backend_spec_provider,
)
from megatron.core.ops.spec_provider import BackendSpecProvider, compose

__all__ = [
    "BackendOptions",
    "BackendSpecProvider",
    "Operation",
    "available_backends",
    "build_spec_provider",
    "compose",
    "get_backend",
    "get_backend_spec_provider",
    "parse_operation",
]
