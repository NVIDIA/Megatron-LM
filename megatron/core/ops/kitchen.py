# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Kitchen quantization backend.

Kitchen ships outside this repository as a partial backend: it takes over its quantized
linear, attention, and expert modules and forwards everything else to a fallback. That is
already the composition shape used here, so it is layered over an assembled provider rather
than split across the operation families, and core never has to know which slots it claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.ops.options import BackendOptions

__all__ = ["kitchen_backend"]


def kitchen_backend(fallback: object, options: "BackendOptions") -> object:
    """Return a backend that owns Kitchen's operations and defers the rest to ``fallback``.

    The dependency check lives in :mod:`megatron.core.ops.resolve` with every other one.
    """
    from megatron.core.extensions.kitchen import KitchenSpecProvider

    return KitchenSpecProvider(
        fallback=fallback,
        use_kitchen_attention=options.use_kitchen_attention,
        kitchen_attention_backend=options.kitchen_attention_backend,
    )
