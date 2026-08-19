# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Kitchen quantization backend.

Kitchen ships outside this repository. It provides a partial backend that takes over its
quantized linear, attention, and expert modules and forwards everything else to a fallback,
which is exactly the composition shape described in the ops design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from megatron.core.ops import _availability

if TYPE_CHECKING:
    from megatron.core.ops.spec_provider import BackendSpecProvider

__all__ = ["kitchen_provider"]

_BACKEND_NAME = "kitchen"


def kitchen_provider(
    fallback: "BackendSpecProvider",
    *,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
) -> "BackendSpecProvider":
    """Return ``fallback`` with Kitchen's quantized operations layered on top."""
    _availability.require("nvidia_kitchen", backend=_BACKEND_NAME)
    from megatron.core.extensions.kitchen import KitchenSpecProvider

    return KitchenSpecProvider(
        fallback=fallback,
        use_kitchen_attention=use_kitchen_attention,
        kitchen_attention_backend=kitchen_attention_backend,
    )
