# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""COMPATIBILITY SHIM -- do not add implementation here.

The context-parallel layout API now lives in
:mod:`megatron.core.context_parallel.layout`. This module only re-exports that
package so older ``import megatron.core.context_parallel_layout`` call sites keep
working. It is TEMPORARY and WILL BE DROPPED once every caller imports from
``megatron.core.context_parallel.layout`` directly.
"""

from megatron.core.context_parallel.layout import *  # noqa: F401,F403
from megatron.core.context_parallel.layout import __all__ as _layout_all

__all__ = list(_layout_all)
