# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated import path; use ``megatron.core.ops.loss.fused_cross_entropy``."""

from megatron.core.ops._compat import deprecated_module

__getattr__, __dir__ = deprecated_module(__name__, "megatron.core.ops.loss.fused_cross_entropy")
