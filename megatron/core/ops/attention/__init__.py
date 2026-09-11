# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attention operation modules and kernels; model assembly and cache lifecycle stay external."""

from megatron.core.ops.attention.kernel_metadata import KERNELS

__all__ = ["KERNELS"]
