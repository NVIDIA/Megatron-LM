# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared SSM operation helpers, convolution, explicit state access and autotuning.

Decode updates mutate caller-owned state using explicit slot indices. The caller
owns slot allocation and global sequence metadata. Operation-level CP exchange,
packing, checkpoint helpers and inference execution also live here. Kernels retain
the existing layouts and dtype constraints; they are not interchangeable with
training kernels that provide backward.
"""

from .kernel_metadata import KERNELS as KERNELS
