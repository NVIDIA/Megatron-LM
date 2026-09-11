# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared local SSM kernels: convolution, explicit state access and autotuning.

Decode updates mutate caller-owned state using explicit slot indices. The caller
owns slot allocation, sequence metadata and CP exchange. Kernel signatures retain
the existing layouts and dtype constraints; they are not interchangeable with
training kernels that provide backward.
"""

from .kernel_metadata import KERNELS as KERNELS
