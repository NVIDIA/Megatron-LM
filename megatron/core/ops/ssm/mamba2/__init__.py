# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba2 state-space duality kernels and optional training targets.

The vendored SSD and selective-state-update implementations serve inference;
mamba-ssm supplies the training/backward kernels. The mixer selects the phase,
owns parameters and recurrent state, and performs CP transformations outside
these local kernels. Packed offsets, initial states and output buffers are
explicit kernel inputs. Keep existing dtype, graph and determinism guards.
"""

from .kernel_metadata import KERNELS as KERNELS
