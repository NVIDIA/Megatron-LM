# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba2 mixer, context-parallel implementation and state-space duality kernels.

The vendored SSD and selective-state-update implementations serve inference;
mamba-ssm supplies the training/backward kernels. The mixer selects the phase,
owns parameters and checkpoint mappings, and uses this family's CP transforms.
Global inference caches remain context-owned. Packed offsets, initial states and output buffers are
explicit kernel inputs. Keep existing dtype, graph and determinism guards.
"""

from .kernel_metadata import KERNELS as KERNELS
