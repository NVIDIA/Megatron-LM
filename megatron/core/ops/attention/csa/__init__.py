# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compressed sparse-attention modules and reference kernels.

Queries are SBHD, shared KV is SBD, and top-k indices are BSJ with -1 for invalid
keys. Attention returns SB(HD); teacher log mass is detached FP32 BHS. Reference
kernels own no parameters or process groups. ``modules`` owns compressors,
indexers, rotary embeddings, learnable sinks and operation checkpoint behavior.
Cross-layer index sharing is coordinated by model assembly. See each
callable for dtype and gradient details; bit-exact determinism is not certified.
"""

from .kernel_metadata import KERNELS as KERNELS
