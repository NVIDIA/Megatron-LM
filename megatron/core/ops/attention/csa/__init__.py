# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compressed sparse-attention modules and kernels.

Queries are SBHD, shared KV is SBD, and top-k indices are BSJ with -1 for invalid
keys; the packed THD path takes final physical indices and explicit process groups.
Attention returns SB(HD); teacher log mass is detached FP32. Kernels own no
parameters. ``modules`` owns compressors, indexers, rotary embeddings, learnable
sinks and operation checkpoint behavior. Cross-layer index sharing is coordinated by
model assembly. See each callable for dtype and gradient details; bit-exact
determinism is not certified.
"""
