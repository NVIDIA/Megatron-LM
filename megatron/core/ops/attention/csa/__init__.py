# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compressed sparse-attention reference kernels.

Queries are SBHD, shared KV is SBD, and top-k indices are BSJ with -1 for invalid
keys. Attention returns SB(HD); teacher log mass is detached FP32 BHS. These
kernels own no parameters or process groups. Compressors, rotary embeddings,
learnable sinks, index sharing and checkpointing stay with the model. See each
callable for dtype and gradient details; bit-exact determinism is not certified.
"""
