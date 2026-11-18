# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Construction-time selection of optional CSA kernels."""

import torch

from megatron.core.ops._backends import require


def select_compact_indexer(precision, *, workspaces, cuda_graphs=False, deterministic=False):
    """Keep native BF16 scoring, or bind the required compact MXFP8 implementation."""
    if precision == "bf16":
        return None
    if precision != "mxfp8":
        raise ValueError(f"Unsupported CSA indexer precision: {precision!r}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("MXFP8 CSA indexer requires an SM100 or later CUDA GPU")
    needed_by = "MXFP8 CSA indexer"
    require(
        "cudnn",
        "DSA.indexer_forward_top_k_wrapper",
        "DSA.compress_topk_cand_buffer_size",
        "DSA.compress_topk_cand_buffer_size_thd",
        needed_by=needed_by,
    )
    require("transformer_engine.pytorch.tensor.mxfp8_tensor", "MXFP8Quantizer", needed_by=needed_by)
    require("triton", needed_by=needed_by)
    module = require(
        "megatron.core.ops.attention.csa.compact_indexer", "CompactIndexer", needed_by=needed_by
    )
    return module.CompactIndexer(
        workspaces=workspaces, cuda_graphs=cuda_graphs, deterministic=deterministic
    )
