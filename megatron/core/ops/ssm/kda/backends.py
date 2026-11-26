# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction-time backend selection for Kimi Delta Attention."""

from typing import Callable

from megatron.core.ops._backends import require


def select_kda_recurrence(backend: str) -> Callable:
    """Return the requested KDA recurrence without silent fallback."""
    if backend == "fla":
        return require("fla.ops.kda", "chunk_kda", needed_by="FLA Kimi Delta Attention").chunk_kda
    if backend == "cudnn":
        require(
            "cudnn.linear_attention.ops",
            "kimi_delta_attention",
            "kimi_delta_attention_summary",
            "kimi_delta_attention_summary_bwd",
            min_version="1.32.0",
            dist="nvidia-cudnn-frontend",
            needed_by="cuDNN Kimi Delta Attention",
        )
        require(
            "cudnn.linear_attention.frost.common.piece_chain",
            "build_state_chain",
            "run_state_chain",
            needed_by="cuDNN Kimi Delta Attention context parallelism",
        )
        from megatron.core.ops.ssm.kda.cudnn_backend import cudnn_chunk_kda

        return cudnn_chunk_kda
    raise ValueError(f"Unsupported KDA backend {backend!r}; expected 'fla' or 'cudnn'.")
