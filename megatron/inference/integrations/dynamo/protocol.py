# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine metadata sent to the Dynamo parent at startup."""

from __future__ import annotations

def engine_metadata(engine, role: str) -> dict:
    """Return the capabilities Dynamo needs to configure this engine."""

    allocator = engine.context.kv_block_allocator
    tokenizer = engine.controller.tokenizer
    bos_token_id = next(
        (
            int(value)
            for name in ("bos", "bos_token_id", "eod")
            if (value := getattr(tokenizer, name, None)) is not None
        ),
        0,
    )
    return {
        "context_length": int(engine.context.max_sequence_length),
        "kv_cache_block_size": int(engine.context.block_size_tokens),
        "total_kv_blocks": max(0, int(allocator.pool_size) - 1),
        "max_num_seqs": int(engine.context.max_requests),
        "max_num_batched_tokens": int(engine.context.max_tokens),
        "role": role,
        "bos_token_id": bos_token_id,
        "enable_prefix_caching": bool(engine.context.enable_prefix_caching),
        "logical_data_parallel_size": 1,
    }
