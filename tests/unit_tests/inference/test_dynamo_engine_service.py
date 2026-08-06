# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.inference.integrations.dynamo.protocol import engine_metadata


def test_engine_metadata_contains_dynamo_configuration():
    engine = SimpleNamespace(
        context=SimpleNamespace(
            kv_block_allocator=SimpleNamespace(pool_size=17),
            max_sequence_length=8192,
            block_size_tokens=64,
            max_requests=8,
            max_tokens=1024,
            enable_prefix_caching=True,
        ),
        controller=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1)),
    )

    assert engine_metadata(engine, "decode") == {
        "context_length": 8192,
        "kv_cache_block_size": 64,
        "total_kv_blocks": 16,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 1024,
        "role": "decode",
        "bos_token_id": 1,
        "enable_prefix_caching": True,
        "logical_data_parallel_size": 1,
    }
