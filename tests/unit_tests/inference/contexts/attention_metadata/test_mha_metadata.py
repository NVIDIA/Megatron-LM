# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.attention_context.mha_metadata import (
    GraphedMHAMetadata,
    MHAMetadata,
    NonGraphedMHAMetadata,
)

MAX_REQUESTS = 4
MAX_KV_BLOCKS = 8


def _make_gpu_view():
    """Build a fake ContextGPUView with the slice attributes mha_metadata reads."""
    return SimpleNamespace(
        mha_query_lengths=torch.zeros(MAX_REQUESTS, dtype=torch.int32),
        mha_cu_query_seq_lengths=torch.zeros(MAX_REQUESTS + 1, dtype=torch.int32),
        mha_cu_kv_seq_lengths=torch.zeros(MAX_REQUESTS + 1, dtype=torch.int32),
        mha_kv_seq_lengths=torch.zeros(MAX_REQUESTS, dtype=torch.int32),
        mha_block_table=torch.zeros((MAX_REQUESTS, MAX_KV_BLOCKS), dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MHAMetadata uses cuda.current_device")
@pytest.mark.parametrize("cls", [MHAMetadata, GraphedMHAMetadata, NonGraphedMHAMetadata])
def test_set_state_data_slices_views_per_active_request_count(cls):
    """After bind_gpu_buffers + set_state_data, state_data contains views sliced
    to [:n] (or [:n+1] for cumulative-length fields). Verified across the base
    class and both Graphed/NonGraphed subclasses which share the same logic."""
    n = 3
    max_seqlen_q, max_seqlen_k = 5, 10
    m = cls(
        block_count_total=64,
        max_kv_block_count=MAX_KV_BLOCKS,
        max_requests=MAX_REQUESTS,
        block_size_tokens=16,
        max_seqlen=512,
    )
    m.bind_gpu_buffers(_make_gpu_view())
    m.set_state_data(
        padded_active_request_count=n, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k
    )

    assert m.state_data["query_lengths"].shape == (n,)
    assert m.state_data["cu_query_seq_lengths"].shape == (n + 1,)
    assert m.state_data["kv_seq_lengths"].shape == (n,)
    assert m.state_data["cu_kv_seq_lengths"].shape == (n + 1,)
    assert m.state_data["block_table"].shape == (n, MAX_KV_BLOCKS)
    assert m.state_data["max_seqlen_q"] == max_seqlen_q
    assert m.state_data["max_seqlen_k"] == max_seqlen_k

    # reset() clears the seqlen scalars; bound view is preserved (no redundant kernel launches).
    m.reset()
    assert m._max_seqlen_q == 0 and m._max_seqlen_k == 0
    assert m._gpu_view is not None
