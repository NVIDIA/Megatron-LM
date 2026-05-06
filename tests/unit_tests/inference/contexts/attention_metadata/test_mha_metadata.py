# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.attention_context.mha_metadata import (
    GraphedMHAMetadata,
    MHAMetadata,
    NonGraphedMHAMetadata,
)


def _make_gpu_view(max_bs=4, max_blocks=8):
    """Build a fake ContextGPUView with the slice attributes mha_metadata reads."""
    return SimpleNamespace(
        mha_query_lengths=torch.zeros(max_bs, dtype=torch.int32),
        mha_cu_query_seq_lengths=torch.zeros(max_bs + 1, dtype=torch.int32),
        mha_cu_kv_seq_lengths=torch.zeros(max_bs + 1, dtype=torch.int32),
        mha_kv_seq_lengths=torch.zeros(max_bs, dtype=torch.int32),
        mha_block_table=torch.zeros((max_bs, max_blocks), dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MHAMetadata uses cuda.current_device")
class TestMHAMetadata:

    def test_init_records_dimensions(self):
        """__init__ stores all dimensional arguments and starts with empty state_data."""
        m = MHAMetadata(
            block_count_total=64,
            max_kv_block_count=8,
            max_requests=4,
            block_size_tokens=16,
            max_seqlen=512,
        )
        assert m.max_blocks == 64
        assert m.max_kv_blocks == 8
        assert m.max_bs == 4
        assert m.max_seqlen == 512
        assert m.state_data == {}
        assert m._gpu_view is None
        assert m._max_seqlen_q == 0
        assert m._max_seqlen_k == 0

    def test_bind_gpu_buffers_stores_view(self):
        """bind_gpu_buffers stores the supplied view under _gpu_view."""
        m = MHAMetadata(64, 8, 4, 16, 512)
        view = _make_gpu_view()
        m.bind_gpu_buffers(view)
        assert m._gpu_view is view

    def test_set_state_data_requires_bound_gpu_view(self):
        """set_state_data asserts that bind_gpu_buffers was called first."""
        m = MHAMetadata(64, 8, 4, 16, 512)
        with pytest.raises(AssertionError):
            m.set_state_data(padded_active_request_count=2, max_seqlen_q=8, max_seqlen_k=16)

    def test_set_state_data_builds_slice_view(self):
        """set_state_data populates state_data with appropriately-sliced views."""
        m = MHAMetadata(64, 8, 4, 16, 512)
        view = _make_gpu_view(max_bs=4, max_blocks=8)
        m.bind_gpu_buffers(view)
        m.set_state_data(padded_active_request_count=3, max_seqlen_q=5, max_seqlen_k=10)
        assert m._max_seqlen_q == 5
        assert m._max_seqlen_k == 10
        # query_lengths sliced to [:n] = [:3]
        assert m.state_data["query_lengths"].shape == (3,)
        # cu_query_seq_lengths sliced to [:n+1] = [:4]
        assert m.state_data["cu_query_seq_lengths"].shape == (4,)
        # cu_kv_seq_lengths sliced to [:n+1]
        assert m.state_data["cu_kv_seq_lengths"].shape == (4,)
        # kv_seq_lengths sliced to [:n]
        assert m.state_data["kv_seq_lengths"].shape == (3,)
        # block_table sliced to [:n, :]
        assert m.state_data["block_table"].shape == (3, 8)
        assert m.state_data["max_seqlen_q"] == 5
        assert m.state_data["max_seqlen_k"] == 10

    def test_reset_clears_seqlen_only(self):
        """reset zeroes the seqlen scalars but leaves bound buffers intact."""
        m = MHAMetadata(64, 8, 4, 16, 512)
        view = _make_gpu_view()
        m.bind_gpu_buffers(view)
        m.set_state_data(padded_active_request_count=2, max_seqlen_q=7, max_seqlen_k=14)
        m.reset()
        assert m._max_seqlen_q == 0
        assert m._max_seqlen_k == 0
        # _gpu_view persists across reset.
        assert m._gpu_view is view

    def test_graphed_and_non_graphed_subclasses_inherit_behaviour(self):
        """GraphedMHAMetadata and NonGraphedMHAMetadata share the parent's logic."""
        for cls in (GraphedMHAMetadata, NonGraphedMHAMetadata):
            m = cls(64, 8, 4, 16, 512)
            assert isinstance(m, MHAMetadata)
            view = _make_gpu_view()
            m.bind_gpu_buffers(view)
            m.set_state_data(1, 1, 1)
            assert "query_lengths" in m.state_data
