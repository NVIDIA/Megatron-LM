# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.gpu_view import ContextGPUView

# Test dimensions. Pulled to module level so views' expected shapes are derivable
# from the constants instead of being repeated as magic numbers in every assert.
MAX_REQUESTS = 4
MAX_TOKENS = 16
MAX_KV_BLOCKS = 8
MAX_MAMBA_CHUNKS = 3

TOKEN_VIEWS_LONG = ("token_to_input_ids", "token_to_pos_ids")
TOKEN_VIEWS_INT32 = (
    "token_to_block_idx",
    "token_to_local_position_within_kv_block",
    "token_to_request_idx",
    "token_to_position_in_request",
)
REQUEST_VIEWS_INT32 = (
    "request_in_prefill_status",
    "request_query_lengths",
    "request_kv_length_offsets",
    "top_k",
    "active_request_last_token_idxs",
)
REQUEST_VIEWS_FLOAT32 = ("temperature", "top_p")
MAMBA_VIEWS = (
    "mamba_batch_indices_decode",
    "mamba_batch_indices_prefill",
    "mamba_seq_idx",
    "mamba_cu_seqlens",
    "mamba_cu_chunk_seqlens",
    "mamba_last_chunk_indices",
    "mamba_seq_idx_for_varlen",
    "mamba_conv_seq_idx",
    "mamba_conv_seq_start",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ContextGPUView allocates CUDA tensors")
class TestContextGPUView:

    @pytest.mark.parametrize("max_mamba_chunks", [0, MAX_MAMBA_CHUNKS])
    def test_layout_with_and_without_mamba(self, max_mamba_chunks):
        """All token / request / MHA views have correct dtype + shape; Mamba views
        are None when max_mamba_chunks == 0 and tensors with expected shapes otherwise.

        Also exercises the internal `assert off == total_bytes` layout guard at the
        end of `__init__`: a layout bug would fire that assertion during construction.
        """
        v = ContextGPUView(
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_kv_blocks=MAX_KV_BLOCKS,
            device=torch.device("cuda"),
            max_mamba_chunks=max_mamba_chunks,
        )
        # Token-level views.
        for name in TOKEN_VIEWS_LONG:
            t = getattr(v, name)
            assert t.dtype == torch.long
            assert t.shape == (MAX_TOKENS,)
        for name in TOKEN_VIEWS_INT32:
            t = getattr(v, name)
            assert t.dtype == torch.int32
            assert t.shape == (MAX_TOKENS,)
        # Request-level views.
        for name in REQUEST_VIEWS_INT32:
            t = getattr(v, name)
            assert t.dtype == torch.int32
            assert t.shape == (MAX_REQUESTS,)
        for name in REQUEST_VIEWS_FLOAT32:
            t = getattr(v, name)
            assert t.dtype == torch.float32
            assert t.shape == (MAX_REQUESTS,)
        # MHA views.
        assert v.mha_query_lengths.shape == (MAX_REQUESTS,)
        assert v.mha_cu_query_seq_lengths.shape == (MAX_REQUESTS + 1,)
        assert v.mha_kv_seq_lengths.shape == (MAX_REQUESTS,)
        assert v.mha_cu_kv_seq_lengths.shape == (MAX_REQUESTS + 1,)
        assert v.mha_block_table.shape == (MAX_REQUESTS, MAX_KV_BLOCKS)
        # Mamba views: all None or all tensors.
        if max_mamba_chunks == 0:
            for name in MAMBA_VIEWS:
                assert getattr(v, name) is None
        else:
            for name in MAMBA_VIEWS:
                assert getattr(v, name) is not None
                assert getattr(v, name).dtype == torch.int32

    def test_views_alias_underlying_buffer(self):
        """Mutating a typed view writes through to `_buf`. This is the central
        load-bearing property of ContextGPUView — the H2D copy targets `_buf`,
        and every kernel reads through the typed views."""
        v = ContextGPUView(
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_kv_blocks=MAX_KV_BLOCKS,
            device=torch.device("cuda"),
            max_mamba_chunks=0,
        )
        sentinel = 0xCAFE
        v.token_to_input_ids[0] = sentinel
        # First 8 bytes are the int64-typed view of token_to_input_ids[0].
        assert v._buf[:8].view(torch.long)[0].item() == sentinel
