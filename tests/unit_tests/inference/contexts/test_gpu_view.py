# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.gpu_view import ContextGPUView


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ContextGPUView allocates CUDA tensors")
class TestContextGPUView:

    def test_init_no_mamba_attaches_all_views(self):
        """Without mamba chunks, all token/request/MHA views are non-None tensor views."""
        device = torch.device("cuda")
        v = ContextGPUView(
            max_requests=4, max_tokens=16, max_kv_blocks=8, device=device, max_mamba_chunks=0
        )
        # Underlying buffer is uint8 on the requested device.
        assert v._buf.dtype == torch.uint8
        assert v._buf.device.type == "cuda"

        # Token-level views (long for input/pos ids; int32 otherwise).
        assert v.token_to_input_ids.dtype == torch.long
        assert v.token_to_input_ids.shape == (16,)
        assert v.token_to_pos_ids.dtype == torch.long
        assert v.token_to_pos_ids.shape == (16,)
        for name in [
            "token_to_block_idx",
            "token_to_local_position_within_kv_block",
            "token_to_request_idx",
            "token_to_position_in_request",
        ]:
            t = getattr(v, name)
            assert t.dtype == torch.int32
            assert t.shape == (16,)

        # Request-level views (5 int32 + 2 float32).
        for name in [
            "request_in_prefill_status",
            "request_query_lengths",
            "request_kv_length_offsets",
            "top_k",
            "active_request_last_token_idxs",
        ]:
            t = getattr(v, name)
            assert t.dtype == torch.int32
            assert t.shape == (4,)
        for name in ["temperature", "top_p"]:
            t = getattr(v, name)
            assert t.dtype == torch.float32
            assert t.shape == (4,)

        # MHA views.
        assert v.mha_query_lengths.shape == (4,)
        assert v.mha_cu_query_seq_lengths.shape == (5,)  # max_bs + 1
        assert v.mha_kv_seq_lengths.shape == (4,)
        assert v.mha_cu_kv_seq_lengths.shape == (5,)
        assert v.mha_block_table.shape == (4, 8)  # (max_bs, max_kv_blocks)

        # Mamba views all None when max_mamba_chunks == 0.
        for name in [
            "mamba_batch_indices_decode",
            "mamba_batch_indices_prefill",
            "mamba_seq_idx",
            "mamba_cu_seqlens",
            "mamba_cu_chunk_seqlens",
            "mamba_last_chunk_indices",
            "mamba_seq_idx_for_varlen",
            "mamba_conv_seq_idx",
            "mamba_conv_seq_start",
        ]:
            assert getattr(v, name) is None

    def test_init_with_mamba_attaches_mamba_views(self):
        """When max_mamba_chunks > 0, all mamba views are non-None tensor views with correct shapes."""
        device = torch.device("cuda")
        v = ContextGPUView(
            max_requests=4, max_tokens=16, max_kv_blocks=8, device=device, max_mamba_chunks=3
        )
        # All shapes per the layout note in the source.
        assert v.mamba_batch_indices_decode.shape == (4,)
        assert v.mamba_batch_indices_prefill.shape == (4,)
        assert v.mamba_seq_idx.shape == (1, 16)
        assert v.mamba_cu_seqlens.shape == (5,)  # max_bs + 1
        assert v.mamba_cu_chunk_seqlens.shape == (4,)  # max_mamba_chunks + 1
        assert v.mamba_last_chunk_indices.shape == (4,)
        assert v.mamba_seq_idx_for_varlen.shape == (3,)
        assert v.mamba_conv_seq_idx.shape == (16,)
        assert v.mamba_conv_seq_start.shape == (16,)
        for name in [
            "mamba_batch_indices_decode",
            "mamba_batch_indices_prefill",
            "mamba_seq_idx",
            "mamba_cu_seqlens",
            "mamba_cu_chunk_seqlens",
            "mamba_last_chunk_indices",
            "mamba_seq_idx_for_varlen",
            "mamba_conv_seq_idx",
            "mamba_conv_seq_start",
        ]:
            assert getattr(v, name).dtype == torch.int32

    def test_buffer_is_zero_initialized(self):
        """The underlying buffer is zero-initialised, so view reads see zeros."""
        device = torch.device("cuda")
        v = ContextGPUView(
            max_requests=2, max_tokens=4, max_kv_blocks=2, device=device, max_mamba_chunks=0
        )
        assert v._buf.eq(0).all().item()
        # Sample a few derived views.
        assert v.token_to_input_ids.eq(0).all().item()
        assert v.temperature.eq(0).all().item()

    def test_views_share_underlying_buffer(self):
        """Mutating one view is observable in the underlying buffer (views are aliasing slices)."""
        device = torch.device("cuda")
        v = ContextGPUView(
            max_requests=2, max_tokens=4, max_kv_blocks=2, device=device, max_mamba_chunks=0
        )
        v.token_to_input_ids[0] = 7
        # Underlying uint8 buffer should now reflect the write (first 8 bytes = int64 little-endian 7).
        first_long = v._buf[:8].view(torch.long)[0].item()
        assert first_long == 7

    def test_total_byte_layout_matches_assertion(self):
        """The internal `assert off == total_bytes` guard does not fire — both with and without mamba."""
        device = torch.device("cuda")
        # Just ensures construction completes without raising the layout-bug assertion.
        ContextGPUView(
            max_requests=8, max_tokens=32, max_kv_blocks=4, device=device, max_mamba_chunks=0
        )
        ContextGPUView(
            max_requests=8, max_tokens=32, max_kv_blocks=4, device=device, max_mamba_chunks=2
        )
