# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for Triton-based KV cache append operations."""

import pytest
import torch

from megatron.core.inference.contexts.fused_kv_append_kernel import triton_append_key_value_cache
from megatron.core.inference.kv_cache import KVCacheLayout, MLACache, create_mhagqa_cache


class TestFusedKVAppend:
    """Test Triton-based KV cache append operations for all layouts."""

    @pytest.fixture
    def cache_params(self):
        """Common cache parameters for testing."""
        return {
            'num_chunks': 8,
            'chunk_size': 64,
            'num_kv_heads': 8,
            'head_dim': 128,
            'dtype': torch.float16,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }

    @pytest.fixture
    def sample_data(self, cache_params):
        """Generate sample K/V data for testing."""
        batch_size = 5
        device = cache_params['device']
        dtype = cache_params['dtype']

        key = torch.randn(
            batch_size, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
            dtype=dtype, device=device
        )
        value = torch.randn(
            batch_size, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
            dtype=dtype, device=device
        )

        token_to_block_idx = torch.tensor([0, 1, 2, 3, 4], device=device)
        token_to_local_pos = torch.tensor([0, 5, 10, 15, 20], device=device)

        return key, value, token_to_block_idx, token_to_local_pos

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "layout",
        [
            KVCacheLayout.M_2NCHD,
            KVCacheLayout.M_N2CHD,
            KVCacheLayout.M_N2HCD,
            KVCacheLayout.S_NCHD,
            KVCacheLayout.S_NHCD,
        ],
    )
    def test_triton_append_mhagqa(self, layout, cache_params, sample_data):
        """Test Triton append for all MHA/GQA layouts."""
        cache = create_mhagqa_cache(layout=layout, **cache_params)
        key, value, block_idx, local_pos = sample_data

        # Verify cache supports Triton
        assert cache.supports_triton(), f"{layout} should support Triton"

        # Append using Triton
        triton_append_key_value_cache(
            key=key,
            value=value,
            cache=cache,
            padded_active_token_count=len(key),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Verify data was written correctly by checking specific positions
        cache_content = cache.get_content()

        # For separate caches, check both K and V
        if isinstance(cache_content, tuple):
            k_cache, v_cache = cache_content
            for i, (b_idx, l_pos) in enumerate(zip(block_idx.tolist(), local_pos.tolist())):
                # Extract from cache based on layout
                if layout == KVCacheLayout.S_NCHD:
                    # [N, C, H, D]
                    cached_k = k_cache[b_idx, l_pos, :, :]
                    cached_v = v_cache[b_idx, l_pos, :, :]
                elif layout == KVCacheLayout.S_NHCD:
                    # [N, H, C, D]
                    cached_k = k_cache[b_idx, :, l_pos, :]
                    cached_v = v_cache[b_idx, :, l_pos, :]

                # Compare with original input
                assert torch.allclose(cached_k, key[i, 0, :, :], rtol=1e-3, atol=1e-3)
                assert torch.allclose(cached_v, value[i, 0, :, :], rtol=1e-3, atol=1e-3)
        else:
            # For merged caches
            for i, (b_idx, l_pos) in enumerate(zip(block_idx.tolist(), local_pos.tolist())):
                if layout == KVCacheLayout.M_2NCHD:
                    # [2, N, C, H, D]
                    cached_k = cache_content[0, b_idx, l_pos, :, :]
                    cached_v = cache_content[1, b_idx, l_pos, :, :]
                elif layout == KVCacheLayout.M_N2CHD:
                    # [N, 2, C, H, D]
                    cached_k = cache_content[b_idx, 0, l_pos, :, :]
                    cached_v = cache_content[b_idx, 1, l_pos, :, :]
                elif layout == KVCacheLayout.M_N2HCD:
                    # [N, 2, H, C, D]
                    cached_k = cache_content[b_idx, 0, :, l_pos, :]
                    cached_v = cache_content[b_idx, 1, :, l_pos, :]

                assert torch.allclose(cached_k, key[i, 0, :, :], rtol=1e-3, atol=1e-3)
                assert torch.allclose(cached_v, value[i, 0, :, :], rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_append_mla(self, cache_params):
        """Test Triton append for MLA cache."""
        kv_reduced_dim = 256
        mla_cache = MLACache(
            num_chunks=cache_params['num_chunks'],
            chunk_size=cache_params['chunk_size'],
            kv_reduced_dim=kv_reduced_dim,
            dtype=cache_params['dtype'],
            device=cache_params['device'],
        )

        # Verify MLA cache supports Triton
        assert mla_cache.supports_triton(), "MLA cache should support Triton"

        batch_size = 5
        kv_concat = torch.randn(
            batch_size, 1, kv_reduced_dim,
            dtype=cache_params['dtype'],
            device=cache_params['device']
        )
        block_idx = torch.tensor([0, 1, 2, 3, 4], device=cache_params['device'])
        local_pos = torch.tensor([0, 5, 10, 15, 20], device=cache_params['device'])

        # Append using Triton (value=None for MLA)
        triton_append_key_value_cache(
            key=kv_concat,
            value=None,
            cache=mla_cache,
            padded_active_token_count=len(kv_concat),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Verify data was written correctly
        cache_content = mla_cache.get_content()
        for i, (b_idx, l_pos) in enumerate(zip(block_idx.tolist(), local_pos.tolist())):
            cached_kv = cache_content[b_idx, l_pos, :]
            assert torch.allclose(cached_kv, kv_concat[i, 0, :], rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "layout",
        [
            KVCacheLayout.M_2NCHD,
            KVCacheLayout.M_N2CHD,
            KVCacheLayout.M_N2HCD,
            KVCacheLayout.S_NCHD,
            KVCacheLayout.S_NHCD,
        ],
    )
    def test_triton_vs_pytorch(self, layout, cache_params, sample_data):
        """Compare Triton kernel output with PyTorch fallback."""
        # Create two identical caches
        cache_triton = create_mhagqa_cache(layout=layout, **cache_params)
        cache_pytorch = create_mhagqa_cache(layout=layout, **cache_params)

        key, value, block_idx, local_pos = sample_data

        # Triton path
        triton_append_key_value_cache(
            key=key,
            value=value,
            cache=cache_triton,
            padded_active_token_count=len(key),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # PyTorch path (using cache's native append method)
        cache_pytorch.append(
            key=key,
            value=value,
            padded_active_token_count=len(key),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Compare results
        content_triton = cache_triton.get_content()
        content_pytorch = cache_pytorch.get_content()

        if isinstance(content_triton, tuple):
            # Separate caches
            k_triton, v_triton = content_triton
            k_pytorch, v_pytorch = content_pytorch
            assert torch.allclose(k_triton, k_pytorch, rtol=1e-3, atol=1e-3)
            assert torch.allclose(v_triton, v_pytorch, rtol=1e-3, atol=1e-3)
        else:
            # Merged cache
            assert torch.allclose(content_triton, content_pytorch, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_vs_pytorch_mla(self, cache_params):
        """Compare Triton kernel output with PyTorch fallback for MLA."""
        kv_reduced_dim = 256

        # Create two identical MLA caches
        cache_triton = MLACache(
            num_chunks=cache_params['num_chunks'],
            chunk_size=cache_params['chunk_size'],
            kv_reduced_dim=kv_reduced_dim,
            dtype=cache_params['dtype'],
            device=cache_params['device'],
        )
        cache_pytorch = MLACache(
            num_chunks=cache_params['num_chunks'],
            chunk_size=cache_params['chunk_size'],
            kv_reduced_dim=kv_reduced_dim,
            dtype=cache_params['dtype'],
            device=cache_params['device'],
        )

        batch_size = 5
        kv_concat = torch.randn(
            batch_size, 1, kv_reduced_dim,
            dtype=cache_params['dtype'],
            device=cache_params['device']
        )
        block_idx = torch.tensor([0, 1, 2, 3, 4], device=cache_params['device'])
        local_pos = torch.tensor([0, 5, 10, 15, 20], device=cache_params['device'])

        # Triton path
        triton_append_key_value_cache(
            key=kv_concat,
            value=None,
            cache=cache_triton,
            padded_active_token_count=len(kv_concat),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # PyTorch path
        cache_pytorch.append(
            key=kv_concat,
            value=None,
            padded_active_token_count=len(kv_concat),
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Compare results
        content_triton = cache_triton.get_content()
        content_pytorch = cache_pytorch.get_content()
        assert torch.allclose(content_triton, content_pytorch, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_batch(self, cache_params):
        """Test handling of empty batches."""
        cache = create_mhagqa_cache(layout=KVCacheLayout.M_2NCHD, **cache_params)

        # Empty tensors
        key = torch.empty(0, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
                         dtype=cache_params['dtype'], device=cache_params['device'])
        value = torch.empty(0, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
                           dtype=cache_params['dtype'], device=cache_params['device'])
        block_idx = torch.empty(0, dtype=torch.long, device=cache_params['device'])
        local_pos = torch.empty(0, dtype=torch.long, device=cache_params['device'])

        # Should not raise an error
        triton_append_key_value_cache(
            key=key,
            value=value,
            cache=cache,
            padded_active_token_count=0,
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_token(self, cache_params):
        """Test appending a single token."""
        cache = create_mhagqa_cache(layout=KVCacheLayout.S_NCHD, **cache_params)

        key = torch.randn(1, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
                         dtype=cache_params['dtype'], device=cache_params['device'])
        value = torch.randn(1, 1, cache_params['num_kv_heads'], cache_params['head_dim'],
                           dtype=cache_params['dtype'], device=cache_params['device'])
        block_idx = torch.tensor([3], device=cache_params['device'])
        local_pos = torch.tensor([42], device=cache_params['device'])

        triton_append_key_value_cache(
            key=key,
            value=value,
            cache=cache,
            padded_active_token_count=1,
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Verify
        k_cache, v_cache = cache.get_content()
        assert torch.allclose(k_cache[3, 42, :, :], key[0, 0, :, :], rtol=1e-3, atol=1e-3)
        assert torch.allclose(v_cache[3, 42, :, :], value[0, 0, :, :], rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_block(self, cache_params):
        """Test filling an entire block."""
        cache = create_mhagqa_cache(layout=KVCacheLayout.M_N2HCD, **cache_params)

        chunk_size = cache_params['chunk_size']
        key = torch.randn(chunk_size, 1, cache_params['num_kv_heads'],
                         cache_params['head_dim'],
                         dtype=cache_params['dtype'], device=cache_params['device'])
        value = torch.randn(chunk_size, 1, cache_params['num_kv_heads'],
                           cache_params['head_dim'],
                           dtype=cache_params['dtype'], device=cache_params['device'])

        # Fill block 0 completely
        block_idx = torch.zeros(chunk_size, dtype=torch.long, device=cache_params['device'])
        local_pos = torch.arange(chunk_size, device=cache_params['device'])

        triton_append_key_value_cache(
            key=key,
            value=value,
            cache=cache,
            padded_active_token_count=chunk_size,
            token_to_block_idx=block_idx,
            token_to_local_position_within_kv_block=local_pos,
        )

        # Verify all positions in block 0 are filled
        cache_content = cache.get_content()  # [N, 2, H, C, D]
        for i in range(chunk_size):
            cached_k = cache_content[0, 0, :, i, :]  # block 0, K, all heads, pos i
            cached_v = cache_content[0, 1, :, i, :]  # block 0, V, all heads, pos i
            assert torch.allclose(cached_k, key[i, 0, :, :], rtol=1e-3, atol=1e-3)
            assert torch.allclose(cached_v, value[i, 0, :, :], rtol=1e-3, atol=1e-3)
