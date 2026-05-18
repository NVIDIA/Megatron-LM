# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

try:
    from megatron.core.inference.contexts.fused_kv_append_kernel import (
        _append_kv_cache_kernel,
        triton_append_key_value_cache,
    )

    HAVE_TRITON = True
except (ImportError, AttributeError):
    HAVE_TRITON = False

from megatron.core.inference.contexts.gpu_view import ContextGPUView


def _gpu_memory_gb() -> float:
    """Return free GPU memory in GiB."""
    free, _ = torch.cuda.mem_get_info()
    return free / (1024**3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBlockIdxDtype:
    """Verify that token_to_block_idx uses int64 in the coalesced buffer layouts.

    Commit 342dd59a narrowed token_to_block_idx from int64 to int32 inside the
    coalesced CPU/GPU bookkeeping buffers.  The Triton KV-append kernel computes
    ``dest_offset = block_idx * stride_cache_block`` in the dtype of the loaded
    block_idx.  When stride_cache_block = block_size_tokens * num_kv_heads *
    head_dim (commonly 32 768), the product overflows int32 for block_idx >=
    65 536 — silently scattering KV data to wrong cache addresses and causing
    accuracy degradation.

    These tests assert the dtype is int64, which prevents the overflow.
    """

    def test_gpu_view_block_idx_is_int64(self):
        view = ContextGPUView(
            max_requests=4, max_tokens=32, max_kv_blocks=4, device="cuda",
        )
        assert view.token_to_block_idx.dtype == torch.int64

    def test_gpu_view_block_idx_is_int64_hybrid(self):
        view = ContextGPUView(
            max_requests=4, max_tokens=32, max_kv_blocks=4, device="cuda",
            max_mamba_chunks=8,
        )
        assert view.token_to_block_idx.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton required")
class TestKVAppendLargeBlockIdx:
    """End-to-end correctness of the Triton KV-append kernel with block indices
    above the int32 overflow boundary.

    The kernel computes ``dest_offset = block_idx * stride_cache_block + …``.
    With a typical stride of 32 768 (= 256 positions × 1 head × 128 dim), the
    product overflows signed int32 at block_idx = 65 536.  When block_idx is
    int64, the multiplication stays in int64 and the offset is correct.

    The "large" variant allocates a cache with stride × block_idx > 2**31 to
    directly demonstrate the overflow; it needs ~8 GiB of free GPU memory and
    is skipped otherwise.  The "small" variant uses a low stride that never
    overflows (regression-safe without the large allocation).
    """

    @staticmethod
    def _reference_append(key_cache, value_cache, key, value, block_idx, local_pos):
        """PyTorch reference: scatter KV into the cache using advanced indexing."""
        n = key.shape[0]
        for i in range(n):
            key_cache[block_idx[i], local_pos[i]] = key[i]
            value_cache[block_idx[i], local_pos[i]] = value[i]

    def _run_append_and_verify(
        self, total_blocks, block_size, num_heads, h_dim, target_block_idx, dtype,
    ):
        """Call the Triton kernel and compare against a PyTorch reference."""
        device = "cuda"
        n_tokens = 1
        layer = 0

        # 6-D memory buffer: (2=KV, num_layers, total_blocks, block_size, heads, hdim)
        memory_buffer = torch.zeros(
            2, 1, total_blocks, block_size, num_heads, h_dim, dtype=dtype, device=device,
        )

        key = torch.randn(n_tokens, 1, num_heads, h_dim, dtype=dtype, device=device)
        value = torch.randn(n_tokens, 1, num_heads, h_dim, dtype=dtype, device=device)
        block_indices = torch.tensor([target_block_idx], dtype=torch.int64, device=device)
        local_positions = torch.zeros(n_tokens, dtype=torch.int32, device=device)

        triton_append_key_value_cache(
            layer_number=layer,
            key=key,
            value=value,
            memory_buffer=memory_buffer,
            padded_active_token_count=n_tokens,
            token_to_block_idx=block_indices,
            token_to_local_position_within_kv_block=local_positions,
        )
        torch.cuda.synchronize()

        expected_key = key.squeeze(1)
        expected_value = value.squeeze(1)
        actual_key = memory_buffer[0, layer, target_block_idx, 0]
        actual_value = memory_buffer[1, layer, target_block_idx, 0]

        assert torch.equal(actual_key, expected_key[0]), (
            f"Key mismatch at block {target_block_idx}"
        )
        assert torch.equal(actual_value, expected_value[0]), (
            f"Value mismatch at block {target_block_idx}"
        )

    @pytest.mark.internal
    def test_kv_append_block_above_65535_small_stride(self):
        """Verify correctness with block_idx > 65535 and a small stride.

        stride_cache_block = 1 * 1 * 128 = 128 — no int32 overflow risk,
        but confirms the kernel handles high block indices correctly.
        """
        self._run_append_and_verify(
            total_blocks=70_000,
            block_size=1,
            num_heads=1,
            h_dim=128,
            target_block_idx=65_537,
            dtype=torch.bfloat16,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or _gpu_memory_gb() < 10.0,
        reason="Needs ≥ 10 GiB free GPU memory to allocate the overflow-sized cache",
    )
    def test_kv_append_block_idx_int32_overflow(self):
        """Directly trigger the int32 overflow boundary.

        stride_cache_block = 256 × 1 × 128 = 32 768.
        block_idx = 65 536  →  offset = 65 536 × 32 768 = 2**31 (overflows int32).

        With int64 block_idx the offset is computed correctly and data lands in
        the right cache slot.  With int32 block_idx the product wraps to
        -2 147 483 648, producing a negative pointer offset and either a CUDA
        memory fault or silent data corruption.
        """
        self._run_append_and_verify(
            total_blocks=65_537,
            block_size=256,
            num_heads=1,
            h_dim=128,
            target_block_idx=65_536,
            dtype=torch.bfloat16,
        )
