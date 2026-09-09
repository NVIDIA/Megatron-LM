# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.fused_kv_append_kernel import (
    HAVE_TRITON,
    triton_append_key_value_cache,
)
from megatron.core.inference.contexts.gpu_view import ContextGPUView


def _gpu_memory_gb() -> float:
    """Return free GPU memory in GiB."""
    free, _ = torch.cuda.mem_get_info()
    return free / (1024**3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton required")
class TestKVAppendLargeBlockIdx:
    """Verify that the Triton KV-append kernel writes to the correct cache
    position when block indices exceed 2^16 - 1.

    The kernel (``_append_kv_cache_kernel``) computes:

        dest_offset = block_idx * stride_cache_block + ...

    Triton performs this multiplication in the dtype of ``block_idx``, which is
    loaded from ``token_to_block_idx``. ``ContextGPUView`` owns that tensor.

    With a typical stride of 2^15 (= 256 pos x 1 head x 128 dim) the product
    overflows **signed int32** the moment block_idx >= 2^16:

        2^16 × 2^15 =  2**31 -> wraps to −2^31

    The wrapped negative offset makes the kernel scatter KV data to an invalid
    address (CUDA fault) or a wrong-but-valid one (silent accuracy corruption).
    """

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or _gpu_memory_gb() < 10.0,
        reason="Needs >= 10 GiB free GPU memory to allocate the overflow-sized cache",
    )
    def test_kv_append_block_idx_overflow(self):
        """Trigger the int32 overflow boundary.

        stride_cache_block = 256 × 1 × 128 = 2^15.
        block_idx = 2^16 -> offset = 2**31  (overflows signed int32).

        The block_idx tensor is created with the same dtype that
        ContextGPUView.token_to_block_idx uses. If that dtype is int32 the
        offset wraps and the kernel writes to the wrong address; if int64 the
        offset is computed correctly and the assertion passes.
        """
        device = "cuda"
        total_blocks = 65_537
        block_size = 256
        num_heads = 1
        h_dim = 128
        target_block = 65_536
        n_tokens = 1
        layer = 0

        view = ContextGPUView(max_requests=4, max_tokens=32, max_kv_blocks=4, device=device)
        block_idx_dtype = view.token_to_block_idx.dtype

        memory_buffer = torch.zeros(
            2, 1, total_blocks, block_size, num_heads, h_dim, dtype=torch.bfloat16, device=device
        )
        key = torch.randn(n_tokens, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)
        value = torch.randn(n_tokens, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)

        block_indices = torch.tensor([target_block], dtype=block_idx_dtype, device=device)
        local_positions = torch.zeros(n_tokens, dtype=torch.int32, device=device)

        triton_append_key_value_cache(
            layer_number=layer,
            key=key,
            value=value,
            memory_buffer=memory_buffer,
            padded_active_token_count=n_tokens,
            token_to_block_idx=block_indices,
            token_to_local_position_within_kv_block=local_positions,
            dummy_block_idx=-1,
        )

        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            pytest.fail(
                f"CUDA error during KV append — likely int32 offset overflow "
                f"(token_to_block_idx dtype is {block_idx_dtype}): {e}"
            )

        expected_key = key.squeeze(1)[0]
        expected_value = value.squeeze(1)[0]
        actual_key = memory_buffer[0, layer, target_block, 0]
        actual_value = memory_buffer[1, layer, target_block, 0]

        assert torch.equal(actual_key, expected_key), (
            f"Key not at expected cache position (block {target_block}). "
            f"token_to_block_idx dtype is {block_idx_dtype}; "
            f"stride_cache_block = {block_size * num_heads * h_dim}, "
            f"block_idx * stride = {target_block * block_size * num_heads * h_dim} "
            f"(overflows int32 at 2**31 = {2**31})."
        )
        assert torch.equal(
            actual_value, expected_value
        ), f"Value not at expected cache position (block {target_block})."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton required")
class TestKVAppendDummyBlockSkip:
    """Tokens routed to the dummy block must not reach the cache.

    ``add_request`` points recomputed prefix tokens at ``dummy_block_idx`` so they
    do not overwrite a shared, already-populated block. The kernel drops those
    stores instead of writing the scratch block nobody reads.
    """

    @pytest.mark.internal
    def test_dummy_block_tokens_are_not_written(self):
        device = "cuda"
        total_blocks = 4
        block_size = 8
        num_heads = 2
        h_dim = 16
        layer = 0
        dummy_block = total_blocks - 1
        real_block = 1

        view = ContextGPUView(max_requests=4, max_tokens=32, max_kv_blocks=4, device=device)
        block_idx_dtype = view.token_to_block_idx.dtype

        memory_buffer = torch.full(
            (2, 1, total_blocks, block_size, num_heads, h_dim),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        )
        untouched = memory_buffer.clone()

        # Two tokens: the first is a normal write, the second is redirected.
        key = torch.randn(2, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)
        value = torch.randn(2, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)
        block_indices = torch.tensor(
            [real_block, dummy_block], dtype=block_idx_dtype, device=device
        )
        local_positions = torch.tensor([0, 3], dtype=torch.int32, device=device)

        triton_append_key_value_cache(
            layer_number=layer,
            key=key,
            value=value,
            memory_buffer=memory_buffer,
            padded_active_token_count=2,
            token_to_block_idx=block_indices,
            token_to_local_position_within_kv_block=local_positions,
            dummy_block_idx=dummy_block,
        )
        torch.cuda.synchronize()

        # The real token landed.
        assert torch.equal(memory_buffer[0, layer, real_block, 0], key.squeeze(1)[0])
        assert torch.equal(memory_buffer[1, layer, real_block, 0], value.squeeze(1)[0])

        # The dummy block is byte-for-byte unchanged.
        assert torch.equal(memory_buffer[:, :, dummy_block], untouched[:, :, dummy_block])

    @pytest.mark.internal
    def test_writes_proceed_when_no_token_is_redirected(self):
        """Control: an out-of-range sentinel masks nothing."""
        device = "cuda"
        block_size, num_heads, h_dim, layer = 8, 2, 16, 0

        view = ContextGPUView(max_requests=4, max_tokens=32, max_kv_blocks=4, device=device)
        memory_buffer = torch.zeros(
            (2, 1, 4, block_size, num_heads, h_dim), dtype=torch.bfloat16, device=device
        )
        key = torch.randn(1, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)
        value = torch.randn(1, 1, num_heads, h_dim, dtype=torch.bfloat16, device=device)

        triton_append_key_value_cache(
            layer_number=layer,
            key=key,
            value=value,
            memory_buffer=memory_buffer,
            padded_active_token_count=1,
            token_to_block_idx=torch.tensor(
                [2], dtype=view.token_to_block_idx.dtype, device=device
            ),
            token_to_local_position_within_kv_block=torch.zeros(
                1, dtype=torch.int32, device=device
            ),
            dummy_block_idx=-1,
        )
        torch.cuda.synchronize()

        assert torch.equal(memory_buffer[0, layer, 2, 0], key.squeeze(1)[0])
        assert torch.equal(memory_buffer[1, layer, 2, 0], value.squeeze(1)[0])
