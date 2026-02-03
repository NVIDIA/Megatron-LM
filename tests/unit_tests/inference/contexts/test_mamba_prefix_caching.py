# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for Mamba prefix caching in hybrid models.

This module tests the Mamba state caching functionality for prefix sharing
in hybrid Mamba-Attention models.
"""

import pytest
import torch

from megatron.core.inference.contexts.attention_context.mamba_metadata import (
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
)
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

from enum import IntEnum


class TestLevel(IntEnum):
    """Priority levels for Mamba prefix caching tests."""
    CRITICAL = 1
    IMPORTANT = 2
    MEDIUM = 3
    LOW = 4


# Set this to control which tests run
TEST_LEVEL = TestLevel.LOW


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class TestMambaPrefixCaching:
    """Test suite for Mamba state caching with prefix sharing."""

    def _setup_model_parallel_group(self, tensor_parallel_size, pipeline_parallel_size):
        self.pp_size = pipeline_parallel_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)

    def _get_dynamic_context(
        self,
        params_dtype,
        num_layers,
        kv_channels,
        num_attention_heads,
        max_sequence_length,
        buffer_size_gb,
        block_size_tokens,
        max_tokens,
        is_hybrid_model=True,
        layer_type_list=None,
        rounder=64,
        paused_buffer_size_gb=None,
        enable_prefix_caching=True,
        prefix_caching_mamba_gb=None,
    ):
        set_rounder(rounder)

        if is_hybrid_model:
            if layer_type_list is None:
                layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
            mamba_conv_states_shape = (544, 4)
            mamba_ssm_states_shape = (8, 64, 16)
            mamba_inference_state_config = MambaInferenceStateConfig(
                layer_type_list, mamba_conv_states_shape, mamba_ssm_states_shape
            )
        else:
            mamba_inference_state_config = None

        dynamic_context = DynamicInferenceContext(
            params_dtype=params_dtype,
            num_layers=num_layers // self.pp_size,
            kv_channels=kv_channels,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_cuda_graphs=None,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=buffer_size_gb,
            paused_buffer_size_gb=(
                0.2 * buffer_size_gb if paused_buffer_size_gb is None else paused_buffer_size_gb
            ),
            block_size_tokens=block_size_tokens,
            max_tokens=max_tokens,
            mamba_inference_state_config=mamba_inference_state_config,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_mamba_gb=prefix_caching_mamba_gb,
        )
        return dynamic_context

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # =========================================================================
    # CRITICAL: Basic functionality tests
    # =========================================================================

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_mamba_cache_allocation(self):
        """Verify Mamba cache is allocated when prefix_caching_mamba_gb > 0."""
        self._setup_model_parallel_group(1, 1)

        # With Mamba prefix caching enabled
        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,  # 10MB
        )

        assert context.max_mamba_cache_slots > 0, "Mamba cache should be allocated"
        assert hasattr(context, 'mamba_cache_conv_states'), "Conv cache should exist"
        assert hasattr(context, 'mamba_cache_ssm_states'), "SSM cache should exist"
        assert hasattr(context, 'block_to_mamba_slot'), "Block to slot mapping should exist"
        assert context.mamba_cache_free_count == context.max_mamba_cache_slots, "All slots should be free"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_mamba_cache_disabled_when_budget_zero(self):
        """Verify Mamba cache is not allocated when prefix_caching_mamba_gb is 0 or None."""
        self._setup_model_parallel_group(1, 1)

        # Without Mamba prefix caching
        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=None,
        )

        assert context.max_mamba_cache_slots == 0, "Mamba cache should not be allocated"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_store_and_retrieve_mamba_state(self):
        """Verify Mamba state can be stored and retrieved for a block."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,
        )

        block_size = context.block_size_tokens
        prompt_length = int(block_size * 2.5)  # 2.5 blocks

        # Create and add a request
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(prompt_length, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=50),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        context.add_request(request)

        # Get the first block ID
        block_0_id = context.request_to_kv_block_ids[0][0].item()

        # Manually set some Mamba state for the request
        mamba_idx = context.mamba_metadata.request_to_mamba_state_idx[0].item()
        context.mamba_conv_states[:, mamba_idx] = 1.0
        context.mamba_ssm_states[:, mamba_idx] = 2.0

        # Store the state for block 0
        context.store_mamba_state_for_block(block_0_id, 0)

        # Verify block now has Mamba state
        assert context.has_mamba_state_for_block(block_0_id), "Block should have Mamba state"

        # Reset the request's Mamba state
        context.mamba_conv_states[:, mamba_idx] = 0.0
        context.mamba_ssm_states[:, mamba_idx] = 0.0

        # Restore from cache
        restored = context.restore_mamba_state_from_block(0, block_0_id)
        assert restored, "Restore should succeed"

        # Verify state was restored
        assert torch.allclose(context.mamba_conv_states[:, mamba_idx], torch.ones_like(context.mamba_conv_states[:, mamba_idx]))
        assert torch.allclose(context.mamba_ssm_states[:, mamba_idx], torch.full_like(context.mamba_ssm_states[:, mamba_idx], 2.0))

    # =========================================================================
    # IMPORTANT: LRU eviction tests
    # =========================================================================

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_mamba_lru_eviction(self):
        """Verify LRU eviction works when Mamba cache is full."""
        self._setup_model_parallel_group(1, 1)

        # Create context with very small Mamba cache (will fit only a few states)
        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.001,  # Very small - only a few slots
        )

        max_slots = context.max_mamba_cache_slots
        assert max_slots > 0, "Should have at least some slots"

        # Fill the cache by storing states for blocks
        # Note: This is a simplified test - in practice blocks would be allocated via add_request
        stored_blocks = []
        for i in range(max_slots):
            block_id = i  # Use block IDs directly
            context.block_to_mamba_slot[block_id] = i
            context.mamba_slot_to_block[i] = block_id
            context.mamba_cache_free_count = 0  # All slots used
            stored_blocks.append(block_id)

        # Now try to allocate another slot - should trigger eviction
        new_block_id = max_slots + 1

        # Set timestamps to make block 0 oldest
        context.block_allocator.block_timestamps[stored_blocks[0]] = 0  # Oldest
        context.block_allocator.block_timestamps[stored_blocks[1]] = 100
        context.block_allocator.block_ref_counts[stored_blocks[0]] = 0  # Not in active use
        context.block_allocator.block_ref_counts[stored_blocks[1]] = 0

        # This should evict the oldest block (block 0)
        slot = context._allocate_mamba_cache_slot(new_block_id)

        assert slot >= 0, "Should get a valid slot"
        assert context.block_to_mamba_slot[stored_blocks[0]].item() == -1, "Block 0 should be evicted"
        assert context.block_to_mamba_slot[new_block_id].item() == slot, "New block should have the slot"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_kv_eviction_invalidates_mamba_state(self):
        """Verify Mamba state is invalidated when KV block is evicted."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.05,  # Small buffer to force eviction
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,
        )

        block_size = context.block_size_tokens

        # Add a request to allocate blocks
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(block_size * 2, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=50),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        context.add_request(request)

        block_0_id = context.request_to_kv_block_ids[0][0].item()

        # Store Mamba state for block 0
        context.store_mamba_state_for_block(block_0_id, 0)
        assert context.has_mamba_state_for_block(block_0_id), "Block should have Mamba state"

        # Invalidate Mamba state (simulating KV eviction)
        context.invalidate_mamba_state_for_block(block_0_id)

        assert not context.has_mamba_state_for_block(block_0_id), "Mamba state should be invalidated"

    # =========================================================================
    # MEDIUM: Edge case tests
    # =========================================================================

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_prompt_shorter_than_block_size(self):
        """Verify handling when prompt is shorter than block size."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=128,  # Large block size
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,
        )

        # Prompt smaller than block size - no complete blocks
        prompt_length = 64
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(prompt_length, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=50),
            block_size_tokens=context.block_size_tokens,
            enable_prefix_caching=True,
        )

        # Should still work without errors
        context.add_request(request)

        # No complete blocks, so no Mamba state storage expected
        assert context.total_request_count == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_mamba_cache_disabled_for_non_hybrid(self):
        """Verify Mamba cache is not allocated for non-hybrid models."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,  # Not hybrid
            rounder=64,
            prefix_caching_mamba_gb=0.01,  # This should be ignored
        )

        assert context.max_mamba_cache_slots == 0, "Non-hybrid models should not have Mamba cache"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_slot_reuse_after_eviction(self):
        """Verify slots are properly reused after eviction."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,
        )

        initial_free_count = context.mamba_cache_free_count
        block_id = 0

        # Allocate a slot
        slot = context._allocate_mamba_cache_slot(block_id)
        assert context.mamba_cache_free_count == initial_free_count - 1

        # Invalidate (return to free pool)
        context.invalidate_mamba_state_for_block(block_id)
        assert context.mamba_cache_free_count == initial_free_count

        # Allocate again - should get the same slot back
        slot2 = context._allocate_mamba_cache_slot(block_id + 1)
        assert slot2 == slot, "Should reuse the same slot"

    # =========================================================================
    # LOW: Memory and metrics tests
    # =========================================================================

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.LOW, reason="Test level not met")
    def test_memory_budget_respected(self):
        """Verify Mamba cache size respects memory budget."""
        self._setup_model_parallel_group(1, 1)

        budget_gb = 0.01  # 10MB
        budget_bytes = int(budget_gb * 1024**3)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=budget_gb,
        )

        # Verify slot count matches budget
        expected_slots = budget_bytes // context.mamba_states_memory_per_request
        assert context.max_mamba_cache_slots == expected_slots, \
            f"Expected {expected_slots} slots, got {context.max_mamba_cache_slots}"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.LOW, reason="Test level not met")
    def test_mamba_states_memory_per_request_stored(self):
        """Verify mamba_states_memory_per_request is stored as instance attribute."""
        self._setup_model_parallel_group(1, 1)

        context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=True,
            rounder=64,
            prefix_caching_mamba_gb=0.01,
        )

        assert hasattr(context, 'mamba_states_memory_per_request'), \
            "mamba_states_memory_per_request should be stored"
        assert context.mamba_states_memory_per_request > 0, \
            "mamba_states_memory_per_request should be positive for hybrid models"
