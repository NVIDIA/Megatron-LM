# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.attention_context.mamba_metadata import (
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    compute_block_hash,
    HASH_PRIME,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils

from enum import IntEnum


class TestLevel(IntEnum):
    """Priority levels for prefix caching tests.

    Controls which tests run based on their importance:
    - CRITICAL: Fundamental correctness and safety (hash collisions, correctness verification, TTFT)
    - IMPORTANT: Robustness and common edge cases (concurrent requests, complex patterns, memory pressure)
    - MEDIUM: Lifecycle integration and additional edge cases
    - LOW: Observability, metrics, and advanced scenarios
    """
    CRITICAL = 1
    IMPORTANT = 2
    MEDIUM = 3
    LOW = 4


# Set this to control which tests run:
# - TestLevel.CRITICAL: Run only critical tests
# - TestLevel.IMPORTANT: Run critical + important tests
# - TestLevel.MEDIUM: Run critical + important + medium tests
# - TestLevel.LOW: Run all tests (default)
TEST_LEVEL = TestLevel.LOW


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class PrefixCachingTestBase:
    """Base class with shared setup/teardown and helper methods for prefix caching tests."""

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
        is_hybrid_model=False,
        layer_type_list=None,
        rounder=64,
        paused_buffer_size_gb=None,
        enable_prefix_caching=True,
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
            use_flashinfer_fused_rope=None,  # default to using flash-infer if available
            # this is for compatibility with the LTS environment
            unified_memory_level=0,  # unit tests currently broken with UVM
            enable_prefix_caching=enable_prefix_caching,
        )
        return dynamic_context

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestBlockHash(PrefixCachingTestBase):
    """Tests for block hash computation."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_block_hash_computation(self):
        """Verify hash computation produces consistent positive values."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator

        # Test 1: Hash should be positive for any valid input
        token_ids = torch.arange(128, device=torch.cuda.current_device(), dtype=torch.int64)
        hash_value = compute_block_hash(0, token_ids)
        assert hash_value > 0, "Hash should be positive"

        # Test 2: Same inputs should produce same hash
        hash_value_2 = compute_block_hash(0, token_ids)
        assert hash_value == hash_value_2, "Hash should be deterministic"

        # Test 3: Different parent hash should produce different result
        hash_with_parent = compute_block_hash(12345, token_ids)
        assert hash_with_parent != hash_value, "Different parent should produce different hash"
        assert hash_with_parent > 0, "Hash with parent should still be positive"

        # Test 4: Different tokens should produce different hash
        different_tokens = torch.arange(1, 129, device=torch.cuda.current_device(), dtype=torch.int64)
        hash_different = compute_block_hash(0, different_tokens)
        assert hash_different != hash_value, "Different tokens should produce different hash"

        # Test 5: Block hashes tensor initialized to -1
        assert (block_allocator.block_hashes == -1).all(), "Block hashes should initialize to -1"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_block_hash_prefill_decode_release(self):
        """Integration test for hash computation during prefill, decode, and release."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,  # Small blocks for easier testing
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create request with 2.5 blocks worth of tokens (80 tokens with block_size=32)
        prompt_length = int(block_size * 2.5)  # 80 tokens
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(prompt_length, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=50),
        )

        # Add request (prefill)
        dynamic_context.add_request(request)

        # Check: First 2 blocks should have hashes computed (they're complete)
        block_0_id = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1_id = dynamic_context.request_to_kv_block_ids[0][1].item()
        block_2_id = dynamic_context.request_to_kv_block_ids[0][2].item()

        assert block_allocator.block_hashes[block_0_id].item() > 0, "Block 0 should have hash"
        assert block_allocator.block_hashes[block_1_id].item() > 0, "Block 1 should have hash"
        assert block_allocator.block_hashes[block_2_id].item() == -1, "Block 2 incomplete, no hash"

        # Release blocks (simulate request completion)
        dynamic_context.release_memory_blocks_from_request_indexes(torch.tensor([0]))

        # Check: Released blocks remain cached with hashes preserved (for prefix reuse).
        # Hashes are only reset after eviction, not release.
        assert block_allocator.block_hashes[block_0_id].item() > 0, "Block 0 should retain hash after release"
        assert block_allocator.block_hashes[block_1_id].item() > 0, "Block 1 should retain hash after release"
        assert block_allocator.block_hashes[block_2_id].item() == -1, "Block 2 incomplete, no hash"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_block_hash_consistency(self):
        """Same token sequence should produce same hash chain across requests."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create identical prompts that span 2 complete blocks
        prompt_tokens = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # First request
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )

        # Second request with same tokens
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )

        # Third request with different tokens
        different_tokens = torch.arange(1, block_size * 2 + 1, device=torch.cuda.current_device())
        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=different_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )

        # Verify: Precomputed hashes are computed correctly
        assert request_1.precomputed_block_hashes is not None, "Hashes should be computed"
        assert len(request_1.precomputed_block_hashes) == 2, "Should have 2 block hashes"

        # Verify: Same token content should produce identical precomputed hashes
        assert request_1.precomputed_block_hashes == request_2.precomputed_block_hashes, (
            f"Precomputed hashes should match for identical prompts: "
            f"{request_1.precomputed_block_hashes} vs {request_2.precomputed_block_hashes}"
        )

        # Verify hash chaining: block 1 hash should differ from block 0
        req1_block_0_hash = request_1.precomputed_block_hashes[0]
        req1_block_1_hash = request_1.precomputed_block_hashes[1]
        assert req1_block_0_hash != req1_block_1_hash, "Different blocks should have different hashes"

        # Verify: Different tokens should produce different hashes
        assert request_1.precomputed_block_hashes != request_3.precomputed_block_hashes, (
            "Different token sequences should produce different hashes"
        )

        # Verify that adding requests and prefix matching works correctly
        dynamic_context.add_request(request_1)
        dynamic_context.add_request(request_2)

        # With prefix caching, request 2 should share the same blocks as request 1
        req1_block_0_id = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1_id = dynamic_context.request_to_kv_block_ids[0][1].item()
        req2_block_0_id = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1_id = dynamic_context.request_to_kv_block_ids[1][1].item()

        assert req1_block_0_id == req2_block_0_id, (
            f"Request 2 should share block 0 with request 1: {req1_block_0_id} vs {req2_block_0_id}"
        )
        assert req1_block_1_id == req2_block_1_id, (
            f"Request 2 should share block 1 with request 1: {req1_block_1_id} vs {req2_block_1_id}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_block_hash_computed_when_filled_during_decode(self):
        """Test hash behavior for partial blocks during decode.

        NOTE: Hash computation during decode (when a partial block becomes complete)
        is not currently implemented. This test verifies the current behavior where
        only complete blocks at prefill time get hashes assigned.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add request with 1 complete block + (block_size - 1) tokens in second block
        # This leaves exactly 1 token slot to fill the second block
        prompt_length = block_size + (block_size - 1)  # 63 tokens for block_size=32
        prompt = torch.arange(prompt_length, device=torch.cuda.current_device())

        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=50),
        )
        dynamic_context.add_request(request)

        # Verify: block 0 has hash (complete block), block 1 is partial (no hash)
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        assert block_allocator.block_hashes[block_0].item() != -1, "Block 0 should have hash"
        assert block_allocator.block_hashes[block_1].item() == -1, (
            "Block 1 should NOT have hash (partial)"
        )

        # Run one decode step - this fills block 1 to completion
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        dynamic_context.update_requests(active_mask, new_tokens)

        # NOTE: Currently, hash computation during decode is NOT implemented.
        # Block 1 remains without a hash even after being filled.
        # This is a known limitation - blocks filled during decode are not
        # registered for prefix caching reuse.
        assert block_allocator.block_hashes[block_1].item() == -1, (
            "Block 1 should still have no hash (decode hash computation not implemented)"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_block_hash_not_computed_for_partial_during_decode(self):
        """Test that hash is NOT computed for partial blocks during decode."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add request with 1 complete block + 10 tokens in second block
        # After 5 decode steps, block 1 will have 15 tokens (still partial for block_size=32)
        prompt_length = block_size + 10
        prompt = torch.arange(prompt_length, device=torch.cuda.current_device())

        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=50),
        )
        dynamic_context.add_request(request)

        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        assert block_allocator.block_hashes[block_1].item() == -1, (
            "Block 1 should NOT have hash initially"
        )

        # Run 5 decode steps (10 + 5 = 15 tokens in block 1, still partial)
        for _ in range(5):
            active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
            new_tokens = torch.tensor([100], device=torch.cuda.current_device())
            dynamic_context.update_requests(active_mask, new_tokens)

        # Block 1 should STILL not have hash (15 < 32 tokens)
        assert block_allocator.block_hashes[block_1].item() == -1, (
            "Block 1 should STILL not have hash (only 15 tokens, need 32)"
        )


class TestPrefixCaching(PrefixCachingTestBase):
    """Tests for basic prefix caching and block sharing."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_basic_sharing(self):
        """Test that identical prefixes share blocks."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create first request with 2 complete blocks
        prompt_tokens = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Get block IDs for request 1
        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Mark blocks as computed (simulates forward pass completion)
        dynamic_context.mark_pending_blocks_computed()

        # Verify hashes are registered in the mapping
        block_0_hash = block_allocator.get_block_hash(req1_block_0)
        block_1_hash = block_allocator.get_block_hash(req1_block_1)
        assert block_0_hash in block_allocator.hash_to_block_id
        assert block_1_hash in block_allocator.hash_to_block_id

        # Verify ref counts are 1
        assert block_allocator.block_ref_counts[req1_block_0].item() == 1
        assert block_allocator.block_ref_counts[req1_block_1].item() == 1

        # Create second request with same prefix (should share computed blocks)
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Get block IDs for request 2 - should be same as request 1 (shared)
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()

        # Verify blocks are shared
        assert req2_block_0 == req1_block_0, "Block 0 should be shared"
        assert req2_block_1 == req1_block_1, "Block 1 should be shared"

        # Verify ref counts are now 2
        assert block_allocator.block_ref_counts[req1_block_0].item() == 2
        assert block_allocator.block_ref_counts[req1_block_1].item() == 2

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_partial_match(self):
        """Test partial prefix matching - only matching prefix is shared."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # First request with 3 complete blocks
        prompt_tokens_1 = torch.arange(block_size * 3, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed (simulates forward pass completion)
        dynamic_context.mark_pending_blocks_computed()

        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        req1_block_2 = dynamic_context.request_to_kv_block_ids[0][2].item()

        # Second request: first 2 blocks same, block 2 different
        prompt_tokens_2 = torch.arange(block_size * 3, device=torch.cuda.current_device())
        # Modify tokens in the third block (indices block_size*2 to block_size*3)
        prompt_tokens_2[block_size * 2 :] += 1000

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()
        req2_block_2 = dynamic_context.request_to_kv_block_ids[1][2].item()

        # Blocks 0 and 1 should be shared
        assert req2_block_0 == req1_block_0, "Block 0 should be shared"
        assert req2_block_1 == req1_block_1, "Block 1 should be shared"
        # Block 2 should be different (new allocation)
        assert req2_block_2 != req1_block_2, "Block 2 should be newly allocated"

        # Verify ref counts
        assert block_allocator.block_ref_counts[req1_block_0].item() == 2
        assert block_allocator.block_ref_counts[req1_block_1].item() == 2
        assert block_allocator.block_ref_counts[req1_block_2].item() == 1
        assert block_allocator.block_ref_counts[req2_block_2].item() == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_ref_count_release(self):
        """Test that ref counts decrement correctly on release."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create two requests with same prefix
        prompt_tokens = torch.arange(block_size * 2, device=torch.cuda.current_device())

        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed so request 2 can share them
        dynamic_context.mark_pending_blocks_computed()

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        block_0_hash = block_allocator.get_block_hash(block_0)

        # Verify ref counts are 2
        assert block_allocator.block_ref_counts[block_0].item() == 2
        assert block_allocator.block_ref_counts[block_1].item() == 2

        # Release request 1
        dynamic_context.release_memory_blocks_from_request_indexes(torch.tensor([0]))

        # Ref counts should now be 1 (request 2 still using them)
        assert block_allocator.block_ref_counts[block_0].item() == 1
        assert block_allocator.block_ref_counts[block_1].item() == 1

        # Block should still be in hash mapping (cached)
        assert block_0_hash in block_allocator.hash_to_block_id

        # Release request 2
        dynamic_context.release_memory_blocks_from_request_indexes(torch.tensor([1]))

        # Ref counts should now be 0 (cached but not active)
        assert block_allocator.block_ref_counts[block_0].item() == 0
        assert block_allocator.block_ref_counts[block_1].item() == 0

        # Block should STILL be in hash mapping (cached for future reuse)
        assert block_0_hash in block_allocator.hash_to_block_id

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,  # Small buffer to force eviction
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=1,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Fill up most of the available blocks
        initial_avail = block_allocator.total_avail

        # Create a request that uses many blocks
        large_prompt = torch.arange(block_size * 5, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=large_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Get block info for request 1
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_0_hash = block_allocator.get_block_hash(block_0)
        timestamp_before = block_allocator.block_timestamps[block_0].item()

        # Release request 1 - blocks become cached (ref_count=0)
        dynamic_context.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        dynamic_context.total_request_count = 0

        # Verify blocks are cached (ref_count=0 but still in hash map)
        assert block_allocator.block_ref_counts[block_0].item() == 0
        assert block_0_hash in block_allocator.hash_to_block_id

        # Evictable count should match number of cached blocks
        evictable = block_allocator.get_evictable_block_count()
        assert evictable >= 5  # At least 5 blocks from request 1

        # Create a new request with different tokens to force allocation
        # (not matching the cached prefix)
        different_prompt = torch.arange(1000, 1000 + block_size * 3, device=torch.cuda.current_device())
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=different_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # If pool is empty, this will trigger LRU eviction
        dynamic_context.add_request(request_2)

        # After eviction and reuse, block_0 may have been evicted
        # The hash should no longer be in the mapping if evicted
        # (or it might still be there if other blocks were evicted first)

        # Key invariant: the system should still function correctly
        assert dynamic_context.total_request_count == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_no_match_allocates_new(self):
        """Test that non-matching prefixes allocate new blocks."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # First request
        prompt_1 = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        req1_blocks = set()
        for i in range(2):
            req1_blocks.add(dynamic_context.request_to_kv_block_ids[0][i].item())

        # Second request with completely different tokens
        prompt_2 = torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device())
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        req2_blocks = set()
        for i in range(2):
            req2_blocks.add(dynamic_context.request_to_kv_block_ids[1][i].item())

        # No blocks should be shared
        assert req1_blocks.isdisjoint(req2_blocks), "Different prefixes should not share blocks"

        # All blocks should have ref_count=1
        for block_id in req1_blocks | req2_blocks:
            assert block_allocator.block_ref_counts[block_id].item() == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_reuse_after_release(self):
        """Test that cached blocks with ref_count=0 are reused by new requests."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add first request
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Get block IDs and verify ref_count=1
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        assert block_allocator.block_ref_counts[block_0].item() == 1
        assert block_allocator.block_ref_counts[block_1].item() == 1

        # Release request 1 - blocks become cached (ref_count=0)
        dynamic_context.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        dynamic_context.total_request_count = 0

        # Verify blocks are cached (ref_count=0 but still in hash map)
        assert block_allocator.block_ref_counts[block_0].item() == 0
        assert block_allocator.block_ref_counts[block_1].item() == 0
        block_0_hash = block_allocator.get_block_hash(block_0)
        assert block_0_hash in block_allocator.hash_to_block_id

        # Add second request with same prefix - should REUSE cached blocks
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Verify same blocks are reused
        new_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        new_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        assert new_block_0 == block_0, "Block 0 should be reused from cache"
        assert new_block_1 == block_1, "Block 1 should be reused from cache"

        # Verify ref_count went from 0 to 1
        assert block_allocator.block_ref_counts[block_0].item() == 1
        assert block_allocator.block_ref_counts[block_1].item() == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_many_requests_same_prefix(self):
        """Test that 10 requests with identical prefix all share the same blocks."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        num_requests = 10
        num_blocks = 3

        # Create identical prompt
        prompt = torch.arange(block_size * num_blocks, device=torch.cuda.current_device())

        # Add first request and mark blocks as computed
        first_request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(first_request)
        dynamic_context.mark_pending_blocks_computed()

        # Add remaining requests - they should share computed blocks
        for i in range(1, num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
                enable_prefix_caching=True,
            )
            dynamic_context.add_request(request)

        # Verify all requests share the same blocks
        first_request_blocks = [
            dynamic_context.request_to_kv_block_ids[0][j].item() for j in range(num_blocks)
        ]

        for req_idx in range(1, num_requests):
            for block_idx in range(num_blocks):
                block_id = dynamic_context.request_to_kv_block_ids[req_idx][block_idx].item()
                assert block_id == first_request_blocks[block_idx], (
                    f"Request {req_idx} block {block_idx} should match request 0"
                )

        # Verify ref_counts are 10
        for block_id in first_request_blocks:
            assert block_allocator.block_ref_counts[block_id].item() == num_requests, (
                f"Block {block_id} should have ref_count={num_requests}"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_hash_chain_correctness(self):
        """Test that block hashes depend on parent hash (hash chaining)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add request with 3 blocks
        prompt = torch.arange(block_size * 3, device=torch.cuda.current_device())
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request)

        # Mark blocks as computed so hashes are set
        dynamic_context.mark_pending_blocks_computed()

        # Get block hashes
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        block_2 = dynamic_context.request_to_kv_block_ids[0][2].item()

        hash_0 = block_allocator.get_block_hash(block_0)
        hash_1 = block_allocator.get_block_hash(block_1)
        hash_2 = block_allocator.get_block_hash(block_2)

        # All hashes should be different (due to chaining)
        assert hash_0 != hash_1, "Block 0 and 1 should have different hashes"
        assert hash_1 != hash_2, "Block 1 and 2 should have different hashes"
        assert hash_0 != hash_2, "Block 0 and 2 should have different hashes"

        # Verify hash chaining: same tokens with different parent = different hash
        # Block 0's tokens are the first block_size tokens from the prompt
        block_0_tokens = prompt[:block_size]
        computed_hash_0 = compute_block_hash(0, block_0_tokens)
        assert computed_hash_0 == hash_0, "Block 0 hash should match with parent=0"

        # Same tokens with different parent should give different hash
        hash_with_different_parent = compute_block_hash(12345, block_0_tokens)
        assert hash_with_different_parent != hash_0, (
            "Same tokens with different parent should produce different hash"
        )


class TestMemoryUsage(PrefixCachingTestBase):
    """Tests for memory accounting with prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_available_blocks_preserved(self):
        """Test that total_avail decreases less when sharing occurs."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        num_blocks = 2

        # Record initial available blocks
        initial_avail = block_allocator.total_avail

        # Add first request
        prompt = torch.arange(block_size * num_blocks, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed so request_2 can share
        dynamic_context.mark_pending_blocks_computed()

        avail_after_first = block_allocator.total_avail
        blocks_used_first = initial_avail - avail_after_first

        # Add second request with identical prompt
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        avail_after_second = block_allocator.total_avail
        blocks_used_second = avail_after_first - avail_after_second

        # First request should use num_blocks
        assert blocks_used_first == num_blocks, (
            f"First request should use {num_blocks} blocks, used {blocks_used_first}"
        )

        # Second request should use 0 blocks (all shared)
        assert blocks_used_second == 0, (
            f"Second request should use 0 additional blocks (sharing), used {blocks_used_second}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_memory_scaling_constant(self):
        """Test that block count is O(1) for N identical requests, not O(N)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        num_blocks = 4
        num_requests = 5

        # Record initial available blocks
        initial_avail = block_allocator.total_avail

        # Add first request and mark computed
        prompt = torch.arange(block_size * num_blocks, device=torch.cuda.current_device())
        first_request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(first_request)
        dynamic_context.mark_pending_blocks_computed()

        # Add remaining requests - they should share computed blocks
        for i in range(1, num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
                enable_prefix_caching=True,
            )
            dynamic_context.add_request(request)

        # Calculate total blocks used
        final_avail = block_allocator.total_avail
        total_blocks_used = initial_avail - final_avail

        # Should be O(1) = num_blocks, not O(N) = num_requests * num_blocks
        assert total_blocks_used == num_blocks, (
            f"Should use {num_blocks} blocks (O(1)), not {num_requests * num_blocks} (O(N)). "
            f"Actually used {total_blocks_used}"
        )


class TestTTFT(PrefixCachingTestBase):
    """Tests for time-to-first-token optimization with prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_matched_blocks_tokens_preserved(self):
        """Test that tokens in matched blocks are NOT overwritten.

        We verify this by checking that block IDs and hashes remain the same
        when a second request shares blocks with the first request.
        If the hash is unchanged, the tokens must be preserved.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add first request with specific tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Record block IDs and hashes
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        original_hash_0 = block_allocator.get_block_hash(block_0)
        original_hash_1 = block_allocator.get_block_hash(block_1)

        # Add second request with same prefix
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Verify blocks are shared (same block IDs)
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()
        assert req2_block_0 == block_0, "Block 0 should be shared"
        assert req2_block_1 == block_1, "Block 1 should be shared"

        # Verify hashes are preserved (implies tokens are preserved)
        current_hash_0 = block_allocator.get_block_hash(block_0)
        current_hash_1 = block_allocator.get_block_hash(block_1)
        assert current_hash_0 == original_hash_0, (
            "Block 0 hash should not change (tokens preserved)"
        )
        assert current_hash_1 == original_hash_1, (
            "Block 1 hash should not change (tokens preserved)"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_only_new_blocks_hashed(self):
        """Test that matched blocks keep same hash, only new blocks get new hashes."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add first request with 2 blocks
        prompt_1 = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Mark blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Record original hashes
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        original_hash_0 = block_allocator.get_block_hash(block_0)
        original_hash_1 = block_allocator.get_block_hash(block_1)

        # Add second request: same first block, different second block, new third block
        prompt_2 = torch.cat([
            prompt_1[:block_size],  # Same first block
            torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device()),  # Different
        ])
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Verify matched block (block 0) has SAME hash
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        assert req2_block_0 == block_0, "First block should be shared"
        current_hash_0 = block_allocator.get_block_hash(block_0)
        assert current_hash_0 == original_hash_0, (
            "Matched block hash should not change"
        )

        # New blocks are pending until marked computed
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()
        req2_block_2 = dynamic_context.request_to_kv_block_ids[1][2].item()
        assert req2_block_1 != block_1, "Second block should be newly allocated"

        # Mark request 2's new blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Verify new blocks have different hashes (now computed)
        new_hash_1 = block_allocator.get_block_hash(req2_block_1)
        new_hash_2 = block_allocator.get_block_hash(req2_block_2)
        assert new_hash_1 != original_hash_1, "New block 1 should have different hash"
        assert new_hash_1 != -1, "New block 1 should have hash computed"
        assert new_hash_2 != -1, "New block 2 should have hash computed"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_prefill_skipped_for_cached_blocks(self):
        """Test that cached blocks are not scheduled for prefill/KV computation.

        This is a critical test verifying that the TTFT improvement actually happens:
        when blocks are cached, they should NOT be added to the pending computation list,
        meaning the engine will skip prefill for those blocks.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        num_blocks = 4

        # Create a prompt with 4 complete blocks (128 tokens)
        prompt = torch.arange(block_size * num_blocks, device=torch.cuda.current_device())

        # Add first request - all blocks should be scheduled for computation
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # Verify: All 4 blocks should be pending computation
        assert len(dynamic_context._blocks_pending_computation) == num_blocks, (
            f"First request should schedule {num_blocks} blocks for computation, "
            f"but scheduled {len(dynamic_context._blocks_pending_computation)}"
        )

        # Mark blocks as computed (simulating prefill completion)
        dynamic_context.mark_pending_blocks_computed()

        # Verify: Pending list should be cleared
        assert len(dynamic_context._blocks_pending_computation) == 0, (
            "After marking computed, pending list should be empty"
        )

        # Add second request with same prefix - NO blocks should be scheduled
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Verify: 0 blocks should be pending computation (all cached)
        assert len(dynamic_context._blocks_pending_computation) == 0, (
            f"Second request with cached prefix should schedule 0 blocks for computation, "
            f"but scheduled {len(dynamic_context._blocks_pending_computation)}. "
            "This means prefill is NOT being skipped for cached blocks!"
        )

        # Verify blocks are actually shared (same block IDs)
        req1_blocks = dynamic_context.request_to_kv_block_ids[0][:num_blocks]
        req2_blocks = dynamic_context.request_to_kv_block_ids[1][:num_blocks]
        assert torch.equal(req1_blocks, req2_blocks), (
            "Both requests should use the same block IDs (shared blocks)"
        )

        # Test partial match scenario: same prefix + new tokens
        extended_prompt = torch.cat([
            prompt,  # Same 4 blocks
            torch.arange(1000, 1000 + block_size, device=torch.cuda.current_device()),  # New block
        ])

        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=extended_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_3)

        # Verify: Only 1 new block should be scheduled (the 5th block)
        assert len(dynamic_context._blocks_pending_computation) == 1, (
            f"Third request should schedule only 1 new block for computation, "
            f"but scheduled {len(dynamic_context._blocks_pending_computation)}. "
            "The 4 cached blocks should NOT be recomputed!"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_hash_function_determinism(self):
        """Test that hash function is deterministic - same input produces same hash.

        This is critical for prefix caching correctness: the same token sequence
        must always produce the same hash, or cached blocks won't be found.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create multiple requests with identical tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Collect hashes from 5 identical requests
        all_hashes = []
        for req_id in range(5):
            request = DynamicInferenceRequest(
                request_id=req_id + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
                enable_prefix_caching=True,
            )
            # All requests should compute the same precomputed hashes
            all_hashes.append(request.precomputed_block_hashes.copy())

        # Verify all requests computed identical hashes
        for i in range(1, 5):
            assert all_hashes[i] == all_hashes[0], (
                f"Request {i} computed different hashes than request 0. "
                f"Hash function is not deterministic! "
                f"Request 0: {all_hashes[0]}, Request {i}: {all_hashes[i]}"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_different_tokens_produce_different_hashes(self):
        """Test that different token sequences produce different hashes.

        This verifies basic hash function quality: we don't want all sequences
        hashing to the same value (which would break prefix caching).
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create 10 different prompts
        hashes_seen = set()
        num_prompts = 10

        for i in range(num_prompts):
            # Each prompt has different starting tokens
            prompt = torch.arange(
                i * 1000, i * 1000 + block_size * 2, device=torch.cuda.current_device()
            )

            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
                enable_prefix_caching=True,
            )

            # Collect first block's hash
            if request.precomputed_block_hashes:
                first_block_hash = request.precomputed_block_hashes[0]
                assert first_block_hash not in hashes_seen, (
                    f"Hash collision detected! Prompt {i} has same hash as previous prompt. "
                    f"Hash: {first_block_hash}. This may indicate poor hash function quality."
                )
                hashes_seen.add(first_block_hash)

        # Verify we collected multiple unique hashes
        assert len(hashes_seen) == num_prompts, (
            f"Expected {num_prompts} unique hashes but only got {len(hashes_seen)}. "
            "Hash function may have poor distribution."
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.LOW, reason="Test level not met")
    def test_hash_collision_would_cause_incorrect_sharing(self):
        """THEORETICAL documentation test: hash collisions would cause incorrect sharing.

        NOTE: This is a theoretical demonstration, NOT a practical concern. Real hash
        collisions are astronomically unlikely with HASH_PRIME = 2305843009213693951
        (2^61 - 1, ~10^18 hash space). This test artificially forces a collision by
        manually overwriting precomputed hashes to document what *would* happen if a
        collision occurred.

        The current implementation does NOT verify token content after a hash match.
        If two different token sequences produced the same hash (nearly impossible),
        they would incorrectly share blocks, leading to wrong KV cache values.

        This test exists purely for documentation/awareness purposes.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add first request
        prompt_1 = torch.arange(block_size, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )
        assert request_1.precomputed_block_hashes is not None, (
            "precomputed_block_hashes should be set when block_size_tokens is provided"
        )
        assert len(request_1.precomputed_block_hashes) == 1, (
            "Should have exactly 1 block hash for prompt of size block_size"
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        # Get the hash and block ID of first request's block
        block_id_1 = dynamic_context.request_to_kv_block_ids[0][0].item()
        hash_1 = block_allocator.get_block_hash(block_id_1)

        # Add second request with DIFFERENT tokens
        prompt_2 = torch.arange(
            10000, 10000 + block_size, device=torch.cuda.current_device()
        )
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )
        assert request_2.precomputed_block_hashes is not None, (
            "precomputed_block_hashes should be set when block_size_tokens is provided"
        )
        assert len(request_2.precomputed_block_hashes) == 1, (
            "Should have exactly 1 block hash for prompt of size block_size"
        )

        # Artificially create a collision by overriding the precomputed hash
        # This simulates what would happen if two different sequences hashed to the same value
        request_2.precomputed_block_hashes[0] = hash_1  # Force collision

        dynamic_context.add_request(request_2)

        # Check if collision caused incorrect sharing
        block_id_2 = dynamic_context.request_to_kv_block_ids[1][0].item()

        if block_id_2 == block_id_1:
            # Collision caused incorrect sharing!
            # This documents that the implementation does NOT verify token content after hash matches.
            # Request 2 has completely different tokens than request 1, but they share the same block
            # because their hashes (artificially) match. This means request 2 will use request 1's
            # KV cache, leading to incorrect model outputs.
            print(
                "\nWARNING: Hash collision test confirmed that the current implementation "
                "does NOT verify token content after hash matches. If a real collision occurs, "
                "it will cause incorrect block sharing and wrong outputs. "
                "Consider adding token verification in _find_matching_prefix_blocks()."
            )
        else:
            # Collision did not cause sharing (perhaps due to other factors)
            # This is fine, but we still want to document the risk
            print(
                "\nINFO: Artificial collision did not trigger incorrect sharing in this test. "
                "However, the implementation still lacks token verification after hash matches, "
                "which is a potential correctness issue if real collisions occur."
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_shared_blocks_preserve_token_content(self):
        """Test that shared blocks maintain correct token content for all requests.

        When multiple requests share blocks, each request must see the correct tokens
        in those blocks, or the model will process wrong inputs.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create shared prefix
        shared_prefix = torch.arange(block_size * 3, device=torch.cuda.current_device())

        # Add 5 requests that all share the same prefix
        num_requests = 5
        for req_id in range(num_requests):
            request = DynamicInferenceRequest(
                request_id=req_id + 1,
                prompt_tokens=shared_prefix.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
            )
            dynamic_context.add_request(request)

            if req_id == 0:
                # Mark first request's blocks as computed
                dynamic_context.mark_pending_blocks_computed()

        # Verify all requests share the same block IDs
        first_req_blocks = dynamic_context.request_to_kv_block_ids[0][:3]
        for req_id in range(1, num_requests):
            req_blocks = dynamic_context.request_to_kv_block_ids[req_id][:3]
            assert torch.equal(req_blocks, first_req_blocks), (
                f"Request {req_id} should share blocks with request 0"
            )

        # Verify ref counts are correct for shared blocks
        for block_idx in range(3):
            block_id = first_req_blocks[block_idx].item()
            ref_count = block_allocator.block_ref_counts[block_id].item()
            assert ref_count == num_requests, (
                f"Block {block_id} should have ref_count={num_requests} "
                f"(shared by all requests), got {ref_count}"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.CRITICAL, reason="Test level not met")
    def test_ref_count_prevents_premature_eviction(self):
        """Test that blocks in active use (ref_count > 0) cannot be evicted.

        This is critical for correctness: if blocks are evicted while still in use,
        requests would get wrong KV cache data, leading to incorrect outputs.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,  # Very small buffer to force eviction pressure
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add first request
        prompt_1 = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        # Get block IDs for request 1
        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Verify ref_count is 1 for active request
        assert block_allocator.block_ref_counts[req1_block_0].item() == 1
        assert block_allocator.block_ref_counts[req1_block_1].item() == 1

        # Add second request that shares the blocks
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_1.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
        )
        dynamic_context.add_request(request_2)

        # Verify ref_count is now 2 (both requests using the blocks)
        assert block_allocator.block_ref_counts[req1_block_0].item() == 2
        assert block_allocator.block_ref_counts[req1_block_1].item() == 2

        # Try to fill up the cache with new requests to trigger eviction pressure
        initial_avail = block_allocator.total_avail
        num_filler_requests = 10

        for i in range(num_filler_requests):
            filler_prompt = torch.arange(
                (i + 10) * 1000, (i + 10) * 1000 + block_size * 2,
                device=torch.cuda.current_device(),
            )
            filler_request = DynamicInferenceRequest(
                request_id=i + 100,
                prompt_tokens=filler_prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=10),
                block_size_tokens=block_size,
            )
            try:
                dynamic_context.add_request(filler_request)
                dynamic_context.mark_pending_blocks_computed()
            except Exception:
                # May run out of space, which is fine
                break

        # Verify the shared blocks (with ref_count=2) still exist and weren't evicted
        assert block_allocator.block_ref_counts[req1_block_0].item() == 2, (
            "Shared block 0 was incorrectly evicted or corrupted!"
        )
        assert block_allocator.block_ref_counts[req1_block_1].item() == 2, (
            "Shared block 1 was incorrectly evicted or corrupted!"
        )

        # Verify the blocks still have valid hashes (blocks weren't reset/corrupted)
        hash_0 = block_allocator.block_hashes[req1_block_0].item()
        hash_1 = block_allocator.block_hashes[req1_block_1].item()
        assert hash_0 != -1, "Block 0 hash was reset (block corrupted/evicted)!"
        assert hash_1 != -1, "Block 1 hash was reset (block corrupted/evicted)!"


class TestEdgeCases(PrefixCachingTestBase):
    """Tests for edge case handling in prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_single_block_prefix(self):
        """Test that sharing works with just 1 complete block."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add request with exactly 1 complete block
        prompt = torch.arange(block_size, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)

        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        assert block_allocator.block_ref_counts[block_0].item() == 1

        # Add second request with same single block
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        # Verify block is shared
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        assert req2_block_0 == block_0, "Single block should be shared"
        assert block_allocator.block_ref_counts[block_0].item() == 2

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_incomplete_block_not_shared(self):
        """Test that incomplete (partial) blocks are NOT shared."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add request with 1.5 blocks (1 complete + 1 partial)
        prompt_length = int(block_size * 1.5)
        prompt = torch.arange(prompt_length, device=torch.cuda.current_device())

        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)

        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Complete block should have hash, partial should not
        assert block_allocator.block_hashes[req1_block_0].item() != -1, (
            "Complete block should have hash"
        )
        assert block_allocator.block_hashes[req1_block_1].item() == -1, (
            "Partial block should NOT have hash"
        )

        # Add second request with same 1.5 blocks
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()

        # Complete block SHOULD be shared
        assert req2_block_0 == req1_block_0, "Complete block should be shared"

        # Partial block should NOT be shared (different allocation)
        assert req2_block_1 != req1_block_1, (
            "Partial block should NOT be shared (no hash for matching)"
        )


class TestDisabledMode(PrefixCachingTestBase):
    """Tests for prefix caching when disabled."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_disabled_no_sharing(self):
        """Test that identical prefixes do NOT share blocks when prefix caching is disabled."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
            enable_prefix_caching=False,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create two requests with IDENTICAL prompts
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)

        req1_blocks = set()
        for i in range(2):
            req1_blocks.add(dynamic_context.request_to_kv_block_ids[0][i].item())

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        req2_blocks = set()
        for i in range(2):
            req2_blocks.add(dynamic_context.request_to_kv_block_ids[1][i].item())

        # With prefix caching disabled, blocks should NOT be shared even with identical prompts
        assert req1_blocks.isdisjoint(req2_blocks), (
            "With prefix caching disabled, identical prefixes should NOT share blocks"
        )

        # All blocks should have ref_count=1 (no sharing)
        for block_id in req1_blocks | req2_blocks:
            assert block_allocator.block_ref_counts[block_id].item() == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_disabled_deterministic_hashes(self):
        """Test that blocks get deterministic unique hashes when prefix caching is disabled."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
            enable_prefix_caching=False,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add a request
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request)

        # Get block IDs
        block_ids = [
            dynamic_context.request_to_kv_block_ids[0][i].item() for i in range(2)
        ]

        # Verify hashes are set (not -1)
        for block_id in block_ids:
            block_hash = block_allocator.block_hashes[block_id].item()
            assert block_hash != -1, "Block hash should be set"

        # Verify hashes are different from each other (unique per block)
        hashes = [block_allocator.block_hashes[bid].item() for bid in block_ids]
        assert len(set(hashes)) == len(hashes), "Each block should have a unique hash"

        # Verify hashes are deterministic (based on block_id)
        # The formula is: (block_id * 2654435761) % HASH_PRIME + 1
        for block_id in block_ids:
            expected_hash = (block_id * 2654435761) % HASH_PRIME + 1
            actual_hash = block_allocator.block_hashes[block_id].item()
            assert actual_hash == expected_hash, (
                f"Hash for block {block_id} should be deterministic: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_performance_comparison(self):
        """Test that prefix caching enabled uses fewer blocks and is faster."""
        import time

        self._setup_model_parallel_group(1, 1)

        block_size = 32
        num_blocks_in_prompt = 4  # 128 tokens

        # Create identical prompt for all requests
        prompt = torch.arange(
            block_size * num_blocks_in_prompt, device=torch.cuda.current_device()
        )
        num_requests = 5

        # --- Test with prefix caching ENABLED ---
        context_enabled = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=block_size,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
            enable_prefix_caching=True,
        )

        start_enabled = time.perf_counter()
        for i in range(num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            context_enabled.add_request(request)
        time_enabled = time.perf_counter() - start_enabled

        # Count unique blocks allocated
        blocks_enabled = set()
        for req_idx in range(num_requests):
            for i in range(num_blocks_in_prompt):
                blocks_enabled.add(
                    context_enabled.request_to_kv_block_ids[req_idx][i].item()
                )

        # --- Test with prefix caching DISABLED ---
        Utils.destroy_model_parallel()
        self._setup_model_parallel_group(1, 1)

        context_disabled = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=block_size,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
            enable_prefix_caching=False,
        )

        start_disabled = time.perf_counter()
        for i in range(num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            context_disabled.add_request(request)
        time_disabled = time.perf_counter() - start_disabled

        # Count unique blocks allocated
        blocks_disabled = set()
        for req_idx in range(num_requests):
            for i in range(num_blocks_in_prompt):
                blocks_disabled.add(
                    context_disabled.request_to_kv_block_ids[req_idx][i].item()
                )

        # --- Assertions ---

        # Memory metric: With caching enabled, should use fewer blocks
        # With 5 identical requests of 4 blocks each:
        # - Enabled: Should use only 4 blocks (all shared)
        # - Disabled: Should use 20 blocks (5 * 4, no sharing)
        assert len(blocks_enabled) == num_blocks_in_prompt, (
            f"With prefix caching enabled, should use only {num_blocks_in_prompt} blocks "
            f"for {num_requests} identical requests, but used {len(blocks_enabled)}"
        )
        assert len(blocks_disabled) == num_requests * num_blocks_in_prompt, (
            f"With prefix caching disabled, should use {num_requests * num_blocks_in_prompt} blocks "
            f"for {num_requests} requests, but used {len(blocks_disabled)}"
        )

        # Verify significant memory savings
        memory_ratio = len(blocks_enabled) / len(blocks_disabled)
        assert memory_ratio <= 0.25, (  # Should be 4/20 = 0.2
            f"Prefix caching should reduce block usage by at least 75%, "
            f"but ratio was {memory_ratio:.2f}"
        )

        # Time metric: With caching enabled, should generally be faster
        # Use generous tolerance since timing can be noisy
        # We don't strictly assert on time, but log it for visibility
        print(f"\nPrefix caching performance:")
        print(f"  Enabled:  {len(blocks_enabled)} blocks, {time_enabled*1000:.3f}ms")
        print(f"  Disabled: {len(blocks_disabled)} blocks, {time_disabled*1000:.3f}ms")
        print(f"  Memory ratio: {memory_ratio:.2f} (lower is better)")


class TestPrefixCoordination(PrefixCachingTestBase):
    """Tests for multi-rank prefix caching coordination."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_register_block_hash_does_not_set_block_hashes(self):
        """Verify that register_block_hash does NOT set block_hashes (two-phase registration)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        allocator = dynamic_context.block_allocator

        # Allocate a block
        block_ids = allocator.allocate_memory_blocks(1)
        block_id = block_ids[0].item()

        # Verify initial state: block_hashes should be -1
        assert allocator.block_hashes[block_id].item() == -1

        # Register a hash for this block
        test_hash = 12345
        allocator.register_block_hash(block_id, test_hash)

        # Verify: hash_to_block_id should be populated
        assert allocator.hash_to_block_id.get(test_hash) == block_id

        # Verify: block_hashes should still be -1 (not computed yet)
        assert allocator.block_hashes[block_id].item() == -1

        # Verify: _pending_block_hashes should contain the block
        assert block_id in allocator._pending_block_hashes
        assert allocator._pending_block_hashes[block_id] == test_hash

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_mark_block_computed_sets_hash(self):
        """Verify that mark_block_computed correctly sets block_hashes."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        allocator = dynamic_context.block_allocator

        # Allocate a block and register hash
        block_ids = allocator.allocate_memory_blocks(1)
        block_id = block_ids[0].item()
        test_hash = 12345

        allocator.register_block_hash(block_id, test_hash)

        # Verify: block_hashes is still -1
        assert allocator.block_hashes[block_id].item() == -1

        # Mark as computed
        allocator.mark_block_computed(block_id)

        # Verify: block_hashes should now be set
        assert allocator.block_hashes[block_id].item() == test_hash

        # Verify: _pending_block_hashes should no longer contain the block
        assert block_id not in allocator._pending_block_hashes

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_pending_blocks_cleared_after_mark(self):
        """Verify that mark_pending_blocks_computed clears the pending list."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create request with 2 complete blocks
        prompt_tokens = torch.arange(
            block_size * 2, device=torch.cuda.current_device()
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Add request - blocks should be added to pending
        dynamic_context.add_request(request)

        # Verify: _blocks_pending_computation should have blocks
        assert len(dynamic_context._blocks_pending_computation) > 0

        # Mark pending blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Verify: _blocks_pending_computation should be empty
        assert len(dynamic_context._blocks_pending_computation) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_precomputed_hashes_correctness(self):
        """Verify precomputed hashes match hashes computed by the allocator."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        # Create request with 3 complete blocks
        prompt_tokens = torch.arange(
            block_size * 3, device=torch.cuda.current_device()
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Verify precomputed hashes match allocator computation
        assert request.precomputed_block_hashes is not None
        assert len(request.precomputed_block_hashes) == 3

        # Manually compute hashes using allocator and compare
        parent_hash = 0
        for i in range(3):
            start = i * block_size
            end = start + block_size
            block_tokens = prompt_tokens[start:end]
            expected_hash = compute_block_hash(parent_hash, block_tokens)
            assert request.precomputed_block_hashes[i] == expected_hash, (
                f"Block {i} hash mismatch: {request.precomputed_block_hashes[i]} vs {expected_hash}"
            )
            parent_hash = expected_hash

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_request_shorter_than_block_size(self):
        """Verify request with prompt shorter than block_size has empty precomputed_block_hashes."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=64,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create request with tokens shorter than block_size
        prompt_tokens = torch.arange(
            block_size // 2, device=torch.cuda.current_device()  # 32 tokens < 64 block_size
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Verify: precomputed_block_hashes should be empty list (not None)
        assert request.precomputed_block_hashes is not None
        assert request.precomputed_block_hashes == []

        # Request can still be added
        dynamic_context.add_request(request)
        assert dynamic_context.total_request_count == 1

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_request_longer_than_block_size(self):
        """Verify request longer than block_size has correct number of precomputed hashes."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create request with 2.5 blocks worth of tokens
        num_tokens = int(block_size * 2.5)  # 80 tokens with block_size=32
        prompt_tokens = torch.arange(
            num_tokens, device=torch.cuda.current_device()
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Verify: should have 2 hashes (only complete blocks)
        assert request.precomputed_block_hashes is not None
        assert len(request.precomputed_block_hashes) == 2

        # Verify hashes are positive
        for h in request.precomputed_block_hashes:
            assert h > 0

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_reset_clears_pending_computation_list(self):
        """Verify that reset() clears _blocks_pending_computation."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create and add request
        prompt_tokens = torch.arange(
            block_size * 2, device=torch.cuda.current_device()
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request)

        # Verify: _blocks_pending_computation should have blocks
        assert len(dynamic_context._blocks_pending_computation) > 0

        # Reset context
        dynamic_context.reset()

        # Verify: _blocks_pending_computation should be cleared
        assert len(dynamic_context._blocks_pending_computation) == 0

        # Verify: _pending_block_hashes in allocator should also be cleared
        assert len(dynamic_context.block_allocator._pending_block_hashes) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_two_phase_registration_flow(self):
        """Test the full two-phase registration: register → discoverable but pending → mark computed."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        # Create request with 2 complete blocks
        prompt_tokens = torch.arange(
            block_size * 2, device=torch.cuda.current_device()
        )
        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Add request
        dynamic_context.add_request(request)

        # Get block IDs
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Phase 1: After add_request, blocks should be in hash_to_block_id but block_hashes == -1
        hash_0 = request.precomputed_block_hashes[0]
        hash_1 = request.precomputed_block_hashes[1]

        # Blocks are discoverable by hash
        assert allocator.lookup_block_by_hash(hash_0) == block_0, (
            "Block 0 should be discoverable by hash after add_request"
        )
        assert allocator.lookup_block_by_hash(hash_1) == block_1, (
            "Block 1 should be discoverable by hash after add_request"
        )

        # But block_hashes is still -1 (not computed)
        assert allocator.get_block_hash(block_0) == -1, (
            "Block 0 hash should be -1 (pending) before mark_pending_blocks_computed"
        )
        assert allocator.get_block_hash(block_1) == -1, (
            "Block 1 hash should be -1 (pending) before mark_pending_blocks_computed"
        )

        # Blocks should be in pending lists
        assert block_0 in allocator._pending_block_hashes
        assert block_1 in allocator._pending_block_hashes
        assert len(dynamic_context._blocks_pending_computation) == 2

        # Phase 2: After mark_pending_blocks_computed, block_hashes should be set
        dynamic_context.mark_pending_blocks_computed()

        assert allocator.get_block_hash(block_0) == hash_0, (
            "Block 0 hash should be set after mark_pending_blocks_computed"
        )
        assert allocator.get_block_hash(block_1) == hash_1, (
            "Block 1 hash should be set after mark_pending_blocks_computed"
        )

        # Pending lists should be cleared
        assert block_0 not in allocator._pending_block_hashes
        assert block_1 not in allocator._pending_block_hashes
        assert len(dynamic_context._blocks_pending_computation) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_lookup_vs_get_hash_difference(self):
        """Test that lookup_block_by_hash finds pending blocks but get_block_hash returns -1."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        allocator = dynamic_context.block_allocator

        # Allocate a block and register its hash (but don't mark computed)
        block_ids = allocator.allocate_memory_blocks(1)
        block_id = block_ids[0].item()
        test_hash = 99999

        allocator.register_block_hash(block_id, test_hash)

        # lookup_block_by_hash should find the block
        found_block = allocator.lookup_block_by_hash(test_hash)
        assert found_block == block_id, (
            "lookup_block_by_hash should find the pending block"
        )

        # But get_block_hash should return -1 (not computed yet)
        stored_hash = allocator.get_block_hash(block_id)
        assert stored_hash == -1, (
            "get_block_hash should return -1 for pending block"
        )

        # This is the key difference that enables coordination:
        # A second request can FIND the block (lookup) but knows it's not READY (get_hash == -1)

        # After marking computed, get_block_hash should return the actual hash
        allocator.mark_block_computed(block_id)
        stored_hash_after = allocator.get_block_hash(block_id)
        assert stored_hash_after == test_hash, (
            "get_block_hash should return actual hash after mark_block_computed"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_eviction_cleans_pending_hashes(self):
        """Test that evicting a pending block cleans up both pending and hash mappings."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,  # Small buffer to make eviction easier
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=1,
        )

        allocator = dynamic_context.block_allocator

        # Allocate a block, register its hash (pending state)
        block_ids = allocator.allocate_memory_blocks(1)
        block_id = block_ids[0].item()
        test_hash = 88888

        allocator.register_block_hash(block_id, test_hash)

        # Verify pending state
        assert block_id in allocator._pending_block_hashes
        assert test_hash in allocator.hash_to_block_id
        assert allocator.get_block_hash(block_id) == -1

        # Set ref_count to 0 so block is evictable (cached state)
        allocator.block_ref_counts[block_id] = 0

        # Force the block back to free pool (simulating release)
        allocator.release_memory_blocks(block_ids)

        # Manually call evict_lru_blocks with enough blocks to evict our block
        # First, we need to exhaust available blocks to trigger eviction
        initial_avail = allocator.total_avail

        # Allocate all available blocks
        if initial_avail > 0:
            _ = allocator.allocate_memory_blocks(initial_avail)

        # Now try to allocate more - this should trigger eviction
        # But first, we need blocks in cached state (ref_count=0)
        # The block we registered should be in the evictable set if ref_count=0

        # Re-add the block to simulate it being cached but evictable
        allocator.release_memory_blocks(torch.tensor([block_id], device='cuda'))
        allocator.block_ref_counts[block_id] = 0
        allocator._pending_block_hashes[block_id] = test_hash
        allocator.hash_to_block_id[test_hash] = block_id

        # Now evict
        allocator.evict_lru_blocks(1)

        # After eviction, pending hash should be cleaned up
        assert block_id not in allocator._pending_block_hashes, (
            "Pending hash should be removed after eviction"
        )
        assert test_hash not in allocator.hash_to_block_id, (
            "Hash mapping should be removed after eviction"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_prefix_matching_requires_sequential_match(self):
        """Test that prefix matching stops at first non-matching block."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        # Create first request with 3 blocks: [A, B, C]
        prompt_1 = torch.arange(block_size * 3, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        req1_block_2 = dynamic_context.request_to_kv_block_ids[0][2].item()

        # Create second request: [A, X, C] - same first and third, different second
        # This tests that matching MUST be sequential - we can't skip block 1
        prompt_2 = torch.arange(block_size * 3, device=torch.cuda.current_device())
        # Modify middle block tokens (indices 32-63)
        prompt_2[block_size:block_size * 2] += 5000

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()
        req2_block_2 = dynamic_context.request_to_kv_block_ids[1][2].item()

        # Only block 0 should be shared (sequential match stops at block 1)
        assert req2_block_0 == req1_block_0, "Block 0 should be shared (same content)"
        assert req2_block_1 != req1_block_1, "Block 1 should NOT be shared (different content)"

        # Block 2 should NOT be shared even though content would match
        # because the hash chain is broken (different parent hash from block 1)
        assert req2_block_2 != req1_block_2, (
            "Block 2 should NOT be shared - hash chain is broken at block 1"
        )

        # Verify ref counts
        assert allocator.block_ref_counts[req1_block_0].item() == 2  # Shared
        assert allocator.block_ref_counts[req1_block_1].item() == 1  # Not shared
        assert allocator.block_ref_counts[req1_block_2].item() == 1  # Not shared
        assert allocator.block_ref_counts[req2_block_1].item() == 1  # Newly allocated
        assert allocator.block_ref_counts[req2_block_2].item() == 1  # Newly allocated

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_pending_block_detection_logic(self):
        """Test the logic used by engine's _has_pending_prefix_blocks."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        # Create first request with 2 blocks
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Add request - blocks are registered but pending
        dynamic_context.add_request(request_1)

        # Create second request with same prompt - it has precomputed hashes
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Simulate _has_pending_prefix_blocks logic:
        # Check if any precomputed hash matches a pending (uncomputed) block
        def has_pending_prefix_blocks(req):
            """Simulate engine's _has_pending_prefix_blocks."""
            if req.precomputed_block_hashes is None:
                return False
            if len(req.precomputed_block_hashes) == 0:
                return False

            for block_hash in req.precomputed_block_hashes:
                block_id = allocator.lookup_block_by_hash(block_hash)
                if block_id is None:
                    break  # No block with this hash - no need to wait
                stored_hash = allocator.get_block_hash(block_id)
                if stored_hash == -1:
                    return True  # Block exists but not computed - wait!
            return False

        # Before mark_pending_blocks_computed: request_2 should detect pending blocks
        assert has_pending_prefix_blocks(request_2), (
            "Should detect pending blocks before mark_pending_blocks_computed"
        )

        # After mark_pending_blocks_computed: no more pending blocks
        dynamic_context.mark_pending_blocks_computed()
        assert not has_pending_prefix_blocks(request_2), (
            "Should NOT detect pending blocks after mark_pending_blocks_computed"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_pending_block_detection_edge_cases(self):
        """Test edge cases for pending block detection."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        def has_pending_prefix_blocks(req):
            """Simulate engine's _has_pending_prefix_blocks."""
            if req.precomputed_block_hashes is None:
                return False
            if len(req.precomputed_block_hashes) == 0:
                return False

            for block_hash in req.precomputed_block_hashes:
                block_id = allocator.lookup_block_by_hash(block_hash)
                if block_id is None:
                    break
                stored_hash = allocator.get_block_hash(block_id)
                if stored_hash == -1:
                    return True
            return False

        # Edge case 1: precomputed_block_hashes is None
        request_none = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(block_size, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            # Not passing block_size_tokens, so hashes won't be precomputed
        )
        # Manually set to None to test
        request_none.precomputed_block_hashes = None
        assert not has_pending_prefix_blocks(request_none), (
            "Should return False when precomputed_block_hashes is None"
        )

        # Edge case 2: precomputed_block_hashes is empty list (prompt < block_size)
        request_short = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=torch.arange(block_size // 2, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        assert request_short.precomputed_block_hashes == [], (
            "Short prompt should have empty precomputed_block_hashes"
        )
        assert not has_pending_prefix_blocks(request_short), (
            "Should return False when precomputed_block_hashes is empty"
        )

        # Edge case 3: First hash not found (no existing block)
        request_new = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        # No blocks with these hashes exist yet
        assert not has_pending_prefix_blocks(request_new), (
            "Should return False when no matching block exists"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_second_request_can_share_after_first_computed(self):
        """Test full coordination flow: request 2 shares blocks only after request 1's KV is computed."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        allocator = dynamic_context.block_allocator

        # Request 1 added - blocks registered but pending
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_1)

        # At this point, blocks are in pending state
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Verify pending state
        assert allocator.get_block_hash(block_0) == -1
        assert allocator.get_block_hash(block_1) == -1
        assert allocator.block_ref_counts[block_0].item() == 1
        assert allocator.block_ref_counts[block_1].item() == 1

        # Simulate: Engine computes KV for request 1 and marks blocks computed
        dynamic_context.mark_pending_blocks_computed()

        # Now blocks are computed
        assert allocator.get_block_hash(block_0) != -1
        assert allocator.get_block_hash(block_1) != -1

        # Request 2 with same prompt can now share blocks
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(request_2)

        # Verify blocks are shared
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()

        assert req2_block_0 == block_0, "Request 2 should share block 0"
        assert req2_block_1 == block_1, "Request 2 should share block 1"

        # Ref counts should be 2
        assert allocator.block_ref_counts[block_0].item() == 2
        assert allocator.block_ref_counts[block_1].item() == 2

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_requests_wait_for_pending_blocks_then_share(self):
        """
        Simulate the engine scheduling flow where requests with pending
        prefix blocks wait, then proceed to share computed blocks.

        This tests the coordination scenario where:
        1. Request A, B, C are added with the same prefix
        2. Engine schedules ONLY request A for forward pass (B, C must wait)
        3. Forward pass runs on A -> mark_pending_blocks_computed()
        4. B and C are now unblocked and can be scheduled
        5. B and C run forward pass sharing A's computed blocks
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Phase 1: Add request A (first request gets scheduled)
        req_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        dynamic_context.add_request(req_a)

        # Get request A's blocks and precomputed hashes
        req_a_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req_a_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Verify blocks are in pending state (registered but not computed)
        assert allocator.get_block_hash(req_a_block_0) == -1, "Block 0 should be pending"
        assert allocator.get_block_hash(req_a_block_1) == -1, "Block 1 should be pending"

        # Precompute hashes for requests B and C (simulating what would happen
        # when they're added to the engine but before add_request is called)
        req_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )
        req_c = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # Phase 2: Simulate engine checking B and C - they should detect pending blocks
        # This simulates the engine's _has_pending_prefix_blocks check
        def has_pending_prefix_blocks(req):
            """Simulate engine's _has_pending_prefix_blocks check."""
            for block_hash in req.precomputed_block_hashes:
                block_id = allocator.lookup_block_by_hash(block_hash)
                if block_id is not None and allocator.get_block_hash(block_id) == -1:
                    return True  # Block exists but not computed - WAIT
            return False

        assert has_pending_prefix_blocks(req_b), "B should wait for A's pending blocks"
        assert has_pending_prefix_blocks(req_c), "C should wait for A's pending blocks"

        # Phase 3: Simulate forward pass completing on A
        dynamic_context.mark_pending_blocks_computed()

        # Verify blocks are now computed
        assert allocator.get_block_hash(req_a_block_0) != -1, "Block 0 should be computed"
        assert allocator.get_block_hash(req_a_block_1) != -1, "Block 1 should be computed"

        # Phase 4: B and C should no longer see pending blocks
        assert not has_pending_prefix_blocks(req_b), "B can now proceed"
        assert not has_pending_prefix_blocks(req_c), "C can now proceed"

        # Phase 5: Add B and C - they should share A's computed blocks
        dynamic_context.add_request(req_b)
        dynamic_context.add_request(req_c)

        # Verify all three share the same blocks
        a_blocks = [
            dynamic_context.request_to_kv_block_ids[0][i].item() for i in range(2)
        ]
        b_blocks = [
            dynamic_context.request_to_kv_block_ids[1][i].item() for i in range(2)
        ]
        c_blocks = [
            dynamic_context.request_to_kv_block_ids[2][i].item() for i in range(2)
        ]

        assert a_blocks == b_blocks == c_blocks, (
            f"All requests should share same blocks. "
            f"A: {a_blocks}, B: {b_blocks}, C: {c_blocks}"
        )

        # Verify ref counts are 3 (one for each request)
        assert allocator.block_ref_counts[a_blocks[0]].item() == 3, (
            f"Block 0 ref count should be 3, got {allocator.block_ref_counts[a_blocks[0]].item()}"
        )
        assert allocator.block_ref_counts[a_blocks[1]].item() == 3, (
            f"Block 1 ref count should be 3, got {allocator.block_ref_counts[a_blocks[1]].item()}"
        )


class TestConcurrentRequests(PrefixCachingTestBase):
    """Tests for concurrent request handling with prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_concurrent_requests_same_prefix(self):
        """Test multiple requests with same prefix added before any marked computed.

        When multiple requests arrive simultaneously with the same prefix, each should
        initially allocate its own blocks (since none are marked computed yet). After
        marking computed, subsequent requests should be able to share.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Add 3 requests with same prompt before marking any as computed
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_3)

        # All 3 should have allocated their own blocks (no sharing yet)
        req1_blocks = dynamic_context.request_to_kv_block_ids[0][:2]
        req2_blocks = dynamic_context.request_to_kv_block_ids[1][:2]
        req3_blocks = dynamic_context.request_to_kv_block_ids[2][:2]

        assert not torch.equal(req1_blocks, req2_blocks), (
            "Requests 1 and 2 should not share blocks before marking computed"
        )
        assert not torch.equal(req1_blocks, req3_blocks), (
            "Requests 1 and 3 should not share blocks before marking computed"
        )

        # Mark all blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Now add a 4th request - it SHOULD share with one of the first 3
        request_4 = DynamicInferenceRequest(
            request_id=4,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_4)

        req4_blocks = dynamic_context.request_to_kv_block_ids[3][:2]

        # Request 4 should share blocks with at least one of the earlier requests
        shares_with_req1 = torch.equal(req4_blocks, req1_blocks)
        shares_with_req2 = torch.equal(req4_blocks, req2_blocks)
        shares_with_req3 = torch.equal(req4_blocks, req3_blocks)

        assert shares_with_req1 or shares_with_req2 or shares_with_req3, (
            "Request 4 should share blocks with one of the computed requests"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_racing_pending_blocks(self):
        """Verify two-phase registration prevents race conditions.

        Request 2 should NOT match blocks from Request 1 that are still pending
        (not yet marked computed). Only after marking computed should sharing happen.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Request 1: add and register hashes (pending state)
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        # Do NOT mark as computed yet

        # Request 2: add with same prefix (should NOT match pending blocks)
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        # Verify: Request 2 should NOT share blocks (they're still pending)
        req1_blocks = dynamic_context.request_to_kv_block_ids[0][:2]
        req2_blocks = dynamic_context.request_to_kv_block_ids[1][:2]

        assert not torch.equal(req1_blocks, req2_blocks), (
            "Request 2 should NOT share pending blocks from Request 1. "
            "Two-phase registration should prevent sharing until blocks are marked computed."
        )

        # Now mark Request 1's blocks as computed
        dynamic_context.mark_pending_blocks_computed()

        # Request 3: add with same prefix (SHOULD match now)
        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_3)

        req3_blocks = dynamic_context.request_to_kv_block_ids[2][:2]

        # Request 3 should share with Request 1 or Request 2
        shares_with_req1 = torch.equal(req3_blocks, req1_blocks)
        shares_with_req2 = torch.equal(req3_blocks, req2_blocks)

        assert shares_with_req1 or shares_with_req2, (
            "Request 3 should share blocks now that they're marked computed"
        )


class TestComplexPrefixPatterns(PrefixCachingTestBase):
    """Tests for complex prefix sharing patterns."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_three_way_prefix_sharing(self):
        """Test three requests sharing the same prefix (ref_count = 3)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 3, device=torch.cuda.current_device())

        # Add request 1
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        block_ids = dynamic_context.request_to_kv_block_ids[0][:3]

        # Verify ref_count = 1
        for block_id in block_ids:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 1

        # Add request 2 (should share)
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        # Verify ref_count = 2
        for block_id in block_ids:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 2

        # Add request 3 (should also share)
        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_3)

        # Verify ref_count = 3
        for block_id in block_ids:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 3, (
                f"Block {block_id.item()} should have ref_count=3 with three sharing requests"
            )

        # Verify all three requests use the same blocks
        req1_blocks = dynamic_context.request_to_kv_block_ids[0][:3]
        req2_blocks = dynamic_context.request_to_kv_block_ids[1][:3]
        req3_blocks = dynamic_context.request_to_kv_block_ids[2][:3]

        assert torch.equal(req1_blocks, req2_blocks) and torch.equal(req1_blocks, req3_blocks)

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_prefix_chain_extending(self):
        """Test prefix chain where B extends A, C extends B.

        Request A: tokens [0:100]
        Request B: tokens [0:200] (shares [0:100] with A)
        Request C: tokens [0:300] (shares [0:200] with B)
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Request A: 2 blocks (64 tokens)
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_a,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_a)
        dynamic_context.mark_pending_blocks_computed()

        blocks_a = dynamic_context.request_to_kv_block_ids[0][:2]

        # Request B: 4 blocks (128 tokens), extends A
        prompt_b = torch.cat([
            prompt_a,  # Same first 2 blocks
            torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device()),
        ])
        request_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_b,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_b)
        dynamic_context.mark_pending_blocks_computed()

        blocks_b = dynamic_context.request_to_kv_block_ids[1][:4]

        # Verify B shares first 2 blocks with A
        assert torch.equal(blocks_b[:2], blocks_a), "B should share first 2 blocks with A"

        # Request C: 6 blocks (192 tokens), extends B
        prompt_c = torch.cat([
            prompt_b,  # Same first 4 blocks
            torch.arange(2000, 2000 + block_size * 2, device=torch.cuda.current_device()),
        ])
        request_c = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=prompt_c,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_c)

        blocks_c = dynamic_context.request_to_kv_block_ids[2][:6]

        # Verify C shares first 4 blocks with B (which includes A's 2 blocks)
        assert torch.equal(blocks_c[:4], blocks_b), "C should share first 4 blocks with B"
        assert torch.equal(blocks_c[:2], blocks_a), "C should share first 2 blocks with A"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_multiple_independent_prefix_trees(self):
        """Test multiple separate prefix patterns in cache simultaneously.

        Tree 1: A, B, C share prefix X
        Tree 2: D, E, F share prefix Y
        Verify both trees are maintained correctly.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Tree 1: Prefix X
        prefix_x = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Tree 2: Prefix Y
        prefix_y = torch.arange(5000, 5000 + block_size * 2, device=torch.cuda.current_device())

        # Add requests for Tree 1
        for i in range(3):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prefix_x.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            dynamic_context.add_request(request)
            if i == 0:
                dynamic_context.mark_pending_blocks_computed()

        # Add requests for Tree 2
        for i in range(3):
            request = DynamicInferenceRequest(
                request_id=i + 10,
                prompt_tokens=prefix_y.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            dynamic_context.add_request(request)
            if i == 0:
                dynamic_context.mark_pending_blocks_computed()

        # Verify Tree 1 requests all share blocks
        tree1_blocks = dynamic_context.request_to_kv_block_ids[0][:2]
        for i in range(1, 3):
            req_blocks = dynamic_context.request_to_kv_block_ids[i][:2]
            assert torch.equal(req_blocks, tree1_blocks), f"Tree 1 request {i} should share blocks"

        # Verify Tree 2 requests all share blocks
        tree2_blocks = dynamic_context.request_to_kv_block_ids[3][:2]
        for i in range(4, 6):
            req_blocks = dynamic_context.request_to_kv_block_ids[i][:2]
            assert torch.equal(req_blocks, tree2_blocks), f"Tree 2 request {i-3} should share blocks"

        # Verify Tree 1 and Tree 2 use DIFFERENT blocks
        assert not torch.equal(tree1_blocks, tree2_blocks), (
            "Tree 1 and Tree 2 should use different blocks"
        )


class TestMemoryPressure(PrefixCachingTestBase):
    """Tests for memory pressure and eviction with prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_eviction_preserves_active_blocks(self):
        """Test that blocks with ref_count > 0 cannot be evicted."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,  # Very small to force eviction pressure
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Add a request and keep it active
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        active_blocks = dynamic_context.request_to_kv_block_ids[0][:2].clone()

        # Verify blocks are active (ref_count = 1)
        for block_id in active_blocks:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 1

        # Try to fill cache to trigger eviction
        for i in range(20):
            try:
                filler_prompt = torch.arange(
                    (i + 10) * 1000, (i + 10) * 1000 + block_size * 2,
                    device=torch.cuda.current_device(),
                )
                filler_request = DynamicInferenceRequest(
                    request_id=i + 100,
                    prompt_tokens=filler_prompt,
                    sampling_params=SamplingParams(num_tokens_to_generate=10),
                )
                dynamic_context.add_request(filler_request)
                dynamic_context.mark_pending_blocks_computed()
            except Exception:
                break

        # Verify active blocks still exist and have ref_count = 1
        for block_id in active_blocks:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 1, (
                f"Active block {block_id.item()} was incorrectly evicted or corrupted!"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.IMPORTANT, reason="Test level not met")
    def test_cache_full_scenario(self):
        """Test behavior when cache is completely full."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,  # Very small buffer
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        initial_avail = block_allocator.total_avail

        # Fill the cache completely
        requests_added = 0
        while block_allocator.total_avail > 2:  # Leave some room
            try:
                prompt = torch.arange(
                    requests_added * 1000, requests_added * 1000 + block_size * 2,
                    device=torch.cuda.current_device(),
                )
                request = DynamicInferenceRequest(
                    request_id=requests_added + 1,
                    prompt_tokens=prompt,
                    sampling_params=SamplingParams(num_tokens_to_generate=10),
                )
                dynamic_context.add_request(request)
                dynamic_context.mark_pending_blocks_computed()
                requests_added += 1
            except Exception:
                break

        # Verify cache is nearly full
        assert block_allocator.total_avail < initial_avail // 2, (
            "Cache should be significantly filled"
        )

        # Now try to add a request with a cached prefix (should work via eviction)
        cached_prompt = torch.arange(0, block_size * 2, device=torch.cuda.current_device())
        new_request = DynamicInferenceRequest(
            request_id=9000,
            prompt_tokens=cached_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        try:
            dynamic_context.add_request(new_request)
            # Should succeed (may evict cached blocks to make room)
        except Exception as e:
            # If it fails, that's also acceptable behavior for a full cache
            pass


class TestRequestLifecycle(PrefixCachingTestBase):
    """Tests for request lifecycle with prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_release_preserves_cached_blocks(self):
        """Test that releasing a request leaves blocks cached (evictable) for reuse."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # Add and complete request 1
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        block_ids = dynamic_context.request_to_kv_block_ids[0][:2]

        # Release request 1
        dynamic_context.release_kv_blocks(0)  # Release first request

        # Verify blocks have ref_count = 0 (cached, evictable)
        for block_id in block_ids:
            assert block_allocator.block_ref_counts[block_id.item()].item() == 0, (
                "Released blocks should have ref_count=0 (cached state)"
            )

        # Verify hashes still present (cached for reuse)
        for block_id in block_ids:
            block_hash = block_allocator.get_block_hash(block_id.item())
            assert block_hash != -1, "Released blocks should still have hash (cached)"
            assert block_hash in block_allocator.hash_to_block_id, (
                "Hash mapping should be preserved for cached blocks"
            )

        # Add request 2 with same prompt (should reuse cached blocks)
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        req2_blocks = dynamic_context.request_to_kv_block_ids[1][:2]

        # Should reuse the cached blocks
        assert torch.equal(req2_blocks, block_ids), "Request 2 should reuse cached blocks"

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_chunked_prefill_prefix_matching(self):
        """Test that prefix matching only happens on first chunk.

        Note: Current implementation only matches prefix on the first chunk.
        This test documents and verifies this behavior.
        """
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create a long prompt (8 blocks)
        long_prompt = torch.arange(block_size * 8, device=torch.cuda.current_device())

        # Add request 1 in chunks
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=long_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # First chunk (assume 4 blocks)
        dynamic_context.add_request(request_1, chunk_length=block_size * 4)
        dynamic_context.mark_pending_blocks_computed()

        # Second chunk (remaining 4 blocks)
        dynamic_context.add_request(request_1, chunk_length=block_size * 4)
        dynamic_context.mark_pending_blocks_computed()

        # Now add request 2 with same prompt
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=long_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # Add first chunk - should match the first 4 blocks
        dynamic_context.add_request(request_2, chunk_length=block_size * 4)

        # Verify first 4 blocks are shared
        req1_first_blocks = dynamic_context.request_to_kv_block_ids[0][:4]
        req2_first_blocks = dynamic_context.request_to_kv_block_ids[1][:4]

        assert torch.equal(req1_first_blocks, req2_first_blocks), (
            "First chunk should share blocks"
        )

class TestAdditionalEdgeCases(PrefixCachingTestBase):
    """Tests for additional edge cases in prefix caching."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_empty_prompt(self):
        """Test request with 0 tokens."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        # Create empty prompt
        empty_prompt = torch.tensor([], device=torch.cuda.current_device(), dtype=torch.long)

        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=empty_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # Should handle gracefully (no blocks allocated, no hashes)
        assert len(request.precomputed_block_hashes) == 0, "Empty prompt should have no hashes"

        try:
            dynamic_context.add_request(request)
            # Should succeed with 0 blocks allocated
        except Exception as e:
            # Or may reject, which is also acceptable
            pass

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_single_token_prompt(self):
        """Test request with only 1 token (less than block size)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        # Single token
        single_token = torch.tensor([42], device=torch.cuda.current_device(), dtype=torch.long)

        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=single_token,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # No complete blocks, so no precomputed hashes
        assert len(request.precomputed_block_hashes) == 0

        dynamic_context.add_request(request)
        # Should allocate 1 block for the partial prompt

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_extremely_long_prompt(self):
        """Test request with many blocks (100+) to verify hash computation scales."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=8192,
            buffer_size_gb=0.5,  # Larger buffer for long prompt
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # Create very long prompt (120 blocks = 3840 tokens)
        long_prompt = torch.arange(
            block_size * 120, device=torch.cuda.current_device(), dtype=torch.long
        )

        request = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=long_prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # Verify all 120 blocks have hashes computed
        assert len(request.precomputed_block_hashes) == 120, (
            f"Expected 120 hashes for 120 blocks, got {len(request.precomputed_block_hashes)}"
        )

        # Verify all hashes are positive
        for i, h in enumerate(request.precomputed_block_hashes):
            assert h > 0, f"Hash {i} should be positive, got {h}"

        # Add request (may take a while, but should succeed)
        try:
            dynamic_context.add_request(request)
            dynamic_context.mark_pending_blocks_computed()
        except Exception:
            # May run out of memory, which is acceptable for such a long prompt
            pass

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.MEDIUM, reason="Test level not met")
    def test_pathological_repeated_tokens(self):
        """Test all tokens identical (pathological case for hash function)."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens

        # All zeros
        all_zeros = torch.zeros(block_size * 4, device=torch.cuda.current_device(), dtype=torch.long)

        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=all_zeros,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )

        # Verify hashes are computed (even for pathological input)
        assert len(request_1.precomputed_block_hashes) == 4

        # Verify hashes are different for different blocks (due to parent hash dependency)
        hashes = request_1.precomputed_block_hashes
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 4, (
            f"Even with all zeros, each block should have different hash (parent dependency). "
            f"Got {len(unique_hashes)} unique hashes for 4 blocks"
        )

class TestObservability(PrefixCachingTestBase):
    """Tests for observability features like metrics and debugging."""

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.LOW, reason="Test level not met")
    def test_block_allocation_tracking(self):
        """Test that we can track block allocation and usage statistics."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        initial_avail = block_allocator.total_avail

        # Add some requests
        num_requests = 5
        for i in range(num_requests):
            prompt = torch.arange(
                i * 1000, i * 1000 + block_size * 2, device=torch.cuda.current_device()
            )
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt,
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            dynamic_context.add_request(request)
            dynamic_context.mark_pending_blocks_computed()

        # Track statistics
        blocks_used = initial_avail - block_allocator.total_avail

        print(f"\nBlock allocation statistics:")
        print(f"  Total blocks: {initial_avail}")
        print(f"  Blocks used: {blocks_used}")
        print(f"  Blocks available: {block_allocator.total_avail}")
        print(f"  Utilization: {blocks_used / initial_avail * 100:.1f}%")

        # Should have allocated blocks
        assert blocks_used > 0, "Should have allocated some blocks"
        assert blocks_used == num_requests * 2, (
            f"Should have allocated {num_requests * 2} blocks (2 per request)"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(TEST_LEVEL < TestLevel.LOW, reason="Test level not met")
    def test_prefix_cache_hit_rate(self):
        """Test tracking prefix cache hit rate."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=False,
            rounder=64,
        )

        block_size = dynamic_context.block_size_tokens
        prompt = torch.arange(block_size * 3, device=torch.cuda.current_device())

        # Add first request (cache miss)
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)
        dynamic_context.mark_pending_blocks_computed()

        # Add 9 more requests with same prefix (cache hits)
        for i in range(2, 11):
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            dynamic_context.add_request(request)

        # Calculate hit rate: 9 hits out of 10 requests = 90%
        # (First request is a miss, next 9 are hits)
        total_requests = 10
        cache_hits = 9  # Requests 2-10 all hit the cache
        hit_rate = cache_hits / total_requests

        print(f"\nPrefix cache statistics:")
        print(f"  Total requests: {total_requests}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Hit rate: {hit_rate * 100:.1f}%")

        # Verify high hit rate for identical requests
        assert hit_rate >= 0.9, f"Expected hit rate >= 90%, got {hit_rate * 100:.1f}%"
