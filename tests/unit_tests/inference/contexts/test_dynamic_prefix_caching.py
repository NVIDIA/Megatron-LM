# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

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


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class TestDynamicPrefixCaching:

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

    # =========================================================================
    # Block hash tests
    # =========================================================================

    @pytest.mark.internal
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
        hash_value = block_allocator.compute_block_hash(0, token_ids)
        assert hash_value > 0, "Hash should be positive"

        # Test 2: Same inputs should produce same hash
        hash_value_2 = block_allocator.compute_block_hash(0, token_ids)
        assert hash_value == hash_value_2, "Hash should be deterministic"

        # Test 3: Different parent hash should produce different result
        hash_with_parent = block_allocator.compute_block_hash(12345, token_ids)
        assert hash_with_parent != hash_value, "Different parent should produce different hash"
        assert hash_with_parent > 0, "Hash with parent should still be positive"

        # Test 4: Different tokens should produce different hash
        different_tokens = torch.arange(1, 129, device=torch.cuda.current_device(), dtype=torch.int64)
        hash_different = block_allocator.compute_block_hash(0, different_tokens)
        assert hash_different != hash_value, "Different tokens should produce different hash"

        # Test 5: Block hashes tensor initialized to -1
        assert (block_allocator.block_hashes == -1).all(), "Block hashes should initialize to -1"

    @pytest.mark.internal
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

        # Check: All released blocks should have hash reset to -1
        assert block_allocator.block_hashes[block_0_id].item() == -1, "Block 0 hash should reset"
        assert block_allocator.block_hashes[block_1_id].item() == -1, "Block 1 hash should reset"
        assert block_allocator.block_hashes[block_2_id].item() == -1, "Block 2 hash should reset"

    @pytest.mark.internal
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

        block_allocator = dynamic_context.block_allocator
        block_size = dynamic_context.block_size_tokens

        # Create identical prompts that span 2 complete blocks
        prompt_tokens = torch.arange(block_size * 2, device=torch.cuda.current_device())

        # First request
        request_1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_1)

        # Get hashes for request 1's blocks
        req1_block_0_id = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1_id = dynamic_context.request_to_kv_block_ids[0][1].item()
        req1_block_0_hash = block_allocator.block_hashes[req1_block_0_id].item()
        req1_block_1_hash = block_allocator.block_hashes[req1_block_1_id].item()

        # Second request with same tokens
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        # Get hashes for request 2's blocks (different block IDs but same content)
        req2_block_0_id = dynamic_context.request_to_kv_block_ids[1][0].item()
        req2_block_1_id = dynamic_context.request_to_kv_block_ids[1][1].item()
        req2_block_0_hash = block_allocator.block_hashes[req2_block_0_id].item()
        req2_block_1_hash = block_allocator.block_hashes[req2_block_1_id].item()

        # Verify: Same token content should produce identical hashes
        assert req1_block_0_hash == req2_block_0_hash, (
            f"Block 0 hashes should match: {req1_block_0_hash} vs {req2_block_0_hash}"
        )
        assert req1_block_1_hash == req2_block_1_hash, (
            f"Block 1 hashes should match: {req1_block_1_hash} vs {req2_block_1_hash}"
        )

        # Verify hash chaining: block 1 hash should differ from block 0
        assert req1_block_0_hash != req1_block_1_hash, "Different blocks should have different hashes"

        # Third request with different tokens
        different_tokens = torch.arange(1, block_size * 2 + 1, device=torch.cuda.current_device())
        request_3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=different_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_3)

        req3_block_0_id = dynamic_context.request_to_kv_block_ids[2][0].item()
        req3_block_0_hash = block_allocator.block_hashes[req3_block_0_id].item()

        # Verify: Different tokens should produce different hash
        assert req1_block_0_hash != req3_block_0_hash, (
            "Different token sequences should produce different hashes"
        )

    @pytest.mark.internal
    def test_block_hash_computed_when_filled_during_decode(self):
        """Test that hash is computed when a block is filled during decode."""
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

        # Verify: block 0 has hash, block 1 is partial (no hash)
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        assert block_allocator.block_hashes[block_0].item() != -1, "Block 0 should have hash"
        assert block_allocator.block_hashes[block_1].item() == -1, (
            "Block 1 should NOT have hash (partial)"
        )

        # Run one decode step - this should fill block 1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        dynamic_context.update_requests(active_mask, new_tokens)

        # Now block 1 should have hash computed
        assert block_allocator.block_hashes[block_1].item() != -1, (
            "Block 1 should have hash after being filled during decode"
        )

    @pytest.mark.internal
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

    # =========================================================================
    # Prefix caching tests
    # =========================================================================

    @pytest.mark.internal
    def test_prefix_caching_basic_sharing(self):
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
        )
        dynamic_context.add_request(request_1)

        # Get block IDs for request 1
        req1_block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        req1_block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()

        # Verify hashes are registered in the mapping
        block_0_hash = block_allocator.get_block_hash(req1_block_0)
        block_1_hash = block_allocator.get_block_hash(req1_block_1)
        assert block_0_hash in block_allocator.hash_to_block_id
        assert block_1_hash in block_allocator.hash_to_block_id

        # Verify ref counts are 1
        assert block_allocator.block_ref_counts[req1_block_0].item() == 1
        assert block_allocator.block_ref_counts[req1_block_1].item() == 1

        # Create second request with same prefix
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
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
    def test_prefix_caching_partial_match(self):
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
        )
        dynamic_context.add_request(request_1)

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
    def test_prefix_caching_ref_count_release(self):
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
        )
        dynamic_context.add_request(request_1)

        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
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
    def test_prefix_caching_lru_eviction(self):
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
        )
        dynamic_context.add_request(request_1)

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
        )

        # If pool is empty, this will trigger LRU eviction
        dynamic_context.add_request(request_2)

        # After eviction and reuse, block_0 may have been evicted
        # The hash should no longer be in the mapping if evicted
        # (or it might still be there if other blocks were evicted first)

        # Key invariant: the system should still function correctly
        assert dynamic_context.total_request_count == 1

    @pytest.mark.internal
    def test_prefix_caching_no_match_allocates_new(self):
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
        )
        dynamic_context.add_request(request_1)

        req1_blocks = set()
        for i in range(2):
            req1_blocks.add(dynamic_context.request_to_kv_block_ids[0][i].item())

        # Second request with completely different tokens
        prompt_2 = torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device())
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt_2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
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
    def test_prefix_caching_reuse_after_release(self):
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
        )
        dynamic_context.add_request(request_1)

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
    def test_prefix_caching_many_requests_same_prefix(self):
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

        # Add 10 requests
        for i in range(num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
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
    def test_prefix_caching_hash_chain_correctness(self):
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
        )
        dynamic_context.add_request(request)

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
        # Block 0's tokens with parent_hash=0
        block_0_tokens = block_allocator.block_to_token_ids[block_0]
        computed_hash_0 = block_allocator.compute_block_hash(0, block_0_tokens)
        assert computed_hash_0 == hash_0, "Block 0 hash should match with parent=0"

        # Same tokens with different parent should give different hash
        hash_with_different_parent = block_allocator.compute_block_hash(12345, block_0_tokens)
        assert hash_with_different_parent != hash_0, (
            "Same tokens with different parent should produce different hash"
        )

    # =========================================================================
    # Memory usage tests
    # =========================================================================

    @pytest.mark.internal
    def test_prefix_caching_available_blocks_preserved(self):
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
        )
        dynamic_context.add_request(request_1)

        avail_after_first = block_allocator.total_avail
        blocks_used_first = initial_avail - avail_after_first

        # Add second request with identical prompt
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
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
    def test_prefix_caching_memory_scaling_constant(self):
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

        # Add N identical requests
        prompt = torch.arange(block_size * num_blocks, device=torch.cuda.current_device())
        for i in range(num_requests):
            request = DynamicInferenceRequest(
                request_id=i + 1,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
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

    # =========================================================================
    # TTFT tests
    # =========================================================================

    @pytest.mark.internal
    def test_prefix_caching_matched_blocks_tokens_preserved(self):
        """Test that tokens in matched blocks are NOT overwritten."""
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
        )
        dynamic_context.add_request(request_1)

        # Record token IDs stored in blocks
        block_0 = dynamic_context.request_to_kv_block_ids[0][0].item()
        block_1 = dynamic_context.request_to_kv_block_ids[0][1].item()
        original_tokens_0 = block_allocator.block_to_token_ids[block_0].clone()
        original_tokens_1 = block_allocator.block_to_token_ids[block_1].clone()

        # Add second request with same prefix
        request_2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(request_2)

        # Verify tokens are preserved (not overwritten)
        current_tokens_0 = block_allocator.block_to_token_ids[block_0]
        current_tokens_1 = block_allocator.block_to_token_ids[block_1]

        assert torch.equal(original_tokens_0, current_tokens_0), (
            "Block 0 tokens should not be overwritten"
        )
        assert torch.equal(original_tokens_1, current_tokens_1), (
            "Block 1 tokens should not be overwritten"
        )

    @pytest.mark.internal
    def test_prefix_caching_only_new_blocks_hashed(self):
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
        )
        dynamic_context.add_request(request_1)

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
        )
        dynamic_context.add_request(request_2)

        # Verify matched block (block 0) has SAME hash
        req2_block_0 = dynamic_context.request_to_kv_block_ids[1][0].item()
        assert req2_block_0 == block_0, "First block should be shared"
        current_hash_0 = block_allocator.get_block_hash(block_0)
        assert current_hash_0 == original_hash_0, (
            "Matched block hash should not change"
        )

        # Verify new blocks have different hashes
        req2_block_1 = dynamic_context.request_to_kv_block_ids[1][1].item()
        req2_block_2 = dynamic_context.request_to_kv_block_ids[1][2].item()
        assert req2_block_1 != block_1, "Second block should be newly allocated"
        new_hash_1 = block_allocator.get_block_hash(req2_block_1)
        new_hash_2 = block_allocator.get_block_hash(req2_block_2)
        assert new_hash_1 != original_hash_1, "New block 1 should have different hash"
        assert new_hash_1 != -1, "New block 1 should have hash computed"
        assert new_hash_2 != -1, "New block 2 should have hash computed"

    # =========================================================================
    # Edge case tests
    # =========================================================================

    @pytest.mark.internal
    def test_prefix_caching_single_block_prefix(self):
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
    def test_prefix_caching_incomplete_block_not_shared(self):
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

    # =========================================================================
    # Disabled mode tests
    # =========================================================================

    @pytest.mark.internal
    def test_prefix_caching_disabled_no_sharing(self):
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
    def test_prefix_caching_disabled_deterministic_hashes(self):
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
            expected_hash = (block_id * 2654435761) % block_allocator.HASH_PRIME + 1
            actual_hash = block_allocator.block_hashes[block_id].item()
            assert actual_hash == expected_hash, (
                f"Hash for block {block_id} should be deterministic: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    @pytest.mark.internal
    def test_prefix_caching_performance_comparison(self):
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
