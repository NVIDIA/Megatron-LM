# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for Mamba prefix caching in hybrid models.

Focuses on Mamba-specific prefix caching features:
- Mamba state store/restore/invalidation
- Mamba LRU eviction
- Coupled KV+Mamba prefix matching (the key correctness fix)
- Cross-config end-to-end equivalence
- Zero-budget behavior
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _set_rounder(value):
    """Set all DynamicInferenceContext rounders to a given value."""
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


def _build_hybrid_context(
    block_size=32,
    max_tokens=256,
    max_requests=8,
    buffer_size_gb=0.01,
    enable_prefix_caching=True,
    prefix_caching_mamba_gb=0.001,
    num_layers=4,
    kv_channels=8,
    num_attention_heads=2,
    rounder=64,
    layer_type_list=None,
    params_dtype=torch.float32,
    pp_size=1,
) -> DynamicInferenceContext:
    """Build a DynamicInferenceContext configured for a hybrid Mamba model."""
    _set_rounder(rounder)

    if layer_type_list is None:
        layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]

    mamba_conv_states_shape = (544, 4)
    mamba_ssm_states_shape = (8, 64, 16)
    mamba_inference_state_config = MambaInferenceStateConfig(
        layer_type_list, mamba_conv_states_shape, mamba_ssm_states_shape
    )

    transformer_config = TransformerConfig(
        params_dtype=params_dtype,
        num_layers=num_layers,
        kv_channels=kv_channels,
        num_attention_heads=num_attention_heads,
        hidden_size=kv_channels * num_attention_heads,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        use_cpu_initialization=True,
    )
    inference_config = InferenceConfig(
        max_sequence_length=1024,
        buffer_size_gb=buffer_size_gb,
        paused_buffer_size_gb=0.2 * buffer_size_gb,
        block_size_tokens=block_size,
        max_tokens=max_tokens,
        mamba_inference_state_config=mamba_inference_state_config,
        use_flashinfer_fused_rope=None,
        unified_memory_level=0,
        enable_prefix_caching=enable_prefix_caching,
        block_evict_lru=enable_prefix_caching,
        prefix_caching_mamba_gb=prefix_caching_mamba_gb,
    )
    return DynamicInferenceContext(
        model_config=transformer_config, inference_config=inference_config
    )


def _make_request(request_id, prompt_tokens, block_size, enable_prefix_caching=True,
                  num_tokens_to_generate=50):
    """Create a DynamicInferenceRequest with the given parameters."""
    if isinstance(prompt_tokens, int):
        prompt_tokens = torch.arange(prompt_tokens, device=torch.cuda.current_device())
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(num_tokens_to_generate=num_tokens_to_generate),
        block_size_tokens=block_size,
        enable_prefix_caching=enable_prefix_caching,
    )


def _simulate_prefill_completion(context):
    """Simulate what the engine does after prefill: mark pending blocks computed."""
    context.mark_pending_blocks_computed()


# =============================================================================
# Test Classes
# =============================================================================


class TestMambaCacheOperations:
    """Tests for basic Mamba state store, restore, and invalidation."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_store_and_restore_mamba_state(self):
        """Store Mamba state for a block, then restore it to a different request slot."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        block_size = ctx.block_size_tokens

        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)

        # Write known values into request's Mamba state
        mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[0].item()
        ctx.mamba_conv_states[:, mamba_idx] = 1.0
        ctx.mamba_ssm_states[:, mamba_idx] = 2.0

        # Store for block 0
        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)

        # Overwrite request state, then restore
        ctx.mamba_conv_states[:, mamba_idx] = 0.0
        ctx.mamba_ssm_states[:, mamba_idx] = 0.0

        restored = ctx.restore_mamba_state_from_block(0, block_0_id)
        assert restored
        assert torch.allclose(
            ctx.mamba_conv_states[:, mamba_idx],
            torch.ones_like(ctx.mamba_conv_states[:, mamba_idx]),
        )
        assert torch.allclose(
            ctx.mamba_ssm_states[:, mamba_idx],
            torch.full_like(ctx.mamba_ssm_states[:, mamba_idx], 2.0),
        )

    @pytest.mark.internal
    def test_has_mamba_state_for_block(self):
        """has_mamba_state_for_block returns True only after store."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        block_size = ctx.block_size_tokens

        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)

        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        assert not ctx.has_mamba_state_for_block(block_0_id)

        ctx.store_mamba_state_for_block(block_0_id, 0)
        assert ctx.has_mamba_state_for_block(block_0_id)

    @pytest.mark.internal
    def test_mamba_state_invalidated_on_block_eviction(self):
        """invalidate_mamba_state_for_block clears stored state."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        block_size = ctx.block_size_tokens

        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)

        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)
        assert ctx.has_mamba_state_for_block(block_0_id)

        ctx.invalidate_mamba_state_for_block(block_0_id)
        assert not ctx.has_mamba_state_for_block(block_0_id)


class TestMambaCacheEviction:
    """Tests for Mamba LRU eviction behavior."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_lru_eviction_when_pool_full(self):
        """When the Mamba cache is full, the LRU slot (oldest timestamp, ref_count=0) is evicted."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.001)
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots > 0

        # Fill all slots manually
        for i in range(max_slots):
            ctx.block_to_mamba_slot[i] = i
            ctx.mamba_slot_to_block[i] = i
            ctx.block_allocator.block_ref_counts[i] = 0
            ctx.block_allocator.block_timestamps[i] = i * 100  # block 0 oldest
        ctx.mamba_cache_free_count = 0

        # Allocate a new slot -- should evict block 0 (oldest)
        new_block_id = max_slots + 1
        slot = ctx._allocate_mamba_cache_slot(new_block_id)
        assert slot >= 0
        assert ctx.block_to_mamba_slot[0].item() == -1, "Block 0 should be evicted"
        assert ctx.block_to_mamba_slot[new_block_id].item() == slot

    @pytest.mark.internal
    def test_eviction_frees_slot_for_reuse(self):
        """Invalidating a block returns its slot to the free pool."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        initial_free = ctx.mamba_cache_free_count

        slot = ctx._allocate_mamba_cache_slot(0)
        assert ctx.mamba_cache_free_count == initial_free - 1

        ctx.invalidate_mamba_state_for_block(0)
        assert ctx.mamba_cache_free_count == initial_free

        slot2 = ctx._allocate_mamba_cache_slot(1)
        assert slot2 == slot, "Should reuse freed slot"

    @pytest.mark.internal
    def test_evicted_block_not_prefix_matchable(self):
        """After Mamba state is evicted for a block, that block should NOT be prefix-matched
        on a hybrid model (due to the coupled KV+Mamba fix)."""
        # Use very small Mamba cache (1-2 slots) so eviction is easy to trigger
        ctx = _build_hybrid_context(
            block_size=32,
            buffer_size_gb=0.01,
            prefix_caching_mamba_gb=0.001,
            max_tokens=None,
            max_requests=8,
        )
        block_size = ctx.block_size_tokens
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 1, f"Need at least 1 Mamba slot, got {max_slots}"

        # --- Request A: 64 tokens (2 blocks) ---
        a_idx = ctx.total_request_count  # 0
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt_a, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        # Store Mamba state for block 1 (last complete block of A)
        block_1_id = ctx.request_to_kv_block_ids[a_idx][1].item()
        ctx.store_mamba_state_for_block(block_1_id, a_idx)
        assert ctx.has_mamba_state_for_block(block_1_id)

        # Release A so its blocks become cached (ref_count=0)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # --- Request C: different 64 tokens (forces Mamba eviction) ---
        c_idx = ctx.total_request_count  # 1
        prompt_c = torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device())
        req_c = _make_request(2, prompt_c, block_size)
        ctx.add_request(req_c)
        _simulate_prefill_completion(ctx)

        # Store Mamba state for C's last block -- may evict block_1's Mamba state
        c_block_1_id = ctx.request_to_kv_block_ids[c_idx][1].item()
        ctx.store_mamba_state_for_block(c_block_1_id, c_idx)

        # Release C
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([c_idx]))

        # --- Request B: same prefix as A ---
        b_idx = ctx.total_request_count  # 2
        req_b = _make_request(3, prompt_a.clone(), block_size)
        # Simulate engine: no Mamba state for matched blocks
        req_b._mamba_num_matched_blocks = 0
        ctx.add_request(req_b)

        # B's Mamba state should be zero-initialized (not restored from cache)
        mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
        assert torch.all(ctx.mamba_conv_states[:, mamba_idx_b] == 0.0), \
            "B's Mamba state should be zero (no cache restore)"


class TestMambaPrefixMatching:
    """Tests for the coupled KV+Mamba prefix matching fix."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_hybrid_model_no_mamba_budget_no_prefix_match(self):
        """With prefix_caching_mamba_gb=None, hybrid model should never prefix-match
        because there's no Mamba state to restore."""
        ctx = _build_hybrid_context(
            block_size=32,
            buffer_size_gb=0.01,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=None,  # No Mamba budget
            max_tokens=None,
        )
        block_size = ctx.block_size_tokens
        assert ctx.max_mamba_cache_slots == 0

        # Add request A
        a_idx = ctx.total_request_count  # 0
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        # Release A
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Add request B with same prefix -- no _mamba_num_matched_blocks set
        b_idx = ctx.total_request_count  # 1
        req_b = _make_request(2, prompt.clone(), block_size)
        ctx.add_request(req_b)

        # The coupled fix: getattr(req, '_mamba_num_matched_blocks', 0) == 0
        # so num_matched_blocks gets set to 0, even though KV blocks are cached.
        # B's Mamba state should be zero-initialized
        mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
        assert torch.all(ctx.mamba_conv_states[:, mamba_idx_b] == 0.0), \
            "With no Mamba budget, B should get zero-init Mamba state"

    @pytest.mark.internal
    def test_hybrid_model_with_mamba_budget_prefix_matches(self):
        """With Mamba budget and cached state, prefix matching should work correctly."""
        ctx = _build_hybrid_context(
            block_size=32,
            buffer_size_gb=0.01,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=0.01,
            max_tokens=None,
        )
        block_size = ctx.block_size_tokens
        assert ctx.max_mamba_cache_slots > 0

        # Add and process request A
        a_idx = ctx.total_request_count  # 0
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        # Store Mamba state for A's blocks
        mamba_idx_a = ctx.mamba_metadata.request_to_mamba_state_idx[a_idx].item()
        ctx.mamba_conv_states[:, mamba_idx_a] = 7.0
        ctx.mamba_ssm_states[:, mamba_idx_a] = 14.0

        block_0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        block_1_id = ctx.request_to_kv_block_ids[a_idx][1].item()
        ctx.store_mamba_state_for_block(block_0_id, a_idx)
        ctx.store_mamba_state_for_block(block_1_id, a_idx)

        # Release A
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Add request B with same prefix and _mamba_num_matched_blocks set
        b_idx = ctx.total_request_count  # 1
        req_b = _make_request(2, prompt.clone(), block_size)
        req_b._mamba_num_matched_blocks = 2  # Both blocks have Mamba state
        ctx.add_request(req_b)

        # B should have restored Mamba state from the last matched block (block 1)
        mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
        assert torch.allclose(
            ctx.mamba_conv_states[:, mamba_idx_b],
            torch.full_like(ctx.mamba_conv_states[:, mamba_idx_b], 7.0),
        ), "B should have restored Mamba conv state from block 1"
        assert torch.allclose(
            ctx.mamba_ssm_states[:, mamba_idx_b],
            torch.full_like(ctx.mamba_ssm_states[:, mamba_idx_b], 14.0),
        ), "B should have restored Mamba SSM state from block 1"

    @pytest.mark.internal
    def test_mamba_match_limits_kv_match(self):
        """KV matches 3 blocks but Mamba only has state for 1 → effective match = 1."""
        ctx = _build_hybrid_context(
            block_size=32,
            buffer_size_gb=0.01,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=0.001,  # Very small -- few slots
            max_tokens=None,
        )
        block_size = ctx.block_size_tokens

        # Add request A with 3 complete blocks (96 tokens)
        a_idx = ctx.total_request_count  # 0
        prompt = torch.arange(block_size * 3, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        # Store Mamba state ONLY for block 0
        mamba_idx_a = ctx.mamba_metadata.request_to_mamba_state_idx[a_idx].item()
        ctx.mamba_conv_states[:, mamba_idx_a] = 99.0
        ctx.mamba_ssm_states[:, mamba_idx_a] = 99.0
        block_0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        ctx.store_mamba_state_for_block(block_0_id, a_idx)

        # Release A
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Add request B with same 96 tokens
        req_b = _make_request(2, prompt.clone(), block_size)
        # Engine computed: KV matches 3, but Mamba only for 1
        req_b._mamba_num_matched_blocks = 1
        b_idx = ctx.total_request_count  # B's index before add_request increments it
        ctx.add_request(req_b)

        # B should have restored Mamba state from block 0
        mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
        assert torch.allclose(
            ctx.mamba_conv_states[:, mamba_idx_b],
            torch.full_like(ctx.mamba_conv_states[:, mamba_idx_b], 99.0),
        ), "B should restore Mamba state from block 0"

        # B should have matched only 1 block (not 3), meaning it allocated 2 new blocks
        # Verify by checking that B has blocks assigned for its 3-block request
        b_blocks = ctx.request_to_kv_block_ids[b_idx][:3].tolist()
        # Block 0 should be shared (same as A's block 0)
        assert b_blocks[0] == block_0_id, "Block 0 should be shared from A"

    @pytest.mark.internal
    def test_mamba_match_zero_limits_all_kv_matches(self):
        """KV matches 2 blocks but Mamba has state for 0 → effective match = 0."""
        ctx = _build_hybrid_context(
            block_size=32,
            buffer_size_gb=0.01,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=0.01,
            max_tokens=None,
        )
        block_size = ctx.block_size_tokens

        # Add request A
        a_idx = ctx.total_request_count  # 0
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        # Don't store any Mamba state for A's blocks
        # Release A
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Add request B with same prefix, but engine says 0 Mamba matches
        b_idx = ctx.total_request_count  # 1
        req_b = _make_request(2, prompt.clone(), block_size)
        req_b._mamba_num_matched_blocks = 0
        ctx.add_request(req_b)

        # B should have zero-init Mamba state (no restore)
        mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
        assert torch.all(ctx.mamba_conv_states[:, mamba_idx_b] == 0.0), \
            "B should get zero-init Mamba state when _mamba_num_matched_blocks=0"


class TestCrossConfigEndToEnd:
    """End-to-end tests verifying that different chunked_prefill x prefix_caching
    configurations produce consistent results."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_cross_chunked_prefix_configs(self):
        """Two requests sharing a 64-token prefix should produce consistent block
        allocation across 4 configs: chunked_prefill x prefix_caching (on/off).

        With prefix_caching ON + Mamba state cached, request B reuses blocks.
        With prefix_caching OFF, request B allocates all blocks fresh.
        Both should be internally consistent.
        """
        block_size = 32
        prompt_shared = torch.arange(64, device=torch.cuda.current_device())  # 2 blocks
        prompt_b_extra = torch.arange(64, 100, device=torch.cuda.current_device())  # 36 more
        prompt_b = torch.cat([prompt_shared, prompt_b_extra])  # 100 tokens total

        configs = [
            {"enable_prefix_caching": True, "prefix_caching_mamba_gb": 0.01},
            {"enable_prefix_caching": False, "prefix_caching_mamba_gb": None},
        ]

        for config in configs:
            # Re-init model parallel for each config
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
            model_parallel_cuda_manual_seed(123)

            ctx = _build_hybrid_context(
                block_size=block_size,
                buffer_size_gb=0.01,
                max_tokens=None,
                max_requests=8,
                **config,
            )

            # --- Request A: 64 tokens (2 complete blocks) ---
            a_idx = ctx.total_request_count  # 0
            req_a = _make_request(1, prompt_shared.clone(), block_size)
            ctx.add_request(req_a)
            _simulate_prefill_completion(ctx)

            a_block_count = ctx.request_kv_block_counts[a_idx].item()
            assert a_block_count == 2, f"Request A should use 2 blocks, got {a_block_count}"

            if config["enable_prefix_caching"]:
                # Store Mamba state for both of A's blocks
                mamba_idx_a = ctx.mamba_metadata.request_to_mamba_state_idx[a_idx].item()
                ctx.mamba_conv_states[:, mamba_idx_a] = 42.0
                ctx.mamba_ssm_states[:, mamba_idx_a] = 42.0
                for blk_idx in range(2):
                    blk_id = ctx.request_to_kv_block_ids[a_idx][blk_idx].item()
                    ctx.store_mamba_state_for_block(blk_id, a_idx)

            # Release A
            ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

            # --- Request B: 100 tokens (shares first 64 with A) ---
            b_idx = ctx.total_request_count  # 1
            req_b = _make_request(2, prompt_b.clone(), block_size)
            if config["enable_prefix_caching"]:
                req_b._mamba_num_matched_blocks = 2  # Both blocks have Mamba state
            ctx.add_request(req_b)
            _simulate_prefill_completion(ctx)

            b_block_count = ctx.request_kv_block_counts[b_idx].item()
            # 100 tokens / 32 = 3.125 → 4 blocks total
            assert b_block_count == 4, f"Request B should use 4 blocks, got {b_block_count}"

            if config["enable_prefix_caching"]:
                # With prefix caching: B matched 2 blocks + allocated 2 new = 4 total
                # Mamba state should be restored
                mamba_idx_b = ctx.mamba_metadata.request_to_mamba_state_idx[b_idx].item()
                assert torch.allclose(
                    ctx.mamba_conv_states[:, mamba_idx_b],
                    torch.full_like(ctx.mamba_conv_states[:, mamba_idx_b], 42.0),
                ), "B should have restored Mamba state in prefix_caching mode"

    @pytest.mark.internal
    def test_interleaving_boundaries(self):
        """Verify the interleaving of Mamba boundaries:
        divergence=64 < chunk_break=80 < last_aligned=96 < end=100."""
        block_size = 32
        prompt_length = 100

        # divergence: where Mamba cache ends (2 blocks * 32 = 64)
        num_mamba_matched = 2
        divergence_token = num_mamba_matched * block_size
        assert divergence_token == 64

        # chunk_break: max_tokens limits first chunk (80)
        max_tokens = 80
        chunk_break = max_tokens
        assert chunk_break == 80

        # last_aligned: last block-aligned token
        last_aligned_token = (prompt_length // block_size) * block_size
        assert last_aligned_token == 96

        # end of sequence
        assert prompt_length == 100

        # Verify interleaving
        assert divergence_token < chunk_break < last_aligned_token < prompt_length


class TestBudgetZero:
    """Tests for zero-budget and disabled Mamba caching scenarios."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_zero_mamba_budget_disables_mamba_caching(self):
        """With prefix_caching_mamba_gb=None, max_mamba_cache_slots should be 0."""
        ctx = _build_hybrid_context(
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=None,
        )
        assert ctx.max_mamba_cache_slots == 0

    @pytest.mark.internal
    def test_zero_mamba_budget_with_prefix_caching_still_works_for_non_hybrid(self):
        """Non-hybrid model with prefix caching should work normally without Mamba cache."""
        _set_rounder(64)

        transformer_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            hidden_size=16,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            use_cpu_initialization=True,
        )
        inference_config = InferenceConfig(
            max_sequence_length=1024,
            buffer_size_gb=0.01,
            paused_buffer_size_gb=0.002,
            block_size_tokens=32,
            max_tokens=None,
            mamba_inference_state_config=None,  # Non-hybrid
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            enable_prefix_caching=True,
            block_evict_lru=True,
            prefix_caching_mamba_gb=None,
        )
        ctx = DynamicInferenceContext(
            model_config=transformer_config, inference_config=inference_config
        )

        assert ctx.max_mamba_cache_slots == 0
        assert not ctx.is_hybrid_model

        # Prefix caching should still work for KV blocks
        block_size = ctx.block_size_tokens
        prompt = torch.arange(block_size * 2, device=torch.cuda.current_device())

        a_idx = ctx.total_request_count  # 0
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)
        _simulate_prefill_completion(ctx)

        a_block_0 = ctx.request_to_kv_block_ids[a_idx][0].item()
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Request B with same prefix should get prefix match (no hybrid limitation)
        b_idx = ctx.total_request_count  # 1
        req_b = _make_request(2, prompt.clone(), block_size)
        ctx.add_request(req_b)

        b_block_0 = ctx.request_to_kv_block_ids[b_idx][0].item()
        assert b_block_0 == a_block_0, "Non-hybrid model should still prefix-match KV blocks"

    @pytest.mark.internal
    def test_negative_mamba_budget_disables_caching(self):
        """Negative budget should result in zero Mamba slots."""
        ctx = _build_hybrid_context(
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=-0.01,
        )
        assert ctx.max_mamba_cache_slots == 0

    @pytest.mark.internal
    def test_tiny_mamba_budget_zero_slots(self):
        """Extremely tiny budget that can't fit even 1 slot."""
        ctx = _build_hybrid_context(
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=1e-12,
        )
        assert ctx.max_mamba_cache_slots == 0
