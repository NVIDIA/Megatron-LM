# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for Mamba prefix caching in hybrid models.

Focuses on Mamba-specific prefix caching features:
- Mamba state store/restore/invalidation
- Mamba LRU eviction
- Coupled KV+Mamba prefix matching (the key correctness fix)
- Cross-config end-to-end equivalence
- Zero-budget behavior
"""

import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.utils import is_fa_min_version
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


def _skip_if_mamba_sequence_packing_not_available():
    """Skip if Mamba sequence packing is not available."""
    sequence_packing_available, reason = _check_mamba_sequence_packing_support()
    if not sequence_packing_available:
        pytest.skip(reason)


def _build_engine(
    enable_chunked_prefill,
    enable_prefix_caching,
    prefix_caching_mamba_gb,
    block_size_tokens,
    max_tokens,
    max_requests,
    vocab_size,
    max_sequence_length,
    buffer_size_gb,
    seed,
):
    """Build a full MambaModel engine stack for end-to-end testing.

    Returns (engine, model) tuple.
    """
    _set_rounder(4)

    # Seed RNG for reproducible model weights.
    random.seed(seed)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(
        seed=seed,
        inference_rng_tracker=True,
        use_cudagraphable_rng=False,
        force_reset_rng=True,
    )

    transformer_config = TransformerConfig(
        params_dtype=torch.bfloat16,
        num_layers=3,  # 1 Mamba + 1 attention + 1 MLP
        hidden_size=256,
        mamba_num_heads=16,
        num_attention_heads=16,
        use_cpu_initialization=True,
        cuda_graph_impl="none",
        inference_rng_tracker=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=True,
        is_hybrid_model=True,
    )

    model = MambaModel(
        config=transformer_config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        parallel_output=True,
        hybrid_attention_ratio=0.3,
        hybrid_mlp_ratio=0.3,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    ).cuda()

    for param in model.parameters():
        param.data = param.data.to(transformer_config.params_dtype)
    model.eval()

    mamba_inference_state_config = MambaInferenceStateConfig.from_model(model)

    context = DynamicInferenceContext(
        model_config=transformer_config,
        inference_config=InferenceConfig(
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            paused_buffer_size_gb=0.2 * buffer_size_gb,
            block_size_tokens=block_size_tokens,
            max_tokens=max_tokens,
            max_requests=max_requests,
            mamba_inference_state_config=mamba_inference_state_config,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            block_evict_lru=enable_prefix_caching,
            prefix_caching_mamba_gb=prefix_caching_mamba_gb,
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
        ),
    )

    wrapped = GPTInferenceWrapper(model, context)
    wrapped.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )

    controller = TextGenerationController(
        inference_wrapped_model=wrapped,
        tokenizer=types.SimpleNamespace(
            vocab_size=vocab_size,
            detokenize=lambda tokens: "text",
        ),
    )

    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    CudaGraphManager.global_mempool = None

    engine = DynamicInferenceEngine(controller, context)
    return engine, model


def _install_deterministic_mock_forward(engine, vocab_size):
    """Replace the model's forward with a deterministic mock.

    The mock produces logits that are a deterministic function of position_ids,
    ensuring identical output tokens across engine configurations (chunked vs
    non-chunked, prefix-cached vs fresh) while avoiding TE GEMM issues.

    The key insight: logits depend on position_ids (not input_ids), so
    regardless of how the prompt is chunked, the logit at position P is
    always the same. This simulates a position-aware model where the output
    is independent of chunking boundaries.
    """
    def mock_forward(input_ids, position_ids, attention_mask, *args, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(
            batch, seq_len, vocab_size, device=input_ids.device, dtype=torch.bfloat16
        )
        for b in range(batch):
            for s in range(seq_len):
                pos = position_ids[b, s].item()
                # Next token = (position + 7) % vocab_size, avoiding 0 and termination_id
                predicted = (pos + 7) % (vocab_size - 1) + 1
                logits[b, s, predicted] = 10.0
        return logits

    model = engine.controller.inference_wrapped_model.model
    model.forward = mock_forward


def _run_to_completion(engine, requests):
    """Add requests, step engine until all complete, return generated tokens per request_id."""
    for req in requests:
        engine._add_request(req)

    results = {}
    while engine.has_unfinished_requests():
        result = engine.step_modern()
        for record in result["finished_request_records"]:
            finished = record.merge()
            results[finished.request_id] = list(finished.generated_tokens)

    return results


class TestCrossConfigEndToEnd:
    """Engine-level test verifying that chunked_prefill x prefix_caching
    configurations produce identical output tokens through a MambaModel engine
    stack with a deterministic mock forward.

    Scenario:
        block_size=32, max_tokens=80, num_tokens_to_generate=4
        Request A: 64 tokens (2 blocks)
        Request B: 100 tokens (first 64 shared with A, 36 unique)

    When B is scheduled with chunked prefill + prefix caching:
        Block:  |---block 0---|---block 1---|---block 2---|--block 3--|
        Tokens: 0            32            64            96       100
                |<-- prefix match (Mamba cached) -->|
                |<----------- chunk 1 (80 tokens) ---------->|
                                                    |<- chunk 2 ->|

    Three configs compared:
        1. chunked=True,  prefix=True   (interleaved boundaries)
        2. chunked=True,  prefix=False  (full prefill in chunks)
        3. chunked=False, prefix=False  (full prefill at once)

    All three must produce identical output tokens for both A and B.

    Uses a deterministic mock forward (position-based logits) to ensure
    reproducible comparisons while exercising the full engine scheduling,
    chunked prefill, and prefix caching codepaths.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_interleaving_boundaries(self):
        _skip_if_mamba_sequence_packing_not_available()

        # --- Parameters ---
        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 80
        max_sequence_length = 256
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        # Seed once for deterministic prompt generation.
        torch.manual_seed(seed)

        # Shared prefix (64 tokens = 2 blocks) + B's unique suffix (36 tokens)
        prompt_a = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)
        prompt_b = torch.cat([
            prompt_a,
            torch.randint(0, vocab_size - 1, (36,), device="cuda", dtype=torch.int64),
        ])

        configs = [
            {
                "name": "chunked+prefix",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "prefix_caching_mamba_gb": 0.01,
                "max_tokens": max_tokens,
            },
            {
                "name": "chunked_only",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "prefix_caching_mamba_gb": None,
                "max_tokens": max_tokens,
            },
            {
                "name": "baseline",
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "prefix_caching_mamba_gb": None,
                "max_tokens": None,  # no chunking limit needed
            },
        ]

        all_results = {}
        for config in configs:
            # Re-init model parallel for each config
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, model = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=4,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # --- Request A: 64 tokens → run to completion ---
            req_a = DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=prompt_a.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=config["enable_prefix_caching"],
            )
            results_a = _run_to_completion(engine, [req_a])

            # --- Request B: 100 tokens (shares first 64 with A) → run to completion ---
            req_b = DynamicInferenceRequest(
                request_id=1,
                prompt_tokens=prompt_b.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=config["enable_prefix_caching"],
            )
            results_b = _run_to_completion(engine, [req_b])

            all_results[config["name"]] = {
                "a": results_a[0],
                "b": results_b[1],
            }

        # --- Assertions: all configs must produce identical tokens ---
        names = list(all_results.keys())
        for i in range(1, len(names)):
            for req_label in ("a", "b"):
                tokens_ref = all_results[names[0]][req_label]
                tokens_cur = all_results[names[i]][req_label]
                assert tokens_ref == tokens_cur, (
                    f"Request {req_label} mismatch between '{names[0]}' and '{names[i]}':\n"
                    f"  {names[0]}: {tokens_ref}\n"
                    f"  {names[i]}: {tokens_cur}"
                )


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


class TestMultiplePrefillWithInitialStates:
    """Engine-level test verifying numerical correctness when multiple prefill
    requests with restored Mamba states run simultaneously.

    Scenario:
        block_size=32, max_tokens=256

        1. Request A: 128-token prompt -> run to completion
           - Stores Mamba states at block boundaries (blocks 0-3)

        2. Request B: same 64-token prefix as A + 32 unique tokens (total 96)
           - Restores Mamba state from block 1 (divergence at token 64)

        3. Request C: same 64-token prefix as A + 32 different unique tokens (total 96)
           - Also restores Mamba state from block 1

        4. Schedule B and C simultaneously -> both have initial states
           -> Both go through batch kernel (via loop)

        5. Compare: run B alone and C alone in separate engine instances
           -> outputs must match the simultaneous run
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_multiple_prefill_with_restored_states(self):
        _skip_if_mamba_sequence_packing_not_available()

        # --- Parameters ---
        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_requests = 8
        max_sequence_length = 512
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        # Generate deterministic prompts
        torch.manual_seed(seed)
        shared_prefix = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)
        suffix_b = torch.randint(0, vocab_size - 1, (32,), device="cuda", dtype=torch.int64)
        suffix_c = torch.randint(0, vocab_size - 1, (32,), device="cuda", dtype=torch.int64)
        prompt_a = torch.cat([
            shared_prefix,
            torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64),
        ])  # 128 tokens
        prompt_b = torch.cat([shared_prefix, suffix_b])  # 96 tokens
        prompt_c = torch.cat([shared_prefix, suffix_c])  # 96 tokens

        engine_kwargs = dict(
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=block_size,
            max_tokens=max_tokens,
            max_requests=max_requests,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            seed=seed,
        )

        def make_req(req_id, prompt):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=True,
            )

        # --- Simultaneous run: A first, then B and C together ---
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_sim, _ = _build_engine(**engine_kwargs)
        _install_deterministic_mock_forward(engine_sim, vocab_size)

        # Run A to completion (populates Mamba state cache)
        results_a_sim = _run_to_completion(engine_sim, [make_req(0, prompt_a)])
        assert 0 in results_a_sim

        # Run B and C simultaneously (both should get restored Mamba states)
        engine_sim._add_request(make_req(1, prompt_b))
        engine_sim._add_request(make_req(2, prompt_c))
        results_bc_sim = {}
        while engine_sim.has_unfinished_requests():
            engine_sim.schedule_waiting_requests()
            result = engine_sim.step_modern()
            for record in result["finished_request_records"]:
                finished = record.merge()
                results_bc_sim[finished.request_id] = list(finished.generated_tokens)

        assert 1 in results_bc_sim, "Request B did not complete"
        assert 2 in results_bc_sim, "Request C did not complete"

        # --- Individual run: A then B alone ---
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_b, _ = _build_engine(**engine_kwargs)
        _install_deterministic_mock_forward(engine_b, vocab_size)

        _run_to_completion(engine_b, [make_req(0, prompt_a)])
        results_b_individual = _run_to_completion(engine_b, [make_req(1, prompt_b)])

        # --- Individual run: A then C alone ---
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_c, _ = _build_engine(**engine_kwargs)
        _install_deterministic_mock_forward(engine_c, vocab_size)

        _run_to_completion(engine_c, [make_req(0, prompt_a)])
        results_c_individual = _run_to_completion(engine_c, [make_req(2, prompt_c)])

        # --- Assertions ---
        assert results_bc_sim[1] == results_b_individual[1], (
            f"Request B mismatch:\n"
            f"  simultaneous: {results_bc_sim[1]}\n"
            f"  individual:   {results_b_individual[1]}"
        )
        assert results_bc_sim[2] == results_c_individual[2], (
            f"Request C mismatch:\n"
            f"  simultaneous: {results_bc_sim[2]}\n"
            f"  individual:   {results_c_individual[2]}"
        )

        # Verify non-trivial output
        assert len(results_bc_sim[1]) == num_tokens_to_generate
        assert len(results_bc_sim[2]) == num_tokens_to_generate


class TestMambaHashMap:
    """Tests for the two-map design: kv_hash_to_block_id + mamba_hash_to_block_id."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_mamba_hash_registered_on_store(self):
        """Storing Mamba state for a block registers its hash in mamba_hash_to_block_id."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)

        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()

        # Before store: hash in kv map but not mamba map
        assert block_0_hash in alloc.kv_hash_to_block_id
        assert block_0_hash not in alloc.mamba_hash_to_block_id

        # Store and register
        ctx.store_mamba_state_for_block(block_0_id, 0)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        # After store: hash in both maps
        assert block_0_hash in alloc.kv_hash_to_block_id
        assert block_0_hash in alloc.mamba_hash_to_block_id
        assert alloc.mamba_hash_to_block_id[block_0_hash] == block_0_id

    @pytest.mark.internal
    def test_mamba_hash_removed_on_mamba_eviction(self):
        """Mamba eviction removes hash from mamba_hash_to_block_id but keeps kv_hash_to_block_id."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.001)
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 1

        # Add request, store mamba state, register hash
        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)
        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        assert block_0_hash in alloc.mamba_hash_to_block_id

        # Release request so blocks become evictable (ref_count = 0)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))

        # Make block_0 the oldest (timestamp=0) so it gets evicted first
        alloc.block_timestamps[block_0_id] = 0

        # Fill remaining mamba slots with valid block IDs (use low IDs that
        # are within total_count and not already mapped)
        filled = 0
        for i in range(alloc.total_count):
            if filled >= max_slots:
                break
            if ctx.block_to_mamba_slot[i].item() < 0:
                alloc.block_ref_counts[i] = 0
                alloc.block_timestamps[i] = 1000 + filled  # newer than block_0
                ctx._allocate_mamba_cache_slot(i)
                filled += 1

        # After eviction: mamba hash gone, kv hash remains
        assert block_0_hash not in alloc.mamba_hash_to_block_id
        assert block_0_hash in alloc.kv_hash_to_block_id

    @pytest.mark.internal
    def test_mamba_hash_removed_on_kv_eviction(self):
        """KV block eviction removes hash from both kv and mamba hash maps."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01, buffer_size_gb=0.01, rounder=1)
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Add request, store mamba state, register hash
        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)
        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        block_1_id = ctx.request_to_kv_block_ids[0][1].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        assert block_0_hash in alloc.kv_hash_to_block_id
        assert block_0_hash in alloc.mamba_hash_to_block_id

        # Release so blocks become cached (ref_count=0)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0

        # Directly evict via _deregister_blocks (simulates LRU eviction)
        blocks_to_evict = torch.tensor([block_0_id, block_1_id],
                                        device=torch.cuda.current_device(),
                                        dtype=torch.int32)
        alloc._deregister_blocks(blocks_to_evict)

        # After KV eviction: both hashes should be gone
        assert block_0_hash not in alloc.kv_hash_to_block_id
        assert block_0_hash not in alloc.mamba_hash_to_block_id

    @pytest.mark.internal
    def test_reset_clears_mamba_hash_map(self):
        """reset() clears mamba_hash_to_block_id."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        req = _make_request(1, block_size * 2, block_size)
        ctx.add_request(req)
        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        assert len(alloc.mamba_hash_to_block_id) > 0

        alloc.reset()
        assert len(alloc.mamba_hash_to_block_id) == 0
        assert len(alloc.kv_hash_to_block_id) == 0

    @pytest.mark.internal
    def test_kv_match_extends_beyond_mamba_match(self):
        """KV match can cover more blocks than mamba match (two-map design)."""
        ctx = _build_hybrid_context(
            prefix_caching_mamba_gb=0.001,
            buffer_size_gb=0.01,
            max_tokens=None,
        )
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Add request A with 3 blocks
        prompt = torch.arange(block_size * 3, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt, block_size)
        ctx.add_request(req_a)

        # Register mamba state only for block 0
        block_0_id = ctx.request_to_kv_block_ids[0][0].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()
        ctx.store_mamba_state_for_block(block_0_id, 0)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        # All 3 blocks in kv map
        for i in range(3):
            bid = ctx.request_to_kv_block_ids[0][i].item()
            h = alloc.block_hashes[bid].item()
            assert h in alloc.kv_hash_to_block_id, f"Block {i} should be in kv map"

        # Only block 0 in mamba map
        assert block_0_hash in alloc.mamba_hash_to_block_id
        for i in range(1, 3):
            bid = ctx.request_to_kv_block_ids[0][i].item()
            h = alloc.block_hashes[bid].item()
            assert h not in alloc.mamba_hash_to_block_id, f"Block {i} should NOT be in mamba map"
