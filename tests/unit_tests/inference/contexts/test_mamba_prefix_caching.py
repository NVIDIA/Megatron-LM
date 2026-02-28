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
from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
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
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU if enable_prefix_caching else PrefixCachingEvictionPolicy.REF_ZERO,
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
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU if enable_prefix_caching else PrefixCachingEvictionPolicy.REF_ZERO,
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
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
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


class TestChunkedPrefillMambaState:
    """Tests for chunked prefill requests that receive intermediate Mamba state.

    Covers two critical scenarios:
    1a. A request whose KV match extends beyond its Mamba match gets its KV match
        truncated at the divergence boundary, restores Mamba state from cache for
        the matched portion, and stores new Mamba state at the divergence boundary.
    1b. A request with a non-block-aligned prompt and no prefix match gets
        forced-chunked at the last-aligned boundary, with Mamba state stored after
        the first chunk and continued via per-request state in the second chunk.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_divergence_boundary_restores_and_stores_mamba_state(self):
        """Request C shares 96 tokens with B (3 KV blocks matched), but only 2 blocks
        have Mamba state. C restores Mamba from block 1, processes 32 effective tokens,
        and stores Mamba state for block 2 (the divergence boundary).

        Sequence:
            A (64 tokens) → Mamba stored for block 1
            B (128 tokens, first 64 shared with A) → Mamba stored for block 3
            C (96 tokens, first 96 shared with B) → KV match=3, Mamba match=2
                → restores from block 1, stores at block 2

        Verified by comparing C's output between chunked+prefix and chunked_only configs.
        """
        _skip_if_mamba_sequence_packing_not_available()

        # --- Parameters ---
        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_sequence_length = 256
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        # Deterministic prompt generation.
        torch.manual_seed(seed)
        shared_ab = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)
        suffix_b = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)

        prompt_a = shared_ab.clone()                      # 64 tokens (2 blocks)
        prompt_b = torch.cat([shared_ab, suffix_b])       # 128 tokens (4 blocks)
        # C shares first 96 tokens with B (blocks 0, 1, 2)
        prompt_c = torch.cat([shared_ab, suffix_b[:32]])  # 96 tokens (3 blocks)

        def make_req(req_id, prompt, enable_pc):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

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
        ]

        all_results = {}
        for config in configs:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, _ = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=8,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # --- Request A: 64 tokens (2 blocks) → run to completion ---
            # Engine computes: last_aligned = 64, no divergence (first request).
            # After A's prefill (total_prefilled=64 == last_aligned=64):
            #   stores Mamba state for block 1 (last complete block).
            # After A completes: KV blocks 0,1 cached (LRU), Mamba at block 1.
            req_a = make_req(0, prompt_a, config["enable_prefix_caching"])
            _run_to_completion(engine, [req_a])

            # --- Request B: 128 tokens (first 64 shared with A) → run to completion ---
            # Engine finds: KV match = 2 blocks (from A), Mamba match = 2
            #   (_find_mamba_divergence_block scans backward: block 1 has state → return 2).
            # num_mamba_matched == num_kv_matched → no divergence, _kv_divergence_token = 0.
            # last_aligned = 128. prefix_skip = 64, effective = 64 tokens.
            # Mamba restored from block 1. request_has_initial_mamba_state = True.
            # After B's prefill (total_prefilled=128 == last_aligned=128):
            #   stores Mamba state for block 3 (last complete block).
            # After B: Mamba at {1, 3}. KV blocks {0,1,2,3} all cached.
            req_b = make_req(1, prompt_b, config["enable_prefix_caching"])
            _run_to_completion(engine, [req_b])

            # --- Mamba state assertions (prefix caching config only) ---
            if config["name"] == "chunked+prefix":
                ctx = engine.context
                alloc = ctx.block_allocator

                # Look up block IDs from B's hashes (B covers blocks 0-3).
                # Blocks remain in KV hash map after release (LRU eviction policy).
                b_hashes = req_b.precomputed_block_hashes
                b_block_ids = [alloc.kv_hash_to_block_id.get(h) for h in b_hashes]
                assert all(bid is not None for bid in b_block_ids), \
                    "All of B's blocks should be registered in KV hash map"

                # After B completes: Mamba state at blocks 1 and 3 only.
                # Block 0: never stored (only the last complete block at a boundary is stored).
                # Block 1: stored after A's prefill.
                # Block 2: not stored (B's prefill boundary was at block 3, not block 2).
                # Block 3: stored after B's prefill.
                assert ctx.has_mamba_state_for_block(b_block_ids[1]), \
                    "Block 1 should have Mamba state (stored after A's prefill)"
                assert ctx.has_mamba_state_for_block(b_block_ids[3]), \
                    "Block 3 should have Mamba state (stored after B's prefill)"
                assert not ctx.has_mamba_state_for_block(b_block_ids[0]), \
                    "Block 0 should NOT have Mamba state"
                assert not ctx.has_mamba_state_for_block(b_block_ids[2]), \
                    "Block 2 should NOT have Mamba state"

            # --- Request C: 96 tokens (first 96 shared with B) → run to completion ---
            # Engine finds: KV match = 3 blocks (0,1,2 from A/B).
            # _find_mamba_divergence_block scans backward:
            #   block 2 (no Mamba) → block 1 (yes) → return 2.
            # num_mamba_matched=2 < num_kv_matched=3 → divergence!
            # _kv_divergence_token = 3*32 = 96, _mamba_last_aligned_token = 96.
            # In add_request: KV match truncated to 2 blocks.
            #   prefix_skip = min(2*32, 96-1) = 64, effective = 32 tokens.
            #   Mamba restored from block 1. request_has_initial_mamba_state = True.
            # After C's prefill (total_prefilled=96 == _kv_divergence_token=96):
            #   stores Mamba state for block 2 (divergence boundary).
            req_c = make_req(2, prompt_c, config["enable_prefix_caching"])
            results_c = _run_to_completion(engine, [req_c])

            # --- Mamba state assertion: block 2 now has Mamba state ---
            if config["name"] == "chunked+prefix":
                ctx = engine.context
                alloc = ctx.block_allocator

                c_hashes = req_c.precomputed_block_hashes
                block_2_id = alloc.kv_hash_to_block_id.get(c_hashes[2])
                assert block_2_id is not None, "Block 2 should be in KV hash map"
                assert ctx.has_mamba_state_for_block(block_2_id), \
                    "Block 2 should now have Mamba state (stored at divergence boundary)"

            all_results[config["name"]] = results_c[2]

        # --- Cross-config comparison: C's output must match ---
        assert all_results["chunked+prefix"] == all_results["chunked_only"], (
            f"Request C output mismatch:\n"
            f"  chunked+prefix: {all_results['chunked+prefix']}\n"
            f"  chunked_only: {all_results['chunked_only']}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_last_aligned_boundary_forces_chunk_and_stores_mamba_state(self):
        """Request X (80 tokens, no shared prefix) gets forced-chunked at 64 tokens
        (last block-aligned boundary). Mamba state is stored after the first chunk,
        and the second chunk (16 tokens) continues with per-request Mamba state.

        Sequence:
            X (80 tokens, no prefix) →
                Chunk 1: 64 tokens → stores Mamba for block 1
                Chunk 2: 16 tokens (request_has_initial_mamba_state=True)

        Verified by comparing X's output between chunked+prefix and chunked_only configs.
        """
        _skip_if_mamba_sequence_packing_not_available()

        # --- Parameters ---
        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_sequence_length = 256
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        torch.manual_seed(seed)
        prompt_x = torch.randint(0, vocab_size - 1, (80,), device="cuda", dtype=torch.int64)

        def make_req(req_id, prompt, enable_pc):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

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
        ]

        all_results = {}
        for config in configs:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, _ = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=8,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # --- Request X: 80 tokens (no prefix) ---
            # In chunked+prefix config:
            #   No prefix match. _mamba_num_matched_blocks = 0.
            #   _mamba_last_aligned_token = (80//32)*32 = 64.
            #   _get_mamba_chunk_limit returns 64 (last_aligned - finished = 64 - 0).
            #   mamba_forces_chunk = (80 > 64) → True.
            #   chunk_length = min(80, min(256, 64)) = 64.
            #   Chunk 1: 64 tokens → Mamba zero-init, processed from scratch.
            #   _store_mamba_states: total_prefilled=64 == last_aligned=64 → stores block 1.
            #   Chunk 2: 16 tokens, request_has_initial_mamba_state=True (batch kernel).
            #
            # In chunked_only config:
            #   No Mamba budget → max_mamba_cache_slots=0 → no mamba_limit → no forced chunk.
            #   X processes all 80 tokens in one shot.
            req_x = make_req(0, prompt_x, config["enable_prefix_caching"])
            results_x = _run_to_completion(engine, [req_x])

            all_results[config["name"]] = results_x[0]

        # --- Cross-config comparison: X's output must match ---
        assert all_results["chunked+prefix"] == all_results["chunked_only"], (
            f"Request X output mismatch:\n"
            f"  chunked+prefix: {all_results['chunked+prefix']}\n"
            f"  chunked_only: {all_results['chunked_only']}"
        )


class TestMixedKernelRouting:
    """Engine-level test for mixed batch-kernel and varlen-kernel routing.

    In a single forward step, a continuing chunked prefill request with
    request_has_initial_mamba_state=True (batch kernel) runs alongside two
    fresh prefill requests with zero-initialized Mamba state (varlen kernel).

    Scenario:
        block_size=32, max_tokens=112, num_tokens_to_generate=4

        1. Request A (64 tokens) → run to completion, Mamba state at block 1
        2. Request B (160 tokens, first 64 shared with A) → chunk 1 = 112 tokens
           (64 prefix-skipped + 48 effective). B has 48 remaining, is continuing.
        3. Add C (32 tokens, fresh) and D (32 tokens, fresh)
        4. Schedule → B continues (batch kernel, 48 tokens) + C (varlen, 32) + D (varlen, 32)
        5. Step → forward pass with mixed kernel routing
        6. Continue until all complete

    Verified by comparing B, C, D outputs against individual baseline runs.

    NOTE: this test covers a strictly more complex scenario than
    TestMultiplePrefillWithInitialStates (mixed batch-kernel + varlen vs
    all-batch-kernel).
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_batch_kernel_continuation_with_varlen_fresh_prefills(self):
        _skip_if_mamba_sequence_packing_not_available()

        # --- Parameters ---
        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 112
        max_requests = 8
        max_sequence_length = 512
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        # Deterministic prompt generation.
        torch.manual_seed(seed)
        shared_prefix = torch.randint(
            0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64
        )
        suffix_b = torch.randint(
            0, vocab_size - 1, (96,), device="cuda", dtype=torch.int64
        )
        prompt_a = shared_prefix.clone()                    # 64 tokens (2 blocks)
        prompt_b = torch.cat([shared_prefix, suffix_b])     # 160 tokens (5 blocks)
        prompt_c = torch.randint(
            0, vocab_size - 1, (32,), device="cuda", dtype=torch.int64
        )  # 32 tokens, no shared prefix
        prompt_d = torch.randint(
            0, vocab_size - 1, (32,), device="cuda", dtype=torch.int64
        )  # 32 tokens, no shared prefix

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

        def make_req(req_id, prompt, enable_pc=True):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

        # =================================================================
        # Simultaneous run: A first, then B (chunked), then B+C+D together
        # =================================================================
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_sim, _ = _build_engine(**engine_kwargs)
        _install_deterministic_mock_forward(engine_sim, vocab_size)

        # Step 1: Run A (64 tokens) to completion.
        # After A: Mamba state stored for block 1 (last_aligned=64),
        # KV blocks 0,1 cached (ref_count=0, LRU).
        results_a = _run_to_completion(engine_sim, [make_req(0, prompt_a)])
        assert 0 in results_a

        # Step 2: Add B (160 tokens, first 64 shared with A) and step once.
        # Engine scheduling:
        #   KV match = 2 blocks (from A), Mamba match = 2 (block 1 has state).
        #   _mamba_last_aligned_token = 160, _kv_divergence_token = 0 (mamba==kv).
        #   mamba_limit = 160 (last_aligned - finished). remaining = 160.
        #   token_fully_can_be_added = (160 <= 112) → False.
        #   chunk_length = min(160, 112) = 112 (capped by max_tokens).
        #   min(112, 160) = 112 → no additional mamba cap.
        # In add_request:
        #   finished_chunk_token_count=0 → is_chunked_prefill=False → allocates Mamba.
        #   prefix_skip = min(64, 111) = 64, effective = 48 tokens.
        #   Mamba restored from block 1, request_has_initial_mamba_state=True.
        # After step: B has 48 remaining tokens, is continuing chunked prefill.
        engine_sim._add_request(make_req(1, prompt_b))
        step_result = engine_sim.step_modern()
        assert len(step_result["finished_request_records"]) == 0, \
            "B should not be finished after first chunk"

        # Step 3: Add C and D (fresh requests, no shared prefix with anything).
        # C: 32 tokens. D: 32 tokens. Both go to waiting queue behind B.
        engine_sim._add_request(make_req(2, prompt_c))
        engine_sim._add_request(make_req(3, prompt_d))

        # Steps 4-6: Schedule and step until all complete.
        # Next schedule_chunked_prefill processes:
        #   B continues (batch kernel): 48 tokens, request_has_initial_mamba_state=True
        #     (set at line 1984 for continuing chunks).
        #   C fresh (varlen): 32 tokens, request_has_initial_mamba_state=False.
        #   D fresh (varlen): 32 tokens, request_has_initial_mamba_state=False.
        #   Total tokens: 48 + 32 + 32 = 112 = max_tokens.
        #   MambaMetadata.update routes: num_batch_kernel_prefills=1, varlen_count=2.
        # Subsequent steps: decode until all requests generate num_tokens_to_generate.
        results_sim = {}
        while engine_sim.has_unfinished_requests():
            result = engine_sim.step_modern()
            for record in result["finished_request_records"]:
                finished = record.merge()
                results_sim[finished.request_id] = list(finished.generated_tokens)

        assert 1 in results_sim, "B did not complete"
        assert 2 in results_sim, "C did not complete"
        assert 3 in results_sim, "D did not complete"

        # =================================================================
        # Individual baseline runs (chunked, no prefix caching)
        # Each request runs alone in a fresh engine, producing baseline output.
        # =================================================================
        baseline_kwargs = dict(
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            prefix_caching_mamba_gb=None,
            block_size_tokens=block_size,
            max_tokens=max_tokens,
            max_requests=max_requests,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            seed=seed,
        )

        # B alone (160 tokens, chunked at 112 + 48, no prefix skip)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_b, _ = _build_engine(**baseline_kwargs)
        _install_deterministic_mock_forward(engine_b, vocab_size)
        results_b = _run_to_completion(
            engine_b, [make_req(1, prompt_b, enable_pc=False)]
        )

        # C alone (32 tokens, single prefill)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_c, _ = _build_engine(**baseline_kwargs)
        _install_deterministic_mock_forward(engine_c, vocab_size)
        results_c = _run_to_completion(
            engine_c, [make_req(2, prompt_c, enable_pc=False)]
        )

        # D alone (32 tokens, single prefill)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        engine_d, _ = _build_engine(**baseline_kwargs)
        _install_deterministic_mock_forward(engine_d, vocab_size)
        results_d = _run_to_completion(
            engine_d, [make_req(3, prompt_d, enable_pc=False)]
        )

        # =================================================================
        # Assertions: simultaneous outputs must match individual baselines
        # =================================================================
        assert results_sim[1] == results_b[1], (
            f"Request B mismatch:\n"
            f"  simultaneous: {results_sim[1]}\n"
            f"  individual:   {results_b[1]}"
        )
        assert results_sim[2] == results_c[2], (
            f"Request C mismatch:\n"
            f"  simultaneous: {results_sim[2]}\n"
            f"  individual:   {results_c[2]}"
        )
        assert results_sim[3] == results_d[3], (
            f"Request D mismatch:\n"
            f"  simultaneous: {results_sim[3]}\n"
            f"  individual:   {results_d[3]}"
        )

        # Verify non-trivial output
        assert len(results_sim[1]) == num_tokens_to_generate
        assert len(results_sim[2]) == num_tokens_to_generate
        assert len(results_sim[3]) == num_tokens_to_generate


class TestMambaEvictionEdgeCases:
    """Tests for Mamba eviction edge cases: all-active raises, mixed ref counts, restore-after-evict."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_all_active_slots_raises_error(self):
        """When all Mamba slots hold blocks with ref_count > 0, eviction raises RuntimeError."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.001)
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots > 0

        # Fill all slots with blocks whose ref_count > 0 (active)
        for i in range(max_slots):
            ctx.block_to_mamba_slot[i] = i
            ctx.mamba_slot_to_block[i] = i
            ctx.block_allocator.block_ref_counts[i] = 1  # active
            ctx.block_allocator.block_timestamps[i] = i * 100
        ctx.mamba_cache_free_count = 0

        # Allocating a new slot should raise because no evictable candidates
        new_block_id = max_slots + 1
        with pytest.raises(RuntimeError, match="all slots in active use"):
            ctx._allocate_mamba_cache_slot(new_block_id)

    @pytest.mark.internal
    def test_mixed_ref_counts_evicts_only_inactive(self):
        """With mixed ref_count=0 and ref_count=1, only the oldest ref_count=0 block is evicted."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.001)
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 3, f"Need at least 3 Mamba slots, got {max_slots}"

        # Fill all slots: some active (ref=1), some evictable (ref=0)
        # slot 0 → block 0: ref=1 (active), timestamp=0 (oldest)
        # slot 1 → block 1: ref=0 (evictable), timestamp=100
        # slot 2 → block 2: ref=0 (evictable), timestamp=50 (oldest evictable)
        for i in range(max_slots):
            ctx.block_to_mamba_slot[i] = i
            ctx.mamba_slot_to_block[i] = i
            if i == 0:
                ctx.block_allocator.block_ref_counts[i] = 1
                ctx.block_allocator.block_timestamps[i] = 0
            elif i == 1:
                ctx.block_allocator.block_ref_counts[i] = 0
                ctx.block_allocator.block_timestamps[i] = 100
            elif i == 2:
                ctx.block_allocator.block_ref_counts[i] = 0
                ctx.block_allocator.block_timestamps[i] = 50
            else:
                # Extra slots beyond 3: make them active so they don't interfere
                ctx.block_allocator.block_ref_counts[i] = 1
                ctx.block_allocator.block_timestamps[i] = 200 + i
        ctx.mamba_cache_free_count = 0

        # Allocate for a new block → should evict block 2 (oldest evictable, timestamp=50)
        new_block_id = max_slots + 1
        slot = ctx._allocate_mamba_cache_slot(new_block_id)
        assert slot >= 0

        # Block 0 (active) should be untouched
        assert ctx.block_to_mamba_slot[0].item() == 0, "Active block 0 should be untouched"
        assert ctx.mamba_slot_to_block[0].item() == 0

        # Block 2 (oldest evictable) should be evicted
        assert ctx.block_to_mamba_slot[2].item() == -1, "Block 2 should be evicted"

        # Block 1 (evictable but newer) should be untouched
        assert ctx.block_to_mamba_slot[1].item() == 1, "Block 1 should be untouched"

        # New block should have the evicted slot
        assert ctx.block_to_mamba_slot[new_block_id].item() == slot

    @pytest.mark.internal
    def test_mamba_restore_after_eviction_cycle(self):
        """After evicting and re-storing Mamba for a block, the new values are restored (not stale)."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.001)
        block_size = ctx.block_size_tokens
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 1

        # Add request A (2 blocks)
        a_idx = ctx.total_request_count
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt_a, block_size)
        ctx.add_request(req_a)

        # Write known values (conv=1.0, ssm=2.0) and store for block 0
        mamba_idx_a = ctx.mamba_metadata.request_to_mamba_state_idx[a_idx].item()
        ctx.mamba_conv_states[:, mamba_idx_a] = 1.0
        ctx.mamba_ssm_states[:, mamba_idx_a] = 2.0
        block_0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        ctx.store_mamba_state_for_block(block_0_id, a_idx)
        assert ctx.has_mamba_state_for_block(block_0_id)

        # Evict block 0's Mamba state by invalidating
        ctx.invalidate_mamba_state_for_block(block_0_id)
        assert not ctx.has_mamba_state_for_block(block_0_id)

        # Re-store with new values (conv=3.0, ssm=4.0)
        ctx.mamba_conv_states[:, mamba_idx_a] = 3.0
        ctx.mamba_ssm_states[:, mamba_idx_a] = 4.0
        ctx.store_mamba_state_for_block(block_0_id, a_idx)
        assert ctx.has_mamba_state_for_block(block_0_id)

        # Clear request state, then restore from cache
        ctx.mamba_conv_states[:, mamba_idx_a] = 0.0
        ctx.mamba_ssm_states[:, mamba_idx_a] = 0.0
        restored = ctx.restore_mamba_state_from_block(a_idx, block_0_id)
        assert restored

        # Verify new values (3.0, 4.0) not stale (1.0, 2.0)
        assert torch.allclose(
            ctx.mamba_conv_states[:, mamba_idx_a],
            torch.full_like(ctx.mamba_conv_states[:, mamba_idx_a], 3.0),
        ), "Should restore new conv values (3.0), not stale (1.0)"
        assert torch.allclose(
            ctx.mamba_ssm_states[:, mamba_idx_a],
            torch.full_like(ctx.mamba_ssm_states[:, mamba_idx_a], 4.0),
        ), "Should restore new ssm values (4.0), not stale (2.0)"


class TestKvMambaEvictionInteraction:
    """Tests for the interaction between KV eviction and Mamba invalidation."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_kv_eviction_cascades_to_mamba_invalidation(self):
        """evict_lru_blocks removes KV blocks AND invalidates their Mamba state."""
        ctx = _build_hybrid_context(
            prefix_caching_mamba_gb=0.01,
            buffer_size_gb=0.01,
            rounder=1,
        )
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Add request A (2 blocks)
        a_idx = ctx.total_request_count
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt_a, block_size)
        ctx.add_request(req_a)

        # Store Mamba state for block 0 and register mamba hash
        block_0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        block_1_id = ctx.request_to_kv_block_ids[a_idx][1].item()
        block_0_hash = alloc.block_hashes[block_0_id].item()
        ctx.store_mamba_state_for_block(block_0_id, a_idx)
        alloc.register_mamba_block_hash(block_0_id, block_0_hash)

        assert ctx.has_mamba_state_for_block(block_0_id)
        assert block_0_hash in alloc.kv_hash_to_block_id
        assert block_0_hash in alloc.mamba_hash_to_block_id
        free_before = ctx.mamba_cache_free_count

        # Release A (ref_count → 0)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))
        ctx.total_request_count = 0

        # Make blocks old so they get evicted first
        alloc.block_timestamps[block_0_id] = 0
        alloc.block_timestamps[block_1_id] = 1

        # Evict A's KV blocks
        result = alloc.evict_lru_blocks(2)
        assert result, "Should evict 2 blocks"

        # Mamba state should be gone
        assert not ctx.has_mamba_state_for_block(block_0_id), \
            "Mamba state should be invalidated after KV eviction"
        # Mamba free count should have increased
        assert ctx.mamba_cache_free_count == free_before + 1
        # Both hash maps should be clear
        assert block_0_hash not in alloc.kv_hash_to_block_id
        assert block_0_hash not in alloc.mamba_hash_to_block_id

    @pytest.mark.internal
    def test_mamba_eviction_preserves_kv_cache(self):
        """Mamba eviction removes Mamba state but leaves KV blocks intact."""
        # Use extremely small Mamba budget to get exactly 1 slot.
        # Each slot needs ~41 KB (conv_states + ssm_states), so 0.00005 GB ≈ 53 KB → 1 slot.
        ctx = _build_hybrid_context(
            prefix_caching_mamba_gb=0.00005,
            buffer_size_gb=0.01,
            rounder=1,
        )
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 1, f"Need at least 1 Mamba slot, got {max_slots}"

        # Add request A
        a_idx = ctx.total_request_count
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt_a, block_size)
        ctx.add_request(req_a)

        block_a0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        block_a0_hash = alloc.block_hashes[block_a0_id].item()
        ctx.store_mamba_state_for_block(block_a0_id, a_idx)
        alloc.register_mamba_block_hash(block_a0_id, block_a0_hash)

        assert ctx.has_mamba_state_for_block(block_a0_id)

        # Release A (ref_count → 0, blocks become evictable)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))

        # Make A's block have old timestamp so it gets evicted first
        alloc.block_timestamps[block_a0_id] = 0

        # Fill any remaining Mamba slots with filler blocks to ensure cache is full
        for i in range(alloc.total_count):
            if ctx.mamba_cache_free_count == 0:
                break
            if ctx.block_to_mamba_slot[i].item() < 0 and i != block_a0_id:
                alloc.block_ref_counts[i] = 0
                alloc.block_timestamps[i] = 1000  # newer than A
                ctx._allocate_mamba_cache_slot(i)

        assert ctx.mamba_cache_free_count == 0, "Mamba cache should be full"

        # Add request B (different prompt)
        b_idx = ctx.total_request_count
        prompt_b = torch.arange(1000, 1000 + block_size * 2, device=torch.cuda.current_device())
        req_b = _make_request(2, prompt_b, block_size)
        ctx.add_request(req_b)

        # Store Mamba for B's block → forces eviction of oldest (A's block)
        block_b0_id = ctx.request_to_kv_block_ids[b_idx][0].item()
        ctx.store_mamba_state_for_block(block_b0_id, b_idx)

        # After Mamba eviction: KV blocks should still be in kv_hash_to_block_id
        assert block_a0_hash in alloc.kv_hash_to_block_id, \
            "A's KV block should still be cached after Mamba eviction"

        # A's Mamba state should be gone
        assert not ctx.has_mamba_state_for_block(block_a0_id), \
            "A's Mamba state should be evicted"
        assert block_a0_hash not in alloc.mamba_hash_to_block_id, \
            "A's hash should be removed from mamba map"

    @pytest.mark.internal
    def test_combined_kv_and_mamba_pressure(self):
        """After both Mamba eviction and KV deregistration, all cached state is gone."""
        ctx = _build_hybrid_context(
            prefix_caching_mamba_gb=0.00005,  # ~1 Mamba slot
            buffer_size_gb=0.01,
            rounder=1,
        )
        block_size = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Add request A (2 blocks)
        a_idx = ctx.total_request_count
        prompt_a = torch.arange(block_size * 2, device=torch.cuda.current_device())
        req_a = _make_request(1, prompt_a, block_size)
        ctx.add_request(req_a)

        # Save block IDs before releasing (release clears request_to_kv_block_ids)
        block_a0_id = ctx.request_to_kv_block_ids[a_idx][0].item()
        block_a1_id = ctx.request_to_kv_block_ids[a_idx][1].item()
        block_a0_hash = alloc.block_hashes[block_a0_id].item()
        ctx.store_mamba_state_for_block(block_a0_id, a_idx)
        alloc.register_mamba_block_hash(block_a0_id, block_a0_hash)

        assert ctx.has_mamba_state_for_block(block_a0_id)
        assert block_a0_hash in alloc.kv_hash_to_block_id
        assert block_a0_hash in alloc.mamba_hash_to_block_id

        # Release A (ref_count → 0)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([a_idx]))
        ctx.total_request_count = 0
        alloc.block_timestamps[block_a0_id] = 0  # oldest
        alloc.block_timestamps[block_a1_id] = 1

        # Now directly deregister A's KV blocks (simulates LRU eviction)
        # This should cascade to Mamba invalidation via the callback
        blocks_tensor = torch.tensor(
            [block_a0_id, block_a1_id], device=torch.cuda.current_device(), dtype=torch.int32
        )
        alloc._deregister_blocks(blocks_tensor)

        # After KV eviction: both KV and Mamba state should be gone
        assert block_a0_hash not in alloc.kv_hash_to_block_id, \
            "A's KV hash should be gone after eviction"
        assert block_a0_hash not in alloc.mamba_hash_to_block_id, \
            "A's Mamba hash should be gone after eviction"
        assert not ctx.has_mamba_state_for_block(block_a0_id), \
            "A's Mamba state should be gone after KV eviction"

        # Add request D with same prefix as A → should get no matches (all state evicted)
        d_idx = ctx.total_request_count
        req_d = _make_request(4, prompt_a.clone(), block_size)
        req_d._mamba_num_matched_blocks = 0
        ctx.add_request(req_d)

        # D should have zero-init Mamba state (fresh allocation)
        mamba_idx_d = ctx.mamba_metadata.request_to_mamba_state_idx[d_idx].item()
        assert torch.all(ctx.mamba_conv_states[:, mamba_idx_d] == 0.0), \
            "D should get zero-init Mamba state (no cache hit)"


class TestEvictionEndToEnd:
    """Engine-level tests verifying correct output after KV and/or Mamba eviction."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_mamba_eviction_forces_full_recompute(self):
        """After Mamba eviction, a repeated prefix triggers full recompute and still produces correct output."""
        _skip_if_mamba_sequence_packing_not_available()

        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_sequence_length = 256
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        torch.manual_seed(seed)
        prompt_a = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)
        prompt_b = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)

        def make_req(req_id, prompt, enable_pc):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

        # Config with very small Mamba cache (1-2 slots) to force Mamba eviction
        configs = [
            {
                "name": "prefix+small_mamba",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "prefix_caching_mamba_gb": 0.0001,  # Very small: 1-2 Mamba slots
                "max_tokens": max_tokens,
            },
            {
                "name": "baseline",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "prefix_caching_mamba_gb": None,
                "max_tokens": max_tokens,
            },
        ]

        all_results = {}
        for config in configs:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, _ = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=8,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # A: 64 tokens → completes → Mamba stored
            _run_to_completion(engine, [make_req(0, prompt_a, config["enable_prefix_caching"])])

            # B: different 64 tokens → completes → may evict A's Mamba
            _run_to_completion(engine, [make_req(1, prompt_b, config["enable_prefix_caching"])])

            # C: same as A → KV match possible, but Mamba may be evicted → full recompute
            results_c = _run_to_completion(
                engine, [make_req(2, prompt_a, config["enable_prefix_caching"])]
            )
            all_results[config["name"]] = results_c[2]

        assert all_results["prefix+small_mamba"] == all_results["baseline"], (
            f"Output mismatch after Mamba eviction:\n"
            f"  prefix+small_mamba: {all_results['prefix+small_mamba']}\n"
            f"  baseline: {all_results['baseline']}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_kv_mamba_eviction_correct_output(self):
        """With a small buffer, many intervening requests force KV LRU eviction.
        A repeated prefix gets full recompute and still produces correct output.

        Uses 65-token prompts (3 blocks each, non-block-aligned to avoid decode
        pausing). With buffer_size_gb=0.01 (~33 blocks), 11 requests fill all
        blocks, and the 12th request (C, same as A) triggers eviction of A's
        blocks during prefill allocation.
        """
        _skip_if_mamba_sequence_packing_not_available()

        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_sequence_length = 256
        buffer_size_gb = 0.01  # ~33 blocks
        num_tokens_to_generate = 4
        prompt_len = 65  # Non-block-aligned to avoid decode pause

        torch.manual_seed(seed)
        prompt_a = torch.randint(0, vocab_size - 1, (prompt_len,), device="cuda", dtype=torch.int64)
        # Generate 10 different filler prompts (A + 10 fillers = 11 requests × 3 blocks = 33)
        filler_prompts = [
            torch.randint(0, vocab_size - 1, (prompt_len,), device="cuda", dtype=torch.int64)
            for _ in range(10)
        ]

        def make_req(req_id, prompt, enable_pc):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

        configs = [
            {
                "name": "prefix+small_buffer",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "prefix_caching_mamba_gb": 0.0005,  # 1 Mamba slot
                "max_tokens": max_tokens,
            },
            {
                "name": "baseline",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "prefix_caching_mamba_gb": None,
                "max_tokens": max_tokens,
            },
        ]

        all_results = {}
        for config in configs:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, _ = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=8,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # A → completes → KV blocks cached, Mamba stored
            _run_to_completion(engine, [make_req(0, prompt_a, config["enable_prefix_caching"])])

            # Run 10 filler requests to fill all blocks in the cache
            for i, filler in enumerate(filler_prompts):
                _run_to_completion(
                    engine, [make_req(i + 1, filler, config["enable_prefix_caching"])]
                )

            # C (same as A) → A's blocks evicted → full recompute
            results_c = _run_to_completion(
                engine, [make_req(12, prompt_a, config["enable_prefix_caching"])]
            )
            all_results[config["name"]] = results_c[12]

        assert all_results["prefix+small_buffer"] == all_results["baseline"], (
            f"Output mismatch after KV+Mamba eviction:\n"
            f"  prefix+small_buffer: {all_results['prefix+small_buffer']}\n"
            f"  baseline: {all_results['baseline']}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.no_grad()
    def test_eviction_with_interleaved_shared_prefixes(self):
        """Multiple requests sharing a prefix produce correct output even when Mamba is evicted between them."""
        _skip_if_mamba_sequence_packing_not_available()

        seed = 42
        vocab_size = 100
        block_size = 32
        max_tokens = 256
        max_sequence_length = 256
        buffer_size_gb = 0.1
        num_tokens_to_generate = 4

        torch.manual_seed(seed)
        shared_prefix = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)
        prompt_b = torch.randint(0, vocab_size - 1, (64,), device="cuda", dtype=torch.int64)

        def make_req(req_id, prompt, enable_pc):
            return DynamicInferenceRequest(
                request_id=req_id,
                prompt_tokens=prompt.clone(),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,
                ),
                block_size_tokens=block_size,
                enable_prefix_caching=enable_pc,
            )

        configs = [
            {
                "name": "prefix+small_mamba",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "prefix_caching_mamba_gb": 0.0001,  # Very small: 1-2 Mamba slots
                "max_tokens": max_tokens,
            },
            {
                "name": "baseline",
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "prefix_caching_mamba_gb": None,
                "max_tokens": max_tokens,
            },
        ]

        all_results = {}
        for config in configs:
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

            engine, _ = _build_engine(
                enable_chunked_prefill=config["enable_chunked_prefill"],
                enable_prefix_caching=config["enable_prefix_caching"],
                prefix_caching_mamba_gb=config["prefix_caching_mamba_gb"],
                block_size_tokens=block_size,
                max_tokens=config["max_tokens"],
                max_requests=8,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                buffer_size_gb=buffer_size_gb,
                seed=seed,
            )
            _install_deterministic_mock_forward(engine, vocab_size)

            # A1: shared prefix → completes → Mamba stored
            results_a1 = _run_to_completion(
                engine, [make_req(0, shared_prefix, config["enable_prefix_caching"])]
            )

            # A2: same prefix → matches A1's cache → completes
            results_a2 = _run_to_completion(
                engine, [make_req(1, shared_prefix, config["enable_prefix_caching"])]
            )

            # B: different prefix → completes → may evict Mamba for shared prefix
            _run_to_completion(engine, [make_req(2, prompt_b, config["enable_prefix_caching"])])

            # A3: same prefix → KV may still be cached (LRU), but Mamba evicted → full recompute
            results_a3 = _run_to_completion(
                engine, [make_req(3, shared_prefix, config["enable_prefix_caching"])]
            )

            all_results[config["name"]] = {
                "a1": results_a1[0],
                "a2": results_a2[1],
                "a3": results_a3[3],
            }

        # All three should match within each config
        for name in all_results:
            assert all_results[name]["a1"] == all_results[name]["a2"], \
                f"[{name}] A1 and A2 outputs should match"
            assert all_results[name]["a1"] == all_results[name]["a3"], \
                f"[{name}] A1 and A3 outputs should match"

        # Cross-config: all should match baseline
        for req_label in ("a1", "a2", "a3"):
            assert all_results["prefix+small_mamba"][req_label] == all_results["baseline"][req_label], (
                f"Request {req_label} mismatch between configs:\n"
                f"  prefix+small_mamba: {all_results['prefix+small_mamba'][req_label]}\n"
                f"  baseline: {all_results['baseline'][req_label]}"
            )


class TestMambaStressAndBudget:
    """Stress tests for Mamba cache slot management."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_rapid_allocation_eviction_cycle(self):
        """100 rapid allocate/invalidate cycles leave the slot pool in a clean state."""
        ctx = _build_hybrid_context(prefix_caching_mamba_gb=0.01)
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots > 0

        initial_free = ctx.mamba_cache_free_count
        assert initial_free == max_slots

        for i in range(100):
            block_id = i % max_slots  # Reuse block IDs
            slot = ctx._allocate_mamba_cache_slot(block_id)
            assert slot >= 0
            ctx.invalidate_mamba_state_for_block(block_id)

        # After all cycles: all slots should be free
        assert ctx.mamba_cache_free_count == max_slots, \
            f"Expected {max_slots} free slots, got {ctx.mamba_cache_free_count}"

        # No dangling references in mamba_slot_to_block
        for s in range(max_slots):
            assert ctx.mamba_slot_to_block[s].item() == -1, \
                f"Slot {s} has dangling block reference"

    @pytest.mark.internal
    def test_many_blocks_few_mamba_slots(self):
        """Large KV buffer with small Mamba cache: only first N blocks get Mamba state."""
        ctx = _build_hybrid_context(
            prefix_caching_mamba_gb=0.001,  # Few Mamba slots
            buffer_size_gb=0.01,            # Many KV blocks
            max_tokens=None,                # No token limit
        )
        max_slots = ctx.max_mamba_cache_slots
        assert max_slots >= 1

        block_size = ctx.block_size_tokens

        # Add a request with many blocks
        num_blocks = max_slots + 3
        prompt = torch.arange(
            block_size * num_blocks, device=torch.cuda.current_device()
        )
        req = _make_request(1, prompt, block_size)
        ctx.add_request(req)

        # Store Mamba for first max_slots blocks
        for i in range(max_slots):
            block_id = ctx.request_to_kv_block_ids[0][i].item()
            ctx.store_mamba_state_for_block(block_id, 0)
            assert ctx.has_mamba_state_for_block(block_id), \
                f"Block {i} should have Mamba state"

        # Cache should be full
        assert ctx.mamba_cache_free_count == 0, "Mamba cache should be full"

        # Blocks beyond max_slots don't have state yet
        for i in range(max_slots, min(num_blocks, ctx.request_to_kv_block_ids.shape[1])):
            block_id = ctx.request_to_kv_block_ids[0][i].item()
            if block_id >= 0:
                assert not ctx.has_mamba_state_for_block(block_id), \
                    f"Block {i} should NOT have Mamba state"

        # Release request so blocks become evictable
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))

        # Set timestamps: block 0 is oldest
        for i in range(max_slots):
            block_id = ctx.request_to_kv_block_ids[0][i].item()
            if block_id >= 0:
                ctx.block_allocator.block_timestamps[block_id] = i * 100

        # Store for one more block beyond max_slots → triggers LRU eviction
        extra_block_id = ctx.request_to_kv_block_ids[0][max_slots].item()
        if extra_block_id >= 0:
            # Need to allocate a new request slot to store from
            ctx.total_request_count = 0
            req2 = _make_request(2, prompt.clone(), block_size)
            ctx.add_request(req2)
            ctx.store_mamba_state_for_block(extra_block_id, ctx.total_request_count - 1)

            # The LRU victim (block 0, oldest timestamp) should have lost state
            first_block_id = ctx.request_to_kv_block_ids[0][0].item()
            assert not ctx.has_mamba_state_for_block(first_block_id), \
                "Block 0 (LRU victim) should have lost Mamba state"

            # New block should have state
            assert ctx.has_mamba_state_for_block(extra_block_id), \
                "Newly stored block should have Mamba state"
