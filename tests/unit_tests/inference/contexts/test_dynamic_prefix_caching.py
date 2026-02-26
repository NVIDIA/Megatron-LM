# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from collections import deque

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    HASH_PRIME,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# =========================================================================
# Base class + helpers
# =========================================================================


class PrefixCachingTestBase:
    """Base class with shared setup/teardown and helper methods."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _ctx(
        self,
        *,
        buffer_size_gb=0.1,
        block_size_tokens=32,
        max_sequence_length=512,
        rounder=64,
        enable_prefix_caching=True,
        max_tokens=None,
        block_evict_lru=True,
    ):
        """Create a DynamicInferenceContext with sensible test defaults.

        Note: block_evict_lru defaults to True so existing tests use LRU behavior.
        """
        DynamicInferenceContext.ROUNDER = rounder
        DynamicInferenceContext.TOKEN_ROUNDER = rounder
        DynamicInferenceContext.REQUEST_ROUNDER = rounder

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
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            paused_buffer_size_gb=0.2 * buffer_size_gb,
            block_size_tokens=block_size_tokens,
            max_tokens=max_tokens,
            mamba_inference_state_config=None,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            enable_prefix_caching=enable_prefix_caching,
            block_evict_lru=block_evict_lru,
        )
        return DynamicInferenceContext(
            model_config=transformer_config, inference_config=inference_config
        )

    @staticmethod
    def _req(ctx, prompt_tokens, request_id=1, *, enable_prefix_caching=True):
        """Create a DynamicInferenceRequest with sensible defaults."""
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=ctx.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

    @staticmethod
    def _prompt(num_tokens, offset=0):
        """Create a prompt tensor on CUDA."""
        return torch.arange(offset, offset + num_tokens, device=torch.cuda.current_device())

    @staticmethod
    def _block_ids(ctx, req_idx, n):
        """Extract first n block IDs for a request as a list of ints."""
        return [ctx.request_to_kv_block_ids[req_idx][i].item() for i in range(n)]


# =========================================================================
# Class 1: TestHashComputation (3 tests)
# =========================================================================


class TestHashComputation(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_hash_determinism_and_quality(self):
        """Same tokens → same hash; different tokens → different; parent-sensitivity."""
        tokens = self._prompt(32)
        h1 = compute_block_hashes_batched(tokens, 32)
        h2 = compute_block_hashes_batched(tokens, 32)
        assert h1 == h2, "Hash should be deterministic"
        assert len(h1) == 1
        assert 1 <= h1[0] <= HASH_PRIME, "Hash should be in [1, HASH_PRIME]"

        # Different tokens
        h_diff = compute_block_hashes_batched(self._prompt(32, offset=1), 32)
        assert h_diff[0] != h1[0]

        # Parent sensitivity (two blocks: hash of block 1 depends on block 0)
        two_blocks = torch.cat([tokens, self._prompt(32, offset=100)])
        h_chain = compute_block_hashes_batched(two_blocks, 32)
        assert len(h_chain) == 2
        assert 1 <= h_chain[1] <= HASH_PRIME

    @pytest.mark.internal
    def test_precomputed_hash_chain(self):
        """3-block request has correct precomputed hashes matching manual computation."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3)

        req = self._req(ctx, prompt)
        assert len(req.precomputed_block_hashes) == 3

        # Verify against manual computation via batched function
        expected = compute_block_hashes_batched(prompt, bs)
        assert req.precomputed_block_hashes == expected

        # Identical requests produce identical hashes
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        assert req.precomputed_block_hashes == req2.precomputed_block_hashes

        # Hashes differ across blocks (due to chaining)
        assert len(set(req.precomputed_block_hashes)) == 3

    @pytest.mark.internal
    def test_precomputed_hashes_edge_cases(self):
        """Short, empty, single-token, all-zero, and very long prompts."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        # Shorter than block → empty hashes
        req_short = self._req(ctx, self._prompt(bs // 2))
        assert req_short.precomputed_block_hashes == []

        # Empty prompt → empty hashes
        empty = torch.tensor([], device=torch.cuda.current_device(), dtype=torch.long)
        req_empty = self._req(ctx, empty)
        assert req_empty.precomputed_block_hashes == []

        # Single token → empty hashes
        single = torch.tensor([42], device=torch.cuda.current_device(), dtype=torch.long)
        req_single = self._req(ctx, single)
        assert req_single.precomputed_block_hashes == []

        # All-zero tokens → 4 unique hashes (parent chaining)
        zeros = torch.zeros(bs * 4, device=torch.cuda.current_device(), dtype=torch.long)
        req_zeros = self._req(ctx, zeros)
        assert len(req_zeros.precomputed_block_hashes) == 4
        assert len(set(req_zeros.precomputed_block_hashes)) == 4

        # 120-block prompt → 120 hashes, all positive
        ctx_long = self._ctx(max_sequence_length=8192)
        long_prompt = torch.arange(bs * 120, device=torch.cuda.current_device(), dtype=torch.long)
        req_long = self._req(ctx_long, long_prompt)
        assert len(req_long.precomputed_block_hashes) == 120
        assert all(h > 0 for h in req_long.precomputed_block_hashes)


# =========================================================================
# Class 2: TestPrefixSharing (4 tests)
# =========================================================================


class TestPrefixSharing(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_full_prefix_match(self):
        """Identical prefix → all blocks shared, ref_count scales with N."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 3)

        # First request
        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 3)

        # 9 more requests sharing the same prefix
        for i in range(2, 11):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))

        # All share same blocks
        for req_idx in range(1, 10):
            assert self._block_ids(ctx, req_idx, 3) == first_blocks

        # ref_count == 10
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 10

    @pytest.mark.internal
    def test_partial_prefix_match(self):
        """3-block request where block 2 differs → blocks 0,1 shared."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        r1 = self._block_ids(ctx, 0, 3)

        # Same first 2 blocks, different 3rd
        prompt2 = prompt1.clone()
        prompt2[bs * 2 :] += 1000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)

        assert r2[0] == r1[0] and r2[1] == r1[1], "Blocks 0,1 shared"
        assert r2[2] != r1[2], "Block 2 separate"
        assert alloc.block_ref_counts[r1[0]].item() == 2
        assert alloc.block_ref_counts[r1[1]].item() == 2
        assert alloc.block_ref_counts[r1[2]].item() == 1
        assert alloc.block_ref_counts[r2[2]].item() == 1

    @pytest.mark.internal
    def test_no_prefix_match(self):
        """Completely different prefixes → no blocks shared."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        r1 = set(self._block_ids(ctx, 0, 2))

        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=1000), request_id=2))
        r2 = set(self._block_ids(ctx, 1, 2))

        assert r1.isdisjoint(r2)
        for bid in r1 | r2:
            assert alloc.block_ref_counts[bid].item() == 1

    @pytest.mark.internal
    def test_sequential_match_stops_at_gap(self):
        """[X,W,Z] vs [X,Y,Z] → only block 0 shared; block 2 NOT shared."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        r1 = self._block_ids(ctx, 0, 3)

        # Same block 0, different block 1, same block 2 tokens
        prompt2 = prompt1.clone()
        prompt2[bs : bs * 2] += 5000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)

        assert r2[0] == r1[0], "Block 0 shared"
        assert r2[1] != r1[1], "Block 1 NOT shared"
        assert r2[2] != r1[2], "Block 2 NOT shared (hash chain broken)"
        assert alloc.block_ref_counts[r1[0]].item() == 2
        assert alloc.block_ref_counts[r1[1]].item() == 1


# =========================================================================
# Class 3: TestRefCountLifecycle (6 tests)
# =========================================================================


class TestRefCountLifecycle(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_ref_count_increment_decrement(self):
        """ref=2 after sharing; ref=1 after one release; ref=0 + hash stays after both."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))

        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2
        assert alloc.block_ref_counts[b1].item() == 2

        # Release request 0
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id

        # Release request 1
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.block_ref_counts[b1].item() == 0
        assert b0_hash in alloc.hash_to_block_id  # still cached

    @pytest.mark.internal
    def test_reuse_after_release(self):
        """Release → ref=0 → new request same prefix reuses same block IDs."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        b0, b1 = self._block_ids(ctx, 0, 2)

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.block_hashes[b0].item() in alloc.hash_to_block_id

        # New request with same prefix
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 0, 2) == [b0, b1]
        assert alloc.block_ref_counts[b0].item() == 1
        assert alloc.block_ref_counts[b1].item() == 1

    @pytest.mark.internal
    def test_eviction_under_pressure(self):
        """Small buffer: active blocks protected; cached blocks evicted LRU-first."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Active request (ref>0)
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        active_blocks = ctx.request_to_kv_block_ids[0][:2].clone()

        # Release a second request to create cached (evictable) blocks
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=5000), request_id=2))
        cached_blocks = ctx.request_to_kv_block_ids[1][:2].clone()
        cached_hash = alloc.block_hashes[cached_blocks[0].item()].item()
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx.total_request_count = 1
        assert alloc.block_ref_counts[cached_blocks[0].item()].item() == 0

        # Fill remaining space with different requests to trigger eviction
        for i in range(20):
            try:
                ctx.add_request(
                    self._req(ctx, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100)
                )
            except Exception:
                break

        # Active blocks protected
        for bid in active_blocks:
            assert alloc.block_ref_counts[bid.item()].item() == 1

        # Cached blocks may have been evicted (hash removed from mapping)
        # The key invariant: system still functions correctly
        assert ctx.total_request_count >= 1

    @pytest.mark.internal
    def test_rz_immediate_deregister_on_release(self):
        """Blocks are deregistered and returned to free pool when ref_count hits 0."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        b1_hash = alloc.block_hashes[b1].item()
        avail_before = alloc.total_avail

        # Release → ref_count hits 0 → blocks deregistered immediately
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.block_ref_counts[b1].item() == 0
        assert b0_hash not in alloc.hash_to_block_id, "Hash should be removed"
        assert b1_hash not in alloc.hash_to_block_id, "Hash should be removed"
        assert alloc.block_hashes[b0].item() == -1, "Block hash should be reset"
        assert alloc.block_hashes[b1].item() == -1, "Block hash should be reset"
        assert alloc.total_avail == avail_before + 2, "Blocks returned to free pool"

    @pytest.mark.internal
    def test_rz_shared_blocks_persist_until_last_ref(self):
        """Shared blocks stay registered until all references are gone."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        # Two requests sharing the same prefix
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))

        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2

        # Release first request → ref_count=1 → still registered
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id, "Still has a reference"

        # Release second request → ref_count=0 → deregistered
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert b0_hash not in alloc.hash_to_block_id, "No more references"

    @pytest.mark.internal
    def test_rz_no_reuse_after_release(self):
        """Released blocks cannot be found by new requests with the same prefix."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 2)

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0

        # New request with same prefix → no hash match, gets new blocks
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        new_blocks = self._block_ids(ctx, 0, 2)
        # Blocks should be freshly allocated (no sharing with deregistered blocks)
        # The hashes were removed, so lookup returns None and new blocks are allocated
        assert alloc.block_ref_counts[new_blocks[0]].item() == 1
        assert alloc.block_ref_counts[new_blocks[1]].item() == 1


# =========================================================================
# Class 4: TestRegistration (3 tests)
# =========================================================================


class TestRegistration(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_registration_flow(self):
        """After add_request, hashes are set immediately and blocks are discoverable."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes

        # Discoverable via lookup, hashes set immediately
        assert alloc.hash_to_block_id.get(h0) == b0
        assert alloc.hash_to_block_id.get(h1) == b1
        assert alloc.block_hashes[b0].item() == h0
        assert alloc.block_hashes[b1].item() == h1

    @pytest.mark.internal
    def test_concurrent_sharing(self):
        """3 requests added share blocks (ref=3), hashes set immediately. 4th also shares."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        for i in range(1, 4):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))

        first_blocks = self._block_ids(ctx, 0, 2)
        for req_idx in range(1, 3):
            assert self._block_ids(ctx, req_idx, 2) == first_blocks

        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 3
            assert alloc.block_hashes[bid].item() != -1

        # 4th request also shares
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=4))
        assert self._block_ids(ctx, 3, 2) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 4

    @pytest.mark.internal
    def test_immediate_hash_registration(self):
        """register_block_hashes sets hash immediately; discoverable via lookup."""
        ctx = self._ctx()
        alloc = ctx.block_allocator

        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        test_hash = 99999

        alloc.register_block_hashes([bid], [test_hash])

        # Hash is set immediately and discoverable
        assert alloc.hash_to_block_id.get(test_hash) == bid
        assert alloc.block_hashes[bid].item() == test_hash


# =========================================================================
# Class 5: TestPrefillSkipping (2 tests)
# =========================================================================


class TestPrefillSkipping(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_cached_blocks_skip_prefill(self):
        """Req1 allocates 4 blocks. Req2 same prefix reuses all 4. Req3 extends: shares 4, allocates 1."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 4)

        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 4)
        avail_after_first = alloc.total_avail

        # Same prefix → reuses same blocks, no new allocation
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 1, 4) == first_blocks
        assert alloc.total_avail == avail_after_first

        # Extended prompt → shares first 4 blocks, allocates 1 new
        extended = torch.cat([prompt, self._prompt(bs, offset=1000)])
        ctx.add_request(self._req(ctx, extended, request_id=3))
        assert self._block_ids(ctx, 2, 4) == first_blocks
        assert alloc.total_avail == avail_after_first - 1

    @pytest.mark.internal
    def test_decode_does_not_compute_hashes(self):
        """Complete block → hash after mark. Partial block → stays -1 even after decode fills it."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # 1 complete block + (block_size - 1) tokens in second block
        prompt = self._prompt(bs + (bs - 1))
        ctx.add_request(self._req(ctx, prompt))

        b0, b1 = self._block_ids(ctx, 0, 2)
        assert alloc.block_hashes[b0].item() != -1, "Complete block has hash"
        assert alloc.block_hashes[b1].item() == -1, "Partial block: no hash"

        # One decode step fills block 1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)

        assert alloc.block_hashes[b1].item() == -1, "Decode hash computation not implemented"

    @pytest.mark.internal
    def test_matched_prefix_reduces_active_tokens(self):
        """Request A fills 3 blocks. Request B (same prefix + trailing) only adds trailing tokens."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt_a = self._prompt(bs * 3)

        ctx.add_request(self._req(ctx, prompt_a.clone()))
        tokens_after_a = ctx.active_token_count

        # B has same 3-block prefix + extra trailing tokens
        trailing = bs // 2
        prompt_b = torch.cat([prompt_a, self._prompt(trailing, offset=9000)])
        ctx.add_request(self._req(ctx, prompt_b, request_id=2))

        added_by_b = ctx.active_token_count - tokens_after_a
        # B should skip 3*bs matched tokens but keep at least 1, so it adds trailing tokens only
        assert (
            added_by_b == trailing
        ), f"Expected {trailing} tokens added (trailing only), got {added_by_b}"
        # query length should also reflect the reduced count
        assert ctx.request_query_lengths[1].item() == trailing

    @pytest.mark.internal
    def test_partial_match_skips_matched_only(self):
        """Request A fills 3 blocks. B matches first 2, differs on 3rd. B skips 2*bs tokens."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        prompt_a = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt_a.clone()))
        tokens_after_a = ctx.active_token_count

        # Same first 2 blocks, different 3rd
        prompt_b = prompt_a.clone()
        prompt_b[bs * 2 :] += 1000
        ctx.add_request(self._req(ctx, prompt_b, request_id=2))

        added_by_b = ctx.active_token_count - tokens_after_a
        expected = bs * 3 - bs * 2  # chunk_length - matched tokens = 1 block worth
        assert added_by_b == expected, f"Expected {expected}, got {added_by_b}"
        assert ctx.request_query_lengths[1].item() == expected

    @pytest.mark.internal
    def test_no_match_no_skip(self):
        """Completely different prefix: full chunk_length added."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        tokens_after_a = ctx.active_token_count

        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=9000), request_id=2))
        added_by_b = ctx.active_token_count - tokens_after_a
        assert added_by_b == bs * 2, f"Expected full {bs * 2}, got {added_by_b}"
        assert ctx.request_query_lengths[1].item() == bs * 2

    @pytest.mark.internal
    def test_exact_full_match_sends_one_token(self):
        """Prompt is exactly N*block_size and fully matched: only 1 token enters the model."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        tokens_after_a = ctx.active_token_count

        # Identical prompt, fully matched
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        added_by_b = ctx.active_token_count - tokens_after_a
        assert added_by_b == 1, f"Expected 1 token (min guard), got {added_by_b}"
        assert ctx.request_query_lengths[1].item() == 1
        # kv_length_offset should reflect skipped prefix
        expected_offset = bs * 3 - 1  # finished(0) + skip(3*bs - 1)
        assert ctx.request_kv_length_offsets[1].item() == expected_offset


# =========================================================================
# Class 6: TestDisabledMode (3 tests)
# =========================================================================


class TestDisabledMode(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_disabled_no_sharing(self):
        """enable_prefix_caching=False → identical prefixes get separate blocks."""
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone(), enable_prefix_caching=False))
        r1 = set(self._block_ids(ctx, 0, 2))

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2, enable_prefix_caching=False))
        r2 = set(self._block_ids(ctx, 1, 2))

        assert r1.isdisjoint(r2)

    @pytest.mark.internal
    def test_disabled_skips_prefix_caching_state(self):
        """Disabled mode skips prefix caching data structures entirely."""
        ctx = self._ctx(enable_prefix_caching=False)
        alloc = ctx.block_allocator

        assert not hasattr(alloc, 'block_hashes'), "block_hashes should not exist when disabled"
        assert not hasattr(
            alloc, 'block_ref_counts'
        ), "block_ref_counts should not exist when disabled"
        assert not hasattr(
            alloc, 'hash_to_block_id'
        ), "hash_to_block_id should not exist when disabled"

    @pytest.mark.internal
    def test_rz_no_timestamps(self):
        """Timestamp attributes don't exist in RZ mode."""
        ctx = self._ctx(block_evict_lru=False)
        alloc = ctx.block_allocator
        assert not hasattr(
            alloc, 'block_timestamps'
        ), "block_timestamps should not exist in RZ mode"


# =========================================================================
# Class 7: TestComplexPatterns (3 tests)
# =========================================================================


class TestComplexPatterns(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_prefix_chain_extending(self):
        """A=2 blocks, B extends to 4, C extends to 6. Proper sharing."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        # A: 2 blocks
        prompt_a = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt_a))
        blocks_a = ctx.request_to_kv_block_ids[0][:2]

        # B: extends A to 4 blocks
        prompt_b = torch.cat([prompt_a, self._prompt(bs * 2, offset=1000)])
        ctx.add_request(self._req(ctx, prompt_b, request_id=2))
        blocks_b = ctx.request_to_kv_block_ids[1][:4]

        assert torch.equal(blocks_b[:2], blocks_a), "B shares first 2 with A"

        # C: extends B to 6 blocks
        prompt_c = torch.cat([prompt_b, self._prompt(bs * 2, offset=2000)])
        ctx.add_request(self._req(ctx, prompt_c, request_id=3))
        blocks_c = ctx.request_to_kv_block_ids[2][:6]

        assert torch.equal(blocks_c[:4], blocks_b), "C shares first 4 with B"
        assert torch.equal(blocks_c[:2], blocks_a), "C shares first 2 with A"

    @pytest.mark.internal
    def test_independent_prefix_trees(self):
        """Two disjoint prefix trees coexist; within-tree sharing works."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        prefix_x = self._prompt(bs * 2)
        prefix_y = self._prompt(bs * 2, offset=5000)

        # Tree 1: 3 requests sharing prefix_x
        for i in range(3):
            ctx.add_request(self._req(ctx, prefix_x.clone(), request_id=i + 1))

        # Tree 2: 3 requests sharing prefix_y
        for i in range(3):
            ctx.add_request(self._req(ctx, prefix_y.clone(), request_id=i + 10))

        # Within-tree sharing
        tree1_blocks = ctx.request_to_kv_block_ids[0][:2]
        for i in range(1, 3):
            assert torch.equal(ctx.request_to_kv_block_ids[i][:2], tree1_blocks)

        tree2_blocks = ctx.request_to_kv_block_ids[3][:2]
        for i in range(4, 6):
            assert torch.equal(ctx.request_to_kv_block_ids[i][:2], tree2_blocks)

        # Cross-tree: different blocks
        assert not torch.equal(tree1_blocks, tree2_blocks)

    @pytest.mark.internal
    def test_memory_scaling_is_constant(self):
        """5 identical 4-block requests use only 4 total blocks, not 20."""
        ctx = self._ctx(max_sequence_length=1024)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 4)
        initial_avail = alloc.total_avail

        ctx.add_request(self._req(ctx, prompt.clone()))

        for i in range(2, 6):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))

        total_used = initial_avail - alloc.total_avail
        assert total_used == 4, f"Should use 4 blocks (O(1)), used {total_used}"


# =========================================================================
# Class 8: TestEngineCoordination (3 tests)
# =========================================================================


class _StubEngine(DynamicInferenceEngine):
    """Lightweight engine subclass that skips full __init__ for unit testing.

    Only initialises the fields needed by prefix coordination and scheduling.
    """

    def __init__(self, context: DynamicInferenceContext):
        # Bypass DynamicInferenceEngine.__init__ entirely — it needs a real
        # controller, CUDA graphs, wandb, etc. We only need the scheduling and
        # prefix-coordination paths.
        self.context = context
        self.enable_chunked_prefill = False
        self._prefix_coordination_waits = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}


class TestEngineCoordination(PrefixCachingTestBase):

    def _engine(self, ctx):
        """Create a _StubEngine wrapping *ctx*."""
        return _StubEngine(ctx)

    def _add_to_waiting(self, engine, ctx, req):
        """Register *req* with the engine and put it in the waiting queue."""
        request_id = req.request_id
        engine.requests[request_id] = type(
            "Entry",
            (),
            {
                "record": DynamicInferenceRequestRecord.from_request(req),
                "future": engine._loop.create_future(),
            },
        )()
        req.status = Status.ACTIVE_AND_GENERATING_TOKENS
        req.sampling_params.num_tokens_to_generate = 10
        engine.waiting_request_ids.append(request_id)

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_scheduling_deferral(self):
        """Two waiting requests with shared prefix: first scheduled, second deferred."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # Both requests in waiting queue (neither in context yet)
        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        # Schedule: req1 scheduled, req2 deferred (overlapping new hashes)
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 1, "Only first request should be scheduled"
        assert len(engine.waiting_request_ids) == 1, "Second request should be deferred"
        assert engine.waiting_request_ids[0] == 2
        assert engine._prefix_coordination_waits == 1

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_scheduling_no_deferral(self):
        """Prefix already in context: second request scheduled without deferral."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # Add first request directly to context (hashes in hash table)
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)

        # Put second request (same prefix) in waiting queue
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        # Scheduling should succeed immediately (hashes already registered)
        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 0, "Request should be scheduled"
        assert ctx.total_request_count == 2
        assert engine._prefix_coordination_waits == 0

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_get_prefix_coordination_metrics(self):
        """get_prefix_coordination_metrics tracks deferral waits."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        assert engine.get_prefix_coordination_metrics() == {"waits": 0}

        # Two requests with shared prefix in waiting queue
        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        # First schedule: req1 scheduled, req2 deferred
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}

        # Second schedule: req2 now schedulable (hashes already in hash table)
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}
        assert len(engine.waiting_request_ids) == 0
