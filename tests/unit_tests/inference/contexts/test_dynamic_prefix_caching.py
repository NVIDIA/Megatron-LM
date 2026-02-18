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
# Infrastructure
# =========================================================================


class PrefixCachingTestBase:
    """Shared setup/teardown and helpers for prefix caching tests."""

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
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=ctx.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

    @staticmethod
    def _prompt(num_tokens, offset=0):
        return torch.arange(offset, offset + num_tokens, device=torch.cuda.current_device())

    @staticmethod
    def _block_ids(ctx, req_idx, n):
        return [ctx.request_to_kv_block_ids[req_idx][i].item() for i in range(n)]

    @staticmethod
    def _simulate_chunk_done(req, chunk_length):
        """Simulate engine post-chunk update for chunked prefill tests."""
        req.remaining_prompt_tokens = req.remaining_prompt_tokens[chunk_length:]
        req.finished_chunk_token_count += chunk_length


class _StubEngine(DynamicInferenceEngine):
    """Lightweight engine that skips full __init__ for unit testing."""

    def __init__(self, context: DynamicInferenceContext):
        self.context = context
        self.enable_chunked_prefill = False
        self._prefix_coordination_waits = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}


# =========================================================================
# Class 1: TestHashContract
# =========================================================================


class TestHashContract(PrefixCachingTestBase):
    """Hash determinism, range, parent-chain, edge cases."""

    @pytest.mark.internal
    def test_determinism_and_range(self):
        """Same tokens → same hash; different → different; range [1, HASH_PRIME]; parent sensitivity."""
        tokens = self._prompt(32)
        h1 = compute_block_hashes_batched(tokens, 32)
        h2 = compute_block_hashes_batched(tokens, 32)
        assert h1 == h2, "Hash must be deterministic"
        assert len(h1) == 1
        assert 1 <= h1[0] <= HASH_PRIME
        h_diff = compute_block_hashes_batched(self._prompt(32, offset=1), 32)
        assert h_diff[0] != h1[0]
        # Parent sensitivity: 2-block chain
        two_blocks = torch.cat([tokens, self._prompt(32, offset=100)])
        h_chain = compute_block_hashes_batched(two_blocks, 32)
        assert len(h_chain) == 2
        assert 1 <= h_chain[1] <= HASH_PRIME

    @pytest.mark.internal
    def test_parent_chain_integrity(self):
        """3-block request: precomputed == batched; identical requests match; all hashes distinct."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3)
        req = self._req(ctx, prompt)
        assert len(req.precomputed_block_hashes) == 3
        assert req.precomputed_block_hashes == compute_block_hashes_batched(prompt, bs)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        assert req.precomputed_block_hashes == req2.precomputed_block_hashes
        assert len(set(req.precomputed_block_hashes)) == 3

    @pytest.mark.internal
    def test_edge_cases(self):
        """Sub-block → []; empty → []; single-token → []; all-zeros → 4 distinct; 120-block → 120 positive."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        dev = torch.cuda.current_device()
        assert self._req(ctx, self._prompt(bs // 2)).precomputed_block_hashes == []
        assert (
            self._req(ctx, torch.tensor([], device=dev, dtype=torch.long)).precomputed_block_hashes
            == []
        )
        assert (
            self._req(
                ctx, torch.tensor([42], device=dev, dtype=torch.long)
            ).precomputed_block_hashes
            == []
        )
        zeros = torch.zeros(bs * 4, device=dev, dtype=torch.long)
        h_zeros = self._req(ctx, zeros).precomputed_block_hashes
        assert len(h_zeros) == 4 and len(set(h_zeros)) == 4
        ctx_long = self._ctx(max_sequence_length=8192)
        h_long = self._req(
            ctx_long, torch.arange(bs * 120, device=dev, dtype=torch.long)
        ).precomputed_block_hashes
        assert len(h_long) == 120 and all(h > 0 for h in h_long)


# =========================================================================
# Class 2: TestPrefixMatchingContract
# =========================================================================


class TestPrefixMatchingContract(PrefixCachingTestBase):
    """Full/partial/no match, gap-breaks-chain."""

    @pytest.mark.internal
    def test_full_match_shares_all_blocks(self):
        """Identical 3-block prompt, 10 requests. All share same block IDs, ref_count==10."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        first_blocks = self._block_ids(ctx, 0, 3)
        for i in range(2, 11):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))
        for req_idx in range(1, 10):
            assert self._block_ids(ctx, req_idx, 3) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 10

    @pytest.mark.internal
    def test_partial_match_shares_common_prefix(self):
        """3-block prompts differing at block[2]: blocks 0,1 shared (ref=2), block 2 distinct."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        ctx.mark_pending_blocks_computed()
        r1 = self._block_ids(ctx, 0, 3)
        prompt2 = prompt1.clone()
        prompt2[bs * 2 :] += 1000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)
        assert r2[0] == r1[0] and r2[1] == r1[1], "Blocks 0,1 shared"
        assert r2[2] != r1[2], "Block 2 separate"
        assert alloc.block_ref_counts[r1[0]].item() == 2
        assert alloc.block_ref_counts[r1[2]].item() == 1

    @pytest.mark.internal
    def test_gap_breaks_chain(self):
        """[X,W,Z] vs [X,Y,Z]: only block 0 shared; block 2 NOT shared despite identical tokens."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        ctx.mark_pending_blocks_computed()
        r1 = self._block_ids(ctx, 0, 3)
        prompt2 = prompt1.clone()
        prompt2[bs : bs * 2] += 5000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)
        assert r2[0] == r1[0], "Block 0 shared"
        assert r2[1] != r1[1], "Block 1 NOT shared"
        assert r2[2] != r1[2], "Block 2 NOT shared (hash chain broken)"
        assert alloc.block_ref_counts[r1[0]].item() == 2

    @pytest.mark.internal
    def test_no_match_disjoint_blocks(self):
        """Completely different prompts → disjoint block sets, all ref_count=1."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        ctx.mark_pending_blocks_computed()
        r1 = set(self._block_ids(ctx, 0, 2))
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=1000), request_id=2))
        r2 = set(self._block_ids(ctx, 1, 2))
        assert r1.isdisjoint(r2)
        for bid in r1 | r2:
            assert alloc.block_ref_counts[bid].item() == 1


# =========================================================================
# Class 3: TestRefCountLifecycle
# =========================================================================


class TestRefCountLifecycle(PrefixCachingTestBase):
    """Increment/decrement, LRU reuse, O(1) memory scaling."""

    @pytest.mark.internal
    def test_share_increments_release_decrements(self):
        """ref=2 after sharing; ref=1 after one release; ref=0 + hash stays after both."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2
        assert alloc.block_ref_counts[b1].item() == 2
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert b0_hash in alloc.hash_to_block_id  # still cached in LRU

    @pytest.mark.internal
    def test_lru_reuse_after_release(self):
        """Release → ref=0 → new request same prefix reuses same block IDs."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        b0, b1 = self._block_ids(ctx, 0, 2)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.block_hashes[b0].item() in alloc.hash_to_block_id
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 0, 2) == [b0, b1]
        assert alloc.block_ref_counts[b0].item() == 1

    @pytest.mark.internal
    def test_memory_scaling_is_constant(self):
        """5 identical 4-block requests consume exactly 4 blocks total, not 20."""
        ctx = self._ctx(max_sequence_length=1024)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 4)
        initial_avail = alloc.total_avail
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        for i in range(2, 6):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))
        assert initial_avail - alloc.total_avail == 4


# =========================================================================
# Class 4: TestEvictionPolicy
# =========================================================================


class TestEvictionPolicy(PrefixCachingTestBase):
    """LRU protects active, RZ deregister, RZ persist-until-last-ref, RZ no-reuse."""

    @pytest.mark.internal
    def test_lru_protects_active_blocks(self):
        """Small buffer: active blocks (ref>0) survive eviction; cached blocks evicted."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        ctx.mark_pending_blocks_computed()
        active_blocks = ctx.request_to_kv_block_ids[0][:2].clone()
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=5000), request_id=2))
        ctx.mark_pending_blocks_computed()
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx.total_request_count = 1
        for i in range(20):
            try:
                ctx.add_request(
                    self._req(ctx, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100)
                )
                ctx.mark_pending_blocks_computed()
            except Exception:
                break
        for bid in active_blocks:
            assert alloc.block_ref_counts[bid.item()].item() == 1

    @pytest.mark.internal
    def test_rz_deregisters_on_ref_zero(self):
        """RZ: release → ref=0 → hash removed, block_hashes reset, total_avail increases."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        b1_hash = alloc.block_hashes[b1].item()
        avail_before = alloc.total_avail
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert b0_hash not in alloc.hash_to_block_id
        assert b1_hash not in alloc.hash_to_block_id
        assert alloc.block_hashes[b0].item() == -1
        assert alloc.block_hashes[b1].item() == -1
        assert alloc.total_avail == avail_before + 2

    @pytest.mark.internal
    def test_rz_shared_persist_until_last_ref(self):
        """RZ: 2 sharers, release one → ref=1, hash stays. Release second → hash removed."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, _ = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert b0_hash not in alloc.hash_to_block_id

    @pytest.mark.internal
    def test_rz_no_reuse_after_release(self):
        """RZ: after release, same prefix gets fresh blocks (no cache)."""
        ctx = self._ctx(block_evict_lru=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        new_blocks = self._block_ids(ctx, 0, 2)
        assert alloc.block_ref_counts[new_blocks[0]].item() == 1
        assert alloc.block_ref_counts[new_blocks[1]].item() == 1


# =========================================================================
# Class 5: TestTwoPhaseRegistration
# =========================================================================


class TestTwoPhaseRegistration(PrefixCachingTestBase):
    """Phase 1→2 flow, concurrent sharing, allocator-level API."""

    @pytest.mark.internal
    def test_phase1_discoverable_phase2_computed(self):
        """After add: hash_to_block_id populated, block_hashes==-1, pending!=−1. After mark: hash set, pending cleared."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        req = self._req(ctx, self._prompt(bs * 2))
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes
        assert alloc.hash_to_block_id.get(h0) == b0
        assert alloc.hash_to_block_id.get(h1) == b1
        assert alloc.block_hashes[b0].item() == -1
        assert alloc._pending_block_hashes[b0].item() != -1
        assert len(ctx._blocks_pending_computation) == 2
        ctx.mark_pending_blocks_computed()
        assert alloc.block_hashes[b0].item() == h0
        assert alloc.block_hashes[b1].item() == h1
        assert alloc._pending_block_hashes[b0].item() == -1
        assert len(ctx._blocks_pending_computation) == 0

    @pytest.mark.internal
    def test_concurrent_sharing_before_computed(self):
        """3 requests before mark → all share (ref=3), hash==-1. After mark + 4th: ref=4, hashes set."""
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
            assert alloc.block_hashes[bid].item() == -1
        ctx.mark_pending_blocks_computed()
        for bid in first_blocks:
            assert alloc.block_hashes[bid].item() != -1
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=4))
        assert self._block_ids(ctx, 3, 2) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 4

    @pytest.mark.internal
    def test_allocator_api_register_then_mark(self):
        """Direct allocator: register → lookup finds block, hash==-1. mark → hash==hash."""
        ctx = self._ctx()
        alloc = ctx.block_allocator
        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        test_hash = 99999
        alloc.register_block_hashes([bid], [test_hash])
        assert alloc.hash_to_block_id.get(test_hash) == bid
        assert alloc.block_hashes[bid].item() == -1
        alloc.mark_blocks_computed([bid])
        assert alloc.block_hashes[bid].item() == test_hash


# =========================================================================
# Class 6: TestPrefillAndDecode
# =========================================================================


class TestPrefillAndDecode(PrefixCachingTestBase):
    """Cached blocks skip pending, decode skips hashes, chain extending."""

    @pytest.mark.internal
    def test_cached_blocks_skip_pending(self):
        """Req1: 4 pending → mark → 0. Req2 same: 0 pending. Req3 extends by 1: 1 pending."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        assert len(ctx._blocks_pending_computation) == 4
        ctx.mark_pending_blocks_computed()
        assert len(ctx._blocks_pending_computation) == 0
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert len(ctx._blocks_pending_computation) == 0
        assert torch.equal(ctx.request_to_kv_block_ids[0][:4], ctx.request_to_kv_block_ids[1][:4])
        extended = torch.cat([prompt, self._prompt(bs, offset=1000)])
        ctx.add_request(self._req(ctx, extended, request_id=3))
        assert len(ctx._blocks_pending_computation) == 1

    @pytest.mark.internal
    def test_decode_does_not_register_hashes(self):
        """Complete block → hash. Partial block → stays -1 even after decode fills it."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs + (bs - 1))
        ctx.add_request(self._req(ctx, prompt))
        ctx.mark_pending_blocks_computed()
        b0, b1 = self._block_ids(ctx, 0, 2)
        assert alloc.block_hashes[b0].item() != -1
        assert alloc.block_hashes[b1].item() == -1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)
        assert alloc.block_hashes[b1].item() == -1

    @pytest.mark.internal
    def test_prefix_chain_extending(self):
        """A=2 blocks, B extends to 4, C extends to 6. Each shares all prior blocks."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt_a = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt_a))
        ctx.mark_pending_blocks_computed()
        blocks_a = ctx.request_to_kv_block_ids[0][:2]
        prompt_b = torch.cat([prompt_a, self._prompt(bs * 2, offset=1000)])
        ctx.add_request(self._req(ctx, prompt_b, request_id=2))
        ctx.mark_pending_blocks_computed()
        blocks_b = ctx.request_to_kv_block_ids[1][:4]
        assert torch.equal(blocks_b[:2], blocks_a)
        prompt_c = torch.cat([prompt_b, self._prompt(bs * 2, offset=2000)])
        ctx.add_request(self._req(ctx, prompt_c, request_id=3))
        blocks_c = ctx.request_to_kv_block_ids[2][:6]
        assert torch.equal(blocks_c[:4], blocks_b)
        assert torch.equal(blocks_c[:2], blocks_a)


# =========================================================================
# Class 7: TestChunkedPrefill
# =========================================================================


class TestChunkedPrefill(PrefixCachingTestBase):
    """Two-halves, many-small-chunks, sub-block-chunks, no-ref-leak."""

    @pytest.mark.internal
    def test_two_halves_full_match(self):
        """Req1 prefills 8 blocks. Req2 in two 4-block chunks: both share all blocks."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 8)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        r1_blocks = ctx.request_to_kv_block_ids[0][:8].clone()
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        chunk_len = bs * 4
        ctx.add_request(req2, chunk_length=chunk_len)
        self._simulate_chunk_done(req2, chunk_len)
        assert torch.equal(ctx.request_to_kv_block_ids[1][:4], r1_blocks[:4])
        ctx.add_request(req2, chunk_length=chunk_len)
        self._simulate_chunk_done(req2, chunk_len)
        assert torch.equal(ctx.request_to_kv_block_ids[1][:8], r1_blocks)

    @pytest.mark.internal
    def test_many_small_chunks_all_match(self):
        """6-block prompt, Req2 in 3x2-block chunks. All match, zero pool allocations."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 6)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        r_a_blocks = ctx.request_to_kv_block_ids[0][:6].clone()
        pool_avail_before = alloc.total_avail
        req_b = self._req(ctx, prompt.clone(), request_id=2)
        chunk_len = bs * 2
        for i in range(3):
            ctx.add_request(req_b, chunk_length=chunk_len)
            self._simulate_chunk_done(req_b, chunk_len)
            start = i * 2
            assert torch.equal(
                ctx.request_to_kv_block_ids[1][start : start + 2], r_a_blocks[start : start + 2]
            )
        assert alloc.total_avail == pool_avail_before

    @pytest.mark.internal
    def test_sub_block_chunks(self):
        """block_size=16, chunk=10, prompt=37. Complete blocks shared across requests."""
        ctx = self._ctx(block_size_tokens=16)
        bs = ctx.block_size_tokens
        prompt = self._prompt(37)
        req1 = self._req(ctx, prompt.clone())
        remaining = 37
        while remaining > 0:
            chunk = min(10, remaining)
            ctx.add_request(req1, chunk_length=chunk)
            ctx.mark_pending_blocks_computed()
            self._simulate_chunk_done(req1, chunk)
            remaining -= chunk
        r1_blocks = ctx.request_to_kv_block_ids[0][:3].clone()
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        remaining = 37
        while remaining > 0:
            chunk = min(10, remaining)
            ctx.add_request(req2, chunk_length=chunk)
            ctx.mark_pending_blocks_computed()
            self._simulate_chunk_done(req2, chunk)
            remaining -= chunk
        r2_blocks = ctx.request_to_kv_block_ids[1][:3]
        assert torch.equal(r2_blocks[:2], r1_blocks[:2])

    @pytest.mark.internal
    def test_no_ref_count_leak(self):
        """4-block prompt, Req2 chunked 2+2. Each block ref_count==2, not inflated."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        r1_block_ids = ctx.request_to_kv_block_ids[0][:4].tolist()
        for bid in r1_block_ids:
            assert alloc.block_ref_counts[bid].item() == 1
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        chunk_len = bs * 2
        ctx.add_request(req2, chunk_length=chunk_len)
        self._simulate_chunk_done(req2, chunk_len)
        ctx.add_request(req2, chunk_length=chunk_len)
        self._simulate_chunk_done(req2, chunk_len)
        for bid in r1_block_ids:
            assert alloc.block_ref_counts[bid].item() == 2


# =========================================================================
# Class 8: TestEngineScheduling
# =========================================================================


class TestEngineScheduling(PrefixCachingTestBase):
    """Pending detection, scheduling deferral, coordination metrics."""

    def _engine(self, ctx):
        return _StubEngine(ctx)

    def _add_to_waiting(self, engine, ctx, req):
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

    @pytest.mark.internal
    def test_pending_detection(self):
        """True when pending, False after mark, False for disabled, False for short prompt."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        assert engine._has_pending_prefix_blocks(req2) is True
        ctx.mark_pending_blocks_computed()
        assert engine._has_pending_prefix_blocks(req2) is False
        req_disabled = self._req(ctx, prompt.clone(), request_id=3, enable_prefix_caching=False)
        assert engine._has_pending_prefix_blocks(req_disabled) is False
        req_short = self._req(ctx, self._prompt(bs // 2), request_id=4)
        assert engine._has_pending_prefix_blocks(req_short) is False

    @pytest.mark.internal
    def test_scheduling_deferral_and_resolution(self):
        """Req2 waits while Req1's blocks pending. After mark: scheduled. Checks waits counter."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)
        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 1
        assert engine._prefix_coordination_waits == 1
        ctx.mark_pending_blocks_computed()
        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 0
        assert ctx.total_request_count == 2

    @pytest.mark.internal
    def test_coordination_metrics_accumulate(self):
        """Waits counter starts 0, increments per deferral, stays after resolution."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)
        assert engine.get_prefix_coordination_metrics() == {"waits": 0}
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 2}
        ctx.mark_pending_blocks_computed()
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 2}


# =========================================================================
# Class 9: TestCPUShadowsAndConfig
# =========================================================================


class TestCPUShadowsAndConfig(PrefixCachingTestBase):
    """Pending set lifecycle, reverse mapping, ref counters, reset, disabled+RZ attrs."""

    @staticmethod
    def _assert_cpu_shadows_consistent(alloc):
        gpu_pending = set(torch.nonzero(alloc._pending_block_hashes != -1).flatten().tolist())
        assert alloc._pending_block_ids_cpu == gpu_pending
        for bid, h in alloc.block_id_to_hash.items():
            assert alloc.hash_to_block_id.get(h) == bid
        for h, bid in alloc.hash_to_block_id.items():
            assert alloc.block_id_to_hash.get(bid) == h
        # Reconcile lazy counter before comparing (matches engine flow)
        alloc.reconcile_blocks_with_refs()
        gpu_blocks_with_refs = int((alloc.block_ref_counts > 0).sum().item())
        gpu_total_ref_count = int(alloc.block_ref_counts.sum().item())
        assert alloc._cpu_blocks_with_refs == gpu_blocks_with_refs
        assert alloc._cpu_total_ref_count == gpu_total_ref_count

    @pytest.mark.internal
    def test_pending_set_tracks_lifecycle(self):
        """_pending_block_ids_cpu: empty → {b0,b1} after add → empty after mark."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        assert alloc._pending_block_ids_cpu == set()
        req = self._req(ctx, self._prompt(bs * 2))
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        assert alloc._pending_block_ids_cpu == {b0, b1}
        self._assert_cpu_shadows_consistent(alloc)
        ctx.mark_pending_blocks_computed()
        assert alloc._pending_block_ids_cpu == set()
        self._assert_cpu_shadows_consistent(alloc)

    @pytest.mark.internal
    def test_reverse_mapping_consistent(self):
        """block_id_to_hash correct after register, mark, and LRU release. RZ release clears it."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        req = self._req(ctx, self._prompt(bs * 2))
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes
        assert alloc.block_id_to_hash[b0] == h0
        assert alloc.block_id_to_hash[b1] == h1
        self._assert_cpu_shadows_consistent(alloc)
        ctx.mark_pending_blocks_computed()
        assert alloc.block_id_to_hash[b0] == h0
        self._assert_cpu_shadows_consistent(alloc)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_id_to_hash[b0] == h0  # still cached in LRU
        self._assert_cpu_shadows_consistent(alloc)
        # RZ mode: release clears reverse mapping
        ctx_rz = self._ctx(block_evict_lru=False)
        alloc_rz = ctx_rz.block_allocator
        req_rz = self._req(ctx_rz, self._prompt(bs * 2))
        ctx_rz.add_request(req_rz)
        ctx_rz.mark_pending_blocks_computed()
        b0_rz, b1_rz = self._block_ids(ctx_rz, 0, 2)
        ctx_rz.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert b0_rz not in alloc_rz.block_id_to_hash
        assert b1_rz not in alloc_rz.block_id_to_hash
        self._assert_cpu_shadows_consistent(alloc_rz)

    @pytest.mark.internal
    def test_cpu_ref_counters(self):
        """_cpu_blocks_with_refs and _cpu_total_ref_count accurate through add→share→release→release."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)
        assert alloc._cpu_blocks_with_refs == 0
        assert alloc._cpu_total_ref_count == 0
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.mark_pending_blocks_computed()
        assert alloc._cpu_blocks_with_refs == 2
        assert alloc._cpu_total_ref_count == 2
        self._assert_cpu_shadows_consistent(alloc)
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert alloc._cpu_blocks_with_refs == 2
        assert alloc._cpu_total_ref_count == 4
        self._assert_cpu_shadows_consistent(alloc)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        alloc.reconcile_blocks_with_refs()
        assert alloc._cpu_blocks_with_refs == 2
        assert alloc._cpu_total_ref_count == 2
        self._assert_cpu_shadows_consistent(alloc)
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        alloc.reconcile_blocks_with_refs()
        assert alloc._cpu_blocks_with_refs == 0
        assert alloc._cpu_total_ref_count == 0
        self._assert_cpu_shadows_consistent(alloc)

    @pytest.mark.internal
    def test_reset_clears_all_shadows(self):
        """After alloc.reset(), all CPU structures zeroed."""
        ctx = self._ctx()
        alloc = ctx.block_allocator
        ctx.add_request(self._req(ctx, self._prompt(ctx.block_size_tokens * 2)))
        ctx.mark_pending_blocks_computed()
        assert len(alloc.block_id_to_hash) > 0
        alloc.reset()
        assert alloc._pending_block_ids_cpu == set()
        assert alloc.block_id_to_hash == {}
        assert alloc._cpu_blocks_with_refs == 0
        assert alloc._cpu_total_ref_count == 0
        self._assert_cpu_shadows_consistent(alloc)

    @pytest.mark.internal
    def test_disabled_and_rz_attribute_absence(self):
        """Disabled: no block_hashes, block_ref_counts, hash_to_block_id. RZ: no block_timestamps."""
        alloc_disabled = self._ctx(enable_prefix_caching=False).block_allocator
        assert not hasattr(alloc_disabled, 'block_hashes')
        assert not hasattr(alloc_disabled, 'block_ref_counts')
        assert not hasattr(alloc_disabled, 'hash_to_block_id')
        alloc_rz = self._ctx(block_evict_lru=False).block_allocator
        assert not hasattr(alloc_rz, 'block_timestamps')
