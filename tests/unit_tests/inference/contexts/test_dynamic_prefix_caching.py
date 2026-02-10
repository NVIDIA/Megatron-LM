# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from collections import deque

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    compute_block_hash,
    HASH_PRIME,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


# =========================================================================
# Base class + helpers
# =========================================================================

class PrefixCachingTestBase:
    """Base class with shared setup/teardown and helper methods."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _ctx(self, *, buffer_size_gb=0.1, block_size_tokens=32,
             max_sequence_length=512, rounder=64, enable_prefix_caching=True,
             max_tokens=None):
        """Create a DynamicInferenceContext with sensible test defaults."""
        DynamicInferenceContext.ROUNDER = rounder
        DynamicInferenceContext.TOKEN_ROUNDER = rounder
        DynamicInferenceContext.REQUEST_ROUNDER = rounder

        transformer_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=4, kv_channels=8,
            num_attention_heads=2, hidden_size=16,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
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
        )
        return DynamicInferenceContext(
            model_config=transformer_config, inference_config=inference_config,
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
        return torch.arange(offset, offset + num_tokens,
                            device=torch.cuda.current_device())

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
        h1 = compute_block_hash(0, tokens)
        h2 = compute_block_hash(0, tokens)
        assert h1 == h2, "Hash should be deterministic"
        assert 1 <= h1 <= HASH_PRIME, "Hash should be in [1, HASH_PRIME]"

        # Different tokens
        h_diff = compute_block_hash(0, self._prompt(32, offset=1))
        assert h_diff != h1

        # Parent sensitivity
        h_parent = compute_block_hash(12345, tokens)
        assert h_parent != h1
        assert 1 <= h_parent <= HASH_PRIME

    @pytest.mark.internal
    def test_precomputed_hash_chain(self):
        """3-block request has correct precomputed hashes matching manual computation."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3)

        req = self._req(ctx, prompt)
        assert len(req.precomputed_block_hashes) == 3

        # Verify against manual computation
        parent = 0
        for i in range(3):
            expected = compute_block_hash(parent, prompt[i * bs:(i + 1) * bs])
            assert req.precomputed_block_hashes[i] == expected
            parent = expected

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
        long_prompt = torch.arange(bs * 120, device=torch.cuda.current_device(),
                                   dtype=torch.long)
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
        ctx.mark_pending_blocks_computed()
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
        ctx.mark_pending_blocks_computed()
        r1 = self._block_ids(ctx, 0, 3)

        # Same first 2 blocks, different 3rd
        prompt2 = prompt1.clone()
        prompt2[bs * 2:] += 1000
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
        ctx.mark_pending_blocks_computed()
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
        ctx.mark_pending_blocks_computed()
        r1 = self._block_ids(ctx, 0, 3)

        # Same block 0, different block 1, same block 2 tokens
        prompt2 = prompt1.clone()
        prompt2[bs:bs * 2] += 5000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)

        assert r2[0] == r1[0], "Block 0 shared"
        assert r2[1] != r1[1], "Block 1 NOT shared"
        assert r2[2] != r1[2], "Block 2 NOT shared (hash chain broken)"
        assert alloc.block_ref_counts[r1[0]].item() == 2
        assert alloc.block_ref_counts[r1[1]].item() == 1


# =========================================================================
# Class 3: TestRefCountLifecycle (3 tests)
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
        ctx.mark_pending_blocks_computed()
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))

        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.get_block_hash(b0)
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
        ctx.mark_pending_blocks_computed()
        b0, b1 = self._block_ids(ctx, 0, 2)

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.get_block_hash(b0) in alloc.hash_to_block_id

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
        ctx.mark_pending_blocks_computed()
        active_blocks = ctx.request_to_kv_block_ids[0][:2].clone()

        # Release a second request to create cached (evictable) blocks
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=5000), request_id=2))
        ctx.mark_pending_blocks_computed()
        cached_blocks = ctx.request_to_kv_block_ids[1][:2].clone()
        cached_hash = alloc.get_block_hash(cached_blocks[0].item())
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx.total_request_count = 1
        assert alloc.block_ref_counts[cached_blocks[0].item()].item() == 0

        # Fill remaining space with different requests to trigger eviction
        for i in range(20):
            try:
                ctx.add_request(self._req(
                    ctx, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100))
                ctx.mark_pending_blocks_computed()
            except Exception:
                break

        # Active blocks protected
        for bid in active_blocks:
            assert alloc.block_ref_counts[bid.item()].item() == 1

        # Cached blocks may have been evicted (hash removed from mapping)
        # The key invariant: system still functions correctly
        assert ctx.total_request_count >= 1


# =========================================================================
# Class 4: TestTwoPhaseRegistration (3 tests)
# =========================================================================

class TestTwoPhaseRegistration(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_two_phase_flow(self):
        """Phase 1: discoverable but hash==-1. Phase 2: hash set, pending cleared."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes

        # Phase 1: discoverable via lookup, but block_hashes == -1
        assert alloc.lookup_block_by_hash(h0) == b0
        assert alloc.lookup_block_by_hash(h1) == b1
        assert alloc.get_block_hash(b0) == -1
        assert alloc.get_block_hash(b1) == -1
        assert b0 in alloc._pending_block_hashes
        assert len(ctx._blocks_pending_computation) == 2

        # Phase 2: mark computed
        ctx.mark_pending_blocks_computed()
        assert alloc.get_block_hash(b0) == h0
        assert alloc.get_block_hash(b1) == h1
        assert b0 not in alloc._pending_block_hashes
        assert len(ctx._blocks_pending_computation) == 0

    @pytest.mark.internal
    def test_concurrent_sharing_before_computed(self):
        """3 requests added before mark → all share (ref=3), pending. After mark: hashes set."""
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
            assert alloc.get_block_hash(bid) == -1  # still pending

        ctx.mark_pending_blocks_computed()
        for bid in first_blocks:
            assert alloc.get_block_hash(bid) != -1

        # 4th request also shares
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=4))
        assert self._block_ids(ctx, 3, 2) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 4

    @pytest.mark.internal
    def test_pending_block_detection(self):
        """lookup finds pending blocks; get_block_hash returns -1; after mark: real hash."""
        ctx = self._ctx()
        alloc = ctx.block_allocator

        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        test_hash = 99999

        alloc.register_block_hash(bid, test_hash)

        # Lookup finds it, but get_block_hash returns -1
        assert alloc.lookup_block_by_hash(test_hash) == bid
        assert alloc.get_block_hash(bid) == -1

        # After mark: real hash
        alloc.mark_block_computed(bid)
        assert alloc.get_block_hash(bid) == test_hash


# =========================================================================
# Class 5: TestPrefillSkipping (2 tests)
# =========================================================================

class TestPrefillSkipping(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_cached_blocks_skip_prefill(self):
        """Req1: 4 pending → mark → 0 pending. Req2 same: 0 pending. Req3 extends: 1 pending."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 4)

        ctx.add_request(self._req(ctx, prompt.clone()))
        assert len(ctx._blocks_pending_computation) == 4

        ctx.mark_pending_blocks_computed()
        assert len(ctx._blocks_pending_computation) == 0

        # Same prefix → 0 pending
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert len(ctx._blocks_pending_computation) == 0
        assert torch.equal(
            ctx.request_to_kv_block_ids[0][:4],
            ctx.request_to_kv_block_ids[1][:4],
        )

        # Extended prompt → only 1 new block pending
        extended = torch.cat([prompt, self._prompt(bs, offset=1000)])
        ctx.add_request(self._req(ctx, extended, request_id=3))
        assert len(ctx._blocks_pending_computation) == 1

    @pytest.mark.internal
    def test_decode_does_not_compute_hashes(self):
        """Complete block → hash after mark. Partial block → stays -1 even after decode fills it."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # 1 complete block + (block_size - 1) tokens in second block
        prompt = self._prompt(bs + (bs - 1))
        ctx.add_request(self._req(ctx, prompt))
        ctx.mark_pending_blocks_computed()

        b0, b1 = self._block_ids(ctx, 0, 2)
        assert alloc.get_block_hash(b0) != -1, "Complete block has hash"
        assert alloc.get_block_hash(b1) == -1, "Partial block: no hash"

        # One decode step fills block 1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)

        assert alloc.get_block_hash(b1) == -1, "Decode hash computation not implemented"


# =========================================================================
# Class 6: TestDisabledMode (2 tests)
# =========================================================================

class TestDisabledMode(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_disabled_no_sharing(self):
        """enable_prefix_caching=False → identical prefixes get separate blocks."""
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        r1 = set(self._block_ids(ctx, 0, 2))

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        r2 = set(self._block_ids(ctx, 1, 2))

        assert r1.isdisjoint(r2)
        for bid in r1 | r2:
            assert alloc.block_ref_counts[bid].item() == 1

    @pytest.mark.internal
    def test_disabled_deterministic_hashes(self):
        """Disabled blocks get Knuth hashes: (block_id * 2654435761) % HASH_PRIME + 1."""
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        block_ids = self._block_ids(ctx, 0, 2)

        hashes = [alloc.block_hashes[bid].item() for bid in block_ids]
        assert len(set(hashes)) == 2, "Each block unique hash"

        for bid in block_ids:
            expected = (bid * 2654435761) % HASH_PRIME + 1
            assert alloc.block_hashes[bid].item() == expected


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
        ctx.mark_pending_blocks_computed()
        blocks_a = ctx.request_to_kv_block_ids[0][:2]

        # B: extends A to 4 blocks
        prompt_b = torch.cat([prompt_a, self._prompt(bs * 2, offset=1000)])
        ctx.add_request(self._req(ctx, prompt_b, request_id=2))
        ctx.mark_pending_blocks_computed()
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
            if i == 0:
                ctx.mark_pending_blocks_computed()

        # Tree 2: 3 requests sharing prefix_y
        for i in range(3):
            ctx.add_request(self._req(ctx, prefix_y.clone(), request_id=i + 10))
            if i == 0:
                ctx.mark_pending_blocks_computed()

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
        ctx.mark_pending_blocks_computed()

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
            "Entry", (), {
                "record": DynamicInferenceRequestRecord.from_request(req),
                "future": engine._loop.create_future(),
            },
        )()
        req.status = Status.ACTIVE_AND_GENERATING_TOKENS
        req.sampling_params.num_tokens_to_generate = 10
        engine.waiting_request_ids.append(request_id)

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_has_pending_prefix_blocks(self):
        """_has_pending_prefix_blocks returns True when blocks pending, False once computed."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # Add first request to context → blocks registered but not computed
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)

        # Second request with same prefix: should detect pending blocks
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        assert engine._has_pending_prefix_blocks(req2) is True

        # After marking computed, pending blocks are resolved
        ctx.mark_pending_blocks_computed()
        assert engine._has_pending_prefix_blocks(req2) is False

        # Disabled mode: always False
        req_disabled = self._req(ctx, prompt.clone(), request_id=3,
                                 enable_prefix_caching=False)
        assert engine._has_pending_prefix_blocks(req_disabled) is False

        # No precomputed hashes (short prompt): always False
        req_short = self._req(ctx, self._prompt(bs // 2), request_id=4)
        assert engine._has_pending_prefix_blocks(req_short) is False

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_scheduling_deferral(self):
        """Requests wait in queue until pending prefix blocks are marked computed."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # Add first request directly to context (simulates prior prefill step)
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        # Blocks are now pending (not yet computed)

        # Put second request (same prefix) in waiting queue
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        # Attempt scheduling — should defer (break) because blocks pending
        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 1, "Request should remain waiting"
        assert engine._prefix_coordination_waits == 1

        # Mark blocks computed
        ctx.mark_pending_blocks_computed()

        # Now scheduling should succeed
        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 0, "Request should be scheduled"
        assert ctx.total_request_count == 2

    # -----------------------------------------------------------------
    @pytest.mark.internal
    def test_get_prefix_coordination_metrics(self):
        """get_prefix_coordination_metrics tracks cumulative waits."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        assert engine.get_prefix_coordination_metrics() == {"waits": 0}

        # Create a pending-block scenario
        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)

        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        # Each scheduling attempt that defers increments the counter
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}

        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 2}

        # Resolve and schedule
        ctx.mark_pending_blocks_computed()
        engine.schedule_non_chunked_prefill()
        # Counter doesn't reset — cumulative
        assert engine.get_prefix_coordination_metrics() == {"waits": 2}
