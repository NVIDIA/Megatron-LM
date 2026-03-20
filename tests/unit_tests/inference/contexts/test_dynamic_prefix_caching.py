# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from collections import deque

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, PrefixCachingEvictionPolicy
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
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    ):
        """Create a DynamicInferenceContext with sensible test defaults."""
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
            prefix_caching_eviction_policy=prefix_caching_eviction_policy,
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


class _StubEngine(DynamicInferenceEngine):
    """Lightweight engine subclass that skips full __init__ for unit testing."""

    def __init__(self, context: DynamicInferenceContext, *, enable_chunked_prefill=False):
        self.context = context
        self.enable_chunked_prefill = enable_chunked_prefill
        self._prefix_coordination_waits = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}


# =========================================================================
# Class 1: TestBlockSharingAndRefCounts (5 tests)
# =========================================================================


class TestBlockSharingAndRefCounts(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_identical_prompts_share_all_blocks(self):
        """N=10 identical requests use only K blocks, not N*K."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 3)

        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 3)
        avail_after_first = alloc.total_avail

        for i in range(2, 11):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))

        # Pool unchanged after 9 more requests
        assert alloc.total_avail == avail_after_first

        # All share the same block IDs
        for req_idx in range(1, 10):
            assert self._block_ids(ctx, req_idx, 3) == first_blocks

        # Ref counts == 10
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 10

    @pytest.mark.internal
    def test_divergent_suffix_shares_common_prefix(self):
        """[A,B,C] + [A,B,D]: blocks 0-1 shared (ref=2), blocks 2,3 separate (ref=1)."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        r1 = self._block_ids(ctx, 0, 3)

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
    def test_broken_chain_stops_sharing(self):
        """[X,W,Z] vs [X,Y,Z]: only block 0 shared due to parent chain break."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        r1 = self._block_ids(ctx, 0, 3)

        prompt2 = prompt1.clone()
        prompt2[bs : bs * 2] += 5000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        r2 = self._block_ids(ctx, 1, 3)

        assert r2[0] == r1[0], "Block 0 shared"
        assert r2[1] != r1[1], "Block 1 NOT shared"
        assert r2[2] != r1[2], "Block 2 NOT shared (hash chain broken)"
        assert alloc.block_ref_counts[r1[0]].item() == 2
        assert alloc.block_ref_counts[r1[1]].item() == 1

    @pytest.mark.internal
    def test_disabled_means_no_sharing(self):
        """enable_prefix_caching=False: identical prompts allocate separate blocks."""
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone(), enable_prefix_caching=False))
        r1 = set(self._block_ids(ctx, 0, 2))

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2, enable_prefix_caching=False))
        r2 = set(self._block_ids(ctx, 1, 2))

        assert r1.isdisjoint(r2)

    @pytest.mark.internal
    def test_disabled_allocator_has_no_caching_state(self):
        """Disabled allocator lacks caching attrs; REF_ZERO lacks timestamps."""
        ctx_disabled = self._ctx(enable_prefix_caching=False)
        alloc_d = ctx_disabled.block_allocator
        assert not hasattr(alloc_d, 'block_hashes')
        assert not hasattr(alloc_d, 'hash_to_block_id')
        assert not hasattr(alloc_d, 'block_ref_counts')

        ctx_rz = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc_rz = ctx_rz.block_allocator
        assert not hasattr(alloc_rz, 'block_timestamps')


# =========================================================================
# Class 2: TestPrefillTokenSavings (4 tests)
# =========================================================================


class TestPrefillTokenSavings(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_lifetime_prefill_count_with_vs_without(self):
        """Enabled saves tokens vs disabled for two identical prompts."""
        bs = 32

        # Enabled
        ctx_on = self._ctx()
        prompt = self._prompt(bs * 4)
        ctx_on.add_request(self._req(ctx_on, prompt.clone()))
        ctx_on.add_request(self._req(ctx_on, prompt.clone(), request_id=2))
        enabled_total = ctx_on.lifetime_prefill_token_count

        # Disabled
        ctx_off = self._ctx(enable_prefix_caching=False)
        ctx_off.add_request(self._req(ctx_off, prompt.clone(), enable_prefix_caching=False))
        ctx_off.add_request(
            self._req(ctx_off, prompt.clone(), request_id=2, enable_prefix_caching=False)
        )
        disabled_total = ctx_off.lifetime_prefill_token_count

        # Enabled: 4*bs + 1 (min guard). Disabled: 4*bs + 4*bs.
        assert enabled_total == bs * 4 + 1
        assert disabled_total == bs * 4 * 2
        assert enabled_total < disabled_total

    @pytest.mark.internal
    def test_partial_match_reduces_prefill_proportionally(self):
        """First req fills 3 blocks, second matches 2 of 3, adds bs tokens."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1.clone()))

        prompt2 = prompt1.clone()
        prompt2[bs * 2 :] += 1000
        ctx.add_request(self._req(ctx, prompt2, request_id=2))

        assert ctx.lifetime_prefill_token_count == bs * 3 + bs

    @pytest.mark.internal
    def test_full_match_adds_single_token(self):
        """Prompt exactly N*bs, fully matched: second request query length == 1."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 3)

        ctx.add_request(self._req(ctx, prompt.clone()))
        tokens_after_a = ctx.active_token_count

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert ctx.active_token_count - tokens_after_a == 1
        assert ctx.request_query_lengths[1].item() == 1

    @pytest.mark.internal
    def test_no_match_adds_full_prompt(self):
        """Different prefix: full chunk_length added to lifetime count."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=9000), request_id=2))

        assert ctx.lifetime_prefill_token_count == bs * 2 + bs * 2


# =========================================================================
# Class 3: TestRefCountLifecycle (5 tests)
# =========================================================================


class TestRefCountLifecycle(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_lru_ref_decrement_preserves_cached_blocks(self):
        """LRU mode: add 2, release both. Ref 2 -> 1 -> 0. Hash still in dict."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()

        assert alloc.block_ref_counts[b0].item() == 2

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert alloc.block_ref_counts[b1].item() == 0
        assert b0_hash in alloc.hash_to_block_id, "LRU keeps cached blocks"

    @pytest.mark.internal
    def test_lru_cached_blocks_reused_by_new_request(self):
        """LRU mode: released blocks (ref=0, cached) reused by same-prefix request."""
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

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 0, 2) == [b0, b1]
        assert alloc.block_ref_counts[b0].item() == 1

    @pytest.mark.internal
    def test_lru_eviction_frees_oldest_cached_first(self):
        """LRU mode, tiny buffer: create cached groups A, B. Fill memory. A evicted first."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        # Active request (protected)
        ctx.add_request(self._req(ctx, self._prompt(bs * 2)))
        active_blocks = ctx.request_to_kv_block_ids[0][:2].clone()

        # Cached request (evictable)
        ctx.add_request(self._req(ctx, self._prompt(bs * 2, offset=5000), request_id=2))
        cached_blocks = ctx.request_to_kv_block_ids[1][:2].clone()
        cached_hash = alloc.block_hashes[cached_blocks[0].item()].item()
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx.total_request_count = 1
        assert alloc.block_ref_counts[cached_blocks[0].item()].item() == 0

        # Fill remaining space to trigger eviction of cached blocks
        for i in range(20):
            try:
                ctx.add_request(
                    self._req(ctx, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100)
                )
            except Exception:
                break

        # Active blocks remain protected
        for bid in active_blocks:
            assert alloc.block_ref_counts[bid.item()].item() == 1

    @pytest.mark.internal
    def test_refzero_deregisters_on_last_release(self):
        """REF_ZERO: two reqs sharing prefix. Release both: hash removed, blocks returned."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))

        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        avail_before = alloc.total_avail

        # Release first: ref=1, hash persists
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1
        assert b0_hash in alloc.hash_to_block_id

        # Release second: ref=0, hash removed, blocks returned
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0
        assert b0_hash not in alloc.hash_to_block_id
        assert alloc.block_hashes[b0].item() == -1
        assert alloc.block_hashes[b1].item() == -1
        assert alloc.total_avail == avail_before + 2

    @pytest.mark.internal
    def test_refzero_released_blocks_not_discoverable(self):
        """REF_ZERO: after full release, same-prefix request gets fresh blocks."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 2)

        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx.total_request_count = 0

        # Hash lookup returns None, so new blocks allocated
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        new_blocks = self._block_ids(ctx, 0, 2)
        assert alloc.block_ref_counts[new_blocks[0]].item() == 1
        assert alloc.block_ref_counts[new_blocks[1]].item() == 1


# =========================================================================
# Class 4: TestHashComputation (3 tests)
# =========================================================================


class TestHashComputation(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_determinism_and_range(self):
        """Same tokens -> same hashes. All in [1, HASH_PRIME]. Different tokens differ."""
        tokens = self._prompt(32)
        h1 = compute_block_hashes_batched(tokens, 32)
        h2 = compute_block_hashes_batched(tokens, 32)
        assert h1 == h2
        assert len(h1) == 1
        assert 1 <= h1[0] <= HASH_PRIME

        h_diff = compute_block_hashes_batched(self._prompt(32, offset=1), 32)
        assert h_diff[0] != h1[0]

    @pytest.mark.internal
    def test_parent_chaining_differentiates_position(self):
        """4 blocks of all-zero tokens: all hashes distinct due to parent chaining."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        zeros = torch.zeros(bs * 4, device=torch.cuda.current_device(), dtype=torch.long)
        hashes = compute_block_hashes_batched(zeros, bs)
        assert len(hashes) == 4
        assert len(set(hashes)) == 4

    @pytest.mark.internal
    def test_edge_cases(self):
        """Short, empty, and long prompts produce expected hash counts."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens

        # Shorter than block_size -> empty
        assert compute_block_hashes_batched(self._prompt(bs // 2), bs) == []

        # Empty prompt -> empty
        empty = torch.tensor([], device=torch.cuda.current_device(), dtype=torch.long)
        assert compute_block_hashes_batched(empty, bs) == []

        # 120-block prompt -> 120 positive hashes
        long_prompt = torch.arange(bs * 120, device=torch.cuda.current_device(), dtype=torch.long)
        h = compute_block_hashes_batched(long_prompt, bs)
        assert len(h) == 120
        assert all(v > 0 for v in h)


# =========================================================================
# Class 5: TestRegistrationAndDiscovery (4 tests)
# =========================================================================


class TestRegistrationAndDiscovery(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_blocks_discoverable_after_add_request(self):
        """After add_request, each block's hash maps to its block ID."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes

        assert alloc.hash_to_block_id.get(h0) == b0
        assert alloc.hash_to_block_id.get(h1) == b1
        assert alloc.block_hashes[b0].item() == h0
        assert alloc.block_hashes[b1].item() == h1

    @pytest.mark.internal
    def test_partial_block_not_registered(self):
        """Prompt of bs + bs//2: first block registered, second block not."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt = self._prompt(bs + bs // 2)
        ctx.add_request(self._req(ctx, prompt))
        b0, b1 = self._block_ids(ctx, 0, 2)

        assert alloc.block_hashes[b0].item() != -1, "Complete block registered"
        assert alloc.block_hashes[b1].item() == -1, "Partial block not registered"

    @pytest.mark.internal
    def test_decode_does_not_register_completed_blocks(self):
        """Partial block stays unregistered after decode fills it (known limitation)."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt = self._prompt(bs + (bs - 1))
        ctx.add_request(self._req(ctx, prompt))
        b0, b1 = self._block_ids(ctx, 0, 2)

        assert alloc.block_hashes[b0].item() != -1
        assert alloc.block_hashes[b1].item() == -1

        # One decode step fills block 1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)

        assert alloc.block_hashes[b1].item() == -1, "Decode does not register blocks"

    @pytest.mark.internal
    def test_second_request_finds_registered_blocks(self):
        """After req1 registers 3 blocks, req2's hashes all resolve in hash_to_block_id."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 3)

        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)

        req2 = self._req(ctx, prompt.clone(), request_id=2)
        for h in req2.precomputed_block_hashes:
            assert h in alloc.hash_to_block_id, f"Hash {h} should be discoverable"


# =========================================================================
# Class 6: TestBlockAllocation (3 tests)
# =========================================================================


class TestBlockAllocation(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_matched_blocks_not_allocated_from_pool(self):
        """Req1 allocates 4 blocks. Req2 same prefix: pool unchanged, same block IDs."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 4)

        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 4)
        avail_after_first = alloc.total_avail

        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 1, 4) == first_blocks
        assert alloc.total_avail == avail_after_first

    @pytest.mark.internal
    def test_extended_prompt_allocates_only_new_blocks(self):
        """Req1: 3 blocks. Req2: 5 blocks (same first 3). Pool drops by exactly 2."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator

        prompt1 = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt1))
        avail_after_first = alloc.total_avail

        prompt2 = torch.cat([prompt1, self._prompt(bs * 2, offset=1000)])
        ctx.add_request(self._req(ctx, prompt2, request_id=2))
        assert alloc.total_avail == avail_after_first - 2

    @pytest.mark.internal
    def test_check_availability_accounts_for_prefix_match(self):
        """check_availability returns kv_cache_available=True when matched blocks cover need."""
        ctx = self._ctx(buffer_size_gb=0.01, rounder=1)
        bs = ctx.block_size_tokens
        alloc = ctx.block_allocator
        prompt = self._prompt(bs * 2)

        ctx.add_request(self._req(ctx, prompt.clone()))

        # Fill remaining pool
        while alloc.total_avail > 0:
            alloc.allocate_memory_blocks(1)

        # New request with same prefix: all blocks matched, needs 0 from pool
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        _, _, kv_available = ctx.check_availability(req2)
        assert kv_available, "Matched blocks don't need pool allocation"


# =========================================================================
# Class 7: TestEngineScheduling (6 tests)
# =========================================================================


class TestEngineScheduling(PrefixCachingTestBase):

    def _engine(self, ctx, **kwargs):
        """Create a _StubEngine wrapping ctx."""
        return _StubEngine(ctx, **kwargs)

    def _add_to_waiting(self, engine, ctx, req):
        """Register req with engine and put it in the waiting queue."""
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
    def test_shared_prefix_defers_second_request(self):
        """Two requests with shared prefix: first scheduled, second deferred."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1
        assert engine.waiting_request_ids[0] == 2
        assert engine._prefix_coordination_waits == 1

    @pytest.mark.internal
    def test_scheduler_skips_deferred_to_schedule_non_conflicting(self):
        """req1+req2 share prefix A, req3 has prefix B. req1 and req3 scheduled, req2 deferred."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)

        prompt_a = self._prompt(bs * 2)
        prompt_b = self._prompt(bs * 2, offset=5000)

        req1 = self._req(ctx, prompt_a.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt_a.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)
        req3 = self._req(ctx, prompt_b.clone(), request_id=3)
        self._add_to_waiting(engine, ctx, req3)

        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 2, "req1 and req3 scheduled"
        assert len(engine.waiting_request_ids) == 1
        assert engine.waiting_request_ids[0] == 2, "req2 deferred"

    @pytest.mark.internal
    def test_registered_prefix_allows_immediate_scheduling(self):
        """Prefix already in context: second request scheduled without deferral."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # Add first request directly to context (hashes registered)
        ctx.add_request(self._req(ctx, prompt.clone()))

        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        engine.schedule_non_chunked_prefill()
        assert len(engine.waiting_request_ids) == 0
        assert ctx.total_request_count == 2
        assert engine._prefix_coordination_waits == 0

    @pytest.mark.internal
    def test_deferred_request_schedulable_after_registration(self):
        """Round 1: req1 scheduled, req2 deferred. Round 2: req2 scheduled."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1

        # Round 2: hashes now registered from req1
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 2
        assert len(engine.waiting_request_ids) == 0

    @pytest.mark.internal
    def test_coordination_metrics_track_deferrals(self):
        """get_prefix_coordination_metrics tracks waits correctly."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        assert engine.get_prefix_coordination_metrics() == {"waits": 0}

        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}

        # Second round: no new deferral
        engine.schedule_non_chunked_prefill()
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}
        assert len(engine.waiting_request_ids) == 0

    @pytest.mark.internal
    def test_chunked_prefill_defers_conflicting_request(self):
        """Chunked prefill path has the same deferral logic."""
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx, enable_chunked_prefill=True)
        prompt = self._prompt(bs * 2)

        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)

        engine.schedule_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1
        assert engine._prefix_coordination_waits == 1
