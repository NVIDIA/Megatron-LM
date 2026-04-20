# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from collections import deque

import numpy as np
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


class PrefixCachingTestBase:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _mamba_config():
        from megatron.core.inference.config import MambaInferenceStateConfig

        return MambaInferenceStateConfig(
            layer_type_list=["*", "M", "*", "M"],
            conv_states_shape=(4, 8),
            ssm_states_shape=(4, 16),
            conv_states_dtype=torch.float32,
            ssm_states_dtype=torch.float32,
        )

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
        mamba_config=None,
        prefix_caching_mamba_gb=None,
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
            mamba_inference_state_config=mamba_config,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_eviction_policy=prefix_caching_eviction_policy,
            prefix_caching_mamba_gb=prefix_caching_mamba_gb,
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
    def _mamba_allocate_and_register(ctx, bids):
        """Allocate Mamba cache slots and register hashes for a list of block IDs."""
        msa = ctx.mamba_slot_allocator
        alloc = ctx.kv_block_allocator
        slots = msa.allocate_slots_batch(bids)
        bid_tensor = torch.tensor(bids, dtype=torch.int64, device=alloc.block_hashes.device)
        hashes = alloc.block_hashes[bid_tensor].tolist()
        msa.register_block_hashes_batch(bids, hashes)
        return slots


class _StubEngine(DynamicInferenceEngine):

    def __init__(self, context: DynamicInferenceContext, *, enable_chunked_prefill=False):
        self.context = context
        self.enable_chunked_prefill = enable_chunked_prefill
        self._prefix_coordination_waits = 0
        self._loop = asyncio.new_event_loop()
        self.waiting_request_ids: deque = deque()
        self.requests = {}
        self._generation_epoch = None


class TestPrefixCachingCore(PrefixCachingTestBase):

    @pytest.mark.internal
    def test_hash_computation(self):
        # determinism and range
        tokens = self._prompt(32)
        h1 = compute_block_hashes_batched(tokens, 32)
        h2 = compute_block_hashes_batched(tokens, 32)
        assert h1 == h2 and len(h1) == 1 and 1 <= h1[0] <= HASH_PRIME
        assert compute_block_hashes_batched(self._prompt(32, offset=1), 32)[0] != h1[0]

        # parent chaining: 4 blocks of all-zero tokens produce distinct hashes
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        zeros = torch.zeros(bs * 4, device=torch.cuda.current_device(), dtype=torch.long)
        hashes = compute_block_hashes_batched(zeros, bs)
        assert len(hashes) == 4 and len(set(hashes)) == 4

        # edge cases: short, empty, long
        assert compute_block_hashes_batched(self._prompt(bs // 2), bs) == []
        empty = torch.tensor([], device=torch.cuda.current_device(), dtype=torch.long)
        assert compute_block_hashes_batched(empty, bs) == []
        long_h = compute_block_hashes_batched(
            torch.arange(bs * 120, device=torch.cuda.current_device(), dtype=torch.long), bs
        )
        assert len(long_h) == 120 and all(v > 0 for v in long_h)

    @pytest.mark.internal
    def test_registration_and_discovery(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # blocks discoverable after add_request
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)
        h0, h1 = req.precomputed_block_hashes
        assert alloc.kv_hash_to_block_id.get(h0) == b0
        assert alloc.kv_hash_to_block_id.get(h1) == b1
        assert alloc.block_hashes[b0].item() == h0 and alloc.block_hashes[b1].item() == h1

        # partial block not registered
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        ctx2.add_request(self._req(ctx2, self._prompt(bs + bs // 2)))
        pb0, pb1 = self._block_ids(ctx2, 0, 2)
        assert alloc2.block_hashes[pb0].item() != -1
        assert alloc2.block_hashes[pb1].item() == -1

        # decode does not register completed blocks
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        ctx3.add_request(self._req(ctx3, self._prompt(bs + (bs - 1))))
        db0, db1 = self._block_ids(ctx3, 0, 2)
        assert alloc3.block_hashes[db0].item() != -1 and alloc3.block_hashes[db1].item() == -1
        active_mask = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx3.update_requests(active_mask, new_tokens)
        assert alloc3.block_hashes[db1].item() == -1

        # second request finds registered blocks
        ctx4 = self._ctx()
        alloc4 = ctx4.kv_block_allocator
        p4 = self._prompt(bs * 3)
        ctx4.add_request(self._req(ctx4, p4.clone()))
        req2 = self._req(ctx4, p4.clone(), request_id=2)
        for h in req2.precomputed_block_hashes:
            assert h in alloc4.kv_hash_to_block_id

    @pytest.mark.internal
    def test_block_sharing_patterns(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # N=10 identical prompts share all blocks
        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 3)
        avail_after_first = alloc.total_avail
        for i in range(2, 11):
            ctx.add_request(self._req(ctx, prompt.clone(), request_id=i))
        assert alloc.total_avail == avail_after_first
        for req_idx in range(1, 10):
            assert self._block_ids(ctx, req_idx, 3) == first_blocks
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 10

        # divergent suffix shares common prefix
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p1 = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p1))
        r1 = self._block_ids(ctx2, 0, 3)
        p2 = p1.clone()
        p2[bs * 2 :] += 1000
        ctx2.add_request(self._req(ctx2, p2, request_id=2))
        r2 = self._block_ids(ctx2, 1, 3)
        assert r2[0] == r1[0] and r2[1] == r1[1] and r2[2] != r1[2]
        assert alloc2.block_ref_counts[r1[0]].item() == 2
        assert alloc2.block_ref_counts[r1[2]].item() == 1

        # broken chain stops sharing: [X,W,Z] vs [X,Y,Z]
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        p3a = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3a))
        r3a = self._block_ids(ctx3, 0, 3)
        p3b = p3a.clone()
        p3b[bs : bs * 2] += 5000
        ctx3.add_request(self._req(ctx3, p3b, request_id=2))
        r3b = self._block_ids(ctx3, 1, 3)
        assert r3b[0] == r3a[0] and r3b[1] != r3a[1] and r3b[2] != r3a[2]
        assert alloc3.block_ref_counts[r3a[0]].item() == 2

    @pytest.mark.internal
    def test_prefill_token_savings(self):
        bs = 32

        # enabled vs disabled – use a non-block-aligned prompt so the second
        # request's effective prefill chunk after prefix skipping is > 1, which
        # avoids the single-token-chunk clamp in _compute_prefix_match.
        tail = 5
        ctx_on = self._ctx()
        prompt = self._prompt(bs * 4 + tail)
        ctx_on.add_request(self._req(ctx_on, prompt.clone()))
        ctx_on.add_request(self._req(ctx_on, prompt.clone(), request_id=2))
        ctx_off = self._ctx(enable_prefix_caching=False)
        ctx_off.add_request(self._req(ctx_off, prompt.clone(), enable_prefix_caching=False))
        ctx_off.add_request(
            self._req(ctx_off, prompt.clone(), request_id=2, enable_prefix_caching=False)
        )
        # With caching: first request prefills all tokens, second skips 4 full blocks.
        assert ctx_on.lifetime_prefill_token_count == (bs * 4 + tail) + tail
        assert ctx_off.lifetime_prefill_token_count == (bs * 4 + tail) * 2

        # partial match reduces proportionally
        ctx2 = self._ctx()
        p2a = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2a.clone()))
        p2b = p2a.clone()
        p2b[bs * 2 :] += 1000
        ctx2.add_request(self._req(ctx2, p2b, request_id=2))
        assert ctx2.lifetime_prefill_token_count == bs * 3 + bs

        # full match: duplicates skip all full cached blocks
        tail = 5
        ctx3 = self._ctx()
        alloc3 = ctx3.kv_block_allocator
        p3 = self._prompt(bs * 3 + tail)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        tokens_after = ctx3.active_token_count
        first_blocks = self._block_ids(ctx3, 0, 3)
        for i in range(5):
            ctx3.add_request(self._req(ctx3, p3.clone(), request_id=i + 2))
            assert ctx3.request_query_lengths[i + 1].item() == tail
        assert ctx3.active_token_count - tokens_after == 5 * tail
        for bid in first_blocks:
            assert alloc3.block_ref_counts[bid].item() == 6
        assert ctx3.lifetime_prefill_token_count == (bs * 3 + tail) + 5 * tail

        # no match: full prompt added
        ctx4 = self._ctx()
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2)))
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2, offset=9000), request_id=2))
        assert ctx4.lifetime_prefill_token_count == bs * 2 + bs * 2

    @pytest.mark.internal
    def test_block_allocation_with_prefix(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator

        # matched blocks not allocated from pool
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        first_blocks = self._block_ids(ctx, 0, 4)
        avail = alloc.total_avail
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        assert self._block_ids(ctx, 1, 4) == first_blocks and alloc.total_avail == avail

        # extended prompt allocates only new blocks
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p2a = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2a))
        avail2 = alloc2.total_avail
        p2b = torch.cat([p2a, self._prompt(bs * 2, offset=1000)])
        ctx2.add_request(self._req(ctx2, p2b, request_id=2))
        assert alloc2.total_avail == avail2 - 2

        # check_availability accounts for prefix match
        ctx3 = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc3 = ctx3.kv_block_allocator
        p3 = self._prompt(ctx3.block_size_tokens * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        while alloc3.total_avail > 0:
            alloc3.allocate_memory_blocks(1)
        _, _, kv_available = ctx3.check_availability(self._req(ctx3, p3.clone(), request_id=2))
        assert kv_available

    @pytest.mark.internal
    def test_ref_count_lru(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 2)

        # decrement preserves cached blocks
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        assert alloc.block_ref_counts[b0].item() == 2
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1 and b0_hash in alloc.kv_hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0 and b0_hash in alloc.kv_hash_to_block_id

        # cached blocks reused by new request
        ctx2 = self._ctx()
        alloc2 = ctx2.kv_block_allocator
        p2 = self._prompt(bs * 2)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        cb0, cb1 = self._block_ids(ctx2, 0, 2)
        ctx2.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx2.total_request_count = 0
        assert alloc2.block_ref_counts[cb0].item() == 0
        ctx2.add_request(self._req(ctx2, p2.clone(), request_id=2))
        assert self._block_ids(ctx2, 0, 2) == [cb0, cb1]
        assert alloc2.block_ref_counts[cb0].item() == 1

        # eviction frees oldest cached first
        ctx3 = self._ctx(buffer_size_gb=0.01, rounder=1)
        alloc3 = ctx3.kv_block_allocator
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 2)))
        active_blocks = ctx3.request_to_kv_block_ids[0][:2].clone()
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 2, offset=5000), request_id=2))
        ctx3.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        ctx3.total_request_count = 1
        for i in range(20):
            try:
                ctx3.add_request(
                    self._req(
                        ctx3, self._prompt(bs * 2, offset=(i + 10) * 1000), request_id=i + 100
                    )
                )
            except Exception:
                break
        for bid in active_blocks:
            assert alloc3.block_ref_counts[bid.item()].item() == 1

    @pytest.mark.internal
    def test_ref_count_refzero(self):
        bs = 32

        # deregisters on last release
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone()))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2))
        b0, b1 = self._block_ids(ctx, 0, 2)
        b0_hash = alloc.block_hashes[b0].item()
        avail_before = alloc.total_avail
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        assert alloc.block_ref_counts[b0].item() == 1 and b0_hash in alloc.kv_hash_to_block_id
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([1]))
        assert alloc.block_ref_counts[b0].item() == 0 and b0_hash not in alloc.kv_hash_to_block_id
        assert alloc.block_hashes[b0].item() == -1 and alloc.block_hashes[b1].item() == -1
        assert alloc.total_avail == avail_before + 2

        # released blocks not discoverable
        ctx2 = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc2 = ctx2.kv_block_allocator
        p2 = self._prompt(bs * 2)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        ctx2.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        ctx2.total_request_count = 0
        ctx2.add_request(self._req(ctx2, p2.clone(), request_id=2))
        new_blocks = self._block_ids(ctx2, 0, 2)
        assert alloc2.block_ref_counts[new_blocks[0]].item() == 1


class TestDisabledAndEngineScheduling(PrefixCachingTestBase):

    def _engine(self, ctx, **kwargs):
        return _StubEngine(ctx, **kwargs)

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
    def test_disabled_mode(self):
        # no sharing
        ctx = self._ctx(enable_prefix_caching=False)
        bs = ctx.block_size_tokens
        prompt = self._prompt(bs * 2)
        ctx.add_request(self._req(ctx, prompt.clone(), enable_prefix_caching=False))
        r1 = set(self._block_ids(ctx, 0, 2))
        ctx.add_request(self._req(ctx, prompt.clone(), request_id=2, enable_prefix_caching=False))
        r2 = set(self._block_ids(ctx, 1, 2))
        assert r1.isdisjoint(r2)

        # no caching attrs on disabled allocator
        alloc_d = ctx.kv_block_allocator
        assert not hasattr(alloc_d, 'block_hashes')
        assert not hasattr(alloc_d, 'kv_hash_to_block_id')
        assert not hasattr(alloc_d, 'block_ref_counts')

        # REF_ZERO lacks timestamps
        ctx_rz = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        assert not hasattr(ctx_rz.kv_block_allocator, 'block_timestamps')

    @pytest.mark.internal
    def test_scheduling_deferral_and_resolution(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx)
        prompt = self._prompt(bs * 2)

        # shared prefix defers second request
        req1 = self._req(ctx, prompt.clone())
        self._add_to_waiting(engine, ctx, req1)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        self._add_to_waiting(engine, ctx, req2)
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1 and engine.waiting_request_ids[0] == 2
        assert engine._prefix_coordination_waits == 1

        # deferred request schedulable after registration (round 2)
        engine.schedule_non_chunked_prefill()
        assert ctx.total_request_count == 2 and len(engine.waiting_request_ids) == 0

        # skip deferred to schedule non-conflicting
        ctx2 = self._ctx()
        engine2 = self._engine(ctx2)
        pa = self._prompt(bs * 2)
        pb = self._prompt(bs * 2, offset=5000)
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pa.clone()))
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pa.clone(), request_id=2))
        self._add_to_waiting(engine2, ctx2, self._req(ctx2, pb.clone(), request_id=3))
        engine2.schedule_non_chunked_prefill()
        assert ctx2.total_request_count == 2
        assert len(engine2.waiting_request_ids) == 1 and engine2.waiting_request_ids[0] == 2

        # registered prefix allows immediate scheduling
        ctx3 = self._ctx()
        engine3 = self._engine(ctx3)
        p3 = self._prompt(bs * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        self._add_to_waiting(engine3, ctx3, self._req(ctx3, p3.clone(), request_id=2))
        engine3.schedule_non_chunked_prefill()
        assert len(engine3.waiting_request_ids) == 0 and ctx3.total_request_count == 2
        assert engine3._prefix_coordination_waits == 0

        # metrics track deferrals
        ctx4 = self._ctx()
        engine4 = self._engine(ctx4)
        p4 = self._prompt(bs * 2)
        assert engine4.get_prefix_coordination_metrics() == {"waits": 0}
        self._add_to_waiting(engine4, ctx4, self._req(ctx4, p4.clone()))
        self._add_to_waiting(engine4, ctx4, self._req(ctx4, p4.clone(), request_id=2))
        engine4.schedule_non_chunked_prefill()
        assert engine4.get_prefix_coordination_metrics() == {"waits": 1}
        engine4.schedule_non_chunked_prefill()
        assert engine4.get_prefix_coordination_metrics() == {"waits": 1}
        assert len(engine4.waiting_request_ids) == 0

    @pytest.mark.internal
    def test_chunked_prefill_deferral(self):
        ctx = self._ctx()
        bs = ctx.block_size_tokens
        engine = self._engine(ctx, enable_chunked_prefill=True)
        prompt = self._prompt(bs * 2)
        self._add_to_waiting(engine, ctx, self._req(ctx, prompt.clone()))
        self._add_to_waiting(engine, ctx, self._req(ctx, prompt.clone(), request_id=2))
        engine.schedule_chunked_prefill()
        assert ctx.total_request_count == 1
        assert len(engine.waiting_request_ids) == 1 and engine._prefix_coordination_waits == 1


class TestMambaPrefixCaching(PrefixCachingTestBase):

    def _mctx(self, **kwargs):
        defaults = dict(
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=256,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return self._ctx(**defaults)

    @pytest.mark.internal
    def test_hybrid_memory_only(self):
        # hybrid model: no prefill skipping, but blocks reused for memory savings
        ctx = self._ctx(mamba_config=self._mamba_config())
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        prompt = self._prompt(bs * 3)
        assert ctx.is_hybrid_model

        req1 = self._req(ctx, prompt.clone())
        ctx.add_request(req1)
        first_blocks = self._block_ids(ctx, 0, 3)
        avail = alloc.total_avail
        tokens_after = ctx.active_token_count

        req2 = self._req(ctx, prompt.clone(), request_id=2)
        # no prefill skipping
        (matched, _, _, _, prefix_skip, eff_chunk) = ctx._compute_prefix_match(req2, len(prompt))
        assert len(matched) == 3 and prefix_skip == 0 and eff_chunk == len(prompt)

        ctx.add_request(req2)
        # blocks reused (pool unchanged), ref counts incremented
        assert alloc.total_avail == avail
        for bid in first_blocks:
            assert alloc.block_ref_counts[bid].item() == 2
        # all tokens processed (none skipped)
        assert ctx.active_token_count - tokens_after == len(prompt)
        assert ctx.request_kv_length_offsets[1].item() == 0

    @pytest.mark.internal
    def test_mamba_cache_lifecycle(self):
        ctx = self._mctx()
        bs = ctx.block_size_tokens

        # allocated when prefix_caching_mamba_gb is set
        assert ctx.mamba_slot_allocator.max_slots > 0
        assert ctx.mamba_slot_allocator.conv_states is not None
        assert ctx.mamba_slot_allocator.free_count == ctx.mamba_slot_allocator.max_slots

        # not allocated when None
        ctx_none = self._mctx(prefix_caching_mamba_gb=None)
        assert ctx_none.mamba_slot_allocator is None

        # store and restore round-trips
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt.clone())
        ctx.add_request(req)
        block_id = ctx.request_to_kv_block_ids[0][0].item()
        slot = ctx.mamba_slot_allocator.allocate_slots_batch([block_id])[0]
        for layer_idx in range(ctx.num_mamba_layers):
            ssm = torch.ones_like(ctx.mamba_slot_allocator.ssm_states[layer_idx, slot]) * (
                layer_idx + 1
            )
            conv = torch.ones_like(ctx.mamba_slot_allocator.conv_states[layer_idx, slot]) * (
                layer_idx + 10
            )
            ctx.mamba_slot_allocator.store_from_tensors(block_id, layer_idx, ssm, conv)
        assert ctx.mamba_slot_allocator.has_state(block_id)
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        ctx.add_request(req2)
        assert ctx.mamba_slot_allocator.restore_to_live(1, block_id)
        mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[1].item()
        for layer_idx in range(ctx.num_mamba_layers):
            assert torch.allclose(
                ctx.mamba_ssm_states[layer_idx, mamba_idx],
                torch.ones_like(ctx.mamba_ssm_states[layer_idx, mamba_idx]) * (layer_idx + 1),
            )

        # invalidate frees slot
        ctx3 = self._mctx()
        p3 = self._prompt(bs * 2)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        bid3 = ctx3.request_to_kv_block_ids[0][0].item()
        ctx3.mamba_slot_allocator.allocate_slots_batch([bid3])
        assert ctx3.mamba_slot_allocator.has_state(bid3)
        free_before = ctx3.mamba_slot_allocator.free_count
        ctx3.mamba_slot_allocator.invalidate_block(bid3)
        assert (
            not ctx3.mamba_slot_allocator.has_state(bid3)
            and ctx3.mamba_slot_allocator.free_count == free_before + 1
        )

        # slot reuse for same block
        ctx4 = self._mctx()
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 2)))
        bid4 = ctx4.request_to_kv_block_ids[0][0].item()
        s1, s2 = ctx4.mamba_slot_allocator.allocate_slots_batch([bid4, bid4])
        assert s1 == s2

        # two-map hash design: kv and mamba maps are independent
        ctx5 = self._mctx()
        alloc5 = ctx5.kv_block_allocator
        p5 = self._prompt(bs * 3)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        msa5 = ctx5.mamba_slot_allocator
        assert len(alloc5.kv_hash_to_block_id) == 3 and len(msa5.hash_to_block_id) == 0
        self._mamba_allocate_and_register(ctx5, self._block_ids(ctx5, 0, 3)[:2])
        assert len(alloc5.kv_hash_to_block_id) == 3 and len(msa5.hash_to_block_id) == 2

        # find_mamba_match_count
        ctx6 = self._mctx()
        alloc6 = ctx6.kv_block_allocator
        p6 = self._prompt(bs * 4)
        ctx6.add_request(self._req(ctx6, p6.clone()))
        msa6 = ctx6.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx6, self._block_ids(ctx6, 0, 4)[:2])
        engine6 = _StubEngine(ctx6)
        assert engine6._find_mamba_match_count(self._req(ctx6, p6.clone(), request_id=2)) == 2
        # no match when no mamba hashes registered
        ctx7 = self._mctx()
        ctx7.add_request(self._req(ctx7, self._prompt(bs * 3)))
        assert (
            _StubEngine(ctx7)._find_mamba_match_count(
                self._req(ctx7, self._prompt(bs * 3), request_id=2)
            )
            == 0
        )

        # allocate, free, re-allocate
        ctx8 = self._mctx()
        ctx8.add_request(self._req(ctx8, self._prompt(bs * 3)))
        bids8 = self._block_ids(ctx8, 0, 3)
        initial_free = ctx8.mamba_slot_allocator.free_count
        ctx8.mamba_slot_allocator.allocate_slots_batch(bids8)
        assert ctx8.mamba_slot_allocator.free_count == initial_free - 3
        ctx8.mamba_slot_allocator.invalidate_block(bids8[0])
        assert (
            ctx8.mamba_slot_allocator.free_count == initial_free - 2
            and not ctx8.mamba_slot_allocator.has_state(bids8[0])
        )
        ctx8.mamba_slot_allocator.allocate_slots_batch([bids8[0]])
        assert (
            ctx8.mamba_slot_allocator.free_count == initial_free - 3
            and ctx8.mamba_slot_allocator.has_state(bids8[0])
        )

    @pytest.mark.internal
    def test_mamba_prefill_skip_and_zero_prefill(self):
        # mamba match limits prefill skip
        ctx = self._mctx()
        bs = ctx.block_size_tokens
        alloc = ctx.kv_block_allocator
        msa = ctx.mamba_slot_allocator
        prompt = self._prompt(bs * 3)
        ctx.add_request(self._req(ctx, prompt.clone()))
        self._mamba_allocate_and_register(ctx, self._block_ids(ctx, 0, 3)[:1])
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        req2._mamba_num_matched_blocks = 1
        (matched, _, _, _, prefix_skip, eff_chunk) = ctx._compute_prefix_match(req2, len(prompt))
        assert len(matched) == 3 and prefix_skip == bs and eff_chunk == len(prompt) - bs

        # no mamba match means no skip
        ctx2 = self._mctx()
        p2 = self._prompt(bs * 3)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        req2b = self._req(ctx2, p2.clone(), request_id=2)
        req2b._mamba_num_matched_blocks = 0
        (m2, _, _, _, ps2, ec2) = ctx2._compute_prefix_match(req2b, len(p2))
        assert len(m2) == 3 and ps2 == 0 and ec2 == len(p2)

        # zero prefill for hybrid (mamba-cached, block-aligned)
        ctx3 = self._mctx()
        p3 = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        self._mamba_allocate_and_register(ctx3, self._block_ids(ctx3, 0, 3))
        req3 = self._req(ctx3, p3.clone(), request_id=2)
        req3._mamba_num_matched_blocks = 3
        (m3, _, _, _, ps3, ec3) = ctx3._compute_prefix_match(req3, len(p3))
        assert len(m3) == 3 and ps3 == 2 * bs and ec3 == bs

        # KV-only prefix skip with non-block-aligned prompt: all 3 full blocks
        # are skipped and only the trailing tokens remain for prefill.
        ctx4 = self._ctx()
        bs4 = ctx4.block_size_tokens
        tail = 5
        p4 = self._prompt(bs4 * 3 + tail)
        req4a = self._req(ctx4, p4.clone())
        ctx4.add_request(req4a)
        req4b = self._req(ctx4, p4.clone(), request_id=2)
        (m4, _, _, _, ps4, ec4) = ctx4._compute_prefix_match(req4b, len(p4))
        assert len(m4) == 3 and ps4 == 3 * bs4 and ec4 == tail
        ctx4.add_request(req4b)

        # KV eviction invalidates mamba
        ctx5 = self._mctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc5 = ctx5.kv_block_allocator
        msa5 = ctx5.mamba_slot_allocator
        p5 = self._prompt(bs * 2)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        bid5 = ctx5.request_to_kv_block_ids[0][0].item()
        bh5 = alloc5.block_hashes[bid5].item()
        self._mamba_allocate_and_register(ctx5, [bid5])
        assert msa5.has_state(bid5) and bh5 in msa5.hash_to_block_id
        ctx5.release_memory_blocks_from_request_indexes([0])
        assert not msa5.has_state(bid5) and bh5 not in msa5.hash_to_block_id

    @pytest.mark.internal
    def test_mamba_intermediate_offsets(self):
        bs = 256

        # KV divergence offsets
        ctx = self._mctx(block_size_tokens=bs)
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        msa = ctx.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx, self._block_ids(ctx, 0, 4)[:2])
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        req2._mamba_num_matched_blocks = 2
        (matched, _, _, overall, prefix_skip, _) = ctx._compute_prefix_match(req2, len(prompt))
        # Copy block IDs to slot 1 so compute_and_store_offsets can resolve EOS block
        ctx.request_to_kv_block_ids[1] = ctx.request_to_kv_block_ids[0]
        msa.compute_and_store_offsets(
            req2,
            1,
            prefix_skip,
            len(prompt),
            len(matched),
            [ctx.request_to_kv_block_ids[0][i].item() for i in range(len(matched))],
            overall,
        )
        # Penultimate block offset (block 2 boundary) is a valid intermediate
        count = msa._intermediate_counts_gpu[1].item()
        if count > 0:
            offsets = msa._intermediate_offsets_gpu[1, :count].tolist()
            for o in offsets:
                assert o > 0 and o % 128 == 0
        assert msa._eos_cache_block_id_gpu[1].item() >= 0

        # non-aligned prompt produces last_aligned intermediate offset
        ctx2 = self._mctx(block_size_tokens=bs)
        prompt_len = bs * 3 + bs // 2
        p2 = self._prompt(prompt_len)
        ctx2.add_request(self._req(ctx2, p2.clone()))
        msa2 = ctx2.mamba_slot_allocator
        self._mamba_allocate_and_register(ctx2, self._block_ids(ctx2, 0, 3)[:2])
        req2b = self._req(ctx2, p2.clone(), request_id=2)
        req2b._mamba_num_matched_blocks = 2
        ctx2.add_request(req2b)
        count2 = msa2._intermediate_counts_gpu[1].item()
        if count2 > 0:
            offsets = msa2._intermediate_offsets_gpu[1, :count2].tolist()
            for o in offsets:
                assert o > 0 and o % 128 == 0
        assert msa2._eos_cache_block_id_gpu[1].item() < 0

        # block-aligned prompts set EOS cache block ID
        ctx3 = self._mctx(block_size_tokens=bs)
        p3 = self._prompt(bs * 3)
        ctx3.add_request(self._req(ctx3, p3.clone()))
        req3 = self._req(ctx3, p3.clone(), request_id=2)
        req3._mamba_num_matched_blocks = 0
        ctx3.add_request(req3)
        assert ctx3.mamba_slot_allocator._eos_cache_block_id_gpu[1].item() >= 0

        # intermediate output buffers are pre-allocated
        ctx4 = self._mctx()
        msa4 = ctx4.mamba_slot_allocator
        assert msa4.intermediate_ssm_out.shape[0] == ctx4.num_mamba_layers
        assert msa4.intermediate_conv_out.shape[0] == ctx4.num_mamba_layers
        assert msa4.intermediate_ssm_out.shape[1] == msa4.max_intermediate_count

        # store_from_live copies all layers
        ctx5 = self._mctx()
        msa5 = ctx5.mamba_slot_allocator
        p5 = self._prompt(ctx5.block_size_tokens * 2)
        ctx5.add_request(self._req(ctx5, p5.clone()))
        bid5 = ctx5.request_to_kv_block_ids[0][0].item()
        slot5 = msa5.allocate_slots_batch([bid5])[0]
        mamba_idx = ctx5.mamba_metadata.request_to_mamba_state_idx[0].item()
        for layer in range(ctx5.num_mamba_layers):
            ctx5.mamba_conv_states[layer, mamba_idx] = layer + 1.0
            ctx5.mamba_ssm_states[layer, mamba_idx] = layer + 100.0
        msa5.store_from_live_batch([slot5], [0])
        for layer in range(ctx5.num_mamba_layers):
            assert torch.allclose(
                ctx5.mamba_slot_allocator.conv_states[layer, slot5],
                torch.full_like(ctx5.mamba_slot_allocator.conv_states[layer, slot5], layer + 1.0),
            )


class TestMixedCachedAndFreshPrefill(PrefixCachingTestBase):

    def _setup_mixed_batch(self, model_type):
        """Set up mixed batch: req0 (decode), reqs 1-4 (mixed cached/fresh prefill).

        Uses 2-block + tail prompts so cached requests skip 2 full blocks and
        prefill only the tail, avoiding the single-token-chunk clamp while still
        producing distinct query lengths for cached vs fresh requests.
        """
        if model_type == "gpt":
            ctx = self._ctx(block_size_tokens=32)
        else:
            ctx = self._ctx(
                mamba_config=self._mamba_config(),
                prefix_caching_mamba_gb=0.01,
                block_size_tokens=256,
                max_sequence_length=4096,
            )
        bs = ctx.block_size_tokens
        tail = 5
        prompt_len = bs * 2 + tail

        prompt0 = self._prompt(prompt_len)
        req0 = self._req(ctx, prompt0.clone())
        ctx.add_request(req0)

        vocab_size = prompt_len + 50
        block_hash = req0.precomputed_block_hashes[0]

        if model_type == "hybrid":
            block_ids_0 = self._block_ids(ctx, 0, 2)
            for bid in block_ids_0:
                bh = ctx.kv_block_allocator.block_hashes[bid].item()
                ctx.mamba_slot_allocator.register_block_hashes_batch([bid], [bh])

        ctx.request_kv_length_offsets[0] += prompt_len
        ctx.request_query_lengths[0] = 1
        ctx.request_last_kv_block_offset[0] = 0
        ctx.num_prefill_requests = 0
        ctx.active_token_count = 1
        ctx.token_to_input_ids[0] = 42
        ctx.token_to_pos_ids[0] = prompt_len
        ctx.token_to_request_idx[0] = 0

        req1 = self._req(ctx, prompt0.clone(), request_id=2)
        req2 = self._req(ctx, self._prompt(prompt_len, offset=50), request_id=3)
        req3 = self._req(ctx, prompt0.clone(), request_id=4)
        req4 = self._req(ctx, self._prompt(prompt_len, offset=40), request_id=5)

        if model_type == "hybrid":
            req1._mamba_num_matched_blocks = 2
            req2._mamba_num_matched_blocks = 0
            req3._mamba_num_matched_blocks = 2
            req4._mamba_num_matched_blocks = 0

        for r in [req1, req2, req3, req4]:
            ctx.add_request(r)

        return ctx, bs, tail, prompt_len, vocab_size, block_hash

    @pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
    @pytest.mark.internal
    def test_mixed_batch(self, model_type):
        ctx, bs, tail, prompt_len, vocab_size, block_hash = self._setup_mixed_batch(model_type)

        # Cached requests (req1/req3) skip 2 full blocks → query_length == tail.
        # Fresh requests (req2/req4) have no match → query_length == prompt_len.
        cached_ql = tail
        fresh_ql = prompt_len

        # query lengths: decode=1, cached=tail, fresh=prompt_len
        assert ctx.request_query_lengths[0].item() == 1
        assert ctx.request_query_lengths[1].item() == cached_ql
        assert ctx.request_query_lengths[2].item() == fresh_ql
        assert ctx.request_query_lengths[3].item() == cached_ql
        assert ctx.request_query_lengths[4].item() == fresh_ql
        assert ctx.active_token_count == 1 + 2 * cached_ql + 2 * fresh_ql

        # last_token_logits
        ctx.initialize_attention_state()
        logits = torch.randn(
            1, ctx.padded_active_token_count, vocab_size, device=torch.cuda.current_device()
        )
        result = ctx.last_token_logits(logits)
        assert result.shape == (5, vocab_size)

        # calculate_log_probs
        new_tokens = torch.randint(0, vocab_size, (5,), device=torch.cuda.current_device())
        log_probs_list, _ = ctx.calculate_log_probs(logits, new_tokens)
        assert len(log_probs_list) == 5
        assert len(log_probs_list[0]) == 1
        assert len(log_probs_list[1]) == cached_ql
        assert len(log_probs_list[2]) == fresh_ql
        assert len(log_probs_list[3]) == cached_ql
        assert len(log_probs_list[4]) == fresh_ql


class TestMambaSlotAllocator(PrefixCachingTestBase):

    def _mctx(self, **kwargs):
        defaults = dict(
            mamba_config=self._mamba_config(),
            prefix_caching_mamba_gb=0.01,
            block_size_tokens=256,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return self._ctx(**defaults)

    @pytest.mark.internal
    def test_allocate_slots_batch(self):
        ctx = self._mctx()
        bs = ctx.block_size_tokens
        msa = ctx.mamba_slot_allocator

        # Basic batch: allocate 3 new slots, verify unique slots and mappings
        prompt = self._prompt(bs * 4)
        ctx.add_request(self._req(ctx, prompt.clone()))
        bids = self._block_ids(ctx, 0, 3)
        initial_free = msa.free_count
        slots = msa.allocate_slots_batch(bids)
        assert len(slots) == 3
        assert len(set(slots)) == 3  # all unique
        assert msa.free_count == initial_free - 3
        for bid, slot in zip(bids, slots):
            assert msa.block_to_slot[bid].item() == slot
            assert msa.slot_to_block[slot].item() == bid

        # Existing slots: same block_ids return same slots without consuming pool
        free_before = msa.free_count
        slots2 = msa.allocate_slots_batch(bids)
        assert slots2 == slots
        assert msa.free_count == free_before

        # Dedup: same block_id twice, only one free slot consumed
        ctx2 = self._mctx()
        ctx2.add_request(self._req(ctx2, self._prompt(bs * 2)))
        bid_new = self._block_ids(ctx2, 0, 1)[0]
        msa2 = ctx2.mamba_slot_allocator
        free_before2 = msa2.free_count
        dup_slots = msa2.allocate_slots_batch([bid_new, bid_new])
        assert dup_slots[0] == dup_slots[1]
        assert msa2.free_count == free_before2 - 1

        # Mixed: pre-allocated + new in one call
        ctx3 = self._mctx()
        ctx3.add_request(self._req(ctx3, self._prompt(bs * 3)))
        bids3 = self._block_ids(ctx3, 0, 3)
        msa3 = ctx3.mamba_slot_allocator
        pre_slot = msa3.allocate_slots_batch([bids3[0]])[0]
        free_before3 = msa3.free_count
        mixed_slots = msa3.allocate_slots_batch(bids3)
        assert mixed_slots[0] == pre_slot
        assert msa3.free_count == free_before3 - 2  # only 2 new

        # Eviction: exhaust free pool, verify eviction fires and returns valid slots
        ctx4 = self._mctx(prefix_caching_mamba_gb=0.001)
        msa4 = ctx4.mamba_slot_allocator
        total_slots = msa4.max_slots
        ctx4.add_request(self._req(ctx4, self._prompt(bs * 4)))
        bids4 = self._block_ids(ctx4, 0, 4)
        # Allocate all available slots by filling the free pool
        fill_bids = bids4[: min(total_slots, 4)]
        fill_slots = msa4.allocate_slots_batch(fill_bids)
        assert len(fill_slots) == len(fill_bids)
        # If we can exhaust the pool, test eviction
        if total_slots <= 4:
            assert msa4.free_count == 0
            # Set ref counts to 0 so blocks are evictable
            for bid in fill_bids:
                ctx4.kv_block_allocator.block_ref_counts[bid] = 0
            # Invalidate old slots, then reallocate to test eviction path
            for bid in fill_bids:
                msa4.invalidate_block(bid)
            evict_slots = msa4.allocate_slots_batch(fill_bids)
            assert len(evict_slots) == len(fill_bids)

    @pytest.mark.internal
    def test_commit_intermediate_states_batched(self):
        ctx = self._mctx(block_size_tokens=256)
        bs = ctx.block_size_tokens
        msa = ctx.mamba_slot_allocator
        alloc = ctx.kv_block_allocator
        metadata = ctx.mamba_metadata

        # Set up context with a prefill request that has block-aligned prompt
        prompt = self._prompt(bs * 3)
        req = self._req(ctx, prompt.clone())
        req._mamba_num_matched_blocks = 0
        ctx.add_request(req)

        # initialize_attention_state sets batch_dimensions and mamba metadata
        ctx.initialize_attention_state()

        # Determine prefill_start for this batch
        prefill_start = ctx.paused_request_count + ctx.batch_dimensions.decode_req_count
        ctx_idx = prefill_start  # first prefill request

        # Write known patterns to intermediate output buffers
        for layer in range(ctx.num_mamba_layers):
            msa.intermediate_ssm_out[layer, 0] = layer + 1.0
            msa.intermediate_conv_out[layer, 0] = layer + 100.0

        # Set up intermediate offsets: 1 intermediate at src_offset=0
        bid0 = ctx.request_to_kv_block_ids[ctx_idx][0].item()
        msa._intermediate_block_ids_gpu[ctx_idx, 0] = bid0
        msa._intermediate_offsets_gpu[ctx_idx, 0] = 128
        msa._intermediate_counts_gpu[ctx_idx] = 1
        msa._has_intermediates = True

        # Set metadata fields that would normally be set by _update_intermediate_offsets
        metadata.intermediate_count = 1
        metadata.per_request_intermediate_counts = [1]

        # Set up EOS block (block-aligned prompt)
        eos_bid = ctx.request_to_kv_block_ids[ctx_idx][2].item()
        msa._eos_cache_block_id_gpu[ctx_idx] = eos_bid

        # Write known patterns to live mamba state for EOS copy
        mamba_idx = metadata.request_to_mamba_state_idx[ctx_idx].item()
        for layer in range(ctx.num_mamba_layers):
            ctx.mamba_conv_states[layer, mamba_idx] = layer + 200.0
            ctx.mamba_ssm_states[layer, mamba_idx] = layer + 300.0

        # Call the batched commit
        msa.commit_intermediate_states()

        # Verify intermediate state was copied to correct slot
        slot0 = msa.block_to_slot[bid0].item()
        assert slot0 >= 0
        for layer in range(ctx.num_mamba_layers):
            assert torch.allclose(
                msa.ssm_states[layer, slot0],
                torch.full_like(msa.ssm_states[layer, slot0], layer + 1.0),
            )
            assert torch.allclose(
                msa.conv_states[layer, slot0],
                torch.full_like(msa.conv_states[layer, slot0], layer + 100.0),
            )

        # Verify EOS state was copied from live buffer
        eos_slot = msa.block_to_slot[eos_bid].item()
        assert eos_slot >= 0
        for layer in range(ctx.num_mamba_layers):
            assert torch.allclose(
                msa.conv_states[layer, eos_slot],
                torch.full_like(msa.conv_states[layer, eos_slot], layer + 200.0),
            )
            assert torch.allclose(
                msa.ssm_states[layer, eos_slot],
                torch.full_like(msa.ssm_states[layer, eos_slot], layer + 300.0),
            )

        # Verify hash_to_block_id updated for valid hashes
        bid0_hash = alloc.block_hashes[bid0].item()
        eos_hash = alloc.block_hashes[eos_bid].item()
        if bid0_hash > 0:
            assert msa.hash_to_block_id.get(bid0_hash) == bid0
        if eos_hash > 0:
            assert msa.hash_to_block_id.get(eos_hash) == eos_bid

        # Verify _has_intermediates cleared
        assert not msa._has_intermediates


class TestPerBlockRouting(PrefixCachingTestBase):
    """Tests for per-block routing storage and reconstruction."""

    @pytest.mark.internal
    def test_store_and_get_block_routing(self):
        """Verify store_block_routing / get_block_routing round-trip."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        num_layers, topk = 4, 2

        # Allocate a block
        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()

        # Store routing for some positions
        positions = np.array([0, 1, 2])
        routing = np.random.randint(-100, 100, size=(3, num_layers, topk), dtype=np.int16)
        alloc.store_block_routing(bid, positions, routing)

        # Retrieve and verify
        stored = alloc.get_block_routing(bid)
        assert stored is not None
        assert isinstance(stored, np.ndarray)
        assert stored.shape == (bs, num_layers, topk)
        assert np.allclose(stored[:3], routing)
        # Remaining positions should be zero
        assert (stored[3:] == 0).all()

    @pytest.mark.internal
    def test_routing_cleared_on_allocate(self):
        """Routing data is cleared when a block is re-allocated."""
        ctx = self._ctx(enable_prefix_caching=False)
        alloc = ctx.kv_block_allocator

        # Allocate, store routing, release, re-allocate
        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        positions = np.array([0])
        routing = np.random.randint(-100, 100, size=(1, 4, 2), dtype=np.int16)
        alloc.store_block_routing(bid, positions, routing)
        assert alloc.get_block_routing(bid) is not None

        alloc.release_memory_blocks(block_ids)
        # After release, routing still present (persists until re-alloc)
        assert alloc.get_block_routing(bid) is not None

        # Re-allocate the same block
        new_ids = alloc.allocate_memory_blocks(1)
        new_bid = new_ids[0].item()
        # The re-allocated block should have routing cleared
        assert alloc.get_block_routing(new_bid) is None

    @pytest.mark.internal
    def test_routing_cleared_on_reset(self):
        """Routing data is cleared on allocator reset."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator

        block_ids = alloc.allocate_memory_blocks(1)
        bid = block_ids[0].item()
        alloc.store_block_routing(
            bid, np.array([0]), np.random.randint(-100, 100, size=(1, 4, 2), dtype=np.int16)
        )
        assert alloc.get_block_routing(bid) is not None

        alloc.reset()
        assert alloc.get_block_routing(bid) is None
        assert len(alloc.block_routing) == 0

    @pytest.mark.internal
    def test_routing_persists_through_deregister(self):
        """Routing data persists through block deregister (needed for reconstruction)."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.REF_ZERO)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        # Add a request so blocks get allocated and registered
        prompt = self._prompt(bs * 2)
        req = self._req(ctx, prompt)
        ctx.add_request(req)
        b0, b1 = self._block_ids(ctx, 0, 2)

        # Store routing for both blocks
        for bid in [b0, b1]:
            alloc.store_block_routing(
                bid, np.arange(bs), np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
            )

        # Release blocks (REF_ZERO deregisters immediately)
        blocks = ctx.request_to_kv_block_ids[0]
        valid_blocks = blocks[blocks >= 0]
        alloc.release_memory_blocks(valid_blocks)

        # Routing data should still be present
        assert alloc.get_block_routing(b0) is not None
        assert alloc.get_block_routing(b1) is not None

    @pytest.mark.internal
    def test_reconstruct_routing_from_blocks(self):
        """Test reconstruction of routing indices from per-block storage."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens
        num_layers, topk = 4, 2

        # Allocate 3 blocks
        block_ids = alloc.allocate_memory_blocks(3)
        bids = block_ids.tolist()

        # Store routing for all positions in first two blocks (full)
        for bid in bids[:2]:
            alloc.store_block_routing(
                bid,
                np.arange(bs),
                np.arange(bs * num_layers * topk, dtype=np.int16).reshape(bs, num_layers, topk)
                + bid,
            )

        # Store routing for partial last block (e.g., 5 tokens)
        partial = 5
        alloc.store_block_routing(
            bids[2],
            np.arange(partial),
            np.arange(partial * num_layers * topk, dtype=np.int16).reshape(
                partial, num_layers, topk
            )
            + bids[2],
        )

        # total_routing_tokens = 2 full blocks + 5 partial = 2*bs + 5
        total_routing_tokens = 2 * bs + partial

        result = alloc.reconstruct_routing_from_blocks(bids, total_routing_tokens)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (total_routing_tokens, num_layers, topk)

        # Verify content: first block
        expected_b0 = (
            np.arange(bs * num_layers * topk, dtype=np.int16).reshape(bs, num_layers, topk)
            + bids[0]
        )
        assert np.allclose(result[:bs], expected_b0)

        # Verify content: partial last block
        expected_partial = (
            np.arange(partial * num_layers * topk, dtype=np.int16).reshape(
                partial, num_layers, topk
            )
            + bids[2]
        )
        assert np.allclose(result[2 * bs :], expected_partial)

    @pytest.mark.internal
    def test_reconstruct_returns_none_for_missing_block(self):
        """Reconstruction returns None if a block has no routing data."""
        ctx = self._ctx()
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        block_ids = alloc.allocate_memory_blocks(2)
        bids = block_ids.tolist()

        # Only store routing for the first block
        alloc.store_block_routing(
            bids[0], np.arange(bs), np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        )

        result = alloc.reconstruct_routing_from_blocks(bids, 2 * bs)
        assert result is None

    @pytest.mark.internal
    def test_routing_survives_prefix_match_lru(self):
        """In LRU mode, matched blocks' routing persists for the new request."""
        ctx = self._ctx(prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU)
        alloc = ctx.kv_block_allocator
        bs = ctx.block_size_tokens

        # First request: 2 full blocks
        prompt = self._prompt(bs * 2)
        req1 = self._req(ctx, prompt, request_id=1)
        ctx.add_request(req1)
        b0, b1 = self._block_ids(ctx, 0, 2)

        # Store routing for both blocks
        routing_b0 = np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        routing_b1 = np.random.randint(-100, 100, size=(bs, 4, 2), dtype=np.int16)
        alloc.store_block_routing(b0, np.arange(bs), routing_b0)
        alloc.store_block_routing(b1, np.arange(bs), routing_b1)

        # Release first request's blocks (LRU: blocks stay cached)
        blocks = ctx.request_to_kv_block_ids[0]
        valid_blocks = blocks[blocks >= 0]
        active_mask = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tokens = torch.tensor([100], device=torch.cuda.current_device())
        ctx.update_requests(active_mask, new_tokens)

        # Second request with same prefix should match
        req2 = self._req(ctx, prompt.clone(), request_id=2)
        ctx.add_request(req2)

        # The matched blocks should still have routing data
        assert alloc.get_block_routing(b0) is not None
        assert np.allclose(alloc.get_block_routing(b0), routing_b0)
        assert alloc.get_block_routing(b1) is not None
        assert np.allclose(alloc.get_block_routing(b1), routing_b1)
