# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise real chunked-prefill scheduling, allocation and lifecycle without a model forward."""

from collections import deque
from concurrent.futures import Future

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, RequestEntry
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord, Status
from tests.unit_tests.inference.contexts.test_dynamic_prefix_caching import PrefixCachingTestBase

pytestmark = pytest.mark.internal


class _NotificationLoop:
    """Synchronously dispose of scheduler notification coroutines."""

    @staticmethod
    def create_task(coroutine):
        coroutine.close()

    @staticmethod
    def call_soon_threadsafe(callback, *args):
        callback(*args)


class TestChunkedPrefillScheduleCharacterization(PrefixCachingTestBase):
    """Exercise scheduler decisions through real context and allocator mutations."""

    def _context(
        self,
        *,
        max_tokens,
        max_requests=2,
        prefix=False,
        mamba=False,
        batch_invariant=False,
        mamba_cache=False,
    ):
        return self._ctx(
            buffer_size_gb=0.01,
            block_size_tokens=8 if mamba else 4,
            max_sequence_length=128,
            rounder=1,
            enable_prefix_caching=prefix,
            max_tokens=max_tokens,
            max_requests=max_requests,
            mamba_config=self._mamba_config(mamba_chunk_size=4) if mamba else None,
            prefix_caching_mamba_gb=0.01 if mamba_cache else None,
            batch_invariant_mode=batch_invariant,
            enable_chunked_prefill=True,
        )

    @staticmethod
    def _engine(context):
        engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
        engine.context = context
        engine.enable_chunked_prefill = True
        engine.cuda_graph_all_prefills = False
        engine._prefix_coordination_waits = 0
        engine._loop = _NotificationLoop()
        engine.waiting_request_ids = deque()
        engine.requests = {}
        engine._generation_epoch = None
        return engine

    def _request(self, context, request_id, length, *, offset=0):
        request = self._req(
            context,
            self._prompt(length, offset=offset),
            request_id=request_id,
            enable_prefix_caching=context.enable_prefix_caching,
        )
        request.status = Status.ACTIVE_AND_GENERATING_TOKENS
        return request

    @staticmethod
    def _queue(engine, request):
        engine.requests[request.request_id] = RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(request), future=Future()
        )
        engine.waiting_request_ids.append(request.request_id)

    @staticmethod
    def _state(engine, request):
        context = engine.context
        count = context.total_request_count
        hidden_idx = context.get_index_of_chunked_prefill_request(safe=False)
        return {
            "remaining": len(request.remaining_prompt_tokens),
            "finished": request.finished_chunk_token_count,
            "chunked_id": context.chunked_prefill_request_id,
            "queue": list(engine.waiting_request_ids),
            "active_ids": context.request_ids[:count].tolist(),
            "query_lengths": context.request_query_lengths[:count].tolist(),
            "kv_offsets": context.request_kv_length_offsets[:count].tolist(),
            "block_counts": context.request_kv_block_counts[:count].tolist(),
            "active_tokens": context.active_token_count,
            "request_count": count,
            "hidden_idx": hidden_idx,
            "cache_hits": context.prefix_cache_hits,
            "computed": context.prefix_cache_prefill_computed_tokens,
            "skipped": context.prefix_cache_prefill_skipped_tokens,
            "cached_tokens": request.num_cached_tokens,
        }

    @staticmethod
    def _consume_step(context, *, survive=True):
        count = context.total_request_count
        active = torch.full((count,), int(survive), dtype=torch.int64)
        context.update_requests(active, torch.zeros(count, dtype=torch.int64))

    @staticmethod
    def _prime_prefix(context, request, block_count):
        """Install an evictable prefix through the real LRU allocator."""
        allocator = context.kv_block_allocator
        blocks = allocator.allocate_memory_blocks(block_count)
        assert blocks is not None and len(blocks) == block_count
        block_ids = blocks.tolist()
        hashes = request.precomputed_block_hashes[:block_count]
        parents = [0, *hashes[:-1]]
        allocator.register_kv_block_hashes(block_ids, hashes, parents)
        allocator.release_memory_blocks(blocks)
        assert [allocator.kv_hash_to_block_id[value] for value in hashes] == block_ids
        return block_ids

    @staticmethod
    def _set_graph(engine, token_count):
        engine.cuda_graph_all_prefills = True
        engine.context.cuda_graph_batch_dimensions_list = [
            InferenceBatchDimensions(
                token_count=token_count, prefill_req_count=1, decode_req_count=0
            )
        ]

    @pytest.mark.parametrize(
        ("phase", "steps", "expected"),
        [
            ("initial-partial", 1, (10, 6, 1, [1], 6, 0, 2)),
            ("continuation", 2, (4, 12, 1, [1], 6, 6, 3)),
            ("final-dequeue", 3, (0, 12, -1, [], 4, 12, 4)),
        ],
    )
    def test_chunk_lifecycle_phases(self, phase, steps, expected):
        """Initial, continuation, and final chunks preserve their distinct state."""
        context = self._context(max_tokens=6, prefix=False)
        engine = self._engine(context)
        request = self._request(context, 1, 16)
        self._queue(engine, request)

        for step in range(steps):
            engine.schedule_chunked_prefill()
            if step + 1 < steps:
                self._consume_step(context)

        state = self._state(engine, request)
        remaining, finished, chunked_id, queue, query, offset, blocks = expected
        assert state["remaining"] == remaining, phase
        assert state["finished"] == finished and state["chunked_id"] == chunked_id
        assert state["queue"] == queue and state["active_ids"] == [1]
        assert state["query_lengths"] == [query] and state["kv_offsets"] == [offset]
        assert state["block_counts"] == [blocks] and state["active_tokens"] == query

    def test_continuation_reoccupies_row_at_max_requests(self):
        context = self._context(max_tokens=5, max_requests=1, prefix=False)
        engine = self._engine(context)
        request = self._request(context, 7, 14)
        self._queue(engine, request)

        engine.schedule_chunked_prefill()
        self._consume_step(context)
        hidden = self._state(engine, request)
        assert hidden["request_count"] == 0 and hidden["hidden_idx"] == 0
        assert context.request_ids[hidden["hidden_idx"]].item() == 7

        engine.schedule_chunked_prefill()
        state = self._state(engine, request)
        assert state["request_count"] == context.max_requests == 1
        assert state["active_ids"] == [7] and state["query_lengths"] == [5]
        assert state["kv_offsets"] == [5] and state["block_counts"] == [3]
        assert (state["remaining"], state["finished"], state["chunked_id"]) == (4, 10, 7)

    def test_cached_logical_span_can_exceed_compute_budget(self):
        context = self._context(max_tokens=4, prefix=True)
        engine = self._engine(context)
        request = self._request(context, 1, 20)
        cached_blocks = self._prime_prefix(context, request, 3)
        self._queue(engine, request)

        engine.schedule_chunked_prefill()
        state = self._state(engine, request)
        assert state["query_lengths"] == [4] and state["kv_offsets"] == [12]
        assert state["finished"] == 16 > context.max_tokens
        assert (state["remaining"], state["computed"], state["skipped"]) == (4, 4, 12)
        assert state["cached_tokens"] == 12 and state["cache_hits"] == 1
        assert context.request_to_kv_block_ids[0, :3].tolist() == cached_blocks

    def test_continuation_keeps_prefix_accounting_cumulative(self):
        """A later chunk adds compute without double-counting the original cache hit."""
        context = self._context(max_tokens=4, prefix=True)
        engine = self._engine(context)
        request = self._request(context, 1, 16)
        self._prime_prefix(context, request, 2)
        self._queue(engine, request)

        engine.schedule_chunked_prefill()
        first = self._state(engine, request)
        assert (first["finished"], first["computed"], first["skipped"]) == (12, 4, 8)
        self._consume_step(context)
        engine.schedule_chunked_prefill()
        final = self._state(engine, request)

        assert final["queue"] == [] and final["chunked_id"] == -1
        assert final["query_lengths"] == [4] and final["kv_offsets"] == [12]
        assert (final["remaining"], final["computed"], final["skipped"]) == (0, 8, 8)
        assert final["cache_hits"] == 1 and final["cached_tokens"] == 8

    def test_cached_single_token_chunk_uses_last_available_token(self):
        context = self._context(max_tokens=5, prefix=True)
        engine = self._engine(context)
        request = self._request(context, 1, 20)
        self._prime_prefix(context, request, 2)
        filler = self._request(context, 9, 4, offset=9000)
        context.add_request(filler)
        self._queue(engine, request)

        one_token_probe = context.check_availability(request, prefill_chunk_length=1)
        matched = context._compute_prefix_match(request, 9)
        expanded_probe = context.check_availability(request, prefill_chunk_length=9)
        assert one_token_probe == (True, True, True)
        assert (matched.prefix_skip_tokens, matched.effective_prefill_chunk_length) == (8, 1)
        assert expanded_probe == (True, True, True)

        engine.schedule_chunked_prefill()
        admitted = self._state(engine, request)
        assert admitted["active_ids"] == [9, 1] and admitted["queue"] == [1]
        assert admitted["query_lengths"] == [4, 1] and admitted["kv_offsets"] == [0, 8]
        assert (admitted["finished"], admitted["remaining"]) == (9, 11)
        assert (admitted["cache_hits"], admitted["skipped"]) == (1, 8)

    @pytest.mark.parametrize(
        ("length", "cached_blocks", "graph_tokens", "max_requests", "bypass"),
        [
            pytest.param(8, 0, None, 3, True, id="full-prefill-fits"),
            pytest.param(12, 0, None, 3, True, id="full-prefill-exact-budget"),
            pytest.param(16, 0, None, 3, False, id="partial-prefill-waits"),
            pytest.param(24, 3, None, 3, True, id="cached-prefix-makes-full-prefill-fit"),
            pytest.param(13, 0, None, 3, False, id="one-token-tail-waits"),
            pytest.param(10, 0, 16, 3, False, id="graph-snap-would-make-partial"),
            pytest.param(8, 0, None, 1, False, id="request-capacity-waits"),
        ],
    )
    def test_prefix_conflict_bypass_finishes_prefill(
        self, length, cached_blocks, graph_tokens, max_requests, bypass
    ):
        context = self._context(max_tokens=20, max_requests=max_requests, prefix=True)
        engine = self._engine(context)
        if graph_tokens:
            engine.cuda_graph_all_prefills = True
            context.cuda_graph_batch_dimensions_list = [
                InferenceBatchDimensions(graph_tokens, 2, 0)
            ]
        first = self._request(context, 1, 8)
        conflict = self._request(context, 2, 8)
        later = self._request(context, 3, length, offset=1000)
        if cached_blocks:
            self._prime_prefix(context, later, cached_blocks)
        for request in (first, conflict, later):
            self._queue(engine, request)

        engine.schedule_chunked_prefill()
        state = self._state(engine, later)
        skipped = cached_blocks * context.block_size_tokens
        assert state["queue"] == ([2] if bypass else [2, 3])
        assert state["chunked_id"] == -1 and state["finished"] == 0
        assert state["active_ids"] == ([1, 3] if bypass else [1])
        assert state["query_lengths"] == ([8, length - skipped] if bypass else [8])
        assert state["kv_offsets"] == ([0, skipped] if bypass else [0])
        assert state["active_tokens"] == (8 + length - skipped if bypass else 8)
        assert state["remaining"] == (0 if bypass else length)
        assert state["skipped"] == (skipped if bypass else 0)
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}
        assert conflict.finished_chunk_token_count == 0
        assert len(conflict.remaining_prompt_tokens) == 8 and conflict.num_cached_tokens == 0

        # Finish A's step: B must now lead admission and reuse A's published prefix.
        self._consume_step(context, survive=False)
        engine.schedule_chunked_prefill()
        resumed = self._state(engine, conflict)
        assert resumed["active_ids"][0] == 2
        assert resumed["kv_offsets"][0] == 7 and resumed["query_lengths"][0] == 1
        assert not conflict.remaining_prompt_tokens.numel()
        assert conflict.num_cached_tokens == 7
        # Any partial C begins only after B is admitted; drain it with the real lifecycle.
        for _ in range(4):
            if not engine.waiting_request_ids:
                break
            if context.chunked_prefill_request_id >= 0:
                assert engine.waiting_request_ids[0] == context.chunked_prefill_request_id == 3
            self._consume_step(context, survive=False)
            engine.schedule_chunked_prefill()
        assert not engine.waiting_request_ids and context.chunked_prefill_request_id == -1
        assert all(
            request.remaining_prompt_tokens.numel() == 0 for request in (first, conflict, later)
        )
        assert engine.get_prefix_coordination_metrics() == {"waits": 1}
        self._consume_step(context, survive=False)
        assert context.total_request_count == context.active_token_count == 0

    @pytest.mark.parametrize(
        ("max_tokens", "length", "graph_tokens", "cached_blocks", "query", "offset", "blocks"),
        [
            pytest.param(10, 20, 6, 0, 6, 0, 2, id="first-chunk-graph-boundary"),
            pytest.param(6, 20, 4, 2, 4, 8, 3, id="prefix-skip-excluded-from-graph-snap"),
            pytest.param(6, 7, 6, 0, 6, 0, 2, id="graph-snap-allows-one-token-tail"),
        ],
    )
    def test_first_chunk_graph_boundaries(
        self, max_tokens, length, graph_tokens, cached_blocks, query, offset, blocks
    ):
        context = self._context(max_tokens=max_tokens, prefix=bool(cached_blocks))
        engine = self._engine(context)
        self._set_graph(engine, graph_tokens)
        request = self._request(context, 1, length)
        if cached_blocks:
            self._prime_prefix(context, request, cached_blocks)
        self._queue(engine, request)

        engine.schedule_chunked_prefill()
        state = self._state(engine, request)
        assert state["query_lengths"] == [query] and state["kv_offsets"] == [offset]
        assert (state["finished"], state["remaining"]) == (query + offset, length - query - offset)
        assert state["active_tokens"] == query and state["chunked_id"] == 1
        assert state["block_counts"] == [blocks] and state["queue"] == [1]
        assert request.cg_wait_iters == 0
        if cached_blocks:
            assert (state["computed"], state["skipped"], state["cached_tokens"]) == (
                query,
                offset,
                offset,
            )
        if length == graph_tokens + 1:
            assert (
                state["query_lengths"][0] == context.cuda_graph_batch_dimensions_list[0].token_count
            )

    @pytest.mark.parametrize(("prefills", "query"), [(1, 2), (0, 10)])
    def test_continuation_uses_graph_or_eager_fallback(self, prefills, query):
        context = self._context(max_tokens=10, prefix=False)
        engine = self._engine(context)
        self._set_graph(engine, 6)
        request = self._request(context, 1, 20)
        self._queue(engine, request)
        engine.schedule_chunked_prefill()
        self._consume_step(context)

        context.cuda_graph_batch_dimensions_list = [
            InferenceBatchDimensions(2, prefills, 1 - prefills)
        ]
        request.cg_wait_iters = 7
        engine.schedule_chunked_prefill()
        state = self._state(engine, request)
        assert state["query_lengths"] == [query] and state["kv_offsets"] == [6]
        assert (state["finished"], state["remaining"]) == (6 + query, 14 - query)
        assert state["chunked_id"] == 1 and request.cg_wait_iters == 0

    @pytest.mark.parametrize(
        ("prefix", "length", "query"),
        [
            pytest.param(False, 25, 12, id="batch-invariant-model-alignment-not-gdp"),
            pytest.param(True, 30, 8, id="hybrid-prefix-kv-block-boundary"),
        ],
    )
    def test_mamba_first_chunk_alignment(self, prefix, length, query):
        context = self._context(
            max_tokens=13, prefix=prefix, mamba=True, batch_invariant=not prefix, mamba_cache=prefix
        )
        engine = self._engine(context)
        request = self._request(context, 1, length)
        self._queue(engine, request)

        engine.schedule_chunked_prefill()
        state = self._state(engine, request)
        assert state["query_lengths"] == [query] and state["kv_offsets"] == [0]
        assert (state["finished"], state["remaining"]) == (query, length - query)
        assert context.mamba_metadata.request_to_mamba_state_idx[0].item() >= 0
        if prefix:
            assert state["finished"] % context.block_size_tokens == 0
            assert context.mamba_slot_allocator._eos_cache_block_id_cpu[0].item() == (
                context.request_last_kv_block_id[0].item()
            )
        else:
            assert context.gdp_num_householder == 0 and context.ssm_chunk_alignment == 4
            assert state["finished"] % context.ssm_chunk_alignment == 0
