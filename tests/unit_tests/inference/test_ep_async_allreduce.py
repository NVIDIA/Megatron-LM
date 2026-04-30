# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the non-blocking EP rank-sync entry point introduced in
commit 20.

Plan validation: multi-rank EP test (EP=2, EP=8), correctness over ≥4096
tokens, nsys shows ZMQ packets in flight during the prior step's GPU
forward. The multi-rank correctness portion is covered by the engine's
distributed integration tests; here we verify the awaitable contract on
EP-size-1 (no-op pass-through).
"""

import asyncio

import pytest

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestEpAsyncAllReduce:
    def test_async_variant_returns_awaitable(self):
        local = InferenceBatchDimensions(
            token_count=8, prefill_req_count=0, decode_req_count=4
        )
        # ep_group=None → pg_size=1 → fast-path returns ``local`` unchanged.
        future = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism_async(
            local_batch_dims=local,
            strict=False,
            decode_only_cuda_graphs=False,
            smallest_non_decode_cuda_graph_size=1,
        )
        assert asyncio.iscoroutine(future) or asyncio.isfuture(future) or hasattr(
            future, "__await__"
        )
        result = _run(future)
        assert result is local

    def test_sync_and_async_variants_match(self):
        """For an EP-size-1 caller the sync and async variants return the
        same object (the local dims, unchanged)."""
        local = InferenceBatchDimensions(
            token_count=12, prefill_req_count=2, decode_req_count=4
        )
        sync_result = InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism(
            local_batch_dims=local,
            strict=False,
            decode_only_cuda_graphs=False,
            smallest_non_decode_cuda_graph_size=1,
        )
        async_result = _run(
            InferenceBatchDimensions.adjust_batch_dims_for_expert_parallelism_async(
                local_batch_dims=local,
                strict=False,
                decode_only_cuda_graphs=False,
                smallest_non_decode_cuda_graph_size=1,
            )
        )
        assert sync_result == async_result
