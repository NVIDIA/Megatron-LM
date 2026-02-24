# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for CUDAGraphBatchDimensionBuilder.match_graph_config with expert parallelism.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from tests.unit_tests.test_utilities import Utils

BD = InferenceBatchDimensions

# Common config shared across tests
MAX_REQUESTS = 256
MAX_TOKENS = 2048
MAX_SEQ_LEN = 4096
TP_SIZE = 1
MIXED_PREFILL_COUNT = 16


def _generate_graphs(num_cuda_graphs, use_non_decode=True):
    """Generate cuda graph batch dimensions using the builder."""
    graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
        tp_size=TP_SIZE,
        num_cuda_graphs=num_cuda_graphs,
        cuda_graph_max_tokens=MAX_REQUESTS,
        cuda_graph_mixed_prefill_count=MIXED_PREFILL_COUNT,
        max_requests=MAX_REQUESTS,
        max_tokens=MAX_TOKENS,
        max_sequence_length=MAX_SEQ_LEN,
        use_cuda_graphs_for_non_decode_steps=use_non_decode,
    )
    return graph_list


def _match(
    real, graph_list, ep_group, strict=False, decode_only=False, explicit_chunked_prefill=False
):
    return CUDAGraphBatchDimensionBuilder.match_graph_config(
        real_batch_dim=real,
        cuda_graph_batch_dimensions_list=graph_list,
        strict=strict,
        decode_only_cuda_graphs=decode_only,
        explicit_chunked_prefill=explicit_chunked_prefill,
        ep_group=ep_group,
        cuda_graph_mixed_prefill_count=MIXED_PREFILL_COUNT,
    )


def _assert_consistent_across_ranks(result, ep_group):
    """Assert that the match result is the same on every EP rank.

    Either all ranks return None, or all ranks return a config with the
    same token_count (which is what the all-reduce synchronises).
    """
    if result is None:
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    else:
        flag = torch.ones(1, dtype=torch.int32, device="cuda")

    # If any rank got None, all must get None; if any rank got a match, all must.
    flag_sum = flag.clone()
    dist.all_reduce(flag_sum, op=dist.ReduceOp.SUM, group=ep_group)
    ep_size = dist.get_world_size(ep_group)
    assert (
        flag_sum.item() == 0 or flag_sum.item() == ep_size
    ), f"Inconsistent match: {flag_sum.item()}/{ep_size} ranks got a match"

    if result is not None:
        tc = torch.tensor([result.token_count], dtype=torch.int32, device="cuda")
        tc_max = tc.clone()
        tc_min = tc.clone()
        dist.all_reduce(tc_max, op=dist.ReduceOp.MAX, group=ep_group)
        dist.all_reduce(tc_min, op=dist.ReduceOp.MIN, group=ep_group)
        assert (
            tc_max.item() == tc_min.item()
        ), f"Token count mismatch across EP ranks: min={tc_min.item()}, max={tc_max.item()}"


class TestCUDAGraphTokenCountAlignment:
    """Verify that mixed/prefill graph token counts are a subset of decode graph token counts."""

    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_mixed_token_counts_subset_of_decode(self, num_cuda_graphs):
        """Every token count in the mixed/prefill graph pool must also appear
        in the decode-only pool. Otherwise, when EP syncs token counts across
        ranks, decode-only ranks cannot find a graph at the same token count
        as prefill ranks, causing inconsistent matching."""
        graph_list = _generate_graphs(num_cuda_graphs)

        decode_token_counts = {bd.token_count for bd in graph_list if bd.prefill_req_count == 0}
        mixed_token_counts = {bd.token_count for bd in graph_list if bd.prefill_req_count > 0}

        mixed_only = mixed_token_counts - decode_token_counts
        assert not mixed_only, (
            f"Mixed/prefill token counts with no decode graph: {sorted(mixed_only)}. "
            f"This will cause EP rank mismatch when some ranks are decode-only "
            f"and others have prefill."
        )

        # Decode-only token counts not in the mixed pool are allowed, but only
        # below MIXED_PREFILL_COUNT. The EP adjustment elevates token counts to
        # at least MIXED_PREFILL_COUNT when any rank has prefill, so any decode
        # token count >= MIXED_PREFILL_COUNT must have a mixed counterpart.
        decode_only = decode_token_counts - mixed_token_counts
        large_decode_only = {tc for tc in decode_only if tc >= MIXED_PREFILL_COUNT}
        assert not large_decode_only, (
            f"Decode-only token counts >= MIXED_PREFILL_COUNT ({MIXED_PREFILL_COUNT}) "
            f"with no mixed/prefill graph: {sorted(large_decode_only)}. "
            f"The EP token count elevation cannot guarantee alignment for these."
        )


class TestMatchGraphConfigWithEP:
    """Tests for match_graph_config with expert parallelism.

    Uses the world group as the EP group (all 8 GPUs form one EP group).
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=Utils.world_size,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _get_ep_group():
        """Return the EP group created by initialize_model_parallel."""
        return ps.get_expert_model_parallel_group()

    # ------------------------------------------------------------------ #
    # 1. All ranks same decode batch → consistent match
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_uniform_decode_batch(self, num_cuda_graphs):
        """All EP ranks have the same decode-only batch → should all match the same graph."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is not None, "Should find a matching graph for uniform decode batch"

    # ------------------------------------------------------------------ #
    # 2. Different token counts across EP ranks → all-reduce takes max
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_varying_decode_token_counts(self, num_cuda_graphs):
        """EP ranks have different decode token counts. The all-reduce
        should take the max, and all ranks should match the same graph."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Each rank gets a different token count: 8, 16, 24, ...
        token_count = (rank + 1) * 8
        real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=token_count)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is not None

    # ------------------------------------------------------------------ #
    # 3. decode_only_cuda_graphs=True, some ranks have prefill → all None
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_decode_only_graphs_with_mixed_ranks(self, num_cuda_graphs):
        """When decode_only_cuda_graphs=True and at least one EP rank has a
        prefill request, ALL ranks should get None (eager mode)."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Rank 0 has a mixed batch (prefill + decode), all others decode-only
        if rank == 0:
            real = BD(token_count=64, prefill_req_count=2, decode_req_count=10)
        else:
            real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)

        result = _match(real, graph_list, ep_group=ep_group, decode_only=True)
        _assert_consistent_across_ranks(result, ep_group)
        assert (
            result is None
        ), "All ranks should run eager when decode_only=True and some rank has prefill"

    # ------------------------------------------------------------------ #
    # 4. explicit_chunked_prefill=True, some ranks prefill → all None
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_explicit_chunked_prefill_with_mixed_ranks(self, num_cuda_graphs):
        """When explicit_chunked_prefill=True and some EP rank has prefill,
        ALL ranks should get None (eager mode)."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        if rank == 0:
            real = BD(token_count=64, prefill_req_count=2, decode_req_count=10)
        else:
            real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)

        result = _match(real, graph_list, ep_group=ep_group, explicit_chunked_prefill=True)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "All ranks should run eager with explicit_chunked_prefill"

    # ------------------------------------------------------------------ #
    # 5. Mixed prefill graphs with strict matching
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_strict_matching_with_mixed_prefill(self, num_cuda_graphs):
        """With strict matching, request counts are synced across EP ranks
        via all-reduce. All ranks should still get a consistent result."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Varying prefill/decode split across ranks
        prefill = min(rank + 1, MIXED_PREFILL_COUNT)
        decode = 16 - prefill
        real = BD(token_count=64, prefill_req_count=prefill, decode_req_count=decode)

        result = _match(real, graph_list, ep_group=ep_group, strict=True)
        _assert_consistent_across_ranks(result, ep_group)

    # ------------------------------------------------------------------ #
    # 6. Non-strict matching with mixed prefill
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_non_strict_matching_with_mixed_prefill(self, num_cuda_graphs):
        """Non-strict matching: prefill slots can serve decode. Token count
        is synced across EP ranks; result must be consistent."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        prefill = min(rank + 1, MIXED_PREFILL_COUNT)
        decode = 16 - prefill
        real = BD(token_count=64, prefill_req_count=prefill, decode_req_count=decode)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)

    # ------------------------------------------------------------------ #
    # 7. Mixed decode/prefill across ranks — strict matching
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_mixed_decode_and_prefill_ranks_strict(self, num_cuda_graphs):
        """Some EP ranks are pure decode, others have prefill requests.
        With strict matching the all-reduce syncs request counts to the
        max across ranks. Result must be consistent."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Even ranks: pure decode (32 tokens)
        # Odd ranks: mixed prefill (64 tokens, 2 prefill + 14 decode)
        if rank % 2 == 0:
            real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)
        else:
            real = BD(token_count=64, prefill_req_count=2, decode_req_count=14)

        result = _match(real, graph_list, ep_group=ep_group, strict=True)
        _assert_consistent_across_ranks(result, ep_group)

    # ------------------------------------------------------------------ #
    # 8. Mixed decode/prefill across ranks — non-strict matching
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_mixed_decode_and_prefill_ranks_non_strict(self, num_cuda_graphs):
        """Some EP ranks are pure decode, others have prefill requests.
        Non-strict matching only syncs token counts (not request counts).
        Result must be consistent."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Even ranks: pure decode (32 tokens)
        # Odd ranks: mixed prefill (64 tokens, 2 prefill + 14 decode)
        if rank % 2 == 0:
            real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)
        else:
            real = BD(token_count=64, prefill_req_count=2, decode_req_count=14)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)

    # ------------------------------------------------------------------ #
    # 9. All ranks decode-only with decode_only_cuda_graphs → should match
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_decode_only_graphs_all_decode(self, num_cuda_graphs):
        """When all EP ranks are decode-only and decode_only_cuda_graphs=True,
        a match should be found."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        token_count = (rank + 1) * 4
        real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=token_count)

        result = _match(real, graph_list, ep_group=ep_group, decode_only=True)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is not None, "All-decode batch with decode_only_cuda_graphs should match"

    # ------------------------------------------------------------------ #
    # 10. Real batch exceeds all graphs → None on all ranks
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_oversized_batch_returns_none(self, num_cuda_graphs):
        """When the real batch is larger than any available graph, all ranks
        should get None."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)

        # Token count exceeds MAX_TOKENS on all ranks
        real = BD(
            token_count=MAX_TOKENS + 100,
            prefill_req_count=0,
            decode_req_count=min(MAX_TOKENS + 100, MAX_REQUESTS),
        )

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "Oversized batch should not match any graph"

    # ------------------------------------------------------------------ #
    # 11. One EP rank has huge batch → all-reduce lifts to max → no match
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_one_rank_oversized_forces_no_match(self, num_cuda_graphs):
        """If one EP rank has a batch exceeding all graph capacities, the
        all-reduce max lifts everyone → no match on any rank."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        if rank == 0:
            # This rank has a batch that exceeds all graphs
            real = BD(
                token_count=MAX_TOKENS + 100,
                prefill_req_count=0,
                decode_req_count=min(MAX_TOKENS + 100, MAX_REQUESTS),
            )
        else:
            real = BD(token_count=8, prefill_req_count=0, decode_req_count=8)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "All-reduce max from oversized rank should cause no match"
