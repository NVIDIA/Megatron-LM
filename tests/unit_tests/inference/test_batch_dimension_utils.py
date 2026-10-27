# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for CUDAGraphBatchDimensionBuilder.match_graph_config with expert parallelism.
"""

import math

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.config import CudaGraphSizingDistribution
from tests.unit_tests.test_utilities import Utils

BD = InferenceBatchDimensions

# Common config shared across tests
MAX_REQUESTS = 256
MAX_TOKENS = 2048
MAX_SEQ_LEN = 4096
TP_SIZE = 1
MIXED_PREFILL_COUNT = 16


def _generate_graphs(
    num_cuda_graphs,
    use_non_decode=True,
    sizing_distribution=CudaGraphSizingDistribution.EXPONENTIAL,
):
    """Generate cuda graph batch dimensions using the builder.

    Defaults to EXPONENTIAL rather than the builder default (HYBRID) because the EP
    matching tests below were written against a single shared ladder; tests that care
    about the shipping default pass it explicitly.
    """
    graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
        tp_size=TP_SIZE,
        num_cuda_graphs=num_cuda_graphs,
        cuda_graph_max_tokens=MAX_REQUESTS,
        cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
        max_requests=MAX_REQUESTS,
        max_tokens=MAX_TOKENS,
        max_sequence_length=MAX_SEQ_LEN,
        use_cuda_graphs_for_non_decode_steps=use_non_decode,
        sizing_distribution=sizing_distribution,
    )
    return graph_list


def _match(real, graph_list, ep_group, strict=False):
    return CUDAGraphBatchDimensionBuilder.match_graph_config(
        real_batch_dim=real,
        cuda_graph_batch_dimensions_list=graph_list,
        strict=strict,
        ep_group=ep_group,
        match_ep_token_counts=True,
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
    """Verify that EP ranks can agree on a CUDA graph.

    Two distributions are covered, and they guarantee different things:

    EXPONENTIAL (the pre-HYBRID default) gives both families one shared ladder, so
    mixed and decode token counts line up exactly -- asserted below.

    HYBRID (the shipping default) deliberately spaces the two families differently, so
    that alignment does not hold. It does not need to: a prefill on any EP rank sends
    every rank eager (`adjust_batch_dims_for_expert_parallelism` returns None as soon as
    the all-reduced non-decode flag is set), so a mixed graph is never selected while
    another rank is decode-only. What EP does rely on is that every rank picks the same
    graph for a decode-only step, which
    `test_decode_only_ep_ranks_select_the_same_graph` covers for both distributions.
    """

    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_mixed_token_counts_subset_of_decode(self, num_cuda_graphs):
        """Every token count in the mixed/prefill graph pool must also appear
        in the decode-only pool. Otherwise, when EP syncs token counts across
        ranks, decode-only ranks cannot find a graph at the same token count
        as prefill ranks, causing inconsistent matching.

        EXPONENTIAL only -- see the class docstring for why HYBRID is exempt."""
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

    @pytest.mark.parametrize(
        "sizing_distribution",
        [CudaGraphSizingDistribution.HYBRID, CudaGraphSizingDistribution.EXPONENTIAL],
        ids=["default_hybrid", "legacy_exponential"],
    )
    @pytest.mark.parametrize("num_cuda_graphs", [1, 4, 16, 32, -1])
    def test_decode_only_ep_ranks_select_the_same_graph(self, num_cuda_graphs, sizing_distribution):
        """On a decode-only step, EP ranks all-reduce-max their token count and then match
        independently against their own (identical) graph list. Every rank must land on the
        same graph, or the ranks replay mismatched captures.

        The all-reduce is emulated here rather than run over a process group: the ranks
        differ only in their local decode_req_count, which is what could pull them onto
        different rungs."""
        graph_list = _generate_graphs(num_cuda_graphs, sizing_distribution=sizing_distribution)
        per_rank_decode_counts = [1, 3, 7, 16, 33, 64, MAX_REQUESTS]

        # adjust_batch_dims_for_expert_parallelism elevates every rank's token count to
        # the group max, leaving decode_req_count local.
        synced_token_count = max(per_rank_decode_counts)
        selected = {
            CUDAGraphBatchDimensionBuilder.match_graph_config(
                real_batch_dim=BD(synced_token_count, 0, decode_req_count),
                cuda_graph_batch_dimensions_list=graph_list,
                match_ep_token_counts=False,
            )
            for decode_req_count in per_rank_decode_counts
        }

        assert len(selected) == 1, (
            f"EP ranks selected different graphs for one decode-only step: "
            f"{sorted(str(s) for s in selected)}"
        )
        assert None not in selected, "Decode-only step at the request limit found no graph"


class TestGenerateCUDAGraphEdgeCases:
    """Single-process tests for graph generation edge cases."""

    def test_generate_cuda_graph_edge_cases(self):
        """Edge cases in graph generation:
        max_tokens > max_requests, small max_tokens, step_size floor, speculative decoding.
        """

        # max_tokens > max_requests: decode graphs capped, prefill graphs span full budget
        g_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=1,
            num_cuda_graphs=8,
            cuda_graph_max_tokens=512,
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=64,
            max_tokens=512,
            max_sequence_length=4096,
            use_cuda_graphs_for_non_decode_steps=True,
        )
        decode_graphs = [g for g in g_list if g.prefill_req_count == 0]
        prefill_graphs = [g for g in g_list if g.prefill_req_count > 0]
        assert all(g.token_count <= 64 for g in decode_graphs)
        assert prefill_graphs and max(g.token_count for g in prefill_graphs) > 64

        # max_tokens < num_cuda_graphs: step_size could round to zero
        g_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=1,
            num_cuda_graphs=32,
            cuda_graph_max_tokens=10,
            cuda_graph_mixed_prefill_request_count=0,
            max_requests=10,
            max_tokens=10,
            max_sequence_length=4096,
            use_cuda_graphs_for_non_decode_steps=False,
        )
        assert len(g_list) > 0 and all(g.token_count > 0 for g in g_list)

        # Step size >= tp_size for various TP sizes
        for tp_size in (1, 2, 4, 8):
            g_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                tp_size=tp_size,
                num_cuda_graphs=64,
                cuda_graph_max_tokens=tp_size,
                cuda_graph_mixed_prefill_request_count=0,
                max_requests=tp_size,
                max_tokens=tp_size,
                max_sequence_length=4096,
                use_cuda_graphs_for_non_decode_steps=False,
            )
            assert len(g_list) > 0
            for g in g_list:
                assert g.token_count % tp_size == 0

        # Speculative decoding with max_tokens >> max_requests * (spec+1)
        for num_spec in (1, 3, 7):
            g_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                tp_size=1,
                num_cuda_graphs=8,
                cuda_graph_max_tokens=1024,
                cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
                max_requests=32,
                max_tokens=1024,
                max_sequence_length=4096,
                use_cuda_graphs_for_non_decode_steps=True,
                num_speculative_tokens=num_spec,
            )
            for g in [g for g in g_list if g.prefill_req_count == 0]:
                assert g.token_count == g.decode_req_count * (num_spec + 1)
                assert g.decode_req_count <= 32


class TestSizingDistributions:
    """Coverage for `sizing_distribution`, in particular the HYBRID default.

    HYBRID names one distribution per graph family -- EXPONENTIAL for prefill/mixed,
    LINEAR for decode-only -- and is resolved inside
    `generate_cuda_graph_batch_dimensions_list`, the only caller that knows which
    family it is building.
    """

    # Decode-only graphs are capped at max_requests * (spec + 1), so with these
    # settings the decode family spans 64 tokens and the prefill/mixed family 512 --
    # the order-of-magnitude gap that motivates a distribution per family.
    MAX_REQUESTS = 64
    CG_MAX_TOKENS = 512

    def _generate(
        self,
        num_cuda_graphs,
        sizing_distribution,
        max_requests=None,
        tp_size=1,
        num_speculative_tokens=0,
    ):
        max_requests = max_requests or self.MAX_REQUESTS
        cuda_graph_max_tokens = max(self.CG_MAX_TOKENS, max_requests * (num_speculative_tokens + 1))
        graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=tp_size,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_max_tokens=cuda_graph_max_tokens,
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=max_requests,
            max_tokens=cuda_graph_max_tokens,
            max_sequence_length=MAX_SEQ_LEN,
            use_cuda_graphs_for_non_decode_steps=True,
            num_speculative_tokens=num_speculative_tokens,
            sizing_distribution=sizing_distribution,
        )
        return graph_list

    @staticmethod
    def _token_counts(graph_list):
        """Token counts of the (prefill/mixed, decode-only) families, descending."""
        return (
            sorted({bd.token_count for bd in graph_list if bd.prefill_req_count > 0}, reverse=True),
            sorted(
                {bd.token_count for bd in graph_list if bd.prefill_req_count == 0}, reverse=True
            ),
        )

    def test_hybrid_ladders(self):
        """HYBRID halves the prefill/mixed family down from cuda_graph_max_tokens while
        stepping the decode family evenly across max_requests, so the coarse halving
        that suits a 512-token range does not also govern a 64-token one. The decode
        ladder reserves its last two rungs for the 1- and 2-request floors."""
        prefill_counts, decode_counts = self._token_counts(
            self._generate(8, CudaGraphSizingDistribution.HYBRID)
        )
        assert prefill_counts == [512, 256, 128, 64, 32, 16, 8]
        assert decode_counts == [64, 60, 48, 36, 24, 12, 2, 1]

    @pytest.mark.parametrize("num_cuda_graphs", [4, 8, 16, -1])
    def test_hybrid_resolves_to_one_distribution_per_family(self, num_cuda_graphs):
        """HYBRID must yield exactly EXPONENTIAL's prefill/mixed graphs and exactly
        LINEAR's decode-only graphs -- not a blend, and not one distribution applied to
        both families. Passing no distribution at all must resolve the same way, so a
        direct caller captures what the inference engine captures."""
        hybrid = self._generate(num_cuda_graphs, CudaGraphSizingDistribution.HYBRID)
        exponential = self._generate(num_cuda_graphs, CudaGraphSizingDistribution.EXPONENTIAL)
        linear = self._generate(num_cuda_graphs, CudaGraphSizingDistribution.LINEAR)

        hybrid_prefill, hybrid_decode = self._token_counts(hybrid)
        exponential_prefill, exponential_decode = self._token_counts(exponential)
        linear_prefill, linear_decode = self._token_counts(linear)

        assert hybrid_prefill == exponential_prefill
        assert hybrid_decode == linear_decode

        # The two families really do diverge here, so the equalities above are not
        # satisfied by every distribution alike.
        assert hybrid_decode != exponential_decode
        assert hybrid_prefill != linear_prefill

        assert self._generate(num_cuda_graphs, None) == hybrid

    @pytest.mark.parametrize("max_requests", [8, 64, 256, 1024])
    @pytest.mark.parametrize("num_cuda_graphs", [2, 4, 16, 32, -1])
    @pytest.mark.parametrize("tp_size, num_speculative_tokens", [(1, 0), (8, 0), (1, 3), (8, 3)])
    def test_decode_family_keeps_its_smallest_graphs(
        self, max_requests, num_cuda_graphs, tp_size, num_speculative_tokens
    ):
        """A linear decode ladder starts at its stride, max_requests / num_cuda_graphs,
        which at a large max_requests is a big graph to run a single decode request in.
        The smallest decode shape must stay captured: one request, at lcm(spec + 1,
        tp_size) tokens -- the smallest count that is both TP-aligned and a whole number
        of speculative steps."""
        graph_list = self._generate(
            num_cuda_graphs,
            CudaGraphSizingDistribution.HYBRID,
            max_requests=max_requests,
            tp_size=tp_size,
            num_speculative_tokens=num_speculative_tokens,
        )
        _, decode_counts = self._token_counts(graph_list)
        min_decode_tokens = math.lcm(num_speculative_tokens + 1, tp_size)

        assert min(decode_counts) == min_decode_tokens
        assert max(decode_counts) == max_requests * (num_speculative_tokens + 1)

    @pytest.mark.parametrize("max_requests", [8, 64, 1024])
    @pytest.mark.parametrize("num_cuda_graphs", [1, 2, 3, 4, 16, 32])
    @pytest.mark.parametrize("sizing_distribution", list(CudaGraphSizingDistribution))
    def test_decode_family_respects_the_graph_budget(
        self, max_requests, num_cuda_graphs, sizing_distribution
    ):
        """Reserving room for the small-decode floors must come out of the caller's graph
        budget, not extend it -- including at num_cuda_graphs=1, where the budget leaves
        room for the max token count alone."""
        _, decode_counts = self._token_counts(
            self._generate(num_cuda_graphs, sizing_distribution, max_requests=max_requests)
        )
        assert 0 < len(decode_counts) <= num_cuda_graphs

    def test_calculate_token_counts_rejects_unresolved_hybrid(self):
        """The per-family generator sees only one range and cannot tell which family it
        serves, so HYBRID must fail loudly there instead of silently falling through to
        exponential decode graphs."""
        with pytest.raises(AssertionError, match="HYBRID must be resolved"):
            CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(
                tp_size=1,
                num_cuda_graphs=8,
                cuda_graph_max_tokens=64,
                sizing_distribution=CudaGraphSizingDistribution.HYBRID,
            )

    @pytest.mark.parametrize(
        "tp_size, num_cuda_graphs, cuda_graph_max_tokens, expected",
        [
            # A single graph is the max token count alone.
            (1, 1, 80, [80]),
            # Exact division: the stride is max / N.
            (1, 2, 80, [80, 40]),
            # 80 / 32 = 2.5 rounds up to a stride of 4. Rounding it down to 2 instead
            # would ladder every even count from 2 to 80: 40 graphs for a budget of 32.
            # fmt: off
            (1, 32, 80, [80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20,
                         16, 12, 8, 4]),
            # fmt: on
            # 64 / 5 = 12.8 rounds up to 14 (CUDA_GRAPH_ROUNDER = 2), so the top gap
            # (64 -> 56) is narrower than the stride.
            (1, 5, 64, [64, 56, 42, 28, 14]),
            # A stride that is already TP-aligned is unchanged by tp_size.
            (2, 5, 64, [64, 56, 42, 28, 14]),
            # 10 / 3 = 3.33 rounds up to 4, leaving a 2-token top gap.
            (1, 3, 10, [10, 8, 4]),
            # tp_size widens the stride (34 -> 40) and floors the max (100 -> 96).
            (8, 3, 100, [96, 80, 40]),
            (4, 7, 100, [100, 96, 80, 64, 48, 32, 16]),
            # More graphs requested than the stride floor allows: the ladder is shorter
            # than num_cuda_graphs rather than denser than it.
            (1, 64, 10, [10, 8, 6, 4, 2]),
            (8, 64, 8, [8]),
            # A max token count that is not TP-aligned is floored before laddering.
            (2, 3, 7, [6, 4]),
        ],
    )
    def test_linear_ladder_rounding(
        self, tp_size, num_cuda_graphs, cuda_graph_max_tokens, expected
    ):
        """Exact linear ladders where the stride, the TP alignment, or the graph budget
        does not divide evenly."""
        counts = CUDAGraphBatchDimensionBuilder._calculate_token_counts_linear(
            tp_size=tp_size,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_max_tokens=cuda_graph_max_tokens,
        )
        assert counts == expected

    @pytest.mark.parametrize("num_cuda_graphs", [1, 2, 3, 5, 8, 16, 32, 64])
    @pytest.mark.parametrize("cuda_graph_max_tokens", [8, 10, 63, 64, 80, 255, 2048])
    @pytest.mark.parametrize("tp_size", [1, 2, 8])
    def test_linear_ladder_invariants(self, num_cuda_graphs, cuda_graph_max_tokens, tp_size):
        """Across every rounding combination the ladder stays within the caller's graph
        budget, stays TP-aligned and descending, and always offers the largest usable
        token count -- a batch at the budget must have a graph to land in."""
        counts = CUDAGraphBatchDimensionBuilder._calculate_token_counts_linear(
            tp_size=tp_size,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_max_tokens=cuda_graph_max_tokens,
        )
        assert len(counts) <= num_cuda_graphs
        assert counts == sorted(set(counts), reverse=True)
        assert all(c > 0 and c % tp_size == 0 for c in counts)
        assert counts[0] == (cuda_graph_max_tokens // tp_size) * tp_size

    @pytest.mark.parametrize("tp_size", [2, 8])
    def test_max_tokens_below_tp_size_yields_no_degenerate_graphs(self, tp_size):
        """Flooring a max token count below tp_size leaves nothing to capture. Graph
        generation must drop those shapes rather than emit a zero-token graph."""
        graph_list, token_counts = (
            CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                tp_size=tp_size,
                num_cuda_graphs=2,
                cuda_graph_max_tokens=4,
                cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
                max_requests=4,
                max_tokens=4,
                max_sequence_length=MAX_SEQ_LEN,
                use_cuda_graphs_for_non_decode_steps=False,
                sizing_distribution=CudaGraphSizingDistribution.HYBRID,
            )
        )
        assert all(bd.token_count > 0 for bd in graph_list)
        assert token_counts is None or all(tc > 0 for tc in token_counts)


class TestMatchGraphConfigWithEP:
    """Tests for match_graph_config with expert parallelism.

    Uses the world group as the EP group (all 8 GPUs form one EP group).
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=Utils.world_size,
        )

    @classmethod
    def teardown_class(cls):
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
    # 3. Any rank has prefill → all ranks fall back to eager (None)
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_any_prefill_rank_forces_eager(self, num_cuda_graphs):
        """When at least one EP rank has a prefill request,
        adjust_batch_dims_for_expert_parallelism returns None and ALL ranks
        get None from match_graph_config (eager mode)."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        # Rank 0 has a mixed batch (prefill + decode), all others decode-only
        if rank == 0:
            real = BD(token_count=64, prefill_req_count=2, decode_req_count=10)
        else:
            real = BD(token_count=32, prefill_req_count=0, decode_req_count=32)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "All ranks should run eager when any rank has prefill"

    # ------------------------------------------------------------------ #
    # 4. Mixed prefill graphs with strict matching
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
    # 9. All ranks decode-only → EP max-reduce finds a matching graph
    # ------------------------------------------------------------------ #
    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, 32, -1])
    def test_all_decode_ranks_match(self, num_cuda_graphs):
        """When all EP ranks are decode-only, the all-reduce max lifts token
        counts to the largest rank's value and a matching graph is found."""
        ep_group = self._get_ep_group()
        graph_list = _generate_graphs(num_cuda_graphs)
        rank = dist.get_rank()

        token_count = (rank + 1) * 4
        real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=token_count)

        result = _match(real, graph_list, ep_group=ep_group)
        _assert_consistent_across_ranks(result, ep_group)
        assert result is not None, "All-decode batch should match a graph"

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


class TestSpeculativeDecodingBatchDimensions:
    """Tests for batch dimensions specifically handling speculative decoding."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=Utils.world_size,
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _get_ep_group():
        return ps.get_expert_model_parallel_group()

    @pytest.mark.parametrize("num_speculative_tokens", [1, 3, 5])
    def test_generate_graphs_with_speculative_tokens(self, num_speculative_tokens):
        """Verify graph generation strictly adheres to the speculative token multiplier."""
        graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=TP_SIZE,
            num_cuda_graphs=4,
            cuda_graph_max_tokens=MAX_REQUESTS * (num_speculative_tokens + 1),
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_sequence_length=MAX_SEQ_LEN,
            use_cuda_graphs_for_non_decode_steps=True,
            num_speculative_tokens=num_speculative_tokens,
        )

        # For pure decode graphs, token_count must exactly equal decode_req_count * (spec_tokens + 1)
        decode_graphs = [g for g in graph_list if g.prefill_req_count == 0]
        assert len(decode_graphs) > 0, "Should generate decode-only graphs"

        for g in decode_graphs:
            expected_tokens = g.decode_req_count * (num_speculative_tokens + 1)
            assert g.token_count == expected_tokens, (
                f"Mismatch in speculative token math: Expected {expected_tokens} tokens "
                f"for {g.decode_req_count} requests with {num_speculative_tokens} spec tokens, got {g.token_count}."
            )

    def test_is_valid_with_speculative_tokens(self):
        """Verify that validation correctly enforces speculative token budgets."""
        num_speculative_tokens = 4
        # 10 decode requests * (4 spec + 1 actual) = 50 tokens required.

        # 49 tokens is not enough -> should be invalid
        bd_invalid = BD(token_count=49, prefill_req_count=0, decode_req_count=10)
        assert not bd_invalid.is_valid(
            max_requests=MAX_REQUESTS,
            max_sequence_length=MAX_SEQ_LEN,
            num_speculative_tokens=num_speculative_tokens,
        ), "Should reject batch dimension without enough tokens for speculative budget."

        # Exactly 50 tokens -> should be valid
        bd_valid = BD(token_count=50, prefill_req_count=0, decode_req_count=10)
        assert bd_valid.is_valid(
            max_requests=MAX_REQUESTS,
            max_sequence_length=MAX_SEQ_LEN,
            num_speculative_tokens=num_speculative_tokens,
        ), "Should accept batch dimension with perfectly matched speculative budget."

    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [1, 16, -1])
    def test_ep_sync_with_speculative_tokens(self, num_cuda_graphs):
        """Verify matching and EP rank syncing scales correctly with speculative tokens."""
        ep_group = self._get_ep_group()
        num_speculative_tokens = 2

        graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=TP_SIZE,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_max_tokens=MAX_REQUESTS * (num_speculative_tokens + 1),
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_sequence_length=MAX_SEQ_LEN,
            use_cuda_graphs_for_non_decode_steps=True,
            num_speculative_tokens=num_speculative_tokens,
        )

        rank = dist.get_rank()

        # Each rank has a different number of decode requests.
        decode_reqs = (rank + 1) * 2
        token_count = decode_reqs * (num_speculative_tokens + 1)
        real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=decode_reqs)

        result = _match(real, graph_list, ep_group=ep_group)

        # All ranks should end up syncing to the maximum requirement and picking the same graph
        _assert_consistent_across_ranks(result, ep_group)
        if result is not None:
            # Confirm the selected graph preserves the speculative token mathematical invariance
            assert result.token_count == result.decode_req_count * (num_speculative_tokens + 1)

    @pytest.mark.internal
    @pytest.mark.parametrize("num_cuda_graphs", [4, 16, -1])
    def test_ep_mixed_decode_prefill_with_speculative_tokens(self, num_cuda_graphs):
        """Verify EP sync when ranks have different request states with speculative tokens.

        Even ranks have decode-only requests; odd ranks have mixed prefill+decode.
        Since any prefill rank causes all ranks to fall back to eager (None),
        the test verifies that all ranks consistently get None.
        """
        ep_group = self._get_ep_group()
        num_speculative_tokens = 2
        ep_size = dist.get_world_size(ep_group)

        if ep_size < 2:
            pytest.skip("Test requires at least 2 EP ranks")

        graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=TP_SIZE,
            num_cuda_graphs=num_cuda_graphs,
            cuda_graph_max_tokens=MAX_REQUESTS * (num_speculative_tokens + 1),
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_sequence_length=MAX_SEQ_LEN,
            use_cuda_graphs_for_non_decode_steps=True,
            num_speculative_tokens=num_speculative_tokens,
        )

        rank = dist.get_rank()

        if rank % 2 == 0:
            # Decode-only: 4 decode requests with speculative tokens.
            decode_reqs = 4
            token_count = decode_reqs * (num_speculative_tokens + 1)
            real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=decode_reqs)
        else:
            # Mixed: 2 decode requests (with speculative) + 1 prefill request (8 tokens).
            decode_reqs = 2
            prefill_reqs = 1
            prefill_tokens = 8
            token_count = decode_reqs * (num_speculative_tokens + 1) + prefill_tokens
            real = BD(
                token_count=token_count,
                prefill_req_count=prefill_reqs,
                decode_req_count=decode_reqs,
            )

        result = _match(real, graph_list, ep_group=ep_group)

        # Any rank has prefill → all ranks get None (eager mode).
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "Any prefill rank should force all ranks to eager mode"

    @pytest.mark.internal
    def test_ep_speculative_decode_to_mixed_graph_transition(self):
        """Verify EP consistency when ranks have mixed prefill/decode states.

        When one EP rank is decode-only and another has prefill, the NCCL
        EP sync detects the prefill and returns None for all ranks (eager mode).
        """
        ep_group = self._get_ep_group()
        num_speculative_tokens = 3
        ep_size = dist.get_world_size(ep_group)

        if ep_size < 2:
            pytest.skip("Test requires at least 2 EP ranks")

        graph_list, _ = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
            tp_size=TP_SIZE,
            num_cuda_graphs=-1,  # Generate all possible graphs
            cuda_graph_max_tokens=MAX_REQUESTS * (num_speculative_tokens + 1),
            cuda_graph_mixed_prefill_request_count=MIXED_PREFILL_COUNT,
            max_requests=MAX_REQUESTS,
            max_tokens=MAX_TOKENS,
            max_sequence_length=MAX_SEQ_LEN,
            use_cuda_graphs_for_non_decode_steps=True,
            num_speculative_tokens=num_speculative_tokens,
        )

        rank = dist.get_rank()

        if rank % 2 == 0:
            # Decode-only: small batch.
            decode_reqs = 2
            token_count = decode_reqs * (num_speculative_tokens + 1)
            real = BD(token_count=token_count, prefill_req_count=0, decode_req_count=decode_reqs)
        else:
            # Prefill-only: forces even ranks out of decode-only graph.
            real = BD(token_count=32, prefill_req_count=2, decode_req_count=0)

        result = _match(real, graph_list, ep_group=ep_group)

        # Odd ranks have prefill → any prefill rank forces all to eager (None).
        _assert_consistent_across_ranks(result, ep_group)
        assert result is None, "Any prefill rank should force all ranks to eager mode"
