# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for GTP weight and token layouts."""

import itertools
import math
from fractions import Fraction
from types import SimpleNamespace

from megatron.core.gtp_parallel_layout import GTPParallelLayout, get_sample_parallel_size


def _by_rank(groups):
    return {rank: group for group in groups for rank in group}


class TestGTPParallelLayout:
    def test_target_nvlink_placement(self):
        for tp, world in ((1, 128), (2, 256)):
            layout = GTPParallelLayout(world, tp, 1, 128, 64 // tp)
            assert layout.sample_size == 1
            assert layout.replica_size == 2 * tp
            for group in layout.weights.get_ranks("tp-gtp_remat"):
                assert len(group) == 64
                assert len({rank // 64 for rank in group}) == 1
            assert all(len(g) == 128 for g in layout.data.get_ranks("cp"))

    def test_complete_orthogonal_contribution_domains(self):
        # Include non-nested sizes to exercise both residual CP and extra data.
        for tp, pp, cp, weight, dp in itertools.product(
            (1, 2), (1, 2), (1, 2, 3, 4, 8, 128), (1, 2, 4, 6, 32), (1, 2)
        ):
            world = tp * pp * math.lcm(cp, weight) * dp
            layout = GTPParallelLayout(world, tp, pp, cp, weight, rank_offset=7)
            weights = _by_rank(layout.weights.get_ranks("gtp_remat"))
            replicas = _by_rank(layout.weights.get_ranks("dp"))
            contexts = _by_rank(layout.data.get_ranks("cp"))
            samples = _by_rank(layout.data.get_ranks("gtp_remat-dp"))
            tokens = _by_rank(layout.data.get_ranks("cp-gtp_remat-dp"))
            for rank in range(7, 7 + world):
                assert set(weights[rank]) & set(replicas[rank]) == {rank}
                assert set(contexts[rank]) & set(samples[rank]) == {rank}
                assert len(contexts[rank]) == cp
                assert len(weights[rank]) == weight
                contributions = [peer for r in replicas[rank] for peer in weights[r]]
                assert len(contributions) == len(set(contributions))
                assert set(contributions) == set(tokens[rank])
                # Replica peers own the same shard; CP peers process the same sequence.
                assert len({weights[r].index(r) for r in replicas[rank]}) == 1
                assert len({samples[r].index(r) for r in contexts[rank]}) == 1

    def test_cp_extension_preserves_weight_and_optimizer_ownership(self):
        for tp, world in ((1, 128), (2, 256)):
            layouts = [
                GTPParallelLayout(world, tp, 1, cp, 64 // tp) for cp in (1, 2, 8, 32, 64, 128)
            ]
            for layout in layouts[1:]:
                for group in ("gtp_remat", "dp", "tp-gtp_remat-pp"):
                    assert layouts[0].weights.get_ranks(group) == layout.weights.get_ranks(group)

    def test_gradient_reductions_count_each_contribution_once(self):
        for cp, weight in ((8, 4), (2, 4), (3, 4), (1, 4)):
            world = math.lcm(cp, weight) * 2
            layout = GTPParallelLayout(world, 1, 1, cp, weight)
            weights = _by_rank(layout.weights.get_ranks("gtp_remat"))
            replicas = _by_rank(layout.weights.get_ranks("dp"))
            # Rank-distinct values detect accidentally reduced unrelated shards.
            gradients = [
                [Fraction((r + 1) * (s + 3), 7) for s in range(weight)] for r in range(world)
            ]
            for per_token in (False, True):
                w_div = 1 if per_token else weight
                r_div = 1 if per_token else layout.replica_size
                token_div = sum(range(1, world + 1)) if per_token else 1
                for rank in range(world):
                    shard = weights[rank].index(rank)
                    matrix_grad = (
                        sum(
                            sum(gradients[p][shard] for p in weights[r]) / w_div
                            for r in replicas[rank]
                        )
                        / r_div
                        / token_div
                    )
                    # Norm/bias: DDP across replicas followed by GTP all-reduce.
                    replicated_grad = (
                        sum(
                            sum(gradients[p][0] for p in replicas[r]) / r_div for r in weights[rank]
                        )
                        / w_div
                        / token_div
                    )
                    divisor = token_div if per_token else world
                    assert matrix_grad == sum(g[shard] for g in gradients) / divisor
                    assert replicated_grad == sum(g[0] for g in gradients) / divisor

    def test_sample_accounting(self):
        legacy = SimpleNamespace(data_parallel_size=2, gtp_weight_remat_size=4)
        assert get_sample_parallel_size(legacy) == 8
        overlap = SimpleNamespace(
            gtp_remat_fold_cp=True,
            world_size=256,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=128,
        )
        assert get_sample_parallel_size(overlap) == 1
        overlap.sample_parallel_size = 3
        assert get_sample_parallel_size(overlap) == 3
