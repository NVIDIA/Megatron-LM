# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the DP-rank shard range computation in DistributedOptimizer.

These tests guard the O(1)-in-DP-world-size computation of the local rank's
grad-buffer shard range against the straightforward reference implementation
that materializes ranges for every DP rank. They run on CPU with mock buffers
so very large world sizes (e.g. 16k ranks) can be checked directly.
"""

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range


class _MockProcessGroup:
    def __init__(self, rank, world_size):
        self._rank = rank
        self._world_size = world_size

    def rank(self):
        return self._rank

    def size(self):
        return self._world_size


class _MockGradData:
    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


class _MockBucket:
    def __init__(self, numel, offset):
        self.grad_data = _MockGradData(numel)
        self.offset = offset


class _MockParamAndGradBuffer:
    def __init__(self, rank, world_size, bucket_numels, param_index_map):
        self.data_parallel_group = _MockProcessGroup(rank, world_size)
        self.buckets = []
        offset = 0
        for numel in bucket_numels:
            self.buckets.append(_MockBucket(numel, offset))
            offset += numel
        self.param_index_map = param_index_map


def _reference_local_world_range(rank, world_size, gbuf_size, bucket_offset):
    """Reference implementation: build all DP ranks' ranges, select the local one."""
    max_gbuf_range_size = gbuf_size // world_size
    all_ranges = []
    for r in range(world_size):
        start = r * max_gbuf_range_size
        end = min(gbuf_size, start + max_gbuf_range_size)
        all_ranges.append(Range(start + bucket_offset, end + bucket_offset))
    return all_ranges[rank]


def _make_params(world_size, bucket_numel):
    """A few params tiled across the bucket so every rank owns at least a shard."""
    params = {}
    boundaries = [0, bucket_numel // 3 + 1, (2 * bucket_numel) // 3 - 1, bucket_numel]
    for i in range(len(boundaries) - 1):
        param = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        params[param] = (boundaries[i], boundaries[i + 1], 0)
    return params


@pytest.mark.parametrize("world_size", [1, 2, 8, 64, 512, 4096, 16384])
def test_local_gbuf_range_matches_reference(world_size):
    bucket_numel = world_size * 24
    param_index_map = _make_params(world_size, bucket_numel)
    # Check first, last, and a middle rank.
    for rank in sorted({0, world_size // 2, world_size - 1}):
        buffer = _MockParamAndGradBuffer(rank, world_size, [bucket_numel], param_index_map)
        expected_world_range = _reference_local_world_range(rank, world_size, bucket_numel, 0)
        expected = DistributedOptimizer._build_model_gbuf_param_range_map(
            param_index_map, expected_world_range, 0
        )
        actual = DistributedOptimizer._build_model_gbuf_range(buffer, 0)["param_map"]
        assert actual.keys() == expected.keys()
        for param in expected:
            for key in ("gbuf_world", "gbuf_world_in_bucket", "gbuf_local", "param"):
                assert actual[param][key].start == expected[param][key].start, (rank, key)
                assert actual[param][key].end == expected[param][key].end, (rank, key)


def test_local_gbuf_range_with_multiple_buckets():
    world_size = 1024
    bucket_numels = [world_size * 8, world_size * 4, world_size * 16]
    rank = world_size - 1
    param_index_map = {}
    offset = 0
    for bucket_index, numel in enumerate(bucket_numels):
        param = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        param_index_map[param] = (offset, offset + numel, bucket_index)
        offset += numel
    buffer = _MockParamAndGradBuffer(rank, world_size, bucket_numels, param_index_map)

    bucket_offset = 0
    for bucket_index, numel in enumerate(bucket_numels):
        expected_world_range = _reference_local_world_range(rank, world_size, numel, bucket_offset)
        expected = DistributedOptimizer._build_model_gbuf_param_range_map(
            param_index_map, expected_world_range, bucket_offset
        )
        actual = DistributedOptimizer._build_model_gbuf_range(buffer, bucket_index)["param_map"]
        assert actual.keys() == expected.keys()
        for param in expected:
            for key in ("gbuf_world", "gbuf_world_in_bucket", "gbuf_local", "param"):
                assert actual[param][key].start == expected[param][key].start
                assert actual[param][key].end == expected[param][key].end
        bucket_offset += numel
