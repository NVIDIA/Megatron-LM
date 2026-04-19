# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for parameter layout computation functions.

These tests verify the pure-computation layout functions without requiring
GPU or distributed setup:
- pad_to_divisor, pad_param_start, pad_bucket_end (shared padding utilities)
- group_params_for_buffers (parameter grouping by dtype/expert)
- _compute_default_per_buffer_param_layout (no-padding layout)
- DistributedOptimizer._compute_per_buffer_param_layout (padded layout)
- DistributedOptimizer.compute_full_param_layout (end-to-end layout)
"""

import math
from unittest import mock

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import (
    _compute_default_per_buffer_param_layout,
    group_params_for_buffers,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.param_layout import (
    BufferKey,
    pad_bucket_end,
    pad_param_start,
    pad_to_divisor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(*shapes, dtype=torch.bfloat16):
    """Create a list of nn.Parameters with the given shapes."""
    return [torch.nn.Parameter(torch.randn(s, dtype=dtype)) for s in shapes]


def _make_param_with_attrs(shape, dtype=torch.bfloat16, **attrs):
    """Create an nn.Parameter with extra attributes (e.g. allreduce, shared_embedding)."""
    param = torch.nn.Parameter(torch.randn(shape, dtype=dtype))
    for k, v in attrs.items():
        setattr(param, k, v)
    return param


# ---------------------------------------------------------------------------
# Tests for shared padding utilities
# ---------------------------------------------------------------------------


class TestPaddingUtilities:

    def test_pad_to_divisor_exact_multiple(self):
        assert pad_to_divisor(128, 64) == 128

    def test_pad_to_divisor_rounds_up(self):
        assert pad_to_divisor(65, 64) == 128

    def test_pad_to_divisor_zero(self):
        assert pad_to_divisor(0, 64) == 0

    def test_pad_param_start(self):
        assert pad_param_start(0) == 0
        assert pad_param_start(1) == 64
        assert pad_param_start(63) == 64
        assert pad_param_start(64) == 64
        assert pad_param_start(65) == 128

    def test_pad_bucket_end_basic(self):
        dp_size = 4
        divisor = math.lcm(dp_size, 128)
        result = pad_bucket_end(1, dp_size, pad_for_high_nccl_busbw=False)
        assert result == divisor
        assert result % dp_size == 0
        assert result % 128 == 0

    def test_pad_bucket_end_high_busbw(self):
        dp_size = 4
        divisor = math.lcm(dp_size, 128, 2**16)
        result = pad_bucket_end(1, dp_size, pad_for_high_nccl_busbw=True)
        assert result == divisor
        assert result % (2**16) == 0

    def test_pad_bucket_end_already_aligned(self):
        dp_size = 2
        divisor = math.lcm(dp_size, 128)
        result = pad_bucket_end(divisor, dp_size, pad_for_high_nccl_busbw=False)
        assert result == divisor


# ---------------------------------------------------------------------------
# Tests for group_params_for_buffers
# ---------------------------------------------------------------------------


class TestGroupParamsForBuffers:

    def test_single_dtype_no_experts(self):
        """All bf16 params with no expert-parallel should go in one group."""
        params = _make_params((100, 100), (50, 50))
        result = group_params_for_buffers(params, grad_reduce_in_fp32=True)

        assert len(result) == 1
        key = list(result.keys())[0]
        assert key == BufferKey(torch.bfloat16, torch.float, False)
        group_params, indices = result[key]
        assert group_params == params
        assert indices == [0, 1]

    def test_grad_reduce_not_fp32(self):
        """When grad_reduce_in_fp32=False, grad_dtype matches param dtype."""
        params = _make_params((100,))
        result = group_params_for_buffers(params, grad_reduce_in_fp32=False)

        key = list(result.keys())[0]
        assert key.grad_dtype == torch.bfloat16

    def test_expert_parallel_separation(self):
        """Params with allreduce=False should be in a separate group."""
        dense = _make_param_with_attrs((100,))
        expert = _make_param_with_attrs((100,), allreduce=False)
        result = group_params_for_buffers([dense, expert], grad_reduce_in_fp32=True)

        assert len(result) == 2
        dense_key = BufferKey(torch.bfloat16, torch.float, False)
        expert_key = BufferKey(torch.bfloat16, torch.float, True)
        assert dense_key in result
        assert expert_key in result
        assert result[dense_key][0] == [dense]
        assert result[expert_key][0] == [expert]

    def test_param_indices_independent_per_group(self):
        """Expert and dense groups should have independent param_indices starting at 0."""
        dense_params = _make_params((100,), (200,))
        expert = _make_param_with_attrs((100,), allreduce=False)
        result = group_params_for_buffers(
            [dense_params[0], expert, dense_params[1]], grad_reduce_in_fp32=True
        )

        dense_key = BufferKey(torch.bfloat16, torch.float, False)
        expert_key = BufferKey(torch.bfloat16, torch.float, True)
        _, dense_indices = result[dense_key]
        _, expert_indices = result[expert_key]
        assert dense_indices == [0, 1]
        assert expert_indices == [0]

    def test_mixed_dtypes(self):
        """Params with different dtypes go in separate groups."""
        bf16_param = _make_params((100,), dtype=torch.bfloat16)[0]
        fp32_param = _make_params((100,), dtype=torch.float32)[0]
        result = group_params_for_buffers([bf16_param, fp32_param], grad_reduce_in_fp32=True)

        assert len(result) == 2
        bf16_key = BufferKey(torch.bfloat16, torch.float, False)
        fp32_key = BufferKey(torch.float32, torch.float, False)
        assert bf16_key in result
        assert fp32_key in result


# ---------------------------------------------------------------------------
# Tests for _compute_default_per_buffer_param_layout
# ---------------------------------------------------------------------------


class TestDefaultParamLayout:

    def test_single_bucket_no_padding(self):
        """With bucket_size=None, all params go in one bucket with no padding."""
        params = _make_params((100, 100), (50, 50))
        layout = _compute_default_per_buffer_param_layout(params, bucket_size=None)

        # Params iterated in reverse: params[1] first (2500 elems), params[0] second (10000).
        assert layout.param_index_map[params[1]] == (0, 2500, 0)
        assert layout.param_index_map[params[0]] == (2500, 12500, 0)
        assert layout.bucket_indices == [(0, 12500)]
        assert layout.per_bucket_numel_unpadded == [12500]

    def test_multiple_buckets(self):
        """Params should split into buckets when exceeding bucket_size."""
        params = _make_params((100, 100), (100, 100), (100, 100))
        layout = _compute_default_per_buffer_param_layout(params, bucket_size=15000)

        # Reverse order: params[2], params[1], params[0]. Each is 10000 elems.
        # After params[2]: 10000 < 15000, continue.
        # After params[1]: 20000 >= 15000, finalize bucket.
        # After params[0]: 10000 < 15000, finalize at end.
        assert len(layout.bucket_indices) == 2
        assert layout.per_bucket_numel_unpadded[0] == 20000
        assert layout.per_bucket_numel_unpadded[1] == 10000

    def test_no_padding_applied(self):
        """Default layout should never add padding."""
        params = _make_params((97,), (103,))
        layout = _compute_default_per_buffer_param_layout(params, bucket_size=None)

        total_numel = 97 + 103
        assert layout.bucket_indices == [(0, total_numel)]
        assert layout.per_bucket_numel_unpadded == [total_numel]

    def test_single_param(self):
        params = _make_params((256,))
        layout = _compute_default_per_buffer_param_layout(params, bucket_size=None)

        assert layout.param_index_map[params[0]] == (0, 256, 0)
        assert layout.bucket_indices == [(0, 256)]


# ---------------------------------------------------------------------------
# Tests for DistributedOptimizer._compute_per_buffer_param_layout
# ---------------------------------------------------------------------------


class TestDistOptParamLayout:

    @staticmethod
    def _make_ddp_config(**overrides):
        defaults = dict(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            bucket_size=None,
            average_in_collective=False,
        )
        defaults.update(overrides)
        return DistributedDataParallelConfig(**defaults)

    def test_param_start_64_alignment(self):
        """Each param's start index should be 64-aligned."""
        # 97 is not a multiple of 64, so second param must be padded.
        params = _make_params((97,), (103,))
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        for param in params:
            start, end, _ = layout.param_index_map[param]
            assert start % 64 == 0, f"Start {start} should be 64-aligned"
            assert end - start == param.numel()

    def test_bucket_end_dp_divisible(self):
        """Each bucket end should be divisible by lcm(dp_size, 128)."""
        params = _make_params((1000,), (1000,))
        dp_size = 4
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=dp_size, ddp_config=ddp_config
        )

        divisor = math.lcm(dp_size, 128)
        for start, end in layout.bucket_indices:
            assert end % divisor == 0, f"Bucket end {end} should be divisible by {divisor}"

    def test_bucket_end_high_busbw_padding(self):
        """With pad_buckets_for_high_nccl_busbw, bucket end should be divisible by 2^16."""
        params = _make_params((1000,))
        dp_size = 2
        ddp_config = self._make_ddp_config(pad_buckets_for_high_nccl_busbw=True)
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=dp_size, ddp_config=ddp_config
        )

        divisor = math.lcm(dp_size, 128, 2**16)
        for _, end in layout.bucket_indices:
            assert end % divisor == 0

    def test_shared_embedding_gets_separate_bucket(self):
        """Params with shared_embedding=True should be placed in their own bucket."""
        regular = _make_param_with_attrs((1000,))
        shared = _make_param_with_attrs((1000,), shared_embedding=True)
        # Reverse order: shared first (since it's last in list), then regular.
        params = [regular, shared]
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        # shared_embedding param should be in its own bucket.
        _, _, shared_bucket = layout.param_index_map[shared]
        _, _, regular_bucket = layout.param_index_map[regular]
        assert shared_bucket != regular_bucket

    def test_shared_embedding_as_first_reversed_param_no_extra_bucket(self):
        """If shared_embedding param is the first in reversed order (last in list),
        it should not create an empty extra bucket before it."""
        shared = _make_param_with_attrs((1000,), shared_embedding=True)
        regular = _make_param_with_attrs((1000,))
        # Reverse order: regular first, then shared.
        params = [shared, regular]
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        # Both params should be in their own buckets (shared splits after regular).
        assert len(layout.bucket_indices) == 2

    def test_multiple_buckets_with_bucket_size(self):
        """Verify bucket splitting with an explicit bucket_size."""
        params = _make_params((5000,), (5000,), (5000,))
        dp_size = 2
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=8000, data_parallel_world_size=dp_size, ddp_config=ddp_config
        )

        # Each param is 5000 elems. With 64-alignment:
        # Bucket 0: params[2] starts at 0, ends at 5000 (< 8000); params[1] starts at 5000
        #   (already 64-aligned since 5000 rounds to 5056), ends at 10056 >= 8000 → finalize.
        # Bucket 1: params[0].
        assert len(layout.bucket_indices) == 2
        assert len(layout.per_bucket_numel_unpadded) == 2

    def test_numel_unpadded_vs_padded(self):
        """per_bucket_numel_unpadded should be <= padded bucket size."""
        params = _make_params((1000,))
        dp_size = 8
        ddp_config = self._make_ddp_config()
        layout = DistributedOptimizer._compute_per_buffer_param_layout(
            params, bucket_size=None, data_parallel_world_size=dp_size, ddp_config=ddp_config
        )

        for i, (start, end) in enumerate(layout.bucket_indices):
            padded_numel = end - start
            assert layout.per_bucket_numel_unpadded[i] <= padded_numel


# ---------------------------------------------------------------------------
# Tests for DistributedOptimizer.compute_full_param_layout
# ---------------------------------------------------------------------------


class TestComputeFullParamLayout:

    @staticmethod
    def _make_ddp_config(**overrides):
        defaults = dict(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            bucket_size=None,
            average_in_collective=False,
        )
        defaults.update(overrides)
        return DistributedDataParallelConfig(**defaults)

    def test_dense_only(self):
        """With only dense params, should produce a single layout."""
        params = _make_params((100, 100), (50, 50))
        ddp_config = self._make_ddp_config()
        full_layout = DistributedOptimizer.compute_full_param_layout(
            params, bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        assert len(full_layout.layouts) == 1
        key = list(full_layout.layouts.keys())[0]
        assert key.is_expert_parallel is False
        layout = full_layout.layouts[key]
        assert set(layout.param_index_map.keys()) == set(params)

    def test_dense_and_expert_separate_layouts(self):
        """Dense and expert-parallel params should get independent layouts."""
        dense = _make_param_with_attrs((100, 100))
        expert = _make_param_with_attrs((100, 100), allreduce=False)
        ddp_config = self._make_ddp_config()
        full_layout = DistributedOptimizer.compute_full_param_layout(
            [dense, expert], bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        assert len(full_layout.layouts) == 2
        dense_key = BufferKey(torch.bfloat16, torch.bfloat16, False)
        expert_key = BufferKey(torch.bfloat16, torch.bfloat16, True)
        assert dense_key in full_layout.layouts
        assert expert_key in full_layout.layouts

        # Each layout should only contain its own params.
        assert set(full_layout.layouts[dense_key].param_index_map.keys()) == {dense}
        assert set(full_layout.layouts[expert_key].param_index_map.keys()) == {expert}

        # Both should start at index 0 (independent index spaces).
        dense_starts = [s for s, _, _ in full_layout.layouts[dense_key].param_index_map.values()]
        expert_starts = [s for s, _, _ in full_layout.layouts[expert_key].param_index_map.values()]
        assert min(dense_starts) == 0
        assert min(expert_starts) == 0

    def test_expert_uses_expert_dp_world_size(self):
        """Expert-parallel layout should use expert_data_parallel_world_size for padding."""
        dense = _make_param_with_attrs((1000,))
        expert = _make_param_with_attrs((1000,), allreduce=False)
        ddp_config = self._make_ddp_config()

        # Dense dp_size=3, expert dp_size=256.
        # lcm(3, 128) = 384, lcm(256, 128) = 256 — different divisors.
        full_layout = DistributedOptimizer.compute_full_param_layout(
            [dense, expert],
            bucket_size=None,
            data_parallel_world_size=3,
            ddp_config=ddp_config,
            expert_data_parallel_world_size=256,
        )

        dense_key = BufferKey(torch.bfloat16, torch.bfloat16, False)
        expert_key = BufferKey(torch.bfloat16, torch.bfloat16, True)

        # Expert bucket end should be divisible by lcm(256, 128) = 256.
        expert_divisor = math.lcm(256, 128)
        assert expert_divisor == 256
        for _, end in full_layout.layouts[expert_key].bucket_indices:
            assert end % expert_divisor == 0

        # Dense bucket end should be divisible by lcm(3, 128) = 384.
        dense_divisor = math.lcm(3, 128)
        assert dense_divisor == 384
        for _, end in full_layout.layouts[dense_key].bucket_indices:
            assert end % dense_divisor == 0

    def test_param_indices_populated(self):
        """compute_full_param_layout should populate param_indices on each layout."""
        params = _make_params((100,), (200,), (300,))
        ddp_config = self._make_ddp_config()
        full_layout = DistributedOptimizer.compute_full_param_layout(
            params, bucket_size=None, data_parallel_world_size=2, ddp_config=ddp_config
        )

        layout = list(full_layout.layouts.values())[0]
        assert len(layout.param_indices) == 3
        assert min(layout.param_indices) == 0
        assert max(layout.param_indices) == 2
