# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Tests for Megatron-FSDP parameter grouping by chunk size factor.

A bucket is padded to a multiple of data_parallel_world_size * chunk_size_factor so that every
DP-Shard slice of it holds a whole number of parameter rows. Parameters whose row sizes cannot
share one factor therefore have to be split into separate groups: folding them into a single
group means the factor, and with it the padding, grows to their least common multiple.

These tests exercise _get_parameter_groups directly and need neither CUDA nor a process group.
"""

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    _get_parameter_groups,
)

# Row size, i.e. Linear.in_features, is what becomes a group's chunk size factor. These three are
# DeepSeek-V3's dense projection widths, and they are pairwise incompatible: 18432 = 2^11 * 3^2,
# 16384 = 2^14 and 7168 = 2^10 * 7, so their least common multiple is 1032192, 56x the largest of
# them. At 512-way data parallelism that factor pads a single bucket by up to 528M elements.
DSV3_ROW_SIZES = (18432, 16384, 7168)
LCM_OF_DSV3_ROW_SIZES = 1032192

# Enough rows that no parameter is smaller than a candidate factor, since a parameter with fewer
# elements than the factor is packed into padding gaps as a fragment rather than splitting a group.
# Real DeepSeek-V3 weights clear that bar by a wide margin; 24 rows is the smallest multiple of
# four that clears it for all three row sizes above.
NUM_ROWS = 24


def build_module(row_sizes) -> torch.nn.Module:
    """Build a module holding one weight per requested row size."""
    return torch.nn.Sequential(
        *[torch.nn.Linear(row_size, NUM_ROWS, bias=False) for row_size in row_sizes]
    )


def group_parameters(module: torch.nn.Module):
    """Group a module's parameters with bucket splitting by size disabled."""
    policy = BucketingPolicy(
        suggested_bucket_size=None,
        fsdp_unit_modules=[],
        data_parallel_sharding_strategy="optim_grads_params",
    )
    bucket_groups, _, _ = _get_parameter_groups(module, policy, meta_device_init_fp8_params={})
    return bucket_groups


class TestChunkSizeFactorGrouping:
    """Grouping must keep each factor at a real row size instead of widening it to the LCM."""

    def test_factor_never_exceeds_largest_row_size(self):
        """
        No group may carry a factor larger than the widest row it holds.

        Widening a group's factor to the LCM of incompatible row sizes inflates it without
        bound, and because padding is a multiple of the factor the wasted memory grows with it.
        """
        groups = group_parameters(build_module(DSV3_ROW_SIZES))

        factors = sorted(group.chunk_size_factor for group in groups)
        assert max(factors) <= max(DSV3_ROW_SIZES), (
            f"chunk size factors {factors} exceed the largest row size {max(DSV3_ROW_SIZES)}, so "
            f"the factor was widened toward the row sizes' common multiple {LCM_OF_DSV3_ROW_SIZES}"
        )
        assert set(factors).issubset(
            set(DSV3_ROW_SIZES)
        ), f"every factor should be one of the row sizes {DSV3_ROW_SIZES}, got {factors}"

    def test_incompatible_row_sizes_are_split(self):
        """Pairwise incompatible row sizes each get their own group."""
        groups = group_parameters(build_module(DSV3_ROW_SIZES))

        assert len(groups) == len(DSV3_ROW_SIZES)
        assert sorted(group.chunk_size_factor for group in groups) == sorted(DSV3_ROW_SIZES)

    def test_every_member_is_compatible_with_its_group_factor(self):
        """
        A group's factor must divide evenly into every member it holds.

        This is the property the padding relies on: each DP-Shard slice of the bucket has to
        contain whole rows. Widening the factor after admitting parameters breaks it, because
        those parameters were only ever checked against the narrower factor.
        """
        groups = group_parameters(build_module(DSV3_ROW_SIZES))

        for group in groups:
            factor = group.chunk_size_factor
            for param in group.params:
                row_size = param.shape[1:].numel()
                assert (
                    factor % row_size == 0
                ), f"factor {factor} is not a multiple of row size {row_size}"
                assert (
                    param.numel() % factor == 0 or param.numel() < factor
                ), f"parameter of {param.numel()} elements does not tile factor {factor}"

    def test_no_parameter_is_dropped_or_duplicated(self):
        """
        Deferring a parameter to a later group must neither lose nor duplicate it.

        Splitting reprocesses the deferred parameters on the next pass, so a mistake there is
        silent: the parameter would simply never receive a buffer.
        """
        module = build_module(DSV3_ROW_SIZES)
        groups = group_parameters(module)

        grouped = [param for group in groups for param in group.params]
        expected = list(module.parameters())
        assert len(grouped) == len(expected)
        assert {id(param) for param in grouped} == {id(param) for param in expected}

    def test_compatible_row_sizes_stay_in_one_group(self):
        """
        Rows that already share a factor must not be split apart.

        Splitting is only worth its extra collectives when the alternative is an inflated
        factor, so a narrower row size that divides the wider one belongs in the same group.
        """
        # 16384 is a multiple of 4096, and both parameters tile a factor of 16384.
        groups = group_parameters(build_module((16384, 4096)))

        assert len(groups) == 1
        assert groups[0].chunk_size_factor == 16384
        assert len(groups[0].params) == 2
