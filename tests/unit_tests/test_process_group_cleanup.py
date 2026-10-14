# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-parallel test helpers must release their groups and preserve other owners."""

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("destroy_between", [True, False], ids=["destroy", "reinitialize"])
def test_model_parallel_group_cleanup(destroy_between):
    Utils.initialize_distributed()
    initial_groups = set(dist.distributed_c10d._world.pg_map)
    independent_group = dist.new_group(list(range(dist.get_world_size())))
    baseline_groups = set(dist.distributed_c10d._world.pg_map)
    try:
        Utils.initialize_model_parallel(1, 1)
        active_group_count = len(dist.distributed_c10d._world.pg_map)
        for _ in range(2):
            if destroy_between:
                Utils.destroy_model_parallel()
                assert set(dist.distributed_c10d._world.pg_map) == baseline_groups
            Utils.initialize_model_parallel(1, 1)
            current_groups = set(dist.distributed_c10d._world.pg_map)
            assert baseline_groups <= current_groups
            assert len(current_groups) == active_group_count

        Utils.destroy_model_parallel()
        assert set(dist.distributed_c10d._world.pg_map) == baseline_groups
        for group in (independent_group, dist.group.WORLD):
            value = torch.ones(1, device="cuda")
            dist.all_reduce(value, group=group)
            assert value.item() == dist.get_world_size(group)
    finally:
        Utils.destroy_model_parallel()
        # Also release leaked groups when this regression test fails on an old helper.
        owned_groups = [
            group for group in dist.distributed_c10d._world.pg_map if group not in initial_groups
        ]
        for group in reversed(owned_groups):
            dist.destroy_process_group(group)
