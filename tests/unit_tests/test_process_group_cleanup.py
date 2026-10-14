# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model-parallel test helpers must release their groups and preserve other owners."""

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
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


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("destroy_between", [True, False], ids=["destroy", "reinitialize"])
def test_fp8_amax_after_model_parallel_reinitialization(destroy_between):
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    available, reason = FP8GlobalStateManager.is_fp8_available()
    if not available:
        pytest.skip(reason)

    try:
        for _ in range(2):
            Utils.initialize_model_parallel(1, 1)
            layer = te.Linear(16, 16, params_dtype=torch.bfloat16, device="cuda")
            inputs = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
            # Delayed scaling retains amax buffers and their reduction group globally.
            # The second forward must not reduce buffers using the previous group.
            with te.fp8_autocast(
                fp8_recipe=DelayedScaling(),
                fp8_group=parallel_state.get_amax_reduction_group(with_context_parallel=True),
            ):
                output = layer(inputs)
            assert torch.isfinite(output).all()
            del layer, inputs, output
            if destroy_between:
                Utils.destroy_model_parallel()
    finally:
        # Keep a failing regression on an old helper from contaminating later tests.
        FP8GlobalStateManager.reset()
        Utils.destroy_model_parallel()
