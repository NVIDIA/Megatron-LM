# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Device-bound process-group tests, scheduled in their own eight-GPU CI bucket.

This module owns WORLD for its entire lifetime. It refuses to reuse another test's WORLD
and destroys the default group at teardown, so eager communicator allocation cannot leak
into later test modules when this file is run outside its dedicated bucket.
"""

import os
from datetime import timedelta

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_process_group_sharing import LAYOUTS, ROLE_GLOBALS

MULTI_RANK_LAYOUTS = [
    "pp2",
    "ep2",
    "ep8_expert_dp1",
    "tp2_cp2_ep4_expert_dp1",
    "pp2_ep4_expert_dp1",
    "cp8_dp1",
]


@pytest.fixture(scope="module")
def device_bound_world():
    """Own a device-bound WORLD without changing an existing default process group."""
    if torch.distributed.is_initialized():
        pytest.skip("Run this file in its dedicated bucket with no pre-existing WORLD")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("Launch this file with torchrun on exactly eight ranks")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=2)
    )
    try:
        world = torch.distributed.distributed_c10d._get_default_group()
        assert world.bound_device_id == device
        yield
    finally:
        torch.distributed.destroy_process_group()


def _run_multi_rank_layout(layout, share_process_groups):
    from tests.unit_tests.test_utilities import Utils

    calls = []
    real_new_group = torch.distributed.new_group

    def recording_new_group(*args, **kwargs):
        calls.append((kwargs.get("group_desc"), tuple(kwargs.get("ranks") or args[0])))
        return real_new_group(*args, **kwargs)

    try:
        torch.distributed.new_group = recording_new_group
        try:
            Utils.initialize_model_parallel(
                share_process_groups=share_process_groups, **LAYOUTS[layout]
            )
        finally:
            torch.distributed.new_group = real_new_group
        every_rank_calls = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(every_rank_calls, calls)
        assert all(rank_calls == calls for rank_calls in every_rank_calls), layout

        # One all-reduce per role, in the same order on every rank.
        for role, name in ROLE_GLOBALS.items():
            group = getattr(ps, name, None)
            if group is None or isinstance(group, list):
                continue
            value = torch.ones(1, device="cuda")
            torch.distributed.all_reduce(value, group=group)
            assert value.item() == torch.distributed.get_world_size(group=group), (layout, role)
        torch.cuda.synchronize()

        groups = {role: getattr(ps, name, None) for role, name in ROLE_GLOBALS.items()}
        if share_process_groups and layout in ("ep8_expert_dp1", "pp2_ep4_expert_dp1"):
            assert groups["ep"] is not groups["dp_cp"]
        if share_process_groups and layout == "cp8_dp1":
            assert groups["cp"] is not groups["dp_cp"]
        return len(calls)
    finally:
        # destroy_model_parallel() drops the NCCL groups without destroying them; destroy each
        # communicator this build created exactly once, shared or not, so layouts do not pile up.
        created = [group for group in ps._global_process_group_list or [] if group is not None]
        Utils.destroy_model_parallel()
        live_groups = torch.distributed.distributed_c10d._world.pg_map
        for group in reversed(created):
            if live_groups.get(group) is not None:
                torch.distributed.destroy_process_group(group)
        ps._global_process_group_list = None


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
@pytest.mark.parametrize("layout", MULTI_RANK_LAYOUTS)
def test_shared_groups_on_a_real_world(layout, device_bound_world):
    shared = _run_multi_rank_layout(layout, share_process_groups=True)
    separate = _run_multi_rank_layout(layout, share_process_groups=False)
    assert shared < separate
    # Initializing again builds new groups that also work.
    assert _run_multi_rank_layout(layout, share_process_groups=True) == shared
