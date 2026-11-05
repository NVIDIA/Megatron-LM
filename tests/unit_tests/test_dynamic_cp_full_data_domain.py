# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime CP may cross GTP peers without changing weight/optimizer groups.

Standalone: torchrun --nproc-per-node=4 -m pytest --noconftest -c /dev/null -q
    tests/unit_tests/test_dynamic_cp_full_data_domain.py
Eight ranks additionally cover TP2/CP2 and multiple weight replicas with TP2.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps


@pytest.fixture(scope="module")
def distributed_world():
    owns_world = not dist.is_initialized()
    if owns_world:
        if "RANK" not in os.environ or not torch.cuda.is_available():
            pytest.skip("Requires torchrun with CUDA ranks")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    yield dist.get_world_size(), dist.get_rank()
    if owns_world:
        dist.destroy_process_group()


def _members(group):
    return dist.get_process_group_ranks(group)


def _check_sum(group, expected_members):
    value = torch.tensor([dist.get_rank() + 1], dtype=torch.int64, device="cuda")
    dist.all_reduce(value, group=group)
    assert value.item() == sum(rank + 1 for rank in expected_members)


@pytest.mark.parametrize(
    "tp,cp,gtp,pp,min_cp",
    [
        (1, 1, 1, 1, 1),
        (2, 1, 1, 1, 1),
        (1, 1, 2, 1, 1),
        (1, 2, 2, 1, 1),
        (2, 1, 2, 1, 1),
        (2, 2, 2, 1, 1),
        (1, 1, 2, 2, 1),
        (1, 2, 2, 1, 2),
    ],
)
def test_runtime_cp_spans_full_data_domain(distributed_world, tp, cp, gtp, pp, min_cp):
    world, rank = distributed_world
    tile = tp * cp * gtp * pp
    if world < tile or world % tile:
        pytest.skip(f"Requires world size divisible by {tile}")
    replicas = world // tile
    ps.destroy_model_parallel()
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            gtp_remat_size=gtp,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=tp,
            dynamic_context_parallel=True,
            min_dynamic_context_parallel_size=min_cp,
            create_gloo_process_groups=False,
        )
        # Independent coordinate arithmetic for the default tp-cp-gtp-dp-pp order.
        pipeline_width = world // pp
        pp_start = rank // pipeline_width * pipeline_width
        tp_coordinate = rank % tp
        domain = list(range(pp_start + tp_coordinate, pp_start + pipeline_width, tp))
        domain_index = domain.index(rank)
        cp_coordinate = domain_index % cp
        gtp_coordinate = domain_index // cp % gtp
        replica_coordinate = domain_index // (cp * gtp)
        weight_group = [
            pp_start + tp_coordinate + tp * (cp_coordinate + cp * (g + gtp * replica_coordinate))
            for g in range(gtp)
        ]
        replica_group = [
            pp_start + tp_coordinate + tp * (c + cp * (gtp_coordinate + gtp * d))
            for d in range(replicas)
            for c in range(cp)
        ]
        data_group = [
            pp_start + tp_coordinate + tp * (cp_coordinate + cp * (gtp_coordinate + gtp * d))
            for d in range(replicas)
        ]
        static_cp = [
            pp_start + tp_coordinate + tp * (c + cp * (gtp_coordinate + gtp * replica_coordinate))
            for c in range(cp)
        ]
        full = ps.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=True)
        assert _members(full) == domain
        assert ps.get_data_parallel_world_size(True, True) == replicas * gtp * cp
        assert _members(ps.get_gtp_weight_remat_group()) == weight_group
        assert _members(ps.get_context_parallel_group()) == static_cp
        assert _members(ps.get_data_parallel_group(with_gtp_remat=False)) == data_group
        replica = ps.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=False)
        assert _members(replica) == replica_group
        partial = ps.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
        )
        assert _members(partial) == replica_group
        optimizer_ag, _ = ps.create_all_gather_groups(for_expert_parallelism=False)
        assert _members(optimizer_ag) == replica_group

        crossed_gtp = False
        for size in ps.get_valid_dynamic_context_parallel_group_sizes(len(domain)):
            if size < min_cp:
                continue
            start = domain_index // size * size
            expected_cp = domain[start : start + size]
            expected_tp_cp = [r - tp_coordinate + t for r in expected_cp for t in range(tp)]
            runtime_cp = ps.get_dynamic_data_context_parallel_groups(group_size=size)
            runtime_tp_cp = ps.get_dynamic_tensor_data_context_parallel_group(group_size=size)
            assert ps.get_dynamic_data_context_parallel_group_map()[size] is runtime_cp
            assert _members(runtime_cp) == expected_cp
            assert _members(runtime_tp_cp) == expected_tp_cp
            assert runtime_cp.size() == size
            assert runtime_tp_cp.size() == tp * size
            _check_sum(runtime_cp, expected_cp)
            if tp > 1:
                _check_sum(runtime_tp_cp, expected_tp_cp)
            if size == len(domain):
                assert runtime_cp is full
            coordinates = {(r - pp_start) // (tp * cp) % gtp for r in expected_cp}
            crossed_gtp |= len(coordinates) > 1
        assert crossed_gtp == (gtp > 1)
        # Runtime group creation must not redirect weight collectives to the full domain.
        _check_sum(ps.get_gtp_weight_remat_group(), weight_group)
        _check_sum(replica, replica_group)
        _check_sum(optimizer_ag, replica_group)
        if gtp == 1:
            assert full is replica
    finally:
        ps.destroy_model_parallel()
