# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from datetime import timedelta
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def distributed_without_model_parallel():
    Utils.initialize_distributed()
    ps.destroy_model_parallel()
    yield
    ps.destroy_model_parallel()


def _assert_world_usable():
    value = torch.ones(2, device="cuda")
    torch.distributed.all_reduce(value)
    assert torch.equal(value, torch.full_like(value, torch.distributed.get_world_size()))


@pytest.mark.skipif(Utils.world_size < 2, reason="Requires a rank outside the subgroup")
def test_destroy_subgroup_preserves_world(distributed_without_model_parallel):
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    group = ps.create_group(ranks=[0], backend="nccl")
    if torch.distributed.get_rank() == 0:
        value = torch.ones(1, device="cuda")
        torch.distributed.all_reduce(value, group=group)
        torch.cuda.synchronize()

    ps.destroy_model_parallel()

    # Nonmembers must not accidentally satisfy a WORLD barrier inside teardown.
    store = torch.distributed.PrefixStore("test_destroy_subgroup_preserves_world", Utils.store)
    if torch.distributed.get_rank() == 0:
        store.set("destroyed", "1")
    else:
        store.wait(["destroyed"], timedelta(seconds=90))

    assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
    _assert_world_usable()


@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "context"])
def test_repeated_lifetimes_release_process_groups(
    distributed_without_model_parallel, parallel_axis
):
    baseline = set(torch.distributed.distributed_c10d._world.pg_map)
    size_key = {
        "tensor": "tensor_model_parallel_size",
        "pipeline": "pipeline_model_parallel_size",
        "context": "context_parallel_size",
    }[parallel_axis]

    for _ in range(2):
        ps.initialize_model_parallel(**{size_key: torch.distributed.get_world_size()})
        created = set(torch.distributed.distributed_c10d._world.pg_map) - baseline
        assert created

        # Initialize actual communicators, including Gloo groups, before checking release.
        for group in ps._global_process_group_list:
            if group is None:
                continue
            device = "cpu" if torch.distributed.get_backend(group) == "gloo" else "cuda"
            value = torch.ones(1, device=device)
            torch.distributed.all_reduce(value, group=group)
            assert value.item() == torch.distributed.get_world_size(group)

        ps.destroy_model_parallel()
        assert not ps.model_parallel_is_initialized()
        assert set(torch.distributed.distributed_c10d._world.pg_map) == baseline
        _assert_world_usable()


@pytest.mark.parametrize("direct_cache_clear", [False, True], ids=["torch_2_6_to_2_9", "current"])
def test_destroy_invalidates_dtensor_sharding_cache(
    distributed_without_model_parallel, monkeypatch, direct_cache_clear
):
    current_group = [object()]

    @lru_cache(maxsize=8)
    def cached_sharding(mesh_layout):
        return current_group[0]

    propagate = SimpleNamespace(cache=cached_sharding)
    if direct_cache_clear:
        propagate.cache_clear = cached_sharding.cache_clear
    dtensor = SimpleNamespace(
        _op_dispatcher=SimpleNamespace(
            sharding_propagator=SimpleNamespace(propagate_op_sharding=propagate)
        )
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", SimpleNamespace(DTensor=dtensor))
    native_cache = {"same_mesh": current_group[0]}
    monkeypatch.setattr(
        torch._C, "_clear_DTensor_sharding_propagator_cache", native_cache.clear, raising=False
    )

    previous_group = cached_sharding("same_mesh")
    ps.destroy_model_parallel()
    current_group[0] = object()

    assert cached_sharding("same_mesh") is current_group[0]
    assert cached_sharding("same_mesh") is not previous_group
    assert not native_cache
