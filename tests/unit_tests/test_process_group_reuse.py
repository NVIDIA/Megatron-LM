# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch

import megatron.core.parallel_state as ps
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def group_factory(monkeypatch):
    """Isolate the cache and record creation without allocating communicators."""
    monkeypatch.setattr(ps, "_PROCESS_GROUP_CACHE", {})
    monkeypatch.setattr(ps, "_PROCESS_GROUP_REUSE_STATS", {"created": 0, "reused": 0})
    monkeypatch.setattr(ps, "_global_process_group_list", None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(ps, "is_torch_min_version", lambda version: True)
    monkeypatch.delenv("MEGATRON_DISABLE_PROCESS_GROUP_REUSE", raising=False)
    monkeypatch.delenv("NCCL_COLLNET_ENABLE", raising=False)
    factory = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(torch.distributed, "new_group", factory)
    return factory


def test_reuse_identical_nccl_group(group_factory):
    """Equivalent membership requests reuse one registered communicator."""
    first = ps.create_group([0, 1], timeout=timedelta(minutes=1), group_desc="first")
    second = ps.create_group(
        [1, 0], timeout=timedelta(minutes=1), backend="nccl", group_desc="second"
    )
    assert second is first
    assert group_factory.call_count == 1
    assert group_factory.call_args.kwargs["group_desc"] == "first"
    assert ps._global_process_group_list == [None, first]
    assert ps._PROCESS_GROUP_REUSE_STATS == {"created": 1, "reused": 1}


@pytest.mark.parametrize("rank", range(4))
def test_nonmembers_make_the_same_reuse_decision(group_factory, monkeypatch, rank):
    """Members and nonmembers enter the same sequence of new_group calls."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    nonmember = torch.distributed.GroupMember.NON_GROUP_MEMBER
    group_factory.side_effect = lambda **kwargs: object() if rank in kwargs["ranks"] else nonmember
    for ranks in ([0, 1], [2, 3]):
        first = ps.create_group(ranks)
        second = ps.create_group(ranks)
        assert second is first
    assert [call.kwargs["ranks"] for call in group_factory.call_args_list] == [[0, 1], [2, 3]]
    assert len(ps._global_process_group_list) == 2


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "gloo"},
        {"backend": "ucc"},
        {"pg_options": object()},
        {"use_local_synchronization": True},
        {"reuse_existing": False},
    ],
)
def test_explicitly_separate_groups_are_not_cached(group_factory, options):
    """Requests requiring an independent group neither read nor populate the cache."""
    cached = ps.create_group([0, 1])
    first = ps.create_group([0, 1], **options)
    second = ps.create_group([0, 1], **options)
    assert first is not cached
    assert second is not first
    assert group_factory.call_count == 3
    assert ps.create_group([0, 1]) is cached


@pytest.mark.parametrize("backend", ["gloo", "cpu:gloo,cuda:nccl"])
def test_default_non_nccl_backend_is_not_reused(group_factory, monkeypatch, backend):
    """backend=None must not silently reuse a non-NCCL WORLD backend."""
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: backend)
    assert ps.create_group([0, 1]) is not ps.create_group([0, 1])
    assert group_factory.call_count == 2
    assert not ps._PROCESS_GROUP_CACHE


def test_membership_timeout_and_sharp_settings_remain_distinct(group_factory, monkeypatch):
    """Different creation settings must not alias a cached communicator."""
    first = ps.create_group([0, 1], timeout=timedelta(minutes=1))
    assert ps.create_group([0, 2], timeout=timedelta(minutes=1)) is not first
    assert ps.create_group([0, 1], timeout=timedelta(minutes=2)) is not first
    monkeypatch.setenv("NCCL_COLLNET_ENABLE", "1")
    assert ps.create_group([0, 1], timeout=timedelta(minutes=1)) is not first
    monkeypatch.delenv("NCCL_COLLNET_ENABLE")
    assert ps.create_group([0, 1], timeout=timedelta(minutes=1)) is first
    assert group_factory.call_count == 4


def test_reuse_can_be_disabled_with_an_existing_cache(group_factory, monkeypatch):
    """The diagnostic override bypasses both cache lookup and insertion."""
    first = ps.create_group([0, 1])
    monkeypatch.setenv("MEGATRON_DISABLE_PROCESS_GROUP_REUSE", "1")
    second = ps.create_group([0, 1])
    assert second is not first
    assert ps.create_group([0, 1]) is not second
    monkeypatch.delenv("MEGATRON_DISABLE_PROCESS_GROUP_REUSE")
    assert ps.create_group([0, 1]) is first
    assert group_factory.call_count == 3


def test_failed_group_creation_is_not_cached(group_factory):
    """A failed collective must be attempted again on a subsequent request."""
    group_factory.side_effect = RuntimeError("creation failed")
    with pytest.raises(RuntimeError, match="creation failed"):
        ps.create_group([0, 1])
    assert not ps._PROCESS_GROUP_CACHE
    group_factory.side_effect = lambda **kwargs: object()
    first = ps.create_group([0, 1])
    assert ps.create_group([0, 1]) is first
    assert group_factory.call_count == 2


def test_all_gather_groups_preserve_independent_communicators(group_factory, monkeypatch):
    """Dense and expert all-gathers stay separate from reduce-scatter groups."""
    monkeypatch.setattr(ps, "is_initialized", lambda: True)
    monkeypatch.setattr(ps, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ps, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ps, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ps, "get_expert_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ps, "get_expert_gtp_weight_remat_world_size", lambda: 1)
    monkeypatch.setattr(ps, "_WEIGHT_REPLICA_RANK_SETS", [[0, 1, 2, 3]])
    dense_rs = ps.create_group([0, 1, 2, 3])
    expert_rs = ps.create_group([0, 2])
    dense_ag, expert_ag = ps.create_all_gather_groups(for_expert_parallelism=True)
    assert dense_ag is not dense_rs
    assert expert_ag is not expert_rs
    assert expert_ag is not None
    next_dense_ag, next_expert_ag = ps.create_all_gather_groups(for_expert_parallelism=True)
    assert next_dense_ag is not dense_ag
    assert next_expert_ag is not expert_ag
    assert ps.create_group([0, 1, 2, 3]) is dense_rs
    assert ps.create_group([0, 2]) is expert_rs


def test_reuse_cache_lifecycle_with_collectives(monkeypatch):
    """Reinitialization gets fresh groups and reused handles perform real collectives."""
    monkeypatch.delenv("MEGATRON_DISABLE_PROCESS_GROUP_REUSE", raising=False)
    Utils.destroy_model_parallel()
    previous = None
    try:
        for _ in range(2):
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
            group = ps.get_data_parallel_group()
            assert group is ps.get_data_parallel_group(with_context_parallel=True)
            assert group is not previous
            value = torch.ones(1, device=torch.cuda.current_device())
            torch.distributed.all_reduce(value, group=group)
            assert value.item() == torch.distributed.get_world_size()
            previous = group
            Utils.destroy_model_parallel()
            assert not ps._PROCESS_GROUP_CACHE
    finally:
        Utils.destroy_model_parallel()
