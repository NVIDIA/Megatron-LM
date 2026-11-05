# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The expert data-parallel group that DDP setup creates when a collection leaves it out."""

import contextlib
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def distributed():
    if Utils.world_size < 2:
        pytest.skip("per-rank and collective group creation coincide on a single rank")
    Utils.initialize_distributed()
    yield
    Utils.destroy_model_parallel()


@contextlib.contextmanager
def _record_group_creation():
    """Record the ranks of every process group created in the block, in creation order."""
    created = []
    new_group = dist.distributed_c10d.new_group

    def recording_new_group(*args, **kwargs):
        ranks = kwargs["ranks"] if "ranks" in kwargs else (args[0] if args else None)
        created.append(None if ranks is None else sorted(ranks))
        return new_group(*args, **kwargs)

    with (
        mock.patch.object(dist, "new_group", recording_new_group),
        mock.patch.object(dist.distributed_c10d, "new_group", recording_new_group),
    ):
        yield created


def _collection_without_expert_dp():
    """Collection of a data-parallel-only grid (TP = PP = EP = 1) that leaves out expt_dp.

    Creating groups is collective: call on every rank.
    """
    world_size = dist.get_world_size()
    single_rank_group, _ = dist.new_subgroups_by_enumeration([[rank] for rank in range(world_size)])
    return ProcessGroupCollection(
        dp=dist.new_group(ranks=list(range(world_size))),
        tp=single_rank_group,
        pp=single_rank_group,
        ep=single_rank_group,
    )


def _assert_same_on_every_rank(created):
    created_on_every_rank = [None] * dist.get_world_size()
    dist.all_gather_object(created_on_every_rank, created)
    assert all(
        created_on_rank == created_on_every_rank[0] for created_on_rank in created_on_every_rank
    ), f"ranks created different process groups: {created_on_every_rank}"


def _assert_later_groups_work():
    """Create a group over every rank and reduce on it, as later setup code does."""
    world_size = dist.get_world_size()
    group = dist.new_group(ranks=list(range(world_size)))
    names = [None] * world_size
    dist.all_gather_object(names, group.group_name)
    assert names == [names[0]] * world_size, f"ranks named the same group differently: {names}"
    value = torch.tensor([float(dist.get_rank())], device="cuda")
    dist.all_reduce(value, group=group)
    assert value.item() == world_size * (world_size - 1) / 2


@pytest.mark.parametrize("expt_dp", ["absent", "none"])
def test_missing_expert_dp_group_is_created_alike_on_every_rank(distributed, expt_dp):
    """Every rank creates the same single-rank groups, keeps its own, and stays in step."""
    pg_collection = _collection_without_expert_dp()
    if expt_dp == "none":
        pg_collection.expt_dp = None
    config = TransformerConfig(num_attention_heads=1, num_layers=1)
    ddp_config = DistributedDataParallelConfig()

    with _record_group_creation() as created:
        groups = ProcessGroupCollection.setup_process_groups_for_ddp(
            pg_collection, config, ddp_config
        )

    _assert_same_on_every_rank(created)
    assert created, "expected an expert data-parallel group to be created"
    expt_dp_group = groups['expt_dp_group']
    assert dist.get_process_group_ranks(expt_dp_group) == [dist.get_rank()]
    assert groups['intra_expt_dp_group'] is expt_dp_group

    value = torch.tensor([float(dist.get_rank())], device="cuda")
    dist.all_reduce(value, group=expt_dp_group)
    assert value.item() == dist.get_rank()
    _assert_later_groups_work()


def test_ddp_without_expert_dp_group_keeps_group_creation_in_step(distributed):
    """DistributedDataParallel on such a collection creates the same groups on every rank."""
    pg_collection = _collection_without_expert_dp()
    config = TransformerConfig(num_attention_heads=1, num_layers=1)
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    module = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))

    with _record_group_creation() as created:
        ddp_model = DistributedDataParallel(
            config, ddp_config=ddp_config, module=module.cuda(), pg_collection=pg_collection
        )

    _assert_same_on_every_rank(created)
    assert dist.get_process_group_ranks(ddp_model.expt_dp_group) == [dist.get_rank()]
    _assert_later_groups_work()


def test_provided_expert_dp_group_is_used_without_creating_groups(distributed):
    pg_collection = _collection_without_expert_dp()
    pg_collection.expt_dp = pg_collection.dp
    config = TransformerConfig(num_attention_heads=1, num_layers=1)
    ddp_config = DistributedDataParallelConfig()

    with _record_group_creation() as created:
        groups = ProcessGroupCollection.setup_process_groups_for_ddp(
            pg_collection, config, ddp_config
        )

    assert created == []
    assert groups['expt_dp_group'] is pg_collection.expt_dp
    assert groups['intra_expt_dp_group'] is pg_collection.expt_dp
