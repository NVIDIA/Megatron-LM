# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Indexer metrics must reduce local contributions independently of runtime CP order."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def logging_groups():
    if not dist.is_initialized():
        if torch.cuda.is_available():
            Utils.initialize_distributed()
        else:
            dist.init_process_group("gloo")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if world_size < 2 or world_size % 2:
        pytest.skip("CP1/CP2 logging requires an even number of distributed ranks")
    owned = []

    def group_for(ranks):
        group = dist.new_group(ranks)
        if rank in ranks:
            owned.append(group)
            return group
        return None

    singleton = pair = stripe = None
    for member in range(world_size):
        group = group_for([member])
        if group is not None:
            singleton = group
    for first in range(0, world_size, 2):
        group = group_for([first, first + 1])
        if group is not None:
            pair = group
    for offset in range(2):
        group = group_for(list(range(offset, world_size, 2)))
        if group is not None:
            stripe = group
    quad_stripe = None
    if world_size % 4 == 0:
        for first in range(0, world_size, 4):
            for offset in range(2):
                group = group_for([first + offset, first + offset + 2])
                if group is not None:
                    quad_stripe = group
    device = (
        torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    )
    yield SimpleNamespace(
        singleton=singleton,
        pair=pair,
        stripe=stripe,
        quad_stripe=quad_stripe,
        world=dist.group.WORLD,
        size=world_size,
        rank=rank,
        device=device,
    )
    dist.barrier()
    for group in reversed(owned):
        dist.destroy_process_group(group)


@pytest.fixture(autouse=True)
def reset_tracker(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def _log(pg_collection, loss_scale=1.0, num_layers=1):
    losses = {}
    DSAIndexerLossLoggingHelper.track_indexer_metrics(
        loss_scale=loss_scale,
        iteration=1,
        writer=None,
        pg_collection=pg_collection,
        total_loss_dict=losses,
        num_layers=num_layers,
        num_indexer_layers=1,
    )
    return losses


@pytest.mark.parametrize("average_cp", [False, True])
@pytest.mark.parametrize("empty_rank", [False, True])
def test_runtime_cp_microbatch_order_does_not_change_logging(
    logging_groups, average_cp, empty_rank
):
    groups = logging_groups
    pg_collection = ProcessGroupCollection(pp=groups.singleton, dp=groups.world, dp_cp=groups.world)
    results = []
    for order in ((1, 2), (2, 1)):
        DSAIndexerLossLoggingHelper.tracker = {}
        for cp_size in order:
            local_loss = (groups.rank + 1) * (1 if cp_size == 1 else 10)
            if empty_rank and groups.rank == groups.size - 1:
                local_loss = 0
            cp_group = groups.pair if cp_size == 2 else None
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                torch.tensor(float(local_loss), device=groups.device),
                layer_number=1,
                num_layers=1,
                reduce_group=cp_group if not average_cp else None,
                avg_group=cp_group if average_cp else None,
                raw_loss=torch.tensor(float(local_loss * 2), device=groups.device),
            )
        results.append(_log(pg_collection, loss_scale=0.5))
    contributing_sum = sum(range(1, groups.size + (0 if empty_rank else 1)))
    expected = contributing_sum / groups.size * (1 + (10 if average_cp else 20)) * 0.5
    for losses in results:
        torch.testing.assert_close(
            losses["indexer loss"], torch.tensor(expected, device=groups.device)
        )
        torch.testing.assert_close(
            losses["indexer raw loss"], torch.tensor(expected * 2, device=groups.device)
        )


@pytest.mark.parametrize("average_cp", [False, True])
@pytest.mark.parametrize("with_gtp_remat", [False, True])
def test_static_cp_and_gtp_remat_normalization_is_preserved(
    logging_groups, average_cp, with_gtp_remat
):
    groups = logging_groups
    if with_gtp_remat and groups.size % 4:
        pytest.skip("CP2 x GTP-remat2 normalization requires a multiple of four ranks")
    pg_collection = ProcessGroupCollection(
        pp=groups.singleton,
        dp=groups.singleton if with_gtp_remat else groups.stripe,
        dp_cp=groups.pair if with_gtp_remat else groups.world,
        gtp_remat=groups.quad_stripe if with_gtp_remat else None,
    )
    local_loss = torch.tensor(float(groups.rank + 1), device=groups.device)
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        local_loss,
        layer_number=1,
        num_layers=1,
        reduce_group=groups.pair if not average_cp else None,
        avg_group=groups.pair if average_cp else None,
        raw_loss=local_loss * 2,
    )
    losses = _log(pg_collection)
    if with_gtp_remat:
        first = groups.rank // 4 * 4
        expected = sum(range(first + 1, first + 5)) / 2
    else:
        expected = sum(range(1, groups.size + 1)) / (groups.size // 2)
    expected /= 2 if average_cp else 1
    torch.testing.assert_close(losses["indexer loss"], torch.tensor(expected, device=groups.device))
    torch.testing.assert_close(
        losses["indexer raw loss"], torch.tensor(expected * 2, device=groups.device)
    )


@pytest.mark.parametrize("average_cp", [False, True])
def test_pipeline_stage_without_indexer_uses_the_same_reduction(logging_groups, average_cp):
    groups = logging_groups
    if groups.size % 4:
        pytest.skip("PP2 x CP2 logging requires a multiple of four ranks")
    pg_collection = ProcessGroupCollection(
        pp=groups.quad_stripe, dp=groups.singleton, dp_cp=groups.pair
    )
    if groups.quad_stripe.rank() == 0:
        local_loss = torch.tensor(float(groups.rank + 1), device=groups.device)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            local_loss,
            layer_number=2,
            num_layers=3,
            reduce_group=groups.pair if not average_cp else None,
            avg_group=groups.pair if average_cp else None,
            raw_loss=local_loss * 2,
        )
    losses = _log(pg_collection, num_layers=3)
    first = groups.rank // 4 * 4
    expected = float(2 * first + 3) / (2 if average_cp else 1)
    torch.testing.assert_close(losses["indexer loss"], torch.tensor(expected, device=groups.device))
    torch.testing.assert_close(
        losses["indexer raw loss"], torch.tensor(expected * 2, device=groups.device)
    )


@pytest.mark.parametrize("average_cp", [False, True])
def test_runtime_cp2_matches_static_cp2_logical_sample_mean(logging_groups, average_cp):
    groups = logging_groups
    results = []
    for static_cp in (1, 2):
        DSAIndexerLossLoggingHelper.tracker = {}
        pg_collection = ProcessGroupCollection(
            pp=groups.singleton,
            cp=groups.singleton if static_cp == 1 else groups.pair,
            dp=groups.world if static_cp == 1 else groups.stripe,
            dp_cp=groups.world,
        )
        local_loss = torch.tensor(float(groups.rank + 1), device=groups.device)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            local_loss,
            1,
            1,
            reduce_group=groups.pair if not average_cp else None,
            avg_group=groups.pair if average_cp else None,
            raw_loss=local_loss * 2,
        )
        results.append(_log(pg_collection))
    expected = sum(range(1, groups.size + 1)) / groups.size * (1 if average_cp else 2)
    for losses in results:
        torch.testing.assert_close(
            losses["indexer loss"], torch.tensor(expected, device=groups.device)
        )
        torch.testing.assert_close(
            losses["indexer raw loss"], torch.tensor(expected * 2, device=groups.device)
        )


@pytest.mark.parametrize("include_singleton_cp", [False, True])
def test_cp1_partial_collection_uses_legacy_dp_group(logging_groups, include_singleton_cp):
    groups = logging_groups
    pg_collection = ProcessGroupCollection(pp=groups.singleton, dp=groups.world)
    if include_singleton_cp:
        pg_collection.cp = groups.singleton
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(float(groups.rank + 1), device=groups.device), 1, 1
    )
    losses = _log(pg_collection)
    torch.testing.assert_close(
        losses["indexer loss"], torch.tensor((groups.size + 1) / 2, device=groups.device)
    )


def test_real_cp_requires_fixed_data_context_group(logging_groups):
    groups = logging_groups
    pg_collection = ProcessGroupCollection(pp=groups.singleton, dp=groups.stripe, cp=groups.pair)
    with pytest.raises(ValueError, match="with CP requires a fixed dp_cp"):
        _log(pg_collection)


@pytest.mark.parametrize("average_cp", [False, True])
def test_runtime_cp_requires_fixed_group_even_without_collection_cp(logging_groups, average_cp):
    groups = logging_groups
    pg_collection = ProcessGroupCollection(pp=groups.singleton, dp=groups.world)
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(1.0, device=groups.device),
        1,
        1,
        reduce_group=groups.pair if not average_cp else None,
        avg_group=groups.pair if average_cp else None,
    )
    with pytest.raises(ValueError, match="with CP requires a fixed dp_cp"):
        _log(pg_collection)
