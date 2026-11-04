# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# This primitive requires only torch; isolate CPU tests from checkpoint I/O,
# Transformer Engine and model imports in the checkpoint package initializer.
_path = Path(__file__).resolve().parents[4] / 'megatron/lite/primitive/ckpt/expert_layout.py'
_spec = importlib.util.spec_from_file_location('_expert_layout_under_test', _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
ExpertLayoutPlan = _module.ExpertLayoutPlan
exchange_expert_layout_tensors = _module.exchange_expert_layout_tensors


@pytest.mark.parametrize('source_ep,target_ep', [(4, 2), (2, 4), (2, 2), (4, 4)])
def test_plan_routes_experts_and_selects_one_replica(source_ep, target_ep):
    ranks = (1, 3, 5, 7)
    plan = ExpertLayoutPlan(8, source_ep, target_ep, ranks)
    for expert in range(8):
        owner = expert // (8 // source_ep)
        target = expert // (8 // target_ep)
        holders = [ranks[i] for i in range(4) if i % source_ep == owner]
        assert plan.source_rank(expert) == holders[target % len(holders)]
        assert plan.destination_ranks(expert) == tuple(
            ranks[i] for i in range(4) if i % target_ep == target
        )
    with pytest.raises(ValueError, match='outside'):
        plan.source_rank(8)


@pytest.mark.parametrize(
    'args',
    [
        (0, 2, 2, (0, 1)),
        (8, 0, 2, (0, 1)),
        (8, 3, 2, (0, 1, 2, 3)),
        (8, 2, 2, (0, 0)),
        (8, 2, 2, ()),
        (8, 2, 4, (0, 1)),
    ],
)
def test_plan_rejects_invalid_topologies(args):
    with pytest.raises(ValueError):
        ExpertLayoutPlan(*args)


def _expert_id(name):
    return int(name.split('.')[1])


def _values(expert, sender):
    # Replica-specific sentinels expose which holder sent each tensor. Scales
    # are scalar, and odd byte payloads precede wider, noncontiguous tensors.
    return [
        (f'experts.{expert}.weight', torch.arange(3 + expert % 3, dtype=torch.uint8)),
        (f'experts.{expert}.fp8', torch.arange(4, dtype=torch.uint8).view(torch.float8_e4m3fn)),
        (f'experts.{expert}.scale', torch.tensor(expert + sender / 8, dtype=torch.float32)),
        (f'experts.{expert}.wide', torch.arange(6, dtype=torch.float64).reshape(2, 3).t()),
        (f'experts.{expert}.empty', torch.empty(0, 2, dtype=torch.float16)),
    ]


def _worker(rank, world_size, rendezvous, source_ep, target_ep, ranks, failure):
    dist.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        group = dist.new_group(list(ranks), backend='gloo', timeout=timedelta(seconds=30))
        if rank not in ranks:
            return
        position = ranks.index(rank)
        plan = ExpertLayoutPlan(8, source_ep, target_ep, ranks)
        weights = [
            item
            for expert in range(8)
            if expert // (8 // source_ep) == position % source_ep
            for item in _values(expert, rank)
        ]
        budget = 17
        if failure == 'duplicate' and rank == ranks[0]:
            weights.append(weights[0])
        if failure == 'wrong_owner' and rank == ranks[0]:
            weights.append(('experts.7.extra', torch.ones(1)))
        if failure == 'budget' and rank == ranks[0]:
            budget = 18
        if failure == 'rank_order':
            plan = ExpertLayoutPlan(8, source_ep, target_ep, tuple(reversed(ranks)))
        if failure == 'empty':
            weights = []
        stream = exchange_expert_layout_tensors(
            iter(weights),
            plan,
            group=group,
            device=torch.device('cpu'),
            expert_id_from_name=_expert_id,
            buffer_max_size_bytes=budget,
        )
        if failure in {'duplicate', 'wrong_owner', 'budget', 'rank_order'}:
            with pytest.raises(ValueError):
                list(stream)
            return
        result = list(stream)
        if failure == 'empty':
            assert not result
            return
        received = dict(result)
        assert len(received) == len(result), 'Duplicate deliveries'
        expected = {}
        for expert in range(8):
            target_owner = expert // (8 // target_ep)
            if target_owner != position % target_ep:
                continue
            source_owner = expert // (8 // source_ep)
            holders = [ranks[i] for i in range(len(ranks)) if i % source_ep == source_owner]
            sender = holders[target_owner % len(holders)]
            expected.update(_values(expert, sender))
        assert received.keys() == expected.keys()
        for name, value in expected.items():
            actual = received[name]
            assert actual.shape == value.shape and actual.dtype == value.dtype
            assert torch.equal(
                actual.reshape(-1).view(torch.uint8),
                value.contiguous().reshape(-1).view(torch.uint8),
            ), name
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('source_ep,target_ep', [(4, 2), (2, 4), (2, 2), (4, 4)])
def test_real_gloo_exchange_preserves_bytes_with_unequal_rounds(tmp_path, source_ep, target_ep):
    # In 2->4, one holder is unselected for a target shard; all ranks must still
    # join every round, even after exhausting their differently sized streams.
    mp.spawn(
        _worker,
        args=(4, str(tmp_path / 'store'), source_ep, target_ep, (0, 1, 2, 3), None),
        nprocs=4,
        join=True,
    )


def test_real_gloo_exchange_uses_explicit_noncontiguous_group(tmp_path):
    mp.spawn(
        _worker, args=(6, str(tmp_path / 'store'), 2, 4, (0, 2, 4, 5), None), nprocs=6, join=True
    )


@pytest.mark.parametrize('failure', ['duplicate', 'wrong_owner', 'budget', 'rank_order', 'empty'])
def test_all_group_members_handle_invalid_or_empty_streams(tmp_path, failure):
    mp.spawn(
        _worker, args=(4, str(tmp_path / 'store'), 2, 4, (0, 1, 2, 3), failure), nprocs=4, join=True
    )
