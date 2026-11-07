# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU topology regressions for runtime CP collections, without collectives."""

from copy import deepcopy

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
from megatron.core.process_groups_config import ProcessGroupCollection


class _Group:
    def __init__(self, ranks, rank):
        self.ranks = list(ranks)
        self.local_rank = self.ranks.index(rank) if rank in self.ranks else -1

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.local_rank


@pytest.fixture
def initialize_groups(monkeypatch):
    # Keep every process-group state change isolated from the distributed test suite.
    for name in vars(parallel_state):
        if name.startswith('_') and name.isupper():
            current = getattr(parallel_state, name)
            monkeypatch.setattr(parallel_state, name, {} if isinstance(current, dict) else None)

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(parallel_state, 'get_nccl_options', lambda *args: None)
    monkeypatch.setattr(parallel_state, '_set_global_memory_buffer', lambda: None)

    def initialize(world_size, tp_size, cp_size, min_cp_size, gtp_size=1, rank=0, dynamic=True):
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world_size)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
        monkeypatch.setattr(
            parallel_state, 'create_group', lambda ranks, **kwargs: _Group(ranks, rank)
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            gtp_remat_size=gtp_size,
            dynamic_context_parallel=dynamic,
            min_dynamic_context_parallel_size=min_cp_size,
            create_gloo_process_groups=False,
        )
        return ProcessGroupCollection.use_mpu_process_groups()

    return initialize


@pytest.mark.parametrize(
    'world_size,tp_size,cp_size,min_cp_size,gtp_size,rank',
    [
        (1, 1, 1, 1, 1, 0),
        (6, 1, 1, 1, 1, 4),
        (6, 1, 1, 6, 1, 0),
        (12, 2, 1, 1, 1, 7),
        (16, 2, 2, 2, 1, 13),
        (24, 2, 1, 1, 2, 5),
    ],
)
def test_initialized_runtime_groups_are_available_in_collection(
    initialize_groups, world_size, tp_size, cp_size, min_cp_size, gtp_size, rank
):
    pgs = initialize_groups(world_size, tp_size, cp_size, min_cp_size, gtp_size, rank)
    total_cp_size = world_size // (tp_size * gtp_size)
    expected_sizes = {
        size
        for size in get_valid_dynamic_context_parallel_group_sizes(total_cp_size)
        if size >= min_cp_size
    }
    assert set(pgs.dynamic_cp) == expected_sizes
    assert set(pgs.dynamic_tp_cp) == expected_sizes
    assert pgs.dynamic_cp[total_cp_size] is pgs.dp_cp
    assert pgs.cp.size() == cp_size

    for size in expected_sizes:
        cp_group = pgs.dynamic_cp[size]
        tp_cp_group = pgs.dynamic_tp_cp[size]
        assert cp_group.size() == size
        assert tp_cp_group.size() == tp_size * size
        assert cp_group.rank() >= 0
        assert tp_cp_group.rank() >= 0
        assert set(cp_group.ranks).issubset(pgs.dp_cp.ranks)
        assert set(cp_group.ranks).issubset(tp_cp_group.ranks)
        assert parallel_state.get_dynamic_data_context_parallel_groups(group_size=size) is cp_group
        assert (
            parallel_state.get_dynamic_tensor_data_context_parallel_group(group_size=size)
            is tp_cp_group
        )
        if tp_size == 1:
            assert tp_cp_group is cp_group

    # Modifying a consumer's snapshot must not alter the compatibility globals.
    del pgs.dynamic_cp[total_cp_size]
    assert total_cp_size in parallel_state.get_dynamic_data_context_parallel_group_map()


def test_disabled_dynamic_cp_has_empty_maps(initialize_groups):
    pgs = initialize_groups(6, 2, 1, 1, dynamic=False)
    assert pgs.dynamic_cp == {}
    assert pgs.dynamic_tp_cp == {}
    assert pgs.cp.size() == 1
    assert pgs.dp_cp.size() == 3


def test_uninitialized_dynamic_maps_do_not_require_distributed_state(monkeypatch):
    monkeypatch.setattr(parallel_state, '_DYNAMIC_DP_CP_GROUPS', {})
    monkeypatch.setattr(parallel_state, '_DYNAMIC_TENSOR_DATA_CONTEXT_PARALLEL_GROUPS', {})

    def unexpected_state_read():
        pytest.fail('Empty dynamic maps must not inspect the uninitialized TP state')

    monkeypatch.setattr(
        parallel_state, 'get_tensor_model_parallel_world_size', unexpected_state_read
    )
    pgs = ProcessGroupCollection.use_mpu_process_groups(['dynamic_cp', 'dynamic_tp_cp'])
    assert pgs.dynamic_cp == {}
    assert pgs.dynamic_tp_cp == {}
    assert pgs.cp is None


def test_explicit_runtime_maps_preserve_group_handles_and_cp_size_keys():
    singleton = _Group([0], 0)
    runtime_cp = _Group([0, 2], 0)
    tp_runtime_cp = _Group([0, 1, 2, 3], 0)
    pgs = ProcessGroupCollection(
        cp=singleton, dynamic_cp={1: singleton, 2: runtime_cp}, dynamic_tp_cp={2: tp_runtime_cp}
    )
    assert pgs.dynamic_cp[1] is singleton
    assert pgs.dynamic_cp[2] is runtime_cp
    assert pgs.dynamic_tp_cp[2] is tp_runtime_cp
    assert deepcopy(pgs) is pgs
    assert pgs.cp is singleton
    assert 'dynamic_cp({1: 1, 2: 2})' in repr(pgs)
    assert 'dynamic_tp_cp({2: 4})' in repr(pgs)


@pytest.mark.parametrize('total_size', [0, -1])
def test_invalid_runtime_cp_domain_is_rejected(total_size):
    with pytest.raises(ValueError, match='must be positive'):
        get_valid_dynamic_context_parallel_group_sizes(total_size)
