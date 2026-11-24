# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""reduce_max_stat_across_model_parallel_group() accepts host values and tensors."""

import pytest
import torch

from megatron.training.utils.common_utils import reduce_max_stat_across_model_parallel_group
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module")
def world():
    Utils.initialize_distributed()
    return torch.distributed.group.WORLD


def test_device_tensor_is_reduced_without_touching_the_callers_buffer(world):
    stat = torch.full((1,), float(Utils.rank), device="cuda")
    reduced = reduce_max_stat_across_model_parallel_group(stat, group=world, return_tensor=True)
    assert reduced.is_cuda
    assert reduced.item() == Utils.world_size - 1
    assert stat.item() == Utils.rank


def test_cpu_tensor_is_moved_to_the_device(world):
    stat = torch.tensor(float(Utils.rank))
    reduced = reduce_max_stat_across_model_parallel_group(stat, group=world, return_tensor=True)
    assert reduced.is_cuda
    assert reduced.item() == Utils.world_size - 1


def test_host_float_and_absent_stat(world):
    assert reduce_max_stat_across_model_parallel_group(2.5, group=world) == 2.5
    assert reduce_max_stat_across_model_parallel_group(None, group=world) is None


def test_multi_element_tensor_is_rejected_before_the_collective(world):
    with pytest.raises(ValueError, match="single-element"):
        reduce_max_stat_across_model_parallel_group(torch.zeros(2, device="cuda"), group=world)
