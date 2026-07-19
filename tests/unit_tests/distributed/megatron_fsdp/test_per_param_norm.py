# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP


class _GradBuffer:
    def __init__(self, grad):
        self.data = grad
        self._grad = grad

    def get_item(self, item_id, only_shard):
        assert item_id == 0
        assert only_shard
        return self._grad


def _fsdp_shell(strategy):
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    module = torch.nn.Module()
    module.register_parameter("weight", param)
    grad_buffer = _GradBuffer(torch.tensor([5.0, 12.0]))
    parameter_group = SimpleNamespace(
        requires_grad=True,
        hfsdp_helper_gbuf=None,
        main_grad_buffer=grad_buffer,
        param_idx={param: 0},
    )

    fsdp = object.__new__(MegatronFSDP)
    torch.nn.Module.__init__(fsdp)
    fsdp.module = module
    fsdp.data_parallel_sharding_strategy = strategy
    fsdp.param_and_grad_buffer = SimpleNamespace(
        param_to_param_group={param: 0}, parameter_groups=[parameter_group]
    )
    fsdp.dist_index = SimpleNamespace(get_dp_group=Mock(return_value="dp-group"))
    return fsdp, param


@pytest.mark.parametrize(
    ("strategy", "expected_collectives"),
    [("no_shard", 0), ("optim", 1), ("optim_grads", 1), ("optim_grads_params", 2)],
)
def test_compute_per_param_norms_reduces_only_sharded_values(strategy, expected_collectives):
    fsdp, _ = _fsdp_shell(strategy)

    with patch("torch.distributed.all_reduce") as all_reduce:
        norms = MegatronFSDP._compute_per_param_norms(fsdp)

    assert norms == {"weight": {"param_norm": 5.0, "grad_norm": 13.0}}
    assert all_reduce.call_count == expected_collectives
    fsdp.dist_index.get_dp_group.assert_called_once_with(is_expert_parallel=False)


def test_compute_per_param_norms_uses_expert_data_parallel_group():
    fsdp, param = _fsdp_shell("optim_grads_params")
    param.allreduce = False

    with patch("torch.distributed.all_reduce"):
        MegatronFSDP._compute_per_param_norms(fsdp)

    fsdp.dist_index.get_dp_group.assert_called_once_with(is_expert_parallel=True)
