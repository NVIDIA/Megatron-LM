# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Local FusedAdam group metadata across distributed-optimizer checkpoint loads."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import distrib_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig

pytestmark = pytest.mark.skipif(
    not (distrib_optimizer.USING_TE_OPTIMIZER or distrib_optimizer.USING_APEX_OPTIMIZER),
    reason="Requires TE or Apex FusedAdam's per-group step metadata",
)


def _optimizer(empty_group: bool = False, reverse: bool = False):
    groups = [
        {
            "params": (
                []
                if empty_group and index == 1
                else [torch.nn.Parameter(torch.ones(4, device="cuda"))]
            ),
            "wd_mult": float(index),
            "lr_mult": 1.0,
        }
        for index in range(2)
    ]
    return distrib_optimizer.Adam(groups[::-1] if reverse else groups, lr=0.01)


def _wrapper(optimizer):
    """Provide local shard sizes without constructing a distributed model."""
    wrapper = distrib_optimizer.DistributedOptimizer.__new__(distrib_optimizer.DistributedOptimizer)
    wrapper.optimizer = optimizer
    wrapper.config = OptimizerConfig(optimizer="adam", lr=0.01)
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.grad_scaler = None
    wrapper.model_param_group_index_map = {
        param: (group_index, param_index)
        for group_index, group in enumerate(optimizer.param_groups)
        for param_index, param in enumerate(group["params"])
    }
    wrapper.gbuf_ranges = [
        {
            torch.float32: [
                {
                    "param_map": {
                        param: {"gbuf_world": distrib_optimizer.Range(0, param.numel())}
                        for param in wrapper.model_param_group_index_map
                    }
                }
            ]
        }
    ]
    return wrapper


def _step(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("empty_step", [None, 0, 7])
@pytest.mark.parametrize("preallocated", [False, True])
def test_load_preserves_empty_fused_adam_group_step(empty_step, preallocated):
    """Global checkpoint steps apply only to groups owning local parameters."""
    source = _optimizer()
    for _ in range(5):
        _step(source)
    checkpoint = _wrapper(source).state_dict()
    assert all(group["step"] == 5 for group in checkpoint["optimizer"]["param_groups"])
    checkpoint_before = deepcopy(checkpoint)

    # A different local ownership/order must still match groups by their identifiers.
    target = _optimizer(empty_group=True, reverse=True)
    if preallocated:
        _step(target)
    empty_group, populated_group = target.param_groups
    if empty_step is not None:
        empty_group["step"] = empty_step
    else:
        assert "step" not in empty_group
    local_param = populated_group["params"][0]

    _wrapper(target).load_state_dict(checkpoint)

    empty_group, populated_group = target.param_groups
    assert empty_group["params"] == []
    assert ("step" in empty_group) == (empty_step is not None)
    if empty_step is not None:
        assert empty_group["step"] == empty_step
    assert populated_group["params"][0] is local_param
    assert populated_group["step"] == 5
    assert [group["wd_mult"] for group in target.param_groups] == [1.0, 0.0]
    assert checkpoint == checkpoint_before
