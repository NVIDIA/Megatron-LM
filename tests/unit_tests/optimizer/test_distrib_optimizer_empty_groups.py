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
    # Populate the fields used by the production save/load methods: config,
    # ddp_config, grad_scaler, model_param_group_index_map and gbuf_ranges.
    # optimizer_state_keys remains the real DistributedOptimizer property.
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


def test_empty_group_flag_matches_installed_fused_adam():
    """FUSED_ADAM_SKIPS_EMPTY_GROUPS must describe what the installed FusedAdam does."""
    optimizer = _optimizer(empty_group=True)
    _step(optimizer)
    empty_group = optimizer.param_groups[1]
    assert empty_group["params"] == []
    assert ("step" not in empty_group) == distrib_optimizer.FUSED_ADAM_SKIPS_EMPTY_GROUPS


@pytest.mark.parametrize("preallocated", [False, True])
def test_restored_empty_group_step_matches_uninterrupted_run(preallocated):
    """After a restore, every group carries the step metadata of an uninterrupted run."""
    source = _optimizer()
    uninterrupted = _optimizer(empty_group=True, reverse=True)
    for _ in range(5):
        _step(source)
        _step(uninterrupted)
    checkpoint = _wrapper(source).state_dict()

    restored = _optimizer(empty_group=True, reverse=True)
    if preallocated:
        _step(restored)
    _wrapper(restored).load_state_dict(checkpoint)

    for restored_group, uninterrupted_group in zip(
        restored.param_groups, uninterrupted.param_groups
    ):
        assert ("step" in restored_group) == ("step" in uninterrupted_group)
        assert restored_group.get("step") == uninterrupted_group.get("step")


@pytest.mark.parametrize("empty_step", [None, 0, 7])
@pytest.mark.parametrize("preallocated", [False, True])
def test_load_preserves_empty_fused_adam_group_step(empty_step, preallocated):
    """Global checkpoint steps reach empty groups only if FusedAdam advances them."""
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
        empty_group.pop("step", None)
    local_param = populated_group["params"][0]

    _wrapper(target).load_state_dict(checkpoint)

    empty_group, populated_group = target.param_groups
    assert empty_group["params"] == []
    expected_empty_step = empty_step if distrib_optimizer.FUSED_ADAM_SKIPS_EMPTY_GROUPS else 5
    assert ("step" in empty_group) == (expected_empty_step is not None)
    assert empty_group.get("step") == expected_empty_step
    assert populated_group["params"][0] is local_param
    assert populated_group["step"] == 5
    assert [group["wd_mult"] for group in target.param_groups] == [1.0, 0.0]
    assert checkpoint == checkpoint_before
