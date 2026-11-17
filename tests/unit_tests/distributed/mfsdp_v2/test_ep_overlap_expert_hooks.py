# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checks for expert unshard hooks in the EP-overlap schedule."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import register_combined_1f1b_hooks
from megatron.core.transformer.moe.experts import TEGroupedMLP


class _Unit(nn.Module, FsdpModule):
    def __init__(self, name, events) -> None:
        nn.Module.__init__(self)
        self.label = name
        self.events = events
        self.ready = False
        self._parameter_groups = ()

    def is_root(self):
        return False

    def unshard(self):
        self.ready = True
        self.events.append(self.label)


class _Expert(TEGroupedMLP):
    def __init__(self, events) -> None:
        nn.Module.__init__(self)
        self.events = events
        self.linear_fc1 = nn.Linear(2, 4, bias=False)
        self.linear_fc2 = nn.Linear(4, 2, bias=False)


class _ShardedExpert(_Expert, _Unit):
    def __init__(self, events) -> None:
        _Unit.__init__(self, "expert", events)
        self.linear_fc1 = nn.Linear(2, 4, bias=False)
        self.linear_fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, inputs):
        assert self.ready
        self.events.append("forward")
        # Fusion reads the parameters without calling the original FC modules.
        output = F.linear(F.linear(inputs, self.linear_fc1.weight), self.linear_fc2.weight)
        return _CheckBackward.apply(output, self)


class _CheckBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, expert):
        ctx.expert = expert
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.expert.ready
        ctx.expert.events.append("backward")
        return grad_output, None


@pytest.fixture
def skip_completion_hooks(monkeypatch):
    # There are no real FSDP buffers to release in these CPU registration checks.
    monkeypatch.setattr(FsdpModule, "register_post_backward_hook", lambda *_: None)


def test_expert_boundary_unshards_before_fused_forward_and_backward(skip_completion_hooks):
    events = []
    root = _Unit("root", events)
    root.expert = _ShardedExpert(events)
    register_combined_1f1b_hooks(root)

    for linear in (root.expert.linear_fc1, root.expert.linear_fc2):
        assert not linear._forward_pre_hooks
        assert not linear._backward_pre_hooks

    inputs = torch.ones(1, 2, requires_grad=True)
    output = root.expert(inputs)
    assert events == ["expert", "forward"]

    # The schedule may release weights between this forward and its backward.
    root.expert.ready = False
    events.clear()
    output.sum().backward()
    assert events == ["expert", "backward"]
    assert inputs.grad is not None


def test_outer_owned_expert_keeps_descendant_hooks(skip_completion_hooks):
    events = []
    root = _Unit("root", events)
    root.expert = _Expert(events)
    register_combined_1f1b_hooks(root)

    for linear, width in ((root.expert.linear_fc1, 2), (root.expert.linear_fc2, 4)):
        assert linear._backward_pre_hooks
        events.clear()
        linear(torch.ones(1, width))
        assert events == ["root"]


def test_nested_unit_keeps_its_own_descendant_hooks(skip_completion_hooks):
    events = []
    root = _Unit("root", events)
    root.expert = _ShardedExpert(events)
    root.expert.nested = _Unit("nested", events)
    root.expert.nested.linear = nn.Linear(2, 2, bias=False)
    register_combined_1f1b_hooks(root)

    assert root.expert.linear_fc1._backward_pre_hooks
    root.expert.linear_fc1(torch.ones(1, 2))
    assert events == ["expert"]

    events.clear()
    assert root.expert.nested.linear._backward_pre_hooks
    root.expert.nested.linear(torch.ones(1, 2))
    assert events == ["nested"]
