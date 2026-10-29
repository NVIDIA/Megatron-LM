# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import copy
import gc
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


class ReverseRegisteredModel(nn.Module):
    """Registers ``second`` before ``first`` but runs ``first`` before ``second``."""

    def __init__(self) -> None:
        super().__init__()
        self.second = nn.Linear(8, 8, bias=False)
        self.first = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``first`` and then ``second``."""
        return self.second(torch.relu(self.first(x)))


@pytest.mark.parametrize(
    'budget, expected',
    [
        (None, [[1], [0], [0], [2], []]),
        (0, [[], [], [], [], []]),
        (2, [[1, 0], [0, 0], [0, 2], [2], []]),
    ],
)
def test_prefetch_replays_occurrences(budget, expected):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append(module)
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.next_item() is successor


def test_unique_calls_resynchronize_and_repeated_calls_validate():
    """Unique VPP chunk calls select their position; ambiguous repeats stay strict."""
    chunk0, layer0, chunk1, layer1 = (nn.Module() for _ in range(4))
    order = ExecutionOrder([chunk0, layer0, chunk1, layer1])
    # Grouped VPP repeats chunk 0 before switching to chunk 1, in one shared order.
    for chunk in (chunk0, chunk0, chunk0, chunk1, chunk1, chunk0):
        layer = layer0 if chunk is chunk0 else layer1
        order.advance(chunk)
        assert order.next_item() is layer
        order.advance(layer)
        assert order.next_item() is (chunk1 if chunk is chunk0 else None)

    order = ExecutionOrder([chunk0, layer0, chunk0])
    order.advance(chunk0)
    assert order.next_item() is layer0
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(chunk0)
    # A rejected repeated call must not consume the expected occurrence.
    order.advance(layer0)
    assert order.next_item() is chunk0
    order.advance(chunk0)
    assert order.next_item() is None
    # Unique calls remain valid anchors even when the sequence also has repeats.
    order.advance(layer0)
    order.advance(chunk0)
    assert order.next_item() is None

    empty = ExecutionOrder()
    assert empty.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(chunk0)


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first, second])
    assert list(order) == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [first, first]
    order.advance(first)
    assert order.next_item() is None
    # Traversal skips dead references, but the demand cursor must not skip a call.
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)


@pytest.mark.parametrize(
    "schedule_policy",
    [SchedulePolicy(), SchedulePolicy(forward_prefetch_size=1024, backward_prefetch_size=1024)],
    ids=["next_module", "element_budget"],
)
def test_backward_prefetch_skips_finished_modules(distributed_setup, schedule_policy):
    """A mispredicted backward prefetch must not carry pre-step weights into the next forward.

    Registration order predicts backward ``first`` -> ``second``, but ``second`` finishes
    first. Prefetching ``second`` from ``first`` gathers weights that no post_backward()
    releases, so they survive optimizer.step().
    """
    rank = distributed_setup.rank
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    torch.manual_seed(1234)
    baseline = ReverseRegisteredModel().to(device)
    model = copy.deepcopy(baseline)
    with fully_shard_context(device=device):
        for module in (model.second, model.first, model):
            fully_shard(module, mesh=mesh, placements=placements, schedule_policy=schedule_policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fully_shard_optimizer(optimizer)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)

    torch.manual_seed(5678 + rank)
    for _ in range(3):
        x = torch.randn(4, 8, device=device)
        baseline_optimizer.zero_grad()
        baseline_loss = baseline(x).square().mean()
        baseline_loss.backward()
        for parameter in baseline.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
        baseline_optimizer.step()

        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert isinstance(parameter, DTensor), f"{name} is still unsharded after backward."
        optimizer.step()
        torch.testing.assert_close(loss, baseline_loss)
