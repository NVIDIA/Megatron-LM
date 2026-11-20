# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-parameter gradient staging lifetimes and exact update parity."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    SchedulePolicy,
    finalize_gradients,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


class SharedModel(nn.Module):
    """One FSDP unit whose gradients become ready at different backward positions."""

    def __init__(self, shared: bool, side_stream: bool) -> None:
        super().__init__()
        self.input = nn.Linear(8, 16)
        self.middle = nn.Linear(16, 16)
        self.output = nn.Linear(16, 4)
        self.shared = shared
        self.side_stream = side_stream
        self._side_stream = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run shared or independent layers on one or two CUDA streams."""
        hidden = self.input(inputs).relu()
        if self.side_stream:
            if self._side_stream is None:
                self._side_stream = torch.cuda.Stream(device=inputs.device)
            current = torch.cuda.current_stream()
            self._side_stream.wait_stream(current)
            with torch.cuda.stream(self._side_stream):
                hidden = self.middle(hidden).relu()
                if self.shared:
                    hidden = self.middle(hidden).relu()
            current.wait_stream(self._side_stream)
        else:
            hidden = self.middle(hidden).relu()
            if self.shared:
                hidden = self.middle(hidden).relu()
        return self.output(hidden)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("grad_comm_dtype", [torch.bfloat16, torch.float32])
def test_staging_preserves_bf16_gradients_and_updates(
    distributed_setup, monkeypatch, shared, side_stream, set_to_none, grad_comm_dtype
):
    """Early staging matches the former module-end path bitwise over four microbatches."""
    setup = distributed_setup
    if setup.world_size < 2:
        pytest.skip("Requires inner gradient communication.")
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    torch.manual_seed(1234)
    staged = SharedModel(shared, side_stream).to(device=setup.device, dtype=torch.bfloat16)
    original = copy.deepcopy(staged)
    policy = MixedPrecisionPolicy(
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.bfloat16,
        grad_comm_dtype=grad_comm_dtype,
    )
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    for model in (original, staged):
        with fully_shard_context(FsdpContext(device=setup.device)):
            fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)
    # Exercise the original reduction path as the reference, on the same code/math/layout.
    original._early_grad_staging = False
    expected_parameters = len(list(staged._trainable_fsdp_parameters()))
    copies, reductions = [], []
    copy_streams = set()
    stage_gradient = FsdpModule._stage_gradient
    reduce_gradients = FsdpParameterGroup.reduce_inner_gradients

    def observe_stage(module, parameter):
        if module is staged:
            assert parameter.grad is not None
            assert parameter.grad.dtype == torch.bfloat16
        stage_gradient(module, parameter)
        if module is staged:
            assert parameter.grad is None
            assert len(module._staged_partial_grads) == 1
            buffer = next(iter(module._staged_partial_grads.values()))
            assert buffer.dtype == grad_comm_dtype
            copy_streams.update(
                module._staged_copy_events[next(iter(module._staged_partial_grads))]
            )
            copies.append(len(module._staged_parameters))

    def observe_reduce(group, partial_grad, **kwargs):
        if any(group is candidate for candidate in staged.parameter_groups):
            assert len(staged._staged_parameters) == expected_parameters
            assert all(p.unsharded.grad is None for p in group.fsdp_parameters)
            reductions.append(len(copies))
        return reduce_gradients(group, partial_grad, **kwargs)

    monkeypatch.setattr(FsdpModule, "_stage_gradient", observe_stage)
    monkeypatch.setattr(FsdpParameterGroup, "reduce_inner_gradients", observe_reduce)
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9) for model in (original, staged)
    ]
    for optimizer in optimizers:
        fully_shard_optimizer(optimizer)

    torch.manual_seed(5678 + setup.rank)
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
        for index in range(4):
            inputs = torch.randn(4, 8, device=setup.device, dtype=torch.bfloat16)
            target = torch.randn(4, 4, device=setup.device, dtype=torch.bfloat16)
            losses = []
            for model in (original, staged):
                with finalize_gradients(model.context, enabled=index == 3):
                    loss = (model(inputs).float() - target.float()).square().mean()
                    loss.backward()
                model.context.finish_grad_sync()
                losses.append(loss.detach())
            torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
            for reference, actual in zip(original.parameters(), staged.parameters()):
                assert reference.grad is not None and actual.grad is not None
                assert actual.grad.dtype == torch.bfloat16
                torch.testing.assert_close(reference.grad, actual.grad, rtol=0, atol=0)
            assert not staged._staged_partial_grads
            assert not staged._staged_parameters
            assert not staged._staged_buffer_ready
            assert not staged._staged_copy_events
        for optimizer in optimizers:
            optimizer.step()
        for reference, actual in zip(original.parameters(), staged.parameters()):
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)

    assert copies == list(range(1, expected_parameters + 1)) * 8
    assert reductions == [expected_parameters * n for n in range(1, 9)]
    assert len(copy_streams) == (2 if side_stream else 1)


def test_partial_buffer_can_be_allocated_before_all_gradients(distributed_setup):
    """The staging allocation uses known group metadata, not other parameters' readiness."""
    setup = distributed_setup
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(8, 8, device=setup.device, dtype=torch.bfloat16)
    policy = MixedPrecisionPolicy(grad_comm_dtype=torch.float32)
    with fully_shard_context(FsdpContext(device=setup.device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements([0], [Shard(0)], [Shard(0)], [Shard(0)]),
            mixed_precision_policy=policy,
        )
    group = model.parameter_groups[0]
    assert all(parameter.unsharded.grad is None for parameter in group.fsdp_parameters)
    with torch.cuda.stream(model.context.inner_reduction_stream):
        partial = group.allocate_partial_grad_buffer()
    assert partial.dtype == torch.float32
    with pytest.raises(RuntimeError, match="Missing gradient"):
        group.copy_gradient_to_partial_buffer(0, partial)


def test_explicit_callback_count_retains_module_end_packing(distributed_setup, monkeypatch):
    """An explicit callback window retains the original local-gradient lifetime."""
    setup = distributed_setup
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    model = nn.Linear(8, 8, device=setup.device, dtype=torch.bfloat16)
    with fully_shard_context(FsdpContext(device=setup.device)):
        fully_shard(
            model,
            mesh=mesh,
            placements=Placements([0], [Shard(0)], [Shard(0)], [Shard(0)]),
            mixed_precision_policy=MixedPrecisionPolicy(grad_comm_dtype=torch.float32),
            schedule_policy=SchedulePolicy(grad_accumulation_count=2),
        )
    assert not model._early_grad_staging

    def forbid_early_staging(*args):
        raise AssertionError("Explicit callback counts must retain module-end packing.")

    monkeypatch.setattr(FsdpModule, "_stage_gradient", forbid_early_staging)
    model(torch.randn(4, 8, device=setup.device, dtype=torch.bfloat16)).float().sum().backward()
    model.context.finish_grad_sync()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert not model._staged_partial_grads and not model._staged_parameters
