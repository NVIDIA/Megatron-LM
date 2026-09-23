# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient completion for a shared parameter under multiple autograd GraphTasks (Combined 1F1B)."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


class TwoNodeUnit(nn.Module):
    """Two parameters behind two graph nodes, one FSDP unit."""

    def __init__(self) -> None:
        """Build the two projections."""
        super().__init__()
        self.first = nn.Linear(16, 16, bias=False)
        self.second = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project twice so each parameter owns a distinct graph node."""
        return self.second(self.first(inputs))


def _sharded_unit(setup, multiplicity: int):
    """Fully shard a ``TwoNodeUnit`` and install a completion hook behind a spy."""
    torch.manual_seed(0)  # identical across ranks: the test is rank-symmetric
    model = TwoNodeUnit().to(setup.device)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=setup.device):
        # ``register_hooks=False`` mirrors production, where the combined
        # scheduler installs its own completion hooks instead of the default.
        fully_shard(model, mesh=mesh, placements=placements, register_hooks=False)
    model.param_grad_readiness.expected.update(
        {parameter.fqns: multiplicity for parameter in model._trainable_fsdp_parameters()}
    )
    finalized = []
    model.register_post_backward_hook(lambda hooked_module: finalized.append(hooked_module))
    return model, finalized


def _graph_task(model: FsdpModule, setup) -> None:
    """Run one forward-backward: one independent autograd GraphTask."""
    if model.is_root():
        model.context.allgather_stream.wait_stream(model.context.current_stream())
    inputs = torch.randn(4, 16, device=setup.device)
    model.unshard()  # forward over the resting sharded state deadlocks in a weight all-gather
    model(inputs).sum().backward()
    model.reshard()


class TestPostBackwardHookAcrossGraphTasks:
    """``register_post_backward_hook`` with one GraphTask per schedule node."""

    def test_the_window_finalizes_once_across_graph_tasks(self, distributed_setup):
        """Two GraphTasks owe two contributions; the hook fires after the second.

        This is the coexistence under test: a finalize per GraphTask would
        reshard and reduce after the first schedule node's backward and drop the
        second node's contribution. The declared multiplicity holds the window
        open until every GraphTask has landed.
        """
        model, finalized = _sharded_unit(distributed_setup, multiplicity=2)

        _graph_task(model, distributed_setup)  # first schedule node
        assert finalized == [], "the window closed after the first GraphTask"
        with pytest.raises(ValueError, match="1/2"):
            model.param_grad_readiness.seal()  # both keys are in flight at half
        _graph_task(model, distributed_setup)  # second schedule node
        assert len(finalized) == 1, f"expected one finalize, got {len(finalized)}"
        model.param_grad_readiness.seal()  # closed window: nothing in flight

    def test_an_undeclared_second_graph_task_fails_loudly(self, distributed_setup):
        """An undeclared second contribution over-fires the declaration of one.

        The declared window finalizes after the first GraphTask and re-arms, so
        the surplus is staged against the re-armed window: a key's declared fire
        followed by the undeclared second GraphTask's fire of the same key. Both
        are delivered at the readiness API rather than from inside a distributed
        backward -- a raise in the autograd engine would leave peer ranks stuck
        in collectives. The surplus must raise over-fired without triggering
        another finalize.
        """
        model, finalized = _sharded_unit(distributed_setup, multiplicity=1)

        _graph_task(model, distributed_setup)
        assert len(finalized) == 1

        key = next(iter(model.param_grad_readiness.expected))
        readiness = model.param_grad_readiness
        readiness.mark(key)  # the re-armed window's declared contribution
        with pytest.raises(ValueError, match="over-fired"):
            readiness.mark(key)  # the undeclared second GraphTask's surplus
        assert len(finalized) == 1, "the surplus must not trigger another finalize"
