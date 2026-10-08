# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Countdown-based gradient completion across combined 1F1B GraphTasks."""

import copy

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import _module_post_backward_hook


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


def _sharded_unit(setup, grad_accumulation_count=None, model=None):
    """Install the scheduler's reduction callback behind a completion spy."""
    torch.manual_seed(0)
    if model is None:
        model = TwoNodeUnit().to(setup.device)
    mesh = init_device_mesh(setup.device.type, (setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=setup.device):
        fully_shard(model, mesh=mesh, placements=placements, register_hooks=False)
    finalized = []

    def post_backward(hooked_module):
        _module_post_backward_hook(hooked_module)
        finalized.append(hooked_module)

    model.register_post_backward_hook(
        post_backward, grad_accumulation_count=grad_accumulation_count
    )
    return model, finalized


def _graph_task(model: FsdpModule, inputs, submodule_name=None) -> torch.Tensor:
    """Run one independently gathered forward/backward schedule node."""
    model.context.allgather_stream.wait_stream(model.context.current_stream())
    model.unshard()
    consumer = model if submodule_name is None else getattr(model, submodule_name)
    loss = consumer(inputs).square().mean()
    loss.backward()
    model.reshard()
    return loss.detach()


class TestPostBackwardHookAcrossGraphTasks:
    """Real reductions wait for every declared wgrad callback, not each dgrad."""

    @pytest.mark.parametrize("num_graph_tasks", [1, 2, 3])
    @pytest.mark.parametrize("set_to_none", [False, True])
    def test_gradients_and_updates_match_baseline(
        self, distributed_setup, num_graph_tasks, set_to_none
    ):
        """Shared contributions match SGD across steps and microbatch windows."""
        setup = distributed_setup
        torch.manual_seed(0)
        baseline = TwoNodeUnit().to(setup.device)
        model, finalized = _sharded_unit(
            setup, grad_accumulation_count=2 * num_graph_tasks, model=copy.deepcopy(baseline)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        fully_shard_optimizer(optimizer)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)

        torch.manual_seed(123 + setup.rank)
        for step in range(3):
            optimizer.zero_grad(set_to_none=set_to_none)
            baseline_optimizer.zero_grad(set_to_none=set_to_none)
            for microbatch_index in range(2):
                previous_finalizations = len(finalized)
                for graph_index in range(num_graph_tasks):
                    inputs = torch.randn(4, 16, device=setup.device)
                    baseline_loss = baseline(inputs).square().mean()
                    baseline_loss.backward()
                    loss = _graph_task(model, inputs)
                    torch.testing.assert_close(loss, baseline_loss.detach())
                    if graph_index < num_graph_tasks - 1:
                        assert len(finalized) == previous_finalizations
                        with pytest.raises(ValueError, match="mid-flight"):
                            model._trainable_parameter_countdown.check_complete()
                assert len(finalized) == step * 2 + microbatch_index + 1
                model._trainable_parameter_countdown.check_complete()

            model.context.post_backward()
            for group in model.parameter_groups:
                replicated_grads = group.main_grad.redistribute((Replicate(),) * group.mesh.ndim)
                for parameter_index, parameter in enumerate(group.fsdp_parameters):
                    reference = baseline.get_parameter(parameter.fqns[0])
                    dist.all_reduce(reference.grad)
                    reference.grad.div_(setup.world_size)
                    torch.testing.assert_close(
                        replicated_grads.get_tensor_view(parameter_index), reference.grad
                    )
            optimizer.step()
            baseline_optimizer.step()
            for group in model.parameter_groups:
                replicated_weights = group.main_weight.redistribute(
                    (Replicate(),) * group.mesh.ndim
                )
                for parameter_index, parameter in enumerate(group.fsdp_parameters):
                    torch.testing.assert_close(
                        replicated_weights.get_tensor_view(parameter_index),
                        baseline.get_parameter(parameter.fqns[0]),
                    )

    def test_different_parameter_multiplicities(self, distributed_setup):
        """The unit total includes two callbacks for first and one for second."""
        setup = distributed_setup
        model, finalized = _sharded_unit(setup, grad_accumulation_count=3)
        for node_index, name in enumerate(["first", "second", "first"]):
            inputs = torch.randn(4, 16, device=setup.device)
            _graph_task(model, inputs, submodule_name=name)
            assert len(finalized) == (1 if node_index == 2 else 0)
        model.context.post_backward()
        model._trainable_parameter_countdown.check_complete()

    def test_default_count_is_one_per_trainable_parameter(self, distributed_setup):
        """The ordinary path keeps its original countdown without a declaration."""
        model, finalized = _sharded_unit(distributed_setup)
        assert model._trainable_parameter_countdown.initial_value == 2
        _graph_task(model, torch.randn(4, 16, device=distributed_setup.device))
        assert len(finalized) == 1
        model.context.post_backward()

    def test_delayed_wgrad_completes_the_countdown(self, distributed_setup):
        """Dgrad and an ordinary gradient cannot substitute for TE's wgrad."""
        device = distributed_setup.device
        model = nn.Sequential(
            te.Linear(
                16,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                device=device,
                delay_wgrad_compute=True,
                fuse_wgrad_accumulation=False,
            ),
            nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16),
        )
        model, finalized = _sharded_unit(distributed_setup, model=model)
        model.context.allgather_stream.wait_stream(model.context.current_stream())
        model.unshard()
        inputs = torch.randn(4, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        model(inputs).float().square().mean().backward()
        assert inputs.grad is not None
        assert finalized == []
        with pytest.raises(ValueError, match="1/2"):
            model._trainable_parameter_countdown.check_complete()
        model[0].backward_dw()
        assert len(finalized) == 1
        model.context.post_backward()
        model._trainable_parameter_countdown.check_complete()

    @pytest.mark.parametrize("count", [0, -1])
    def test_invalid_trainable_count_fails(self, distributed_setup, count):
        """Trainable modules cannot use the no-parameter fallback or negative totals."""
        with pytest.raises(ValueError):
            _sharded_unit(distributed_setup, grad_accumulation_count=count)

    @pytest.mark.parametrize("count", [None, 0])
    def test_parameter_free_module_uses_full_backward_hook(self, distributed_setup, count):
        """Zero callbacks preserve the module-level completion fallback."""
        model, finalized = _sharded_unit(
            distributed_setup, grad_accumulation_count=count, model=nn.ReLU()
        )
        inputs = torch.randn(4, 16, device=distributed_setup.device, requires_grad=True)
        _graph_task(model, inputs)
        assert inputs.grad is not None
        assert len(finalized) == 1
        model.context.post_backward()
        model._trainable_parameter_countdown.check_complete()

    def test_parameter_free_module_rejects_nonzero_count(self, distributed_setup):
        """A parameter-free unit cannot wait for callbacks that never arrive."""
        with pytest.raises(ValueError, match="owns no trainable parameters"):
            _sharded_unit(distributed_setup, grad_accumulation_count=1, model=nn.ReLU())
