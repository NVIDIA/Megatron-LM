# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Invocation-specific prefetch must preserve own-weight gathering and backward."""

import copy
from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    SchedulePolicy,
    fully_shard,
    fully_shard_context,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.residual_connection import ResidualConnection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _Residual(ResidualConnection):
    """Independent read and write weights called around one branch."""

    def __init__(self):
        super().__init__(residual_stream_hidden_size=16, branch_hidden_size=16)
        self.read_weight = nn.Parameter(torch.randn(16))
        self.write_weight = nn.Parameter(torch.randn(16))

    def _read(self, hidden_states):
        return hidden_states * self.read_weight, ()

    def _write(self, branch_output, state, *, dropout_probability, training):
        return state[0] + branch_output * self.write_weight


class _ResidualBlock(nn.Module):
    """The residual's registered successor executes between its two invocations."""

    def __init__(self):
        super().__init__()
        self.residual = _Residual()
        self.branch = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        branch_input, state = self.residual(x, operation="read")
        return self.residual(
            self.branch(branch_input),
            operation="write",
            state=state,
            dropout_probability=0.0,
            training=self.training,
        )


@pytest.mark.parametrize("guarded", [False, True])
def test_adapter_residual_write_does_not_retain_branch(distributed_setup, guarded):
    """Read prefetches the branch; write must not gather it after its final forward use."""
    Utils.initialize_model_parallel(1, 1)
    try:
        torch.manual_seed(123)
        block = _ResidualBlock().to(distributed_setup.device)
        reference = copy.deepcopy(block)
        wrapped = FullyShardedDataParallel(
            config=TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4),
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
            ),
            module=block,
            fsdp_unit_modules=[_Residual, nn.Linear],
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert block.residual._schedule_policy.forward_prefetch_predicate is not None
        if not guarded:
            block.residual._schedule_policy = replace(
                block.residual._schedule_policy, forward_prefetch_predicate=None
            )
        branch_was_prefetched = []
        block.branch.register_forward_pre_hook(
            lambda module, args: branch_was_prefetched.append(module._unshard_event is not None),
            prepend=True,
        )
        x = torch.randn(4, 16, device=distributed_setup.device)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(x), reference(x), rtol=0, atol=0)
        torch.cuda.synchronize()
        assert branch_was_prefetched == [True]
        live_bytes = sum(
            group._unsharded_model_weight.local_buffer.untyped_storage().nbytes()
            for group in block.branch.parameter_groups
        )
        assert (live_bytes == 0) is guarded
        block.branch.reshard()
    finally:
        Utils.destroy_model_parallel()


class _KeywordLinear(nn.Linear):
    """Linear operation with a keyword used only by the scheduling predicate."""

    def forward(self, x, *, operation):
        return super().forward(x)


@pytest.mark.parametrize("reentrant", [None, False, True])
def test_predicate_preserves_own_gather_and_checkpoint_backward(distributed_setup, reentrant):
    """Suppress only successor gathering, including when backward replays the module."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(123)
    layer = _KeywordLinear(16, 16, bias=False).to(device)
    successor = nn.Linear(16, 16, bias=False).to(device)
    reference = copy.deepcopy(layer)
    reference_successor = copy.deepcopy(successor)
    predicate_calls = []

    def predicate(kwargs):
        predicate_calls.append(kwargs)
        return kwargs.get("operation") != "write"

    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device):
        fully_shard(
            layer,
            mesh=mesh,
            placements=placements,
            schedule_policy=SchedulePolicy(forward_prefetch_predicate=predicate),
        )
        fully_shard(successor, mesh=mesh, placements=placements)
    torch.manual_seed(900 + distributed_setup.rank)
    x = torch.randn(4, 16, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    operation = partial(layer, operation="write")
    intermediate = (
        operation(x) if reentrant is None else checkpoint(operation, x, use_reentrant=reentrant)
    )
    assert successor._unshard_event is None
    output = successor(intermediate)
    expected = reference_successor(reference(reference_x, operation="write"))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for module, baseline in ((layer, reference), (successor, reference_successor)):
        dist.all_reduce(baseline.weight.grad, op=dist.ReduceOp.AVG)
        torch.testing.assert_close(module.weight.grad.full_tensor(), baseline.weight.grad)
    assert predicate_calls == [{"operation": "write"}]
