# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for recording MFSDP demand-unshard order."""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    iteration as iteration_scope,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import microbatch


def test_shared_module_recording_matches_dense_training(distributed_setup):
    """Record and replay MTP-like module reuse, preserving gradients and updates."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(8, 8, bias=False)
            self.second = nn.Linear(8, 8, bias=False)

        def forward(self, inputs):
            hidden = self.first(inputs)
            # MTP and main logits call the same output layer consecutively
            # on different hidden states; both uses contribute gradients.
            auxiliary = self.second(hidden + 0.1)
            main = self.second(hidden)
            return (auxiliary + main).relu()

    torch.manual_seed(42)
    dense, model = Model().to(device), Model()
    model.load_state_dict(dense.state_dict())
    placements = Placements([0], [Shard(0)], [Shard(0)], [Shard(0)])
    with fully_shard_context(device=device) as context:
        fully_shard(model.first, mesh=mesh, placements=placements)
        fully_shard(model.second, mesh=mesh, placements=placements)
        fully_shard(model, mesh=mesh, placements=placements)
    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    num_microbatches = 3
    base_inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16
    for iteration in range(2):
        dense_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        with iteration_scope(context):
            for index in range(num_microbatches):
                inputs = base_inputs + 0.1 * distributed_setup.rank + 0.02 * (iteration + index)
                expected = dense(inputs)
                (expected.square().mean() / num_microbatches).backward()
                with microbatch(context, is_last=index == num_microbatches - 1):
                    actual = model(inputs)
                    (actual.square().mean() / num_microbatches).backward()
                torch.testing.assert_close(actual, expected)
            context.finish_grad_sync()
        demands = context.iteration_order._orders["default"]
        assert [module_ref() for module_ref, phase in demands if phase == "forward"] == [
            model,
            model.first,
            model.second,
            model.second,
        ] * num_microbatches
        assert [module_ref() for module_ref, phase in demands if phase == "backward"] == [
            model,
            model.second,
            model.second,
            model.first,
        ] * num_microbatches
        for reference in dense.parameters():
            dist.all_reduce(reference.grad, group=mesh.get_group())
            reference.grad.div_(distributed_setup.world_size)
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.grad.full_tensor(), reference.grad)
        dense_optimizer.step()
        optimizer.step()
        for reference, sharded in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(sharded.full_tensor(), reference)
        assert all(module._unshard_event is None for module in (model, model.first, model.second))
