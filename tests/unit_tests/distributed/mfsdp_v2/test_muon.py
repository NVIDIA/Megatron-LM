# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed owner-compute Muon steps compared with unsharded Emerging Optimizers Muon."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.range import intersect_ranges

pytest.importorskip("emerging_optimizers")
from emerging_optimizers.orthogonalized_optimizers.muon import Muon

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.muon import FsdpMuon


def _placements():
    return Placements(
        dp_axes=[0], parameter=[RowAtomic()], gradient=[RowAtomic()], optimizer=[RowAtomic()]
    )


class TinyModel(nn.Module):
    """Two buffers, each with a split matrix and a matrix wholly on one rank."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 4, bias=False)
        )
        self.block2 = nn.Sequential(
            nn.Linear(4, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
            nn.Linear(1, 1),  # Output projection and bias belong to another optimizer.
        )

    def forward(self, x):
        """Run both blocks."""
        return self.block2(self.block1(x))


def _parameter_groups(params):
    return [
        {"params": [params["block1.0.weight"], params["block2.0.weight"]], "lr": 0.05},
        {
            "params": [params["block1.2.weight"], params["block2.2.weight"]],
            "lr": 0.02,
            "momentum": 0.8,
        },
    ]


def _reference_shard(param, full):
    group = get_containing_parameter_group(param)
    buffer = group.main_weight
    index = next(i for i, fp in enumerate(group.fsdp_parameters) if fp.sharded is param)
    tensor_range = buffer.layout.get_tensor_range(index)
    local_range = buffer.layout.get_local_range(buffer.mesh, buffer.placements)
    shard = intersect_ranges(tensor_range, local_range)
    if shard.numel == 0:
        return full.new_empty(0)
    return full.detach().view(-1).narrow(0, shard.start - tensor_range.start, shard.numel)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("nesterov,decay_method", [(False, "decoupled"), (True, "l2")])
def test_muon_matches_reference(distributed_setup, dtype, nesterov, decay_method):
    """Match weights and momentum over steps, including a checkpoint round trip."""
    world_size = distributed_setup.world_size
    if world_size < 2 or world_size % 2:
        pytest.skip("Requires an even number of ranks for two-rank DP meshes.")
    device = distributed_setup.device
    # With more than two GPUs, also exercise meshes whose global ranks do not start at zero.
    mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("replica", "dp"))["dp"]
    torch.manual_seed(1234)
    model = TinyModel().to(device=device, dtype=dtype)
    reference = copy.deepcopy(model)
    reference_params = {
        name: nn.Parameter(param.detach().float().clone())
        for name, param in reference.named_parameters()
    }
    with fully_shard_context(device=device):
        fully_shard(model.block1, mesh=mesh, placements=_placements())
        fully_shard(model.block2, mesh=mesh, placements=_placements())
    params = dict(model.named_parameters())
    kwargs = dict(
        momentum=0.9, weight_decay=0.1, nesterov=nesterov, weight_decay_method=decay_method
    )
    optimizer = FsdpMuon(_parameter_groups(params), dp_mesh=mesh, **kwargs)
    reference_optimizer = Muon(_parameter_groups(reference_params), **kwargs)
    excluded_names = ["block2.3.weight", "block2.3.bias"]
    other_optimizer = torch.optim.SGD([params[n] for n in excluded_names], lr=0.01, foreach=False)
    fully_shard_optimizer(other_optimizer)
    reference_other = torch.optim.SGD([reference_params[n] for n in excluded_names], lr=0.01)

    for block in (model.block1, model.block2):
        buffer = get_containing_parameter_group(block[0].weight).main_weight
        holder_counts = [
            sum(
                intersect_ranges(
                    buffer.layout.get_tensor_range(i),
                    buffer.layout.get_rank_range(mesh, buffer.placements, rank),
                ).numel
                > 0
                for rank in mesh.mesh.tolist()
            )
            for i in range(2)
        ]
        assert sorted(holder_counts) == [1, 2]

    for step in range(3):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = torch.randn(4, 8, device=device, dtype=dtype)
        actual = model(x)
        expected = reference(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        expected.sum().backward()
        if step == 1:
            # A previously active matrix can skip a step without losing its momentum.
            params["block1.2.weight"].grad = None
            reference.block1[2].weight.grad = None
        for name, param in reference.named_parameters():
            reference_params[name].grad = None if param.grad is None else param.grad.float().clone()

        excluded_before = [params[n].to_local().clone() for n in excluded_names]
        optimizer.step()
        for name, before in zip(excluded_names, excluded_before):
            torch.testing.assert_close(params[name].to_local(), before, atol=0, rtol=0)
            assert params[name] not in optimizer.state
        other_optimizer.step()
        reference_optimizer.step()
        reference_other.step()

        for name, param in params.items():
            torch.testing.assert_close(
                param.to_local().reshape(-1),
                _reference_shard(param, reference_params[name]),
                atol=0,
                rtol=0,
            )
            if param in optimizer.state:
                momentum = optimizer.state[param]["momentum_buffer"]
                assert isinstance(momentum, DTensor)
                torch.testing.assert_close(
                    momentum.to_local().reshape(-1),
                    _reference_shard(
                        param, reference_optimizer.state[reference_params[name]]["momentum_buffer"]
                    ),
                    atol=0,
                    rtol=0,
                )
        with torch.no_grad():
            for name, param in reference.named_parameters():
                param.copy_(reference_params[name])
        if step == 0:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = FsdpMuon(_parameter_groups(params), dp_mesh=mesh, **kwargs)
            optimizer.load_state_dict(state)


@pytest.mark.parametrize("shape", [(8,), (2, 4, 8)])
def test_muon_rejects_non_matrix_parameters(distributed_setup, shape):
    """Reject unsupported dimensions on every rank before owner communication."""
    mesh = init_device_mesh("cuda", (distributed_setup.world_size,))
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(shape, device=distributed_setup.device))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_placements())
    with pytest.raises(ValueError, match="only supports 2-D"):
        FsdpMuon(model.parameters(), dp_mesh=mesh)
